import torch
import torch.nn as nn
import torch.nn.functional as F

from pytorch_lightning import LightningModule
from ImplicitNeuralRepr.datasets import val2idx
from ImplicitNeuralRepr.linear_transforms import LinearTransform, FiniteDiff, load_linear_transform
from ImplicitNeuralRepr.objectives.data_consistency import DCLoss, DCLossMetric
from ImplicitNeuralRepr.objectives.regularization import RegLoss, RegLossMetric
from torchmetrics import MetricCollection
from .utils import load_optimizer
from collections import defaultdict
from einops import rearrange
from typing import List


class TrainSpatial(LightningModule):
    def __init__(self, model: nn.Module, measurement: torch.Tensor, lin_tfm: LinearTransform,  params: dict):
        """
        Using spatial gradient (TV) regularization.

        params: lr, lamda_reg, if_pred_res, step_size (StepLR), gamma (StepLR)
        logging tags: dc_loss, reg_loss, loss
        """
        super().__init__()
        self.params = params
        self.model = model
        self.measurement = measurement  # img: (H, W)
        self.lin_tfm = lin_tfm
        self.zf = lin_tfm.conj_op(self.measurement)  # (H, W)
        self.step_outputs = defaultdict(list)  # keys: train
        self.dc_loss = DCLoss(self.lin_tfm)
        self.reg_loss_x = RegLoss(FiniteDiff(-1))
        self.reg_loss_y = RegLoss(FiniteDiff(-2))
    
    def collate_pred(self, pred):
        if self.params["if_pred_res"]:
            pred = pred + self.zf

        return pred
        
    def training_step(self, batch, batch_idx):
        """
        batch: ((B, H * W, 3), (B, 1)), with B = 1
        """
        H, W = self.zf.shape
        x, mask = batch
        x = x[..., 1:]  # remove axis 0 (T) 
        x = x[mask[:, 0], ...]  # (B', H * W, 2)
        x = rearrange(x, "B (H W) D -> B H W D", H=H)  # (B, H, W, 2)
        # Note we only have one image and we don't take slices, so there's no need to call val2idx(.)
        pred_res = self.model(x).squeeze(-1)  # (B, H, W, 1) -> (B, H, W)
        pred = self.collate_pred(pred_res)  # (B, H, W)
        dc_loss = self.dc_loss(pred, self.measurement)
        reg_loss = self.reg_loss_x(pred, self.params["lamda_reg"]) + self.reg_loss_y(pred, self.params["lamda_reg"])
        loss = dc_loss + reg_loss

        # logging
        log_dict = {
            "dc_loss": dc_loss,
            "reg_loss": reg_loss,
            "loss": loss
        }
        self.log_dict(log_dict, prog_bar=True, on_step=True)
        log_dict.update({
            "num_samples": x.shape[0]
        })
        self.step_outputs["train"].append(log_dict)

        return loss
    
    def on_training_epoch_end(self):
        mode = "train"
        log_dict = defaultdict(list)
        num_samples = 0
        for log_dict_iter in self.step_outputs[mode]:
            num_samples_iter = log_dict_iter.pop("num_samples")
            num_samples += num_samples_iter
            for key in log_dict_iter:
                log_dict[key].append(log_dict_iter[key] * num_samples_iter)
        
        for key in log_dict:
            log_dict[key] = sum(log_dict[key]) / num_samples
        
        self.log_dict(log_dict,prog_bar=True, on_epoch=True)
        self.step_outputs[mode] = []
    
    def predict_step(self, batch, batch_idx):
        """
        batch: (B, H * W, 3), with B = 1
        """
        H, W = self.zf.shape
        x = batch[..., 1:]  # remove axis 0 (T) 
        x = rearrange(x, "B (H W) D -> B H W D", H=H)  # (B, H, W, 2)
        # Note we only have one image and we don't take slices, so there's no need to call val2idx(.)
        pred_res = self.model(x).squeeze(-1)  # (B, H, W, 1) -> (B, H, W)
        pred = self.collate_pred(pred_res)  # (B, H, W)

        return pred

    def configure_optimizers(self):
        opt = torch.optim.AdamW(self.model.parameters(), lr=self.params["lr"])
        scheduler = torch.optim.lr_scheduler.StepLR(opt, step_size=self.params["step_size"], gamma=self.params["gamma"])
        opt_config = {
            "optimizer": opt,
            "lr_scheduler": {
                "scheduler": scheduler
            }
        }

        return opt_config


class Train2DTime(LightningModule):
    def __init__(self, siren: nn.Module, grid_sample: nn.Module, measurement: torch.Tensor, config: dict,  params: dict):
        """
        params: lamda_reg, siren_weight, grid_sample_weight
        """
        super().__init__()
        self.params = params
        self.config = config
        self.in_shape = config["transforms"]["dc"]["in_shape"]  # (T, H, W)
        self.siren = siren
        self.grid_sample = grid_sample
        self.measurement = measurement  # img: (T, H, W)
        dc_lin_tfm = load_linear_transform("2d+time", "dc")
        self.dc_loss = DCLoss(dc_lin_tfm)
        reg_lin_tfm = load_linear_transform("2d+time", "reg")
        self.reg_loss = RegLoss(reg_lin_tfm)
        self.dc_metric, self.reg_metric = DCLossMetric(dc_lin_tfm), RegLossMetric(reg_lin_tfm) 
        # self.loss_metric = self.dc_metric + self.reg_metric  # can't do this due to different pred and label
        # self.metrics = MetricCollection({
        #     "loss_metric": loss_metric,
        #     "dc_metric": dc_metric,
        #     "reg_metric": reg_metric 
        # })
    
    def collate_pred(self, pred_siren: torch.Tensor, pred_grid_sample: torch.Tensor) -> torch.Tensor:
        pred = pred_siren * self.params["siren_weight"] + pred_grid_sample * self.params["grid_sample_weight"]
        
        return pred 
    
    def training_step(self, batch, batch_idx, optimizer_idx):
        """
        batch: (B, H * W, 3), (B, T, 3), (B, 2); coord: (t, y, x)
        """
        T, H, W = self.in_shape
        x_s, x_t, mask = batch
        dc_loss, reg_loss = 0., 0.

        x_s = x_s[mask[:, 0], ...]  # (B', H * W, 3)
        if x_s.shape[0] > 0:
            x_s = rearrange(x_s, "B (H W) D -> B H W D", H=H)  # (B', H, W, 3)
            pred_siren = self.siren(x_s).squeeze(-1)  # (B', H, W)
            pred_grid_sample = self.grid_sample(x_s).squeeze(0)  # (1, B', H, W) -> (B', H, W)
            pred_s = self.collate_pred(pred_siren, pred_grid_sample)  # (B', H, W)

            # x_s: (B', H, W, 3); # coord order: (t, y, x)
            t_inds = torch.round((x_s[:, 0, 0, 0] + 1) / 2 * (T - 1))  # (B,); each sample of shape (H, W, 3) has the same t
            t_inds = t_inds.long()
            measurement_t_slices = self.measurement[..., t_inds, :, :, :]  # .measurement: (..., T, H, W, num_sens) -> (..., B', H, W, num_sens)

            dc_loss = self.dc_loss(pred_s, measurement_t_slices, t_inds)
            dc_metric = self.dc_metric(pred_s, measurement_t_slices, t_inds)

        x_t = x_t[mask[:, 1], ...]  # (B', T, 3)
        if x_t.shape[0] > 0:
            pred_siren = self.siren(x_t).squeeze(-1)  # (B', T)
            x_t = rearrange(x_t, "(H W) T D -> T H W D", H=1)  # dummy spatial shape: (T, 1, B', 3); all coord (resampling) info has been encoded in the last dim
            pred_grid_sample = self.grid_sample(x_t)  # (1, T, 1, B')
            pred_grid_sample = rearrange(pred_grid_sample, "C T H W -> (C H W) T", C=pred_grid_sample.shape[0])  # (B', T)
            pred_t = self.collate_pred(pred_siren, pred_grid_sample)
        
            reg_loss = self.reg_loss(pred_t, self.params["lamda_reg"])
            reg_metric = self.reg_metric(pred_t, self.params["lamda_reg"])

        loss = reg_loss
        if dc_loss > 0:
            loss = loss + dc_loss

        # logging
        log_dict = {
            "loss": loss,
            "dc_loss": dc_loss,
            "reg_loss": reg_loss
        }
        self.log_dict(log_dict, prog_bar=True, on_step=True)

        return loss

    def on_train_epoch_end(self) -> None:
        dc_loss = self.dc_metric.compute()
        reg_loss = self.reg_metric.compute()
        loss = dc_loss + reg_loss
        log_dict = {
            "epoch_loss": loss,
            "epoch_dc_loss": dc_loss,
            "epoch_reg_loss": reg_loss
        }
        self.log_dict(log_dict, prog_bar=True, on_epoch=True)

        self.dc_metric.reset()
        self.reg_metric.reset()

    def predict_step(self, batch, batch_idx):
        # batch: (B, H * W, 3)
        T, H, W = self.in_shape
        x_s = batch
        x_s = rearrange(x_s, "B (H W) D -> B H W D", H=H)  # (B', H, W, 3)
        pred_siren = self.siren(x_s).squeeze(-1)  # (B', H, W)
        pred_grid_sample = self.grid_sample(x_s).squeeze(0)  # (1, B', H, W) -> (B', H, W)
        pred_s = self.collate_pred(pred_siren, pred_grid_sample)  # (B', H, W)

        return pred_s
    
    @staticmethod
    def pred2vol(preds: List[torch.Tensor]) -> torch.Tensor:
        # >= 2 dimensions for indices: use coord_ds.idx2sub(.); 2D + time: only 1-dim index (T)
        # preds: list[(B', H, W)]
        pred = torch.cat(preds, dim=0)  # (T, H, W)

        return pred
    
    def configure_optimizers(self):
        opt_siren, scheduler_siren = load_optimizer(self.config["optimization"]["siren"], self.siren)
        opt_grid_sample, scheduler_grid_sample = load_optimizer(self.config["optimization"]["grid_sample"], self.grid_sample)
        opt_siren_dict = {
            "optimizer": opt_siren
        }
        if scheduler_siren is not None:
            opt_siren_dict.update({
                "lr_scheduler": scheduler_siren
            })
        
        opt_grid_sample_dict = {
            "optimizer": opt_grid_sample
        }
        if scheduler_grid_sample is not None:
            opt_grid_sample_dict.update({
                "lr_scheduler": scheduler_grid_sample
            })
        
        return opt_siren_dict, opt_grid_sample_dict


class Train2DTimeReg(LightningModule):
    pass
