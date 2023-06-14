import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import os
import pickle
import ImplicitNeuralRepr.utils.pytorch_utils as ptu

from pytorch_lightning import LightningModule
from ImplicitNeuralRepr.datasets import val2idx
from ImplicitNeuralRepr.linear_transforms import LinearTransform, FiniteDiff, load_linear_transform
from ImplicitNeuralRepr.objectives.data_consistency import DCLoss, DCLossMetric
from ImplicitNeuralRepr.objectives.regularization import RegLoss, RegLossMetric
from ImplicitNeuralRepr.objectives.regularization_profile import RegProfileLoss, RegProfileLossMetric
from torchmetrics import MetricCollection
from .utils import (
    load_optimizer,
    load_reg_profile
)
from ImplicitNeuralRepr.utils.utils import (
    save_vol_as_gif,
    expand_dim_as
)
from ImplicitNeuralRepr.models.liif import sliding_window_inference
from collections import defaultdict
from einops import rearrange
from typing import Any, List, Optional, Union
from pytorch_lightning.utilities.types import STEP_OUTPUT
from ImplicitNeuralRepr.configs import IMAGE_KEY, MEASUREMENT_KEY, ZF_KEY, COORD_KEY


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
    def __init__(self, siren: nn.Module, grid_sample: nn.Module, measurement: torch.Tensor, config: dict,  params: dict, ZF: Union[None, torch.Tensor] = None):
        """
        params: lamda_reg, siren_weight, grid_sample_weight
        """
        super().__init__()
        self.params = params
        self.config = config
        self.in_shape = config["transforms"]["dc"]["in_shape"]  # (T, H, W)
        self.T, self.H, self.W = self.in_shape
        self.siren = siren
        self.grid_sample = grid_sample
        self.measurement = measurement  # img: (T, H, W)
        self.ZF = ZF  # (T, H, W)
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
    
    def __dc_step(self, batch, if_pred=False):
        """
        Shared with .predict(.)
        """
        x_coord = batch
        # x_coord = x_coord.float()  # (B, H, W, 3)
        t_vals = x_coord[:, 0, 0, 0]  # (B,)
        t_inds = (t_vals + 1) / 2 * (self.T - 1)
        t_inds = t_inds.long()
        s_gt = self.measurement[..., t_inds, :, :, :]  # (..., T, H, W, num_sens) -> (..., B, H, W, num_sens)
        
        siren_img = self.siren(x_coord).squeeze(-1)  # (B, H, W)
        x_coord_grid_sample_in = x_coord.unsqueeze(1)  # (B, T=1, H, W, 3)
        grid_sample_img = self.grid_sample(x_coord_grid_sample_in).reshape(*siren_img.shape)  # (B, C=1, T=1, H, W) -> (B, H, W)
        img = siren_img * self.params["siren_weight"] + grid_sample_img * self.params["grid_sample_weight"]
        
        if self.ZF is not None:
            img = img + self.ZF[t_inds, ...]
        
        if if_pred:
            # (B, H, W)
            return img.detach()
        dc_loss = self.dc_loss(img, s_gt, t_inds)
        self.dc_metric(img, s_gt, t_inds)

        return dc_loss
    
    def __reg_step(self, batch):
        x_coord = batch
        x_coord = x_coord.float()  # (B, T, 3)
        
        siren_img = self.siren(x_coord).squeeze(-1)  # (B, T)
        B, T, D = x_coord.shape
        x_coord_grid_sample_in = x_coord.reshape(B, T, 1, 1, D)  # (B, T, H=1, W=1, 3)
        grid_sample_img = self.grid_sample(x_coord_grid_sample_in).reshape(siren_img.shape)  # (B, C=1, T, H=1, W=1) -> (B, T)
        img = siren_img * self.params["siren_weight"] + grid_sample_img * self.params["grid_sample_weight"]
        
        if self.ZF is not None:
            y_vals, x_vals = x_coord[:, 0, 1], x_coord[:, 0, 2]  # (B,) each
            y_inds = (y_vals + 1) / 2 * (self.H - 1)
            y_inds = y_inds.long()
            x_inds = (x_vals + 1) / 2 * (self.W - 1)
            x_inds = x_inds.long()
            img = img + self.ZF[:, y_inds, x_inds].T  # (T, B) -> (B, T)
        
        reg_loss = self.reg_loss(img, self.params["lamda_reg"])
        self.reg_metric(img, self.params["lamda_reg"])

        return reg_loss
    
    def training_step(self, batch, batch_idx, optimizer_idx):
        """
        batch: (B1, H, W, 3), (B2, T, 3)
        """
        x_s, x_t = batch
        dc_loss = self.__dc_step(x_s)
        reg_loss = self.__reg_step(x_t)
        loss = dc_loss + reg_loss

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
        # batch: (B, H, W, 3)
        pred_s = self.__dc_step(batch, if_pred=True)  # (B, H, W)

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
    def __init__(self, siren: nn.Module, measurement: torch.Tensor, config: dict,  params: dict):
        """
        params: lamda_reg_profile
        """
        super().__init__()
        self.params = params
        self.config = config
        self.in_shape = config["transforms"]["dc"]["in_shape"]  # (Lambda, T, H, W)
        self.siren = siren
        self.measurement = measurement  # img: (T, H, W)
        dc_lin_tfm = load_linear_transform("2d+time+reg", "dc")
        self.dc_loss = DCLoss(dc_lin_tfm)
        reg_lin_tfm = load_linear_transform("2d+time+reg", "reg")
        self.reg_loss = RegLoss(reg_lin_tfm)
        profile_name, profile_params = load_reg_profile(config["transforms"]["reg_profile"])
        self.reg_profile_loss = RegProfileLoss(profile_name, profile_kwargs=profile_params)
        self.dc_metric = DCLossMetric(dc_lin_tfm)
        self.reg_metric = RegLossMetric(reg_lin_tfm)
        self.reg_profile_metric = RegProfileLossMetric(profile_name, profile_kwargs=profile_params)
    
    def training_step(self, batch, batch_idx):
        """
        batch: (B, H * W, 4), (B, T, 4), (B, Lambda, 4); coord: (lamda, t, y, x)
        """
        Lambda, T, H, W = self.in_shape
        x_s, x_t, x_reg, mask = batch
        dc_loss, reg_loss, reg_profile_loss = 0., 0., 0.

        x_s = x_s[mask[:, 0], ...]  # (B', H * W, 4)
        if x_s.shape[0] > 0:
            x_s = rearrange(x_s, "B (H W) D -> B H W D", H=H)  # (B', H, W, 4)
            pred_siren = self.siren(x_s).squeeze(-1)  # (B', H, W)
            pred_s = pred_siren

            # x_s: (B', H, W, 4); # coord order: (lamda, t, y, x)
            t_inds = torch.round((x_s[:, 0, 0, 1] + 1) / 2 * (T - 1))  # (B,); each sample of shape (H, W, 4) has the same t (repeated for different lamda's)
            t_inds = t_inds.long()
            measurement_t_slices = self.measurement[..., t_inds, :, :, :]  # .measurement: (..., T, H, W, num_sens) -> (..., B', H, W, num_sens)

            dc_loss = self.dc_loss(pred_s, measurement_t_slices, t_inds)
            dc_metric = self.dc_metric(pred_s, measurement_t_slices, t_inds)

        x_t = x_t[mask[:, 1], ...]  # (B', T, 4)
        if x_t.shape[0] > 0:
            pred_siren = self.siren(x_t).squeeze(-1)  # (B', T)
            pred_t = pred_siren
        
            reg_loss = self.reg_loss(pred_t, 1.)
            reg_metric = self.reg_metric(pred_t, 1.)
        
        x_reg = x_reg[mask[:, 2], ...]  # (B', Lambda, 4)
        if x_reg.shape[0] > 0:
            pred_siren = self.siren(x_reg).squeeze(-1)  # (B', Lambda)
            pred_reg = pred_siren

            reg_profile_loss = self.reg_profile_loss(pred_reg, x_reg[..., 0], self.params["lamda_reg_profile"])
            reg_profile_metric = self.reg_profile_metric(pred_reg, x_reg[..., 0], self.params["lamda_reg_profile"])
        
        loss = reg_profile_loss
        if reg_loss > 0:
            loss = loss + reg_loss
        if dc_loss > 0:
            loss = loss + dc_loss
        
        # logging
        log_dict = {
            "loss": loss,
            "dc_loss": dc_loss,
            "reg_loss": reg_loss,
            "reg_profile_loss": reg_profile_loss
        }
        self.log_dict(log_dict, prog_bar=True, on_step=True)

        return loss

    def on_train_epoch_end(self) -> None:
        dc_loss = self.dc_metric.compute()
        reg_loss = self.reg_metric.compute()
        reg_profile_loss = self.reg_profile_metric.compute()
        loss = dc_loss + reg_loss + reg_profile_loss
        log_dict = {
            "epoch_loss": loss,
            "epoch_dc_loss": dc_loss,
            "epoch_reg_loss": reg_loss,
            "epoch_reg_profile_loss": reg_profile_loss
        }
        self.log_dict(log_dict, prog_bar=True, on_epoch=True)

        self.dc_metric.reset()
        self.reg_metric.reset()
        self.reg_profile_metric.reset()

    def predict_step(self, batch, batch_idx):
        # batch: (B, H * W, 4)
        Lambda, T, H, W = self.in_shape
        x_s = batch
        x_s = rearrange(x_s, "B (H W) D -> B H W D", H=H)  # (B', H, W, 4)
        pred_siren = self.siren(x_s).squeeze(-1)  # (B', H, W)
        pred_s = pred_siren

        return pred_s

    def pred2vol(self, preds: List[torch.Tensor]) -> torch.Tensor:
        """
        Use coord_ds.idx2sub(.) or just rearrange(.) since no shuffling is applied.
        Note this is a simplified approach to choose reg per voxel, since this method chooses one common reg for the whole volume.

        preds: list[(B', H, W)]
        """
        Lambda, T, H, W = self.in_shape
        preds_out = torch.cat(preds, dim=0)  # ((Lambda * T), H, W)
        preds_out = rearrange(preds_out, "(L T) H W -> L T H W", L=Lambda)

        # (Lambda, T, H, W)
        return preds_out

    @staticmethod
    def save_preds(preds: torch.Tensor, save_dir: str, **kwargs):
        """
        kwargs: extra Tensors to save (e.g. original.pt)
        preds: (Lambda, T, H, W)

        + save_dir
            + lamda_0
                - desc.txt: lamda = -1.
                - original.pt
                - reconstructions.pt
                - recons_mag.gif
                - recons_phase.gif
            ...
        """
        preds = preds.detach().cpu()
        Lambda = preds.shape[0]
        lamda_grid = np.arange(Lambda) / (Lambda - 1)
        lamda_grid = lamda_grid * 2 - 1
        for i in range(Lambda):
            lamda_iter = lamda_grid[i]
            cur_dir = os.path.join(save_dir, f"lamda_{i}")
            if not os.path.isdir(cur_dir):
                os.makedirs(cur_dir)
            desc_dict = {
                "lamda_idx": i,
                "lamda_total": Lambda,
                "lamda": lamda_iter
            }
            with open(os.path.join(cur_dir, "desc.txt"), "w") as wf:
                for key, val in desc_dict.items():
                    wf.write(f"{key}: {val}\n")
            with open(os.path.join(cur_dir, "args_dict.pkl"), "wb") as wf:
                pickle.dump(desc_dict, wf)

            for key, val in kwargs.items():
                assert isinstance(val, torch.Tensor)
                torch.save(val, os.path.join(cur_dir, f"{key}.pt"))
            torch.save(preds[i, ...], os.path.join(cur_dir, "reconstructions.pt"))  # (T, H, W)
            save_vol_as_gif(torch.abs(preds[i, ...].unsqueeze(1)), cur_dir, "recons_mag.gif")  # (T, H, W) -> (T, 1, H, W)
            save_vol_as_gif(torch.angle(preds[i, ...].unsqueeze(1)), cur_dir, "recons_phase.gif")

    def configure_optimizers(self):
        opt_siren, scheduler_siren = load_optimizer(self.config["optimization"]["siren"], self.siren)
        opt_siren_dict = {
            "optimizer": opt_siren
        }
        if scheduler_siren is not None:
            opt_siren_dict.update({
                "lr_scheduler": scheduler_siren
            })

        return opt_siren_dict


class Train2DTimeExplicitReg(LightningModule):
    def __init__(self, siren: nn.Module, grid_sample: nn.Module, measurement: torch.Tensor, config: dict,  params: dict, ZF: Union[None, torch.Tensor] = None):
        """
        params: siren_weight, grid_sample_weight
        """
        super().__init__()
        self.params = params
        self.config = config
        self.in_shape = config["transforms"]["dc"]["in_shape"]  # (T, H, W)
        self.T, self.H, self.W = self.in_shape
        self.siren = siren
        self.grid_sample = grid_sample
        self.measurement = measurement  # img: (T, H, W)
        self.ZF = ZF
        dc_lin_tfm = load_linear_transform("2d+time", "dc")
        self.dc_loss = DCLoss(dc_lin_tfm)
        reg_lin_tfm = load_linear_transform("2d+time", "reg")
        self.reg_loss = RegLoss(reg_lin_tfm)
        self.dc_metric, self.reg_metric = DCLossMetric(dc_lin_tfm), RegLossMetric(reg_lin_tfm)

    def __dc_step(self, batch, if_pred=False):
        if if_pred:
            x_s = batch
        else:
            x_s, lam_s = batch  # (Bs, H, W, 4), (Bs,); (lam, t, y, x)
        t_vals = x_s[:, 0, 0, 1]  # (B,)
        t_inds = (t_vals + 1) / 2 * (self.T - 1)
        t_inds = t_inds.long()
        s_gt = self.measurement[..., t_inds, :, :, :]  # (..., T, H, W, num_sens) -> (..., B, H, W, num_sens)

        siren_img = self.siren(x_s).squeeze(-1)  # (B, H, W)
        x_s_grid_sample_in = x_s.unsqueeze(1)  # (B, T=1, H, W, 4)
        grid_sample_img = self.grid_sample(x_s_grid_sample_in[..., 1:]).reshape(siren_img.shape)  # (B, C=1, T=1, H, W) -> (B, H, W)
        img = siren_img * self.params["siren_weight"] + grid_sample_img * self.params["grid_sample_weight"]

        if self.ZF is not None:
            img = img + self.ZF[t_inds, ...]

        if if_pred:
            return img.detach()
        
        # lam_s_sqrt = torch.sqrt(lam_s)
        # img = img / expand_dim_as(lam_s_sqrt, img)
        # s_gt = s_gt / expand_dim_as(lam_s_sqrt, s_gt)
        dc_loss = self.dc_loss(img, s_gt, t_inds)
        self.dc_metric(img, s_gt, t_inds)

        return dc_loss

    def __reg_step(self, batch):
        x_t, lam_t = batch  # (Bt, T, 4), (Bt,)

        siren_img = self.siren(x_t).squeeze(-1)  # (B, T)
        B, T, D = x_t.shape
        x_t_grid_sample_in = x_t.reshape(B, T, 1, 1, D)  # (B, T, H=1, W=1, 4)
        grid_img = self.grid_sample(x_t_grid_sample_in[..., 1:]).reshape(siren_img.shape)  # (B, C=1, T, H=1, W=1) -> (B, T)
        img = siren_img + grid_img

        if self.ZF is not None:
            y_vals, x_vals = x_t[:, 0, 2], x_t[:, 0, 3]  # (B,) each
            y_inds = (y_vals + 1) / 2 * (self.H - 1)
            y_inds = y_inds.long()
            x_inds = (x_vals + 1) / 2 * (self.W - 1)
            x_inds = x_inds.long()
            img = img + self.ZF[:, y_inds, x_inds].T  # (T, B) -> (B, T)
        
        img = img * expand_dim_as(lam_t, img)
        reg_loss = self.reg_loss(img, 1.)
        self.reg_metric(img, 1.)

        return reg_loss
    
    def training_step(self, batch, batch_idx, optimizer_idx):
        batch_s, batch_t = batch

        dc_loss = self.__dc_step(batch_s)
        reg_loss = self.__reg_step(batch_t)
        loss = dc_loss + reg_loss

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
        # batch: (B, H, W, 4)
        pred_s = self.__dc_step(batch, if_pred=True)  # (B, H, W)

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


class TrainLIIF(LightningModule):
    def __init__(self, model: nn.Module, config: dict, lin_tfm: LinearTransform):
        super().__init__()
        self.model = model
        self.config = config
        self.lin_tfm = lin_tfm

    @staticmethod
    def compute_l1_loss_complex(x_pred: torch.Tensor, x: torch.Tensor, eps=1e-4):
        # x_pred, x: (B, T, H, W)
        num = torch.abs(torch.abs(x_pred) - torch.abs(x)).sum(dim=(1, 2, 3))  # (B,)
        den = torch.abs(x).sum(dim=(1, 2, 3)) + eps  # (B,)
        loss = (num / den).mean()
        
        return loss
    
    def __shared_step(self, batch: Any, batch_idx: int, t_coord: Union[torch.Tensor, None] = None) -> Union[STEP_OUTPUT, None]:
        img, measurement = batch  # (B, T, H, W), (B, T, H, W, num_sens)
        img_zf = self.lin_tfm.conj_op(measurement)
        img_pred = self.model(img_zf, t_coord)  # (B, T, H, W)
        loss = self.compute_l1_loss_complex(img_pred, img)

        return loss, img_pred
    
    def training_step(self, batch: Any, batch_idx: int) -> STEP_OUTPUT:
        loss, _ = self.__shared_step(batch, batch_idx)

        self.log("train_loss", loss, prog_bar=True, on_step=True)
        self.log("epoch_train_loss", loss, prog_bar=True, on_epoch=True)

        return loss
    
    def validation_step(self, batch: Any, batch_idx: int) -> Union[STEP_OUTPUT, None]:
        loss, _ = self.__shared_step(batch, batch_idx)

        self.log("epoch_val_loss", loss, prog_bar=True, on_epoch=True)

        return loss

    def predict_step(self, batch: Any, batch_idx: int, **kwargs) -> Any:
        """
        kwargs: t_coord
        """
        measurement = batch
        t_coord = kwargs.get("t_coord", None)
        img_zf = self.lin_tfm.conj_op(measurement)
        img_pred = self.model(img_zf, t_coord)  # (B, T, H, W)

        return img_pred
    
    def configure_optimizers(self):
        opt, scheduler = load_optimizer(self.config["optimization"], self.model)
        opt_dict = {
            "optimizer": opt
        }
        if scheduler is not None:
            opt_dict.update({
                "lr_scheduler": scheduler
            })
        
        return opt_dict


class TrainLIIF3DConv(LightningModule):
    def __init__(self, model: nn.Module, config: dict):
        super().__init__()
        self.model = model
        self.config = config
        self.finite_diff_t = FiniteDiff(dim=1)  # for (B, T, H, W)

    @staticmethod
    def compute_l1_loss_complex(x_pred: torch.Tensor, x: torch.Tensor, eps=1e-4, if_reduce=True):
        # x_pred, x: (B, T, H, W)
        num = torch.abs(torch.abs(x_pred) - torch.abs(x)).sum(dim=(1, 2, 3))  # (B,)
        den = torch.abs(x).sum(dim=(1, 2, 3)) + eps  # (B,)
        loss = num / den
        if if_reduce:
            loss = loss.mean()
        
        return loss

    def compute_diff_t_loss(self, x_pred: torch.Tensor, x: torch.Tensor, t_coord: torch.Tensor, eps=1e-4, if_reduce=True):
        # x_pred, x: (B, T, H, W); t_coord: (B, T)
        t_coord = t_coord.reshape(*t_coord.shape, 1, 1)
        partial_t = self.finite_diff_t(t_coord)
        partial_x_pred = self.finite_diff_t(torch.abs(x_pred))
        partial_x = self.finite_diff_t(torch.abs(x))
        partial_x_pred_partial_t = partial_x_pred / partial_t
        partial_x_partial_t = partial_x / partial_t
        num = torch.abs(partial_x_pred_partial_t - partial_x_partial_t).sum(dim=(1, 2, 3))  # (B,)
        den = torch.abs(partial_x_partial_t).sum(dim=(1, 2, 3)) + eps  # (B,)
        finite_t_loss = num / den
        finite_t_loss *= self.config["training"]["diff_t_weight"]
        if if_reduce:
            finite_t_loss = finite_t_loss.mean()
        
        return finite_t_loss

    def shared_step(self, batch: Any, batch_idx: int, **kwargs) -> Union[STEP_OUTPUT, None]:
        # for .training_step(.) and .validation_step(.)
        img = batch[IMAGE_KEY]  # (B, T0, H, W)
        img_zf = batch[ZF_KEY]
        t_coord = batch[COORD_KEY]  # (B, T0)
        img_pred = self.model(img_zf, t_coord)  # (B, T0, H, W)
        # img_pred = self.model(**batch)  # (B, T0, H, W)
        if_reduce = kwargs.get("if_reduce", True)
        if_return_zero_order = kwargs.get("if_return_zero_order", False)
        loss_zero_order = self.compute_l1_loss_complex(img_pred, img, if_reduce=if_reduce)
        loss_first_order = self.compute_diff_t_loss(img_pred, img, t_coord, if_reduce=if_reduce)
        loss = loss_zero_order + loss_first_order
        if if_return_zero_order:
            return loss_zero_order, img_pred

        return loss, img_pred
    
    def training_step(self, batch: Any, batch_idx: int) -> STEP_OUTPUT:
        loss, _ = self.shared_step(batch, batch_idx)

        self.log("train_loss", loss, prog_bar=True, on_step=True)
        self.log("epoch_train_loss", loss, prog_bar=True, on_epoch=True)

        return loss
    
    def validation_step(self, batch: Any, batch_idx: int) -> Union[STEP_OUTPUT, None]:
        loss, _ = self.shared_step(batch, batch_idx)

        self.log("epoch_val_loss", loss, prog_bar=True, on_epoch=True)

        return loss

    # def predict_step(self, batch: Any, batch_idx: int, **kwargs) -> Any:
    #     """
    #     kwargs: upsample_rate, roi_size, overlap
    #     """
    #     img = batch[IMAGE_KEY]  # (B, T0, H, W)
    #     img_zf = batch[ZF_KEY]
    #     upsampling_rate = kwargs.get("upsample_rate", 1.)
    #     roi_size = kwargs.get("roi_size", 8)
    #     overlap = kwargs.get("overlap", 0.25)

    #     img_pred = sliding_window_inference(img_zf, upsampling_rate, roi_size, overlap, self.model)  # (B, T, H, W)
    #     error_val = None
    #     if img_pred.shape == img.shape:
    #         error_val = self.compute_l1_loss_complex(img_pred, img, if_reduce=False)
    #         error_val = ptu.to_numpy(error_val)

    #     return img_pred, error_val

    def predict_step(self, batch: Any, batch_idx: int, **kwargs) -> Any:
        error_val, img_pred = self.shared_step(batch, batch_idx, if_reduce=False, if_return_zero_order=True)

        return img_pred, error_val
    
    def configure_optimizers(self):
        opt, scheduler = load_optimizer(self.config["optimization"], self.model)
        opt_dict = {
            "optimizer": opt
        }
        if scheduler is not None:
            opt_dict.update({
                "lr_scheduler": {
                    "scheduler": scheduler
                }
            })
        if isinstance(scheduler, torch.optim.lr_scheduler.ReduceLROnPlateau):
            opt_dict["lr_scheduler"].update({
                "monitor": "epoch_val_loss"
            })
        
        return opt_dict


class TrainTemporalTV(LightningModule):
    def __init__(self, model: nn.Module, config: dict):
        super().__init__()
        self.model = model
        self.config = config

    def training_step(self, batch: Any, batch_idx: int) -> STEP_OUTPUT:
        recons, loss = self.model()
        img = batch[IMAGE_KEY]  # (1, T', H, W)
        _, val_loss = self.upsample_recons_and_compute_error(recons, img)
        
        self.log("train_loss", loss, prog_bar=True, on_step=True)
        self.log("val_loss", val_loss, prog_bar=True, on_step=True)

        return loss
    
    @staticmethod
    def upsample_recons_and_compute_error(recons: torch.Tensor, img: torch.Tensor):
        # recons: (1, T, H, W), img: (1, T', H, W)
        resize_factor = img.shape[1] / recons.shape[1]
        recons = recons.unsqueeze(1)  # (1, 1, T, H, W)
        recons = F.interpolate(torch.abs(recons), scale_factor=(resize_factor, 1, 1), mode="trilinear")  # (1, 1, T', H, W)
        recons = recons.to(torch.complex64).squeeze(1)  # (1, T', H, W)
        rel_mag_error = TrainLIIF3DConv.compute_l1_loss_complex(recons, img)

        # (1, T', H, W)
        return recons, rel_mag_error
    
    # def validation_step(self, batch: Any, batch_idx: int) -> Union[STEP_OUTPUT, None]:
    #     recons, _ = self.model()  # (1, T, H, W)
    #     img = batch[IMAGE_KEY]  # (1, T', H, W)
    #     _, val_loss = self.upsample_recons_and_compute_error(recons, img)
    #     self.log("val_loss", val_loss, prog_bar=True, on_step=True)

    def configure_optimizers(self):
        opt, scheduler = load_optimizer(self.config["optimization"], self.model)
        opt_dict = {
            "optimizer": opt
        }
        if scheduler is not None:
            opt_dict.update({
                "lr_scheduler": {
                    "scheduler": scheduler
                }
            })
        if isinstance(scheduler, torch.optim.lr_scheduler.ReduceLROnPlateau):
            opt_dict["lr_scheduler"].update({
                "monitor": "val_loss"
            })
        
        return opt_dict
