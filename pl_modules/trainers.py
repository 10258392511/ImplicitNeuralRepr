import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import os
import pickle

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
    save_vol_as_gif
)
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
        self.T, self.H, self.W = self.in_shape
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
        # print(f"batch {batch_idx}: x_s {type(x_s)}, {x_s[0, 0, 0, :]}, x_t: {type(x_t)}, dc_loss: {dc_loss.item()}")

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
    def __init__(self, siren: nn.Module, grid_sample: nn.Module, measurement: torch.Tensor, config: dict,  params: dict):
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
        dc_lin_tfm = load_linear_transform("2d+time", "dc")
        self.dc_loss = DCLoss(dc_lin_tfm)
        reg_lin_tfm = load_linear_transform("2d+time", "reg")
        self.reg_loss = RegLoss(reg_lin_tfm)
        self.dc_metric, self.reg_metric = DCLossMetric(dc_lin_tfm), RegLossMetric(reg_lin_tfm)

    def __dc_step(self, batch, if_pred=False):
        if if_pred:
            x_s = batch
        else:
            x_s, lam_s = batch  # (B, H, W, 4), (B,); (lam, t, y, x)
        t_vals = x_s[:, 0, 0, 1]  # (B,)
        t_inds = (t_vals + 1) / 2 * (self.T - 1)
        t_inds = t_inds.long()
        s_gt = self.measurement[..., t_inds, :, :, :]  # (..., T, H, W, num_sens) -> (..., B, H, W, num_sens)

        siren_img = self.siren(x_s).squeeze(-1)  # (B, H, W)
        x_s_grid_sample_in = x_s.unsqueeze(1)  # (B, T=1, H, W, 4)
        grid_sample_img = self.grid_sample(x_s_grid_sample_in[..., 1:]).reshape(siren_img.shape)  # (B, C=1, T=1, H, W) -> (B, H, W)
        img = siren_img * self.params["siren_weight"] + grid_sample_img * self.params["grid_sample_weight"]
        if if_pred:
            return img.detach()
        dc_loss = self.dc_loss(img, s_gt, t_inds)
        self.dc_metric(img, s_gt, t_inds)

        return dc_loss

    def __reg_step(self, batch):
        x_t, lam_t = batch  # (B, T, 4), (B,)

        siren_img = self.siren(x_t).squeeze(-1)  # (B, T)
        B, T, D = x_t.shape
        x_t_grid_sample_in = x_t.reshape(B, T, 1, 1, D)  # (B, T, H=1, W=1, 4)
        grid_img = self.grid_sample(x_t_grid_sample_in[..., 1:]).reshape(siren_img.shape)  # (B, C=1, T, H=1, W=1) -> (B, T)
        img = siren_img + grid_img
        lam_t = lam_t.unsqueeze(1)  # (B, 1)
        reg_loss = self.reg_loss(img * lam_t, 1.)
        self.reg_metric(img * lam_t, 1.)

        return reg_loss
    
    def training_step(self, batch, batch_idx, optimizer_idx):
        ### using WrapperDM or simplified_sampling_datasets ###
        batch_s, batch_t = batch
        ######################

        # ### using ZipDM ###
        # batch_s, batch_t, batch_lam, mask = batch  # (B, H, W, 4), (B, T, 4), (B,), (B,)
        # batch_s = (batch_s[mask], None)
        # batch_t = (batch_t, batch_lam)
        # ###################

        ### using WrapperDM ###
        dc_loss = self.__dc_step(batch_s)
        ######################

        # ### using ZipDM ###
        # if not torch.any(mask):
        #     dc_loss = 0
        # else:
        #     dc_loss = self.__dc_step(batch_s)
        # ##################
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
