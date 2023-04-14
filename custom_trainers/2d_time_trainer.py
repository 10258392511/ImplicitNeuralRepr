import numpy as np
import torch
import torch.nn as nn
import os
import ImplicitNeuralRepr.utils.pytorch_utils as ptu

from .base_trainer import BaseTrainer
from torch.utils.data import DataLoader
from ImplicitNeuralRepr.linear_transforms import LinearTransform, FiniteDiff, load_linear_transform
from ImplicitNeuralRepr.objectives.data_consistency import DCLoss, DCLossMetric
from ImplicitNeuralRepr.objectives.regularization import RegLoss, RegLossMetric
from ImplicitNeuralRepr.pl_modules.utils import load_optimizer
from ImplicitNeuralRepr.utils.utils import (
    save_vol_as_gif
)
from typing import Dict, Any


class SpatialTemporalTrainer(BaseTrainer):
    def __init__(self, params: dict, model_dict: Dict[str, nn.Module], config: dict, loader_dict: Dict[str, DataLoader], measurement: torch.Tensor):
        """
        params: num_epochs, if_notebook, log_dir, lam_reg, siren_weight, grid_sample_weight, log_dir, eval_interval, output_dir
        loader_dict: keys: spatial, temporal, spatial_for_pred
        model_dict: keys: siren, grid_sample
        logger tags: dc_loss, reg_loss, loss, epoch_dc_loss, epoch_reg_loss, epoch_loss
        """
        super().__init__(params, model_dict, config)
        self.loader_dict = loader_dict
        self.in_shape = config["transforms"]["dc"]["in_shape"]  # (T, H, W)
        self.T, self.H, self.W = self.in_shape
        self.siren = model_dict["siren"]
        self.grid_sample = model_dict["grid_sample"]
        self.measurement = measurement  # img: (T, H, W)
        dc_lin_tfm = load_linear_transform("2d+time", "dc")
        self.dc_loss = DCLoss(dc_lin_tfm)
        reg_lin_tfm = load_linear_transform("2d+time", "reg")
        self.reg_loss = RegLoss(reg_lin_tfm)
        self.dc_metric, self.reg_metric = DCLossMetric(dc_lin_tfm), RegLossMetric(reg_lin_tfm)
        self.global_steps = {"train": 0}
        self.callback_states["best_metric"] = float("inf")
    
    def train_epoch(self, *args, **kwargs) -> dict:
        pbar = tqdm(zip(*self.loader_dict.values()), total=len(self.loader_dict["spatial"]), leave=False)
        for spatial_batch, temporal_batch in pbar:
            dc_loss = self.__dc_step(spatial_batch)
            reg_loss = self.__reg_step(temporal_batch)
            loss = dc_loss + reg_loss
            for opt_config_iter in self.opt_configs:
                opt_config_iter["optimizer"].zero_grad()
            loss.backward()
            for opt_config_iter in self.opt_configs:
                opt_config_iter["optimizer"].step()
            
            # logging
            log_dict_step = {
                "dc_loss": dc_loss.item(),
                "reg_loss": reg_loss.item(),
                "loss": loss.item()
            }
            for tag, val in log_dict_step.items():
                self.logger.add_scalar(tag, val, self.global_steps["train"])
            self.global_steps["train"] += 1
            desc = self.dict2str(log_dict_step)
            pbar.set_description(desc)
        
        # on_epoch_end:
        dc_loss = self.dc_metric.compute()
        reg_loss = self.reg_metric.compute()
        loss = dc_loss + reg_loss
        log_dict = {
            "epoch_loss": loss,
            "epoch_dc_loss": dc_loss,
            "epoch_reg_loss": reg_loss
        }

        self.dc_metric.reset()
        self.reg_metric.reset()

        return log_dict

    def __dc_step(self, batch, if_pred=False):
        """
        Shared with .predict(.)
        """
        x_coord = batch
        x_coord = x_coord.float()  # (B, H, W, 3)
        t_vals = x_coord[:, 0, 0, 0]  # (B,)
        t_inds = (t_vals + 1) / 2 * (self.T - 1)
        t_inds = t_inds.long()
        s_gt = self.measurement[..., t_inds, ...]  # (..., T, H, W, num_sens) -> (..., B', H, W, num_sens)
        
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
        reg_loss = self.reg_loss(img, self.params["lam_reg"])
        self.reg_metric(img, self.params["lam_reg"])

        return reg_loss

    @torch.no_grad()
    def callbacks(self) -> None:
        super().callbacks()
        self.__save_model_ckpt_callback()
        self.__save_screenshot_callback()
    
    def __save_model_ckpt_callback(self):
        eval_metric_name = "epoch_loss"
        epoch_metrics = self.callback_states["epoch_metrics"]
        metric_candidate = epoch_metrics[eval_metric_name]
        if metric_candidate < self.callback_states["best_metric"]:
            self.callback_states["best_metric"] = metric_candidate
            self.save_models(os.path.join(self.params["log_dir"], self.time_stamp, "ckpt.pt"))

    def __save_screenshot_callback(self):
        if not (self.callback_states["epoch"] % self.params["eval_interval"] or \
            self.callback_states["epoch"] == self.params["num_epochs"] - 1):
            return
        pred = self.predict()  # (T, H, W)
        pred = pred.unsqueeze(1)  # (T, 1, H, W)
        output_dir = os.path.join(self.params["output_dir"], "screenshots")
        torch.save(pred.detach().cpu(), os.path.join(output_dir, f"recons_{self.callback_states['epoch']}.pt"))
        save_vol_as_gif(torch.abs(pred), save_dir=output_dir, filename=f"mag_{self.callback_states['epoch']}.gif")
        save_vol_as_gif(torch.angle(pred), save_dir=output_dir, filename=f"phase_{self.callback_states['epoch']}.gif")

        low_res = self.grid_sample.get_low_res().detach().cpu()  # (1, 1, T0, H0, W0)
        low_res = low_res.squeeze().unsqueeze(1)  # (T0, 1, H0, W0)
        duration = pred.shape[0]  # set duration equal to "pred"
        save_vol_as_gif(torch.abs(low_res), save_dir=output_dir, filename=f"mag_low_res_{self.callback_states['epoch']}.gif", duration=duration)
        save_vol_as_gif(torch.angle(low_res), save_dir=output_dir, filename=f"phase_low_res_{self.callback_states['epoch']}.gif", duration=duration)
    
    @torch.no_grad()
    def predict(self, *args, **kwargs) -> Any:
        super().predict(*args, **kwargs)
        preds = []
        pred_loader = self.loader_dict["spatial_for_pred"]
        pbar = tqdm(pred_loader, total=len(pred_loader), leave=False)
        for batch in pbar:
            pred_iter = self.__dc_step(batch, if_pred=True)
            preds.append(pred_iter)  # list[(B, H, W)]
        preds = torch.cat(preds, axis=0)  # (T, H, W)

        return preds
        
    def configure_optimizers(self) -> Dict[str, dict]:
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
        
        opt_dict_all = {
            self.siren: opt_siren_dict,
            self.grid_sample: opt_grid_sample_dict
        }
        
        return opt_dict_all
