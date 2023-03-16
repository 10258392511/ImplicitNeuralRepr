import torch
import torch.nn as nn
import torch.nn.functional as F

from pytorch_lightning import LightningModule
from ImplicitNeuralRepr.datasets import val2idx
from ImplicitNeuralRepr.linear_transforms import LinearTransform, FiniteDiff
from ImplicitNeuralRepr.objectives.data_consistency import DCLoss
from ImplicitNeuralRepr.objectives.regularization import RegLoss
from collections import defaultdict
from einops import rearrange


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
