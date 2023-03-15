import torch
import torch.nn as nn
import torch.nn.functional as F

from pytorch_lightning import LightningModule
from ImplicitNeuralRepr.linear_transforms import LinearTransform, FiniteDiff
from ImplicitNeuralRepr.objectives.data_consistency import DCLoss
from ImplicitNeuralRepr.objectives.regularization import RegLoss
from collections import defaultdict
from einops import rearrange


class TrainSpatial(LightningModule):
    def __init__(self, model: nn.Module, measurement: torch.Tensor, lin_tfm: LinearTransform,  params: dict):
        """
        Using spatial gradient (TV) regularization.

        params: lr, lamda_reg
        logging tags: dc_loss, reg_loss, loss
        """
        super().__init__()
        self.params = params
        self.model = model
        self.measurement = measurement  # img: (H, W)
        self.lin_tfm = lin_tfm
        self.zf = lin_tfm.conj_op(self.measurement)  # (H, W)
        self.zf = rearrange(self.zf, "(B C H) W -> B C H W", B=1, C=1)  # (1, 1, H, W), , to facilitate grid_sample(.)
        self.step_outputs = defaultdict(list)  # keys: train
        self.dc_loss = DCLoss(self.lin_tfm)
        self.reg_loss_x = RegLoss(FiniteDiff(-1))
        self.reg_loss_y = RegLoss(FiniteDiff(-2))
    
    def training_step(self, batch, batch_idx):
        """
        batch: ((B, H * W, 2), (B, 1))
        """
        _, H, W = self.zf.shape
        x, mask = batch
        x = x[mask[:, 0]]  # (B', H * W, 2)
        x = rearrange(x, "B (H W) D -> B H W D")  # (B, H, W, 2)




    def training_epoch_end(self):
        pass

    def configure_optimizers(self):
        opt = torch.optim.AdamW(self.model.parameters(), lr=self.params["lr"])
        opt_config = {
            "optimizer": opt
        }

        return opt_config
