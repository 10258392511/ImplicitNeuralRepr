import torch

from torchmetrics import Metric
from ImplicitNeuralRepr.linear_transforms import LinearTransform
from typing import Union


class DCLoss(object):
    """
    dc_loss = 0.5 * ||AX - S||_2^2
    """
    def __init__(self, lin_tfm: LinearTransform):
        self.lin_tfm = lin_tfm
    
    def __call__(self, X: torch.Tensor, S: torch.Tensor, t_indices: Union[torch.Tensor, None] = None):
        S_pred = self.lin_tfm(X, t_indices)
        loss = 0.5 * torch.norm(S_pred - S) ** 2
        
        return loss


class DCLossMetric(Metric):
    """
    For logging.
    """
    def __init__(self, lin_tfm: LinearTransform):
        super().__init__()
        self.lin_tfm = lin_tfm
        self.add_state("loss", default=torch.tensor(0.), dist_reduce_fx="sum")
    
    def update(self, X: torch.Tensor, S: torch.Tensor, t_indices: Union[torch.Tensor, None] = None):
        S_pred = self.lin_tfm(X, t_indices)
        loss = 0.5 * torch.norm(S_pred - S) ** 2
        self.loss += loss  # reduce: sum in DCLoss

    def compute(self):  

        return self.loss
