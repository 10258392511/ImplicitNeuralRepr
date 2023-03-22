import torch

from torchmetrics import Metric
from ImplicitNeuralRepr.linear_transforms import LinearTransform
from typing import Union


class RegLoss(object):
    """
    reg_loss = lamda * norm(lin_tfm(X)) 
    """
    def __init__(self, lin_tfm: LinearTransform, norm_order: Union[int, str] = 1):
        self.lin_tfm = lin_tfm
        self.norm_order = norm_order
    
    def __call__(self, X: torch.Tensor, lamda: float = 1.):
        loss = lamda * (torch.norm(self.lin_tfm(X), self.norm_order) ** self.norm_order)

        return loss


class RegLossMetric(Metric):
    """
    For logging.
    """
    def __init__(self, lin_tfm: LinearTransform, norm_order: Union[int, str] = 1):
        super().__init__()
        self.lin_tfm = lin_tfm
        self.norm_order = norm_order
        self.add_state("loss", default=torch.tensor(0.), dist_reduce_fx="sum")
    
    def update(self, X: torch.Tensor, lamda: float = 1):
        loss = lamda * (torch.norm(self.lin_tfm(X), self.norm_order) ** self.norm_order)
        self.loss += loss  # reduce: sum in RegLoss

    def compute(self):  
              
        return self.loss
