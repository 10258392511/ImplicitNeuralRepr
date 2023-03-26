import matplotlib.pyplot as plt
import torch

from torchmetrics import Metric
from functools import partial
from ImplicitNeuralRepr.utils.utils import dict2str
from typing import Union


def plot_profile(profile_name: str, profile_kwargs: Union[dict, None] = None, num_grid_pts: int = 1000, **kwargs):
    """
    kwargs: figsize_unit
    """
    figsize_unit = kwargs.get("figsize_unit", 3.6)

    if profile_kwargs is None:
        profile_kwargs = {}
    profile = partial(NAME2PROFILE[profile_name], **profile_kwargs)
    xx = torch.linspace(-1, 1, num_grid_pts)
    weights = profile(xx)
    fig, axis = plt.subplots(figsize=(figsize_unit, figsize_unit))
    axis.plot(xx, weights)
    axis.grid(axis="y")
    axis.set_xlim((-1, 1))
    axis.set_ylim((0, 1))
    title_dict = {
        "name": profile_name,
        **profile_kwargs
    }
    axis.set_title(dict2str(title_dict))

    return fig


def profile_linear(coord: torch.Tensor, k: float = 0.5):
    """
    g(x) = -k * (x + 1) + 1  (i.e. g(-1) = 1)
    
    coord: any shape but with value between -1 and 1
    """
    assert torch.all(torch.logical_and(-1 <= coord, coord <= 1))
    assert k <= 0.5 , "weights can't be negative"
    weights = -k * (coord + 1) + 1

    return weights


def profile_log(coord: torch.Tensor, alpha: float = 1.):
    """
    g(x) = exp(-alpha * (x + 1))  (i.e. g(-1) = 1)
    
    coord: any shape but with value between -1 and 1
    """
    assert torch.all(torch.logical_and(-1 <= coord, coord <= 1))
    weights = torch.exp(-alpha * (coord + 1))

    return weights


NAME2PROFILE = {
    "linear": profile_linear,
    "log": profile_log
}


class RegProfileLoss(object):
    """
    reg_profile_loss = lamda * norm(X - X_tgt)
    """
    def __init__(self, profile_name: str, norm_order: Union[int, str] = 2, profile_kwargs: Union[dict, None] = None):
        if profile_kwargs is None:
            profile_kwargs = {}
        self.profile = partial(NAME2PROFILE[profile_name], **profile_kwargs)
        self.norm_order = norm_order
    
    def __call__(self, X: torch.Tensor, coord: torch.Tensor, lamda: float = 1.):
        """ 
        X: (B, Lambda) 
        coord: (B, Lambda), i.e. extracted from e.g. (lamda, t, y, x)
        """
        weights = self.profile(coord)  # (B, Lambda)
        coord_min, coord_min_inds = coord.min(dim=-1, keepdim=True)  # (B, 1)
        assert torch.allclose(coord_min, torch.tensor(-1.).to(coord_min.device))
        X_tgt = X[coord_min_inds].detach() * weights
        loss = lamda * torch.norm(X - X_tgt, self.norm_order) ** self.norm_order

        return loss


class RegProfileLossMetric(Metric):
    """
    For logging.
    """
    def __init__(self, profile_name: str, norm_order: Union[int, str] = 1, profile_kwargs: Union[dict, None] = None):
        super().__init__()
        self.loss_call = RegProfileLoss(profile_name, norm_order, profile_kwargs)
        self.add_state("loss", default=torch.tensor(0.), dist_reduce_fx="sum")
    
    def update(self, X: torch.Tensor, coord: torch.Tensor, lamda: float = 1.):
        loss = self.loss_call(X, coord, lamda)
        self.loss += loss
    
    def compute(self):
        
        return self.loss
