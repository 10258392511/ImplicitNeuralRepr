import torch

from ImplicitNeuralRepr.linear_transforms import LinearTransform


class DCLoss(object):
    """
    dc_loss = 0.5 * ||AX - S||_2^2
    """
    def __init__(self, lin_tfm: LinearTransform):
        self.lin_tfm = lin_tfm
    
    def __call__(self, X: torch.Tensor, S: torch.Tensor):
        S_pred = self.lin_tfm(X)
        loss = 0.5 * torch.norm(S_pred - S) ** 2
        
        return loss
