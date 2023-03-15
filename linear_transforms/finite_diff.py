import torch

from .base import LinearTransform
from typing import Union, Tuple


class FiniteDiff(LinearTransform):
    def __init__(self, dim: int):
        self.dim = dim

    def __call__(self, X: torch.Tensor) -> torch.Tensor:
        X_forward = torch.roll(X, -1, self.dim)
        X_out = X_forward - X

        return X_out
    
    def conj_op(self, S: torch.Tensor) -> torch.Tensor:
        S_backward = torch.roll(S, 1, self.dim)
        S_out = S_backward - S

        return S_out
    
    def log_lh_grad(self, X: torch.Tensor, S: torch.Tensor = None, lamda: float = 1) -> torch.Tensor:
        """
        grad = -lamda * nabla' @ (sign(nabla @ X)) 
        """
        grad = -lamda * self.conj_op(torch.sign(self(X)))

        return grad
