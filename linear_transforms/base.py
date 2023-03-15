import numpy as np
import torch
import abc


class LinearTransform(abc.ABC):
    """
    All inputs: (B, C, H, W)
    """
    @abc.abstractmethod
    def __call__(self, X: torch.Tensor) -> torch.Tensor:
        return X

    @abc.abstractmethod
    def conj_op(self, S: torch.Tensor) -> torch.Tensor:
        return S

    def log_lh_grad(self, X: torch.Tensor, S: torch.Tensor, lamda: float = 1.) -> torch.Tensor:
        """
        grad = -lamda * A'(Ax - s)
        """
        diff = self(X) - S  # (B, C, H_s, W_s)
        grad = -self.conj_op(diff) * lamda

        return grad
    

class Indentity(LinearTransform):
    def __call__(self, X: torch.Tensor) -> torch.Tensor:
        return X
    
    def conj_op(self, S: torch.Tensor) -> torch.Tensor:
        return S


def i2k_complex(X, dims=(-1, -2)):
    """
    X: (..., H, W) 
    """
    X = X.to(torch.complex64)
    X = torch.fft.ifftshift(X, dim=dims)
    X_k_space = torch.fft.fftn(X, dim=dims, norm="ortho")
    X_k_space_shifted = torch.fft.fftshift(X_k_space, dim=dims)

    return X_k_space_shifted


def k2i_complex(X, dims=(-1, -2)):
    """
    X: (..., H, W)
    """
    X = X.to(torch.complex64)
    X_i_shifted = torch.fft.ifftshift(X, dim=dims)
    X_img = torch.fft.ifftn(X_i_shifted, dim=dims, norm="ortho")
    X_img = torch.fft.fftshift(X_img, dim=dims)

    return X_img


def generate_mask(T: int, N: int, sw=0.3, sm=0.7, sa=0.045, T_max=1000, dev=0.01, seed=None):
    # default to R = 4
    np.random.seed(seed)
    x = np.linspace(-1, 1, N)
    p = np.exp(-np.abs(x) / sw) * sm + sa
    masks = np.random.rand(N, T_max) <= p[:, None]
    masks[masks.shape[0] // 2 - 1:masks.shape[0] // 2 + 1, :] = 1
    selected = np.abs(masks.mean(axis=0) - masks.mean()) < dev
    masks_selected = masks[:, selected]
    indices = np.random.choice(masks_selected.shape[1], T)
    masks_out = masks_selected[:, indices].T  # (N, T) -> (T, N)
    if T == 1:
        # (1, N)
        return torch.tensor(masks_out[0:1, :])
    else:
        # (T, 1, N)
        return torch.tensor(masks_out[:, None, :])
