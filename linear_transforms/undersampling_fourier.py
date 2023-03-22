import numpy as np
import torch

from scipy.spatial import distance_matrix
from .base import i2k_complex, k2i_complex, generate_mask, LinearTransform
from typing import Union


class RandomUndersamplingFourier(LinearTransform):
    def __init__(self, in_shape, mask_params, seed=None):
        """
        in_shape: (..., H, W) or (..., T, H, W)
        mask_params: sw, sm, sa, T_max, if_temporal (True: view in_shape[-3] as T)
        """
        self.in_shape = in_shape
        self.mask_params = mask_params  # for debug
        self.seed = seed
        T, N = 1, self.in_shape[-1]
        if self.mask_params.get("if_temporal", False):
            T = self.in_shape[-3]
        if "if_temporal" in self.mask_params:
            self.mask_params.pop("if_temporal")
        self.mask = generate_mask(T, N, seed=seed, **self.mask_params)  # (H, W) or (T, H, W)

    def __call__(self, X: torch.Tensor, t_indices: Union[torch.Tensor, None]) -> torch.Tensor:
        # X: (..., H, W) or (..., T, H, W)
        mask = self.mask.to(X.device)
        if t_indices is not None:
            mask = mask[t_indices, :, :]
        S = mask * i2k_complex(X)  # (..., H, W) or (..., T, H, W)

        return S

    def conj_op(self, S: torch.Tensor) -> torch.Tensor:
        X = k2i_complex(S)

        return X
    

class SENSE(LinearTransform):
    def __init__(self, sens_type, num_sens, in_shape, mask_params, seed=None):
        """
        in_shape, mask_params: see RandomUndersamplingFourier
        """
        assert sens_type in ["exp"]
        self.random_under_fourier = RandomUndersamplingFourier(in_shape, mask_params, seed)
        sens_maps = []
        for i in range(num_sens):
            seed = self.random_under_fourier.seed
            if seed is not None:
                seed += i
            sens_maps.append(self._generate_sens_map(sens_type, seed))

        sens_maps = torch.stack(sens_maps, dim=0)  # [(H, W)...] -> (num_sens, H, W)
        normalize_fractor = (torch.abs(sens_maps) ** 2).sum(dim=0)  # (num_sens, H, W) -> (H, W)
        normalize_fractor = torch.sqrt(normalize_fractor)  # (H, W)
        self.sens_maps = sens_maps / normalize_fractor  # (num_sens, H, W)

        energy = (torch.abs(self.sens_maps) ** 2).sum(dim=0)
        assert torch.allclose(energy,  torch.ones_like(energy))

    def _generate_sens_map(self, sens_type, seed=0, **kwargs):
        sens_map = torch.ones(self.random_under_fourier.in_shape)
        if sens_type == "exp":
            # kwargs: l, anchor: np.ndarray
            # exp(-||x - x0||^2 / (2 * l))
            anchor = kwargs.get("anchor", None)

            if anchor is None:
                H, W = self.random_under_fourier.in_shape[-2:]
                np.random.seed(seed)
                anchor_h, anchor_w = np.random.choice(H), np.random.choice(W)
                anchor = np.array([anchor_h, anchor_w])[None, :]  # (1, 2)
                ww, hh = np.mgrid[0:W, 0:H]  # (H, W) each
                coords = np.stack([ww.flatten(), hh.flatten()], axis=1)  # (HW, 2)
                dist_mat = distance_matrix(coords, anchor, p=2)  # (HW, 1), not squared
                dist_mat_tensor = torch.tensor(dist_mat.reshape((H, W)))  # (H, W)
                l = kwargs.get("l", dist_mat.max() / 2)
                sens_map = torch.exp(- dist_mat_tensor / (2 * l))

        return sens_map

    def __call__(self, X: torch.Tensor, t_indices: Union[torch.Tensor, None] = None) -> torch.Tensor:
        # X: (..., H, W)
        S = []
        sens_maps = self.sens_maps.to(X.device)
        for i in range(sens_maps.shape[0]):
            sens_map_iter = sens_maps[i]  # (H, W)
            S.append(self.random_under_fourier(sens_map_iter * X, t_indices))

        S = torch.stack(S, dim=-1)  # [(..., H, W)...] -> (..., H, W, num_sens)

        return S

    def conj_op(self, S: torch.Tensor) -> torch.Tensor:
        # S: (..., H, W, num_sens)
        sens_maps = self.sens_maps.to(S.device)
        X_out = torch.zeros(S.shape[:-1], dtype=S.dtype).to(S.device)  # (..., H, W)
        for i in range(S.shape[-1]):
            X_out += sens_maps[i].conj() * self.random_under_fourier.conj_op(S[..., i])

        # (B, C, H, W)
        return X_out

    def SSOS(self, S: torch.Tensor) -> torch.Tensor:
        # S: (..., H, W, num_sens)
        X_out = torch.zeros(S.shape[:-1], dtype=torch.float32).to(S.device)  # (..., H, W)
        for i in range(S.shape[-1]):
            X_out += torch.abs(self.random_under_fourier.conj_op(S[..., i])) ** 2

        X_out = torch.sqrt(X_out)

        return X_out
