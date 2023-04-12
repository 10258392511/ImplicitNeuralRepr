import torch

from typing import Sequence


class SpatialTimeRegGenerator(object):
    """ 
    coord: (lam, t, y, x)
    """
    def __init__(self, in_shape: Sequence[int], lam_max: float, lam_min: float):
        self.T, self.H, self.W = in_shape
        self.lam_max = lam_max
        self.lam_min = lam_min

    def generate_spatial(self, lam_samples: torch.Tensor, t_samples: torch.Tensor):
        # lam_samples, t_samples: 1D, unnormalized
        lam_samples = self.normalize_lam(lam_samples)
        t_samples = self.normalize_coord(t_samples, self.T)

        yy, xx = torch.meshgrid(torch.linspace(-1, 1, self.H), torch.linspace(-1, 1, self.W), indexing="ij")  # (H, W) each
        yx_grid = torch.stack([yy, xx], dim=-1)  # (H, W, 2)
        all_coords = torch.zeros((len(lam_samples), len(t_samples), self.H, self.W, 4))
        for i, lam_iter in enumerate(lam_samples):
            for j, t_iter in enumerate(t_samples):
                all_coords[i, j, ..., 2:] = yx_grid
                all_coords[i, j, ..., 0] = lam_iter
                all_coords[i, j, ..., 1] = t_iter

        # (Lam', T', H, W, 4)
        return all_coords
    
    def generate_temporal(self, lam_samples: torch.Tensor, y_samples: torch.Tensor, x_samples: torch.Tensor):
        # lam_samples, y/x_samples: 1D, unnormalized
        # lam, y, x are majority, so use meshgrid on them instead
        lam_samples = self.normalize_lam(lam_samples)
        y_samples = self.normalize_coord(y_samples, self.H)
        x_samples = self.normalize_coord(x_samples, self.W)
        lam_lam, yy, xx = torch.meshgrid(lam_samples, y_samples, x_samples, indexing="ij")
        lam_y_x_grid = torch.stack([lam_lam, yy, xx], dim=-1)  # (Lam', H', W', 3)
        all_coords = torch.zeros((len(lam_samples), self.T, len(y_samples), len(x_samples), 4))
        for i, t_iter in enumerate(torch.linspace(-1, 1, self.T)):
            all_coords[:, i, ..., [0, 2, 3]] = lam_y_x_grid
            all_coords[:, i, ..., 1] = t_iter

        # (Lam', T, H', W', 4)
        return all_coords        

    def normalize_lam(self, lam: torch.Tensor):
        # TODO: add exponential
        lam = 2 * (lam - self.lam_min) / (self.lam_max - self.lam_min) - 1

        return lam
    
    def normalize_coord(self, coord: torch.Tensor, dim: int):
        coord = 2 * coord / (dim - 1) - 1

        return coord
