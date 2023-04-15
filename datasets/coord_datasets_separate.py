import numpy as np
import torch

from torch.utils.data import Dataset
from typing import Sequence


class Spatial2DTimeCoordDataset(Dataset):
    """
    (t, y, x) 
    """
    def __init__(self, in_shape: Sequence[int]):
        super().__init__()
        self.T, self.H, self.W = in_shape
        yx_grid = torch.meshgrid(torch.linspace(-1, 1, self.H), torch.linspace(-1, 1, self.W), indexing="ij")
        self.yx_grid = torch.stack(yx_grid, dim=-1)  # (H, W, 2)
        self.t_grid = torch.linspace(-1, 1, self.T)
    
    def __len__(self):
        return self.T
    
    def __getitem__(self, idx: int):
        sample = torch.zeros((self.H, self.W, 3))
        sample[..., -2:] = self.yx_grid
        sample[..., 0] = self.t_grid[idx]

        # (T, H, W)
        return sample.float()


class Temporal2DTimeCoordDataset(Dataset):
    """
    (t, y, x)
    """
    def __init__(self, in_shape: Sequence[int]):
        super().__init__()
        self.T, self.H, self.W = in_shape
        self.t_grid = torch.linspace(-1, 1, self.T)
        self.y_grid = torch.linspace(-1, 1, self.H)
        self.x_grid = torch.linspace(-1, 1, self.W)
    
    def __len__(self):
        return self.H * self.W
    
    def __getitem__(self, idx: int):
        y_idx, x_idx = np.unravel_index(idx, (self.H, self.W))
        sample = torch.zeros((self.T, 3))
        sample[:, 0] = self.t_grid
        sample[:, -2] = self.y_grid[y_idx]
        sample[:, -1] = self.x_grid[x_idx]

        # (T, H, W)
        return sample.float()


class Spatial2DTimeRegCoordDataset(Dataset):
    """
    (lam, t, y, x)
    """
    def __init__(self, in_shape: Sequence[int], lam_min: int, lam_max: int, lam_tfm=lambda lam : lam):
        """
        lam: [lam_min, lam_max)
        """
        self.T, self.H, self.W = in_shape
        self.Lam = lam_max - lam_min
        self.lam_max = lam_max
        self.lam_min = lam_min
        self.lam_tfm = lam_tfm
        yx_grid = torch.meshgrid(torch.linspace(-1, 1, self.H), torch.linspace(-1, 1, self.W), indexing="ij")
        self.yx_grid = torch.stack(yx_grid, dim=-1)  # (H, W, 2)
        self.t_grid = torch.linspace(-1, 1, self.T)
        self.lam_grid = torch.linspace(-1, 1, self.Lam)
        self.lam_grid_val = lam_tfm(torch.arange(self.lam_min, self.lam_max).float())  # used for explicit reg scaler
    
    def __len__(self):
        return self.Lam * self.T
    
    def __getitem__(self, idx: int):
        lam_idx, t_idx = np.unravel_index(idx, (self.Lam, self.T))
        sample = torch.zeros((self.H, self.W, 4))
        sample[..., -2:] = self.yx_grid
        sample[..., 0] = self.lam_grid[lam_idx]
        sample[..., 1] = self.t_grid[t_idx]

        # (H, W, 4)
        return sample.float(), self.lam_grid_val[lam_idx]
    

class Temporal2DTimeRegCoordDataset(Dataset):
    """
    (lam, t, y, x)
    """
    def __init__(self, in_shape: Sequence[int], lam_min: int, lam_max: int, lam_tfm=lambda lam : lam):
        """
        lam: [lam_min, lam_max)
        """
        self.T, self.H, self.W = in_shape
        self.Lam = lam_max - lam_min
        self.lam_max = lam_max
        self.lam_min = lam_min
        self.lam_tfm = lam_tfm
        self.t_grid = torch.linspace(-1, 1, self.T)
        self.lam_grid = torch.linspace(-1, 1, self.Lam)
        self.y_grid = torch.linspace(-1, 1, self.H)
        self.x_grid = torch.linspace(-1, 1, self.W)
        self.lam_grid_val = lam_tfm(torch.arange(self.lam_min, self.lam_max).float())  # used for explicit reg scaler
    
    def __len__(self):
        return self.Lam * self.H * self.W
    
    def __getitem__(self, idx: int):
        lam_idx, y_idx, x_idx = np.unravel_index(idx, (self.Lam, self.H, self.W))
        sample = torch.zeros((self.T, 4))
        sample[..., 1] = self.t_grid
        sample[..., 0] = self.lam_grid[lam_idx]
        sample[..., -2] = self.y_grid[y_idx]
        sample[..., -1] = self.x_grid[x_idx]
        

        # (H, W, 4)
        return sample, self.lam_grid_val[lam_idx]


class Spatial2DTimeRegCoordPredDataset(Dataset):
    """
    (fixed lam, t, y, x) 
    """
    def __init__(self, in_shape: Sequence[int], lam: float):
        super().__init__()
        self.T, self.H, self.W = in_shape
        yx_grid = torch.meshgrid(torch.linspace(-1, 1, self.H), torch.linspace(-1, 1, self.W), indexing="ij")
        self.yx_grid = torch.stack(yx_grid, dim=-1)  # (H, W, 2)
        self.t_grid = torch.linspace(-1, 1, self.T)
        self.lam = lam
    
    def __len__(self):
        return self.T
    
    def __getitem__(self, idx: int):
        sample = torch.zeros((self.H, self.W, 4))
        sample[..., -2:] = self.yx_grid
        sample[..., 1] = self.t_grid[idx]
        sample[..., 0] = self.lam

        # (T, H, W)
        return sample.float()
    