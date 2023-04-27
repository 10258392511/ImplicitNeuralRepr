import numpy as np
import torch
import random

from .coord_datasets_separate import (
    Spatial2DTimeCoordDataset,
    Temporal2DTimeCoordDataset,
    Spatial2DTimeRegCoordDataset,
    Temporal2DTimeRegCoordDataset,
    Spatial2DTimeRegCoordPredDataset
)
from torch.utils.data import DataLoader, Subset, TensorDataset
from torch.distributions import Uniform
from pytorch_lightning import LightningDataModule
from collections import defaultdict
from typing import Optional


class SpatialTemporalSamplingDM(LightningDataModule):
    def __init__(self, params: dict):
        """
        params: in_shape = (T, H, W), spatial_batch_size, pred_batch_size, num_temporal_repeats, num_workers
        """
        super().__init__()
        self.params = params
        self.spatial_ds = Spatial2DTimeCoordDataset(self.params["in_shape"])
        self.temporal_ds = Temporal2DTimeCoordDataset(self.params["in_shape"])
        self.pred_ds = Spatial2DTimeCoordDataset(self.params["in_shape"])
        self.T, self.H, self.W = self.spatial_ds.T, self.spatial_ds.H, self.spatial_ds.W
        self.spatial_batch_size = self.params["spatial_batch_size"]
        self.temporal_batch_size = int(np.ceil(self.spatial_batch_size * self.H * self.W / self.T))
        self.spatial_subset = None
        self.temporal_subset = None
        self.resample()
    
    def prepare_data(self) -> None:
        return super().prepare_data()
    
    def setup(self, stage: Optional[str] = None) -> None:
        return super().setup(stage)
    
    def train_dataloader(self):
        spatial_loader = DataLoader(self.spatial_subset, self.spatial_batch_size, pin_memory=True, num_workers=0)
        temporal_loader = DataLoader(self.temporal_subset, self.temporal_batch_size, pin_memory=True, num_workers=self.params["num_workers"])

        return [spatial_loader, temporal_loader]
    
    def predict_dataloader(self):
        loader = DataLoader(self.pred_ds, self.params["pred_batch_size"], pin_memory=True, num_workers=self.params["num_workers"])

        return loader

    def resample(self) -> None:
        spatial_inds = np.random.choice(len(self.spatial_ds), (self.spatial_batch_size,))
        temporal_inds = np.random.choice(len(self.temporal_ds), (self.params["num_temporal_repeats"] * self.temporal_batch_size,))
        self.spatial_subset = Subset(self.spatial_ds, spatial_inds)
        self.temporal_subset = Subset(self.temporal_ds, temporal_inds)


class SpatialTemporalRegSamplingDM(LightningDataModule):
    def __init__(self, params: dict, lam_tfm=lambda lam : lam):
        """
        params: in_shape = (T, H, W), lam_min, lam_max, lam_pred, spatial_batch_size, pred_batch_size, num_temporal_repeats, num_workers
        """
        super().__init__()
        self.params = params
        self.spatial_ds = Spatial2DTimeRegCoordDataset(self.params["in_shape"], self.params["lam_min"], self.params["lam_max"], lam_tfm)
        self.temporal_ds = Temporal2DTimeRegCoordDataset(self.params["in_shape"], self.params["lam_min"], self.params["lam_max"], lam_tfm)
        self.pred_ds = Spatial2DTimeRegCoordPredDataset(self.params["in_shape"], self.params["lam_pred"])
        self.T, self.H, self.W = self.spatial_ds.T, self.spatial_ds.H, self.spatial_ds.W
        self.Lam = self.spatial_ds.Lam
        self.spatial_batch_size = self.params["spatial_batch_size"]
        self.temporal_batch_size = int(np.ceil(self.spatial_batch_size * self.H * self.W / self.T))
        self._spatial_lam2idx = None
        self.__spatial_lam2idx()
        self._temporal_lam2idx = None
        self.__temporal_lam2idx()
        self.spatial_subset = None
        self.temporal_subset = None
        self.resample()
    
    def __spatial_lam2idx(self):
        self._spatial_lam2idx = defaultdict(list)
        for idx in range(len(self.spatial_ds)):
            lam, t = np.unravel_index(idx, (self.Lam, self.T))
            self._spatial_lam2idx[lam].append(idx)
    
    def __temporal_lam2idx(self):
        self._temporal_lam2idx = defaultdict(list)
        for idx in range(len(self.temporal_ds)):
            lam, y, x = np.unravel_index(idx, (self.Lam, self.H, self.W))
            self._temporal_lam2idx[lam].append(idx)

    def prepare_data(self) -> None:
        return super().prepare_data()
    
    def setup(self, stage: Optional[str] = None) -> None:
        return super().setup(stage)
    
    def train_dataloader(self):
        spatial_loader = DataLoader(self.spatial_subset, self.spatial_batch_size, pin_memory=True, num_workers=0)
        temporal_loader = DataLoader(self.temporal_subset, self.temporal_batch_size, pin_memory=True, num_workers=self.params["num_workers"])

        return [spatial_loader, temporal_loader]
    
    def predict_dataloader(self):
        loader = DataLoader(self.pred_ds, self.params["pred_batch_size"], pin_memory=True, num_workers=self.params["num_workers"])

        return loader
    
    def resample(self):
        lam = np.random.randint(0, self.Lam)
        candidate_list_s = self._spatial_lam2idx[lam]
        candidate_list_t = self._temporal_lam2idx[lam]
        spatial_inds = random.sample(candidate_list_s, k=self.spatial_batch_size)
        temporal_inds = random.choices(candidate_list_t, k=self.params["num_temporal_repeats"] * self.temporal_batch_size)
        self.spatial_subset = Subset(self.spatial_ds, spatial_inds)
        self.temporal_subset = Subset(self.temporal_ds, temporal_inds)


class FracSpatialTemporalRegDM(LightningDataModule):
    def __init__(self, params: dict, lam_tfm=lambda lam : lam):
        """
        params: in_shape = (T, H, W), lam_min, lam_max, num_lams, lam_pred, spatial_batch_size, pred_batch_size, num_temporal_repeats, num_workers
        """
        super().__init__()
        assert params["spatial_batch_size"] % params["num_lams"] == 0

        self.params = params
        self.pred_ds = Spatial2DTimeRegCoordPredDataset(self.params["in_shape"], self.params["lam_pred"])
        self.T, self.H, self.W = self.params["in_shape"]
        self.lam_tfm = lam_tfm
        self.spatial_batch_size = self.params["spatial_batch_size"]
        self.temporal_batch_size = int(np.ceil(self.spatial_batch_size * self.H * self.W / self.T))
        self.spatial_batch_size_each = self.spatial_batch_size // self.params["num_lams"]
        self.temporal_batch_size_each = self.temporal_batch_size // self.params["num_lams"]
        yx_grid = torch.meshgrid(torch.linspace(-1, 1, self.H), torch.linspace(-1, 1, self.W), indexing="ij")
        self.yx_grid = torch.stack(yx_grid, dim=-1)  # (H, W, 2)
        self.t_grid = torch.linspace(-1, 1, self.T)  # (T,)
        self.spatial_subset = None
        self.temporal_subset = None
        self.unif = Uniform(-1, 1)
        self.unif_lam = Uniform(self.params["lam_min"], self.params["lam_max"])
        self.resample()
    
    def prepare_data(self) -> None:
        return super().prepare_data()
    
    def setup(self, stage: Optional[str] = None) -> None:
        return super().setup(stage)

    def __normalize_lam(self, lam):
        lam_normed = (lam - self.params["lam_min"]) / (self.params["lam_max"] - self.params["lam_min"]) * 2 - 1

        return lam_normed
    
    def train_dataloader(self):
        spatial_loader = DataLoader(self.spatial_subset, self.spatial_batch_size, pin_memory=True, num_workers=0)
        temporal_loader = DataLoader(self.temporal_subset, self.temporal_batch_size, pin_memory=True, num_workers=self.params["num_workers"])

        return [spatial_loader, temporal_loader]

    def predict_dataloader(self):
        loader = DataLoader(self.pred_ds, self.params["pred_batch_size"], pin_memory=True, num_workers=self.params["num_workers"])

        return loader

    def resample(self):
        """
        S1, T1, lam1; S2, T2, lam1; S3, T3, lam2; S4, T4, lam2; S1, T5, lam1...
        """
        lam_samples = self.unif_lam.sample((self.params["num_lams"],))  # (num_lams,), [lam_1, lam_2], Unif[.lam_min, .lam_max]
        lam_all = self.lam_tfm(lam_samples)
        lam_samples = self.__normalize_lam(lam_samples)  # Unif[-1, 1]
        
        spatial_all = []
        lam_s_all = []
        for lam_iter, lam_val_iter in zip(lam_samples, lam_all):
            spatial_iter = torch.zeros((self.spatial_batch_size_each, self.H, self.W, 4))
            spatial_iter[..., -2:] = self.yx_grid
            t_samples_idx = torch.randperm(self.T)[:self.spatial_batch_size_each]
            t_samples = self.t_grid[t_samples_idx]
            spatial_iter[..., 1] = t_samples.reshape(t_samples.shape[0], 1, 1)
            spatial_iter[..., 0] = lam_iter
            spatial_all.append(spatial_iter)
            lam_s_all.append(torch.ones((self.spatial_batch_size_each,)) * lam_val_iter)
        spatial_all = torch.cat(spatial_all, dim=0)  # (.spatial_batch_size, H, W, 4)
        lam_s_all = torch.cat(lam_s_all, dim=0)  # (.spatial_batch_size,)

        temporal_all = []
        lam_t_all = []
        for _ in range(self.params["num_temporal_repeats"]):
            for lam_iter, lam_val_iter in zip(lam_samples, lam_all):
                temporal_iter = torch.zeros((self.temporal_batch_size_each, self.T, 4))
                temporal_iter[..., 1] = self.t_grid
                yx_samples = self.unif.sample((self.temporal_batch_size_each, 2))
                temporal_iter[..., -2:] = yx_samples.unsqueeze(1)  # left: (.temporal_batch_size_each, .T, 2)
                temporal_iter[..., 0] = lam_iter
                temporal_all.append(temporal_iter)
                lam_t_all.append(torch.ones((self.temporal_batch_size_each,)) * lam_val_iter)
        temporal_all = torch.cat(temporal_all, dim=0)  # (.temporal_batch_size, T, 4)
        lam_t_all = torch.cat(lam_t_all, dim=0)  # (.temporal_batch_size,)

        self.spatial_subset = TensorDataset(spatial_all, lam_s_all)
        self.temporal_subset = TensorDataset(temporal_all, lam_t_all)
