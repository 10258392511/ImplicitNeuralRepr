import numpy as np
import random

from .coord_datasets_separate import (
    Spatial2DTimeCoordDataset,
    Temporal2DTimeCoordDataset,
    Spatial2DTimeRegCoordDataset,
    Temporal2DTimeRegCoordDataset,
    Spatial2DTimeRegCoordPredDataset
)
from torch.utils.data import DataLoader, Subset
from pytorch_lightning import LightningDataModule
from collections import defaultdict
from itertools import cycle
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
