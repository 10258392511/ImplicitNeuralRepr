import numpy as np
import torch

from .coord_datasets_separate import Spatial2DTimeRegCoordDataset,  Temporal2DTimeRegCoordDataset
from torch.utils.data import Dataset, DataLoader
from pytorch_lightning import LightningDataModule
from collections import defaultdict, deque
from typing import Sequence, Optional


class ZipDataset(Dataset):
    """
    (lam, t, y, x)
    """
    def __init__(self, in_shape: Sequence[int], lam_min: int, lam_max: int, lam_tfm=lambda lam : lam):
        """
        spatial_ds: Lam * T samples, each: (H, W, 4)
        temporal_ds: Lam * H * W samples, each: (T, 4)
        """
        super().__init__()
        self.spatial_ds = Spatial2DTimeRegCoordDataset(in_shape, lam_min, lam_max, lam_tfm)
        self.temporal_ds = Temporal2DTimeRegCoordDataset(in_shape, lam_min, lam_max, lam_tfm)
        self.spatial_lam2idx = None  # {0: [spatial_ds_idx...]...}
        self.reload_lam2idx()
    
    def __len__(self):
        return len(self.temporal_ds)
    
    def __getitem__(self, idx: int):
        temporal_sample, lam_sample = self.temporal_ds[idx]
        lam, y, x = np.unravel_index(idx, (self.temporal_ds.Lam, self.temporal_ds.H, self.temporal_ds.W))
        spatial_sample = torch.zeros_like(self.spatial_ds[0][0])  # (H, W, 4)
        spatial_valid = False
        if len(self.spatial_lam2idx[lam]) > 0:
            spatial_idx = self.spatial_lam2idx[lam].popleft()
            spatial_sample = self.spatial_ds[spatial_idx][0]
            spatial_valid = True
        
        # (H, W, 4), (T, 4), float, bool
        return spatial_sample, temporal_sample, lam_sample, spatial_valid

    def reload_lam2idx(self):
        self.spatial_lam2idx = defaultdict(deque)
        for idx in range(len(self.spatial_ds)):
            lam, t = np.unravel_index(idx, (self.spatial_ds.Lam, self.spatial_ds.T))
            self.spatial_lam2idx[lam].append(idx)


class ZipDM(LightningDataModule):
    def __init__(self, ds: ZipDataset, batch_size: int, pred_ds: Dataset, pred_batch_size: int, num_workers: int = 0):
        super().__init__()
        self.ds = ds
        self.batch_size = batch_size
        self.pred_ds = pred_ds
        self.pred_batch_size = pred_batch_size
        self.num_workers = num_workers
    
    def prepare_data(self) -> None:
        return super().prepare_data()
    
    def setup(self, stage: Optional[str] = None) -> None:
        return super().setup(stage)
    
    def train_dataloader(self):
        loader = DataLoader(self.ds, self.batch_size, shuffle=True, pin_memory=True, num_workers=self.num_workers)

        return loader

    def predict_dataloader(self):
        loader = DataLoader(self.pred_ds, self.pred_batch_size, pin_memory=True, num_workers=self.num_workers)

        return loader

    def resample(self):
        self.ds.reload_lam2idx()
        