import torch
import os

from .cine import (
    load_cine,
    add_phase,
    CINEImageKSPDM
)
from .coord import MetaCoordDM, CoordDataset, WrapperDM
from .coord_datasets_separate import (
    Spatial2DTimeCoordDataset, 
    Temporal2DTimeCoordDataset, 
    Spatial2DTimeRegCoordDataset, 
    Temporal2DTimeRegCoordDataset,
    Spatial2DTimeRegCoordPredDataset
)
from .spatial_time_reg import (
    ZipDataset, 
    ZipDM
)
from .simplified_sampling_datasets import (
    SpatialTemporalSamplingDM,
    SpatialTemporalRegSamplingDM,
    FracSpatialTemporalRegDM
)
from typing import Sequence, Union
from einops import rearrange


ROOT_DIR = os.path.abspath(__file__)
for _ in range(2):
    ROOT_DIR = os.path.dirname(ROOT_DIR)

DATA_PATHS = {
    "CINE64": os.path.join(ROOT_DIR, "data/cine_64"),
    "CINE127": os.path.join(ROOT_DIR, "data/cine_127"),
}

LOADER = {
    "CINE64": load_cine,
    "CINE127": load_cine
}


def load_data(ds_name: str, mode: str, **kwargs):
    assert ds_name in DATA_PATHS.keys()
    assert mode in ["train", "val", "test"]

    data_dir = DATA_PATHS[ds_name]
    loader = LOADER[ds_name]
    ds = loader(data_dir, mode=mode, **kwargs)

    return ds


def val2idx(X: torch.Tensor, dim_ranges: Sequence[int]):
    """
    From [-1, 1] to indices.

    For example, for 2D + time with spatial data:
    X: (..., 3), dim_ranges: [T, H, W]
    """
    if not isinstance(dim_ranges, torch.Tensor):
        dim_ranges = torch.tensor(dim_ranges, dtype=X.dtype, device=X.device)
    X_idx = torch.round((X + 1) / 2 * (dim_ranges - 1))

    return X_idx.long()


def idx2val(X_idx: torch.Tensor, dim_ranges: Sequence[int]):
    """
    From indices to [-1, 1]. Inverse of input2val(.).
    """
    if not isinstance(dim_ranges, torch.Tensor):
        dim_ranges = torch.tensor(dim_ranges, dtype=X_idx.dtype, device=X_idx.device)
    X = (X_idx / (dim_ranges - 1)) * 2 - 1

    return X.float()


