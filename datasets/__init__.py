import torch
import os

from .cine import load_cine
from .coord import MetaCoordDM, CoordDataset
from .coord_data_generator import SpatialTimeRegGenerator
from monai.transforms import Resize as monai_Resize
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


def add_phase(imgs: torch.Tensor, init_shape: Union[tuple, int] = (5, 5), seed=None, mode="spatial"):
    # imgs: (B, C, H, W) or (T, C, H, W)
    assert mode in ["spatial", "2d+time"]
    if seed is not None:
        torch.manual_seed(seed)
    imgs_out = imgs
    if mode == "spatial":
        # add smooth phase for each spatial slice
        B, C, H, W = imgs.shape
        imgs_out = torch.empty_like(imgs, dtype=torch.complex64)
        for i in range(B):
            img_iter = imgs[i, ...]  # (C, H, W)
            phase_init_patch = torch.randn(C, *init_shape, device=img_iter.device)
            resizer = monai_Resize((H, W), mode="bicubic", align_corners=True)
            phase = resizer(phase_init_patch)  # (C, H, W)
            imgs_out[i, ...] = img_iter * torch.exp(1j * phase)
    elif mode == "2d+time":
        # use 3D phase map for each channel for (T, C, H, W)
        assert len(init_shape) == 3
        T, C, H, W = imgs.shape
        phase = torch.randn(C, *init_shape, device=imgs.device)  # e.g. (init_x, init_y, init_z)
        resizer = monai_Resize((T, H, W), mode="trilinear", align_corners=True)
        phase = resizer(phase)  # (C, T, H, W)
        imgs_out = imgs * torch.exp(1j * rearrange(phase, "C T H W -> T C H W"))

    return imgs_out
