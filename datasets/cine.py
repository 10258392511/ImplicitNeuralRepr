import torch
import scipy.io as sio
import os
import glob

from torch.utils.data import TensorDataset
from einops import rearrange


def load_cine(root_dir, mode="train", img_key="imgs", ds_type="2d+time"):
    assert mode in ["train", "val", "test"]
    assert ds_type in ["2d+time", "spatial"]
    if mode == "val":
        mode = "test"
    filename = glob.glob(os.path.join(root_dir, f"*{mode}*.mat"))[0]
    ds = sio.loadmat(filename)[img_key]  # (H, W, T, N)
    ds = rearrange(ds, "H W T N -> N T H W")
    ds = (ds - ds.min(axis=(1, 2, 3), keepdims=True)) / (ds.max(axis=(1, 2, 3), keepdims=True) -
                                                         ds.min(axis=(1, 2, 3), keepdims=True))
    if ds_type == "spatial":
        ds = rearrange(ds, "N T H W -> (N T) H W")

    ds = torch.FloatTensor(ds)
    ds = TensorDataset(ds)

    return ds
