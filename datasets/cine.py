import numpy as np
import torch
import scipy.io as sio
import os
import glob

from torch.utils.data import TensorDataset, Dataset, DataLoader
from einops import rearrange
from ImplicitNeuralRepr.linear_transforms import LinearTransform
from pytorch_lightning import LightningDataModule
from pytorch_lightning.utilities.types import EVAL_DATALOADERS, TRAIN_DATALOADERS
from monai.transforms import Resize as monai_Resize
from typing import Union

ROOT_DIR = os.path.abspath(__file__)
for _ in range(2):
    ROOT_DIR = os.path.dirname(ROOT_DIR)

DATA_PATHS = {
    "CINE64": os.path.join(ROOT_DIR, "data/cine_64"),
    "CINE127": os.path.join(ROOT_DIR, "data/cine_127"),
}


def load_cine(root_dir, mode="train", img_key="imgs", ds_type="2d+time"):
    """
    "mode" is not used
    """
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


class CINEImageKSPDataset(Dataset):
    def __init__(self, params: dict, lin_tfm: LinearTransform):
        """
        params: mode {train, test}, res {64, 127}, seed: int or None, train_test_split: float
        """
        super().__init__()
        self.params = params
        self.lin_tfm = lin_tfm
        self.resizer = None
        if self.params["res"] == 64:
            vol_name = "CINE64"
        elif self.params["res"] == 127:
            vol_name = "CINE127"
            self.resizer = monai_Resize((25, 128, 128), mode="trilinear", align_corners=True)

        else:
            raise NotImplementedError
        self.cine_ds = load_cine(DATA_PATHS[vol_name])  # (N, T, H, W)
        seed = self.params.get("seed", None)
        if seed is not None:
            np.random.seed(seed)
        indices = np.arange(len(self.cine_ds))
        np.random.shuffle(indices)
        split_idx = int(len(self.cine_ds) * self.params["train_test_split"])
        if self.params["mode"] == "train":
            self.indices = indices[:split_idx]
        else:
            self.indices = indices[split_idx:]
    
    def __len__(self):

        return len(self.indices)
    
    def __getitem__(self, idx):
        seed = self.params.get("seed", 0) + idx
        ds_idx = self.indices[idx]
        img = self.cine_ds[ds_idx][0]  # (T, H, W)
        if self.resizer is not None:
            img = self.resizer(img[None, ...])  # (C, T, H, W)
            img = img[0, ...]  # (T, H, W)
        img = add_phase(img[:, None, ...], init_shape=(5, 5, 5),  seed=seed, mode="2d+time")
        img = img[:, 0, ...]  # (T, 1, H, W) -> (T, H, W)
        measurement = self.lin_tfm(img)  # (..., T, H, W, num_sense)

        return img, measurement


class CINEImageKSPDM(LightningDataModule):
    def __init__(self, params: dict, lin_tfm: LinearTransform):
        """
        params: see CINEImageKSPDataset
                    batch_size: int, num_workers: int = 0
        """
        super().__init__()
        self.params = params
        params["mode"] = "train"
        self.train_ds = CINEImageKSPDataset(params, lin_tfm)
        params["mode"] = "test"
        self.test_ds = CINEImageKSPDataset(params, lin_tfm)

    def prepare_data(self) -> None:
        return super().prepare_data()
    
    def setup(self, stage: Union[str, None] = None) -> None:
        return super().setup(stage)
    
    def train_dataloader(self) -> TRAIN_DATALOADERS:
        loader = DataLoader(self.train_ds, self.params["batch_size"], shuffle=True, pin_memory=True)

        return loader
    
    def __shared_val_dataloader(self):
        loader = DataLoader(self.test_ds, self.params["batch_size"], pin_memory=True)

        return loader
    
    def val_dataloader(self) -> EVAL_DATALOADERS:

        return self.__shared_val_dataloader()
    
    def predict_dataloader(self) -> EVAL_DATALOADERS:
        
        return self.__shared_val_dataloader()
    