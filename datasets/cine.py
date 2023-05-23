import numpy as np
import torch
import scipy.io as sio
import os
import glob

from torch.utils.data import TensorDataset, Dataset, DataLoader
from einops import rearrange
from ImplicitNeuralRepr.linear_transforms import LinearTransform, SENSE
from pytorch_lightning import LightningDataModule
from pytorch_lightning.utilities.types import EVAL_DATALOADERS, TRAIN_DATALOADERS
from monai.transforms import Resize as monai_Resize
from ImplicitNeuralRepr.configs import IMAGE_KEY, MEASUREMENT_KEY, ZF_KEY, COORD_KEY
from typing import Union

ROOT_DIR = os.path.abspath(__file__)
for _ in range(2):
    ROOT_DIR = os.path.dirname(ROOT_DIR)

DATA_PATHS = {
    "CINE64": os.path.join(ROOT_DIR, "data/cine_64"),
    "CINE127": os.path.join(ROOT_DIR, "data/cine_127"),
}


def load_cine(root_dir, mode="train", img_key="imgs", ds_type="2d+time", if_return_tensor=False):
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
    if if_return_tensor:
        return ds
    
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


def kspace_binning(measurement: torch.Tensor, T0: float, mask: torch.Tensor, reduce: str = "mean"):
    """
    See kspace_binning_real(.)
    """
    measurement_out = kspace_binning_real(torch.real(measurement), T0, mask, reduce) + \
        1j * kspace_binning_real(torch.imag(measurement), T0, mask, reduce)

    return measurement_out


def kspace_binning_real(measurement: torch.Tensor, T0: float, mask: torch.Tensor, reduce: str = "mean"):
    """
    measurement: (B, T, H, W, ...)
    mask: (T, 1, W)
    reduce: {mean | none}

    T := (T0 - 1) * n + n_mod
    """
    broadcast_shape_tail = tuple([1 for _ in range(measurement.ndim - 4)])
    mask = mask.reshape(1, *mask.shape, *broadcast_shape_tail).expand_as(measurement)  # (1, T, 1, W, ...) -> (B, T, H, W, ...)
    B, T, H, W = measurement.shape[:4]
    n_mod = T % (T0 - 1)
    n = T // (T0 - 1)
    measurement = measurement.clone()
    measurement[~mask] = torch.nan
    measurement_n = measurement[:, :-n_mod, ...]  # (B, T', H, W, ...) where T' := (T0 - 1) * n
    measurement_n = measurement_n.reshape(B, -1, n, *measurement_n.shape[2:])  # (B, T0 - 1, n, H, W, ...)
    measurement_n_mod = measurement[:, -n_mod:, ...].unsqueeze(1)  # (B, n_mod, H, W) -> (B, 1, n_mod, H, W, ...)
    measurement_out = None
    if reduce == "mean":
        # (B, T0, H, W, ...)
        measurement_out = torch.zeros((B, T0, H, W) + measurement.shape[4:], dtype=measurement.dtype, device=measurement.device)
        measurement_out[:, :-1, ...] = torch.nanmean(measurement_n, dim=2)  # (B, T0 - 1, H, W, ...)
        measurement_out[:, -1:, ...] = torch.nanmean(measurement_n_mod, dim=2)  # (B, 1, H, W, ...)

    elif reduce == "none":
        # measurement_out = torch.zeros_like(measurement)  # (B, T, H, W, ...)
        measurement_n_mean = torch.nanmean(measurement_n, dim=2, keepdim=True).expand_as(measurement_n)  # (B, T0 - 1, 1 -> n, H, W, ...)
        measurement_n_mod_mean = torch.nanmean(measurement_n_mod, dim=2, keepdim=True).expand_as(measurement_n_mod)  # (B, 1, 1 -> n_mod, H, W, ...)
        
        mask_iter = measurement_n.isnan()
        measurement_n[mask_iter] = measurement_n_mean[mask_iter]
        mask_iter = measurement_n_mod.isnan()
        measurement_n_mod[mask_iter] = measurement_n_mod_mean[mask_iter]

        measurement_n = measurement_n.reshape(B, -1, *measurement.shape[2:])  # (B, T', H, W, ...)
        measurement_n_mod = measurement_n_mod.reshape(B, -1, *measurement.shape[2:])  # (B, n_mod, H, W, ...)
        measurement_out = torch.cat([measurement_n, measurement_n_mod], dim=1)  # (B, T, H, W, ...)
        measurement_out = torch.nan_to_num(measurement_out, 0.)

    else:
        raise NotImplementedError
    
    return measurement_out


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
    

class CINEImageKSDownSampleDataset(Dataset):
    def __init__(self, params: dict, mask_config: dict):
        """
        params: mode: str {train | val | test}, res: int {64 | 127}, T = 25, num_sens = 2, test_undersample_rate: float, data_aug_p: float, 
                seed: int or None, train_val_split: float
        """
        super().__init__()
        self.params = params

        # fixed settings
        self.input_T = self.params["input_T"]

        # init SENSE for all R
        self.mask_config = mask_config
        self.lin_tfm_dict = {}
        self.res = self.params["res"]
        if self.res == 127:
            self.res = 128
        
        self.test_lin_tfm = self.__init_lin_tfm(1., self.params["test_undersample_rate"], mode="test")
        
        # load CINE64/127
        if self.params["res"] == 64:
            vol_name = "CINE64"
        elif self.params["res"] == 127:
            vol_name = "CINE127"
            self.resizer = monai_Resize((25, 128, 128), mode="trilinear", align_corners=True)
        else:
            raise NotImplementedError
        self.cine_ds = load_cine(DATA_PATHS[vol_name], if_return_tensor=True)  # (N, T, H, W)
        if self.params["res"] == 127:
            self.cine_ds = self.resizer(self.cine_ds)  # (N, T, H', W')

        # train-val split
        seed = self.params.get("seed", None)
        if seed is not None:
            np.random.seed(seed)
        indices = np.arange(len(self.cine_ds))
        np.random.shuffle(indices)
        if self.params["mode"] == "test":
            self.indices = indices  # return all images for evaluation without cropping in T-axis; callback: using the last image
        else:
            split_idx = int(len(self.cine_ds) * self.params["train_val_split"])
            if self.params["mode"] == "train":
                self.indices = indices[:split_idx]
            elif self.params["mode"] == "val":
                self.indices = indices[split_idx:]
            else:
                raise KeyError("Invalid mode")
        
        self.cine_ds = self.cine_ds[self.indices, ...]  # (N', T, H', W')
        self.t_grid = torch.linspace(-1, 1, self.cine_ds.shape[1])  # (T,)
    
    def __init_lin_tfm(self, scale: float, undersampling_rate: float, mode=None):
        """
        scale * undersampling_rate = .max_undersampling_rate
        """
        mask_params = self.mask_config[undersampling_rate]
        if mode is None:
            mode = self.params["mode"]
        T_in = self.input_T if mode != "test" else self.params["T"]
        seed = self.params["seed"]

        lin_tfm_params = {
            "sens_type": "exp",
            "num_sens": 2,
            "in_shape": (T_in, self.res, self.res),
            "mask_params": {"if_temporal": True, **mask_params},
            "undersample_t": scale,
            "seed": seed
        }
        lin_tfm = SENSE(**lin_tfm_params)
        self.lin_tfm_dict[undersampling_rate] = lin_tfm

        return lin_tfm

    def __len__(self):
        return self.cine_ds.shape[0]
    
    def __getitem__(self, idx):
        img = self.cine_ds[idx, ...]  # (T, H, W)

        # adding phase
        if np.random.rand() <= self.params["data_aug_p"]:    
            img = add_phase(img[:, None, ...], init_shape=(5, 5, 5),  seed=None, mode="2d+time")
            img = img[:, 0, ...]  # (T, 1, H, W) -> (T, H, W)
        else:
            img = img.to(torch.complex64)
        img = img[None, ...]  # (1, T, H, W)

        measurement = self.test_lin_tfm(img)  # (1, T, H', W', num_sens)
        measurement = kspace_binning(measurement, self.input_T, self.test_lin_tfm.random_under_fourier.mask)  # (1, T0, H', W', num_sens)

        img = img.squeeze(0)  # (1, T0, H, W) -> (T0, H, W)
        measurement = measurement.squeeze(0)  # (1, T0, H, W, num_sens) -> (T0, H, W, num_sens)
        
        out_dict = {
            IMAGE_KEY: img,  # (T0, H, W)
            MEASUREMENT_KEY: measurement,
            ZF_KEY: self.test_lin_tfm.conj_op(measurement),
            COORD_KEY: self.t_grid  # (T0,)
        }
        
        return out_dict


class CINEImageKSDownSampleDM(LightningDataModule):
    def __init__(self, params: dict, mask_config: dict):
        """
        params: see CINEImageKSDownSampleDataset
                    batch_size: int, (test_batch_size: int), num_workers: int = 0
        """
        super().__init__()
        self.params = params.copy()
        self.params["mode"] = "train"
        self.train_ds = CINEImageKSDownSampleDataset(self.params, mask_config)
        self.params = params.copy()
        self.params["mode"] = "val"
        self.val_ds = CINEImageKSDownSampleDataset(self.params, mask_config)
        self.params = params.copy()
        self.params["mode"] = "test"
        self.test_ds = CINEImageKSDownSampleDataset(self.params, mask_config)

    def prepare_data(self) -> None:
        return super().prepare_data()
    
    def setup(self, stage: Union[str, None] = None) -> None:
        return super().setup(stage)
    
    def train_dataloader(self) -> TRAIN_DATALOADERS:
        loader = DataLoader(self.train_ds, self.params["batch_size"], shuffle=True, num_workers=self.params["num_workers"], 
                            pin_memory=True)

        return loader
    
    def val_dataloader(self) -> EVAL_DATALOADERS:
        loader = DataLoader(self.val_ds, self.params["batch_size"], num_workers=self.params["num_workers"], 
                            pin_memory=True)

        return loader
    
    def predict_dataloader(self) -> EVAL_DATALOADERS:
        batch_size = self.params.get("test_batch_size", self.params["batch_size"])
        loader = DataLoader(self.test_ds, batch_size, num_workers=self.params["num_workers"], 
                            pin_memory=True)

        return loader
    