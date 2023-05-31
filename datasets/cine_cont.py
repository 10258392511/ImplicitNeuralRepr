import numpy as np
import torch
import torch.nn.functional as F
import einops
import h5py
import os

from ImplicitNeuralRepr.linear_transforms import i2k_complex, k2i_complex
from ImplicitNeuralRepr.utils.pytorch_utils import grid_sample_cplx
from torch.utils.data import Dataset, DataLoader
from pytorch_lightning import LightningDataModule
from torch.distributions import Uniform
from ImplicitNeuralRepr.configs import (
    IMAGE_KEY,
    MEASUREMENT_KEY,
    ZF_KEY,
    COORD_KEY,
    MASK_KEY
)
from pytorch_lightning.utilities.types import EVAL_DATALOADERS, TRAIN_DATALOADERS
from typing import Union

ROOT_DIR = os.path.abspath(__file__)
for _ in range(2):
    ROOT_DIR = os.path.dirname(ROOT_DIR)
DATA_PATH = os.path.join(ROOT_DIR, "data/cont_cine/cine_disps.h5")


def resample(imgs, disp_f, disp_b, taus, if_pad=True):
    B, T, Nx, Ny = imgs.shape

    assert torch.max(taus) <= 1 and torch.min(taus) >= 0
    taus = torch.clamp(taus, 1e-6, 1 - 1e-6)
    taus = taus * (T - 1)

    g1, g2 = torch.meshgrid(torch.linspace(-1,1, Nx), torch.linspace(-1,1, Ny), indexing='ij')  
    g1 = einops.rearrange(g1, 'h w -> 1 h w 1')   
    g2 = einops.rearrange(g2, 'h w -> 1 h w 1')
    g = torch.cat((g1, g2), dim=-1).to(disp_f.device)

    taus = taus.to(disp_b.device)
    tauf = torch.floor(taus).long()
    k = taus - tauf
    ka = einops.repeat(k, 't -> (b t) 1 1 1', b=B)
    
    img1 = einops.rearrange(imgs[:, tauf, ...], 'b t h w -> (b t) 1 h w')
    img2 = einops.rearrange(imgs[:, tauf + 1, ...], 'b t h w -> (b t) 1 h w')
    d1 = einops.rearrange(disp_f[:, tauf, ...], 'b t c h w -> (b t) h w c')
    d2 = einops.rearrange(disp_b[:, tauf, ...], 'b t c h w -> (b t) h w c')
    ns = einops.rearrange(np.array([Nx, Ny]), 'C -> 1 1 1 C')
    ns = torch.tensor(ns, dtype=torch.float32).to(disp_f)

    d1 = d1 / (ns - 1) * 2. * ka + g
    d2 = d2 / (ns - 1) * 2. * (1 - ka) + g
    d1 = d1[..., [1, 0]]
    d2 = d2[..., [1, 0]]

    imd1 = grid_sample_cplx(img1, d1, align_corners=True, mode='bicubic')
    imd2 = grid_sample_cplx(img2, d2, align_corners=True, mode='bicubic')
    imhr2 = (1 - ka) * imd1 + ka * imd2

    imgs = einops.rearrange(imhr2, '(b t) 1 h w -> b t h w', b=B)
    if if_pad:
        imgs = F.pad(imgs, (0, 1, 0, 1), mode="constant", value=0)  # (B, T, 127, 127) -> (B, T, 128, 128), padded at right and bottom one 0
    
    return imgs

def gen_mask_laplace_1d(B, Nx, R, sgm=0.3, dlt=0.01):
    gx = torch.linspace(-1,1, Nx)
    px = torch.exp(-gx.abs() / sgm) + dlt
    px = px / px.max()
    q = px.mean() / (1/R)
    m = torch.rand(B, Nx) * q < einops.rearrange(px, 'w -> 1 w') 
    m[:, Nx//2] = 1.
    m = m.float()
    
    return m

def leave_only_first_nonzero(mask, dim=-1):
    mask_f = mask * (mask.float().cumsum(dim=dim) == 1)
    mask_f = mask_f.to(mask)
    
    return mask_f


class ContDataSampler(object):
    def __init__(self, imgs, disp_f, disp_b, T_high=32, T_low=8):
        # T_high is time resolution at which the kspace is generated (in theory the higher -- the closer we are to continuous)
        # T_low is # of bins that you actually receive 
        self.imgs = imgs.astype(np.complex64)
        self.disp_f = disp_f.astype(np.float32)
        self.disp_b = disp_b.astype(np.float32)
        self.T_high = T_high
        self.T_low = T_low
        self.device = torch.device("cpu")

    def get_image_at_t(self, batch_idxs, taus):
        # you have to provide batch indices and time points at which to generate the image
        imgs = self.imgs[batch_idxs, ...]  # (B, T, H, W)
        disp_f = self.disp_f[batch_idxs, ...]
        disp_b = self.disp_b[batch_idxs, ...]
        imgs = torch.tensor(imgs, dtype=torch.complex64).to(self.device)
        disp_f = torch.tensor(disp_f, dtype=torch.float32).to(self.device)
        disp_b = torch.tensor(disp_b, dtype=torch.float32).to(self.device)
        if not isinstance(taus, torch.Tensor):
            taus = torch.tensor(taus)
        taus = taus.float().to(self.device)
        
        return resample(imgs, disp_f, disp_b, taus)
    
    def sample_us_kspace(self, batch_idxs, R_25):
        # R_25 is the undersampling factor at 25 frames per second. There are a lot of binning so it's approximate
        R_high = R_25 * self.T_high / 25  # keeping number of measurements, remap the acceleration rate
        R_low = R_25 * self.T_low / 25
        B = len(batch_idxs)
        taus = torch.linspace(0, 1, self.T_high).float().to(self.device)
        imgs_hr = self.get_image_at_t(batch_idxs, taus) # b Tlow*f h w

        Nx = imgs_hr.shape[-1]
        mask_hr = gen_mask_laplace_1d(B * self.T_high, Nx, R_high).to(self.device)
        mask_hr = einops.rearrange(mask_hr, '(b T_low f) w -> b T_low f w', \
                                   T_low=int(self.T_low), f=int(self.T_high//self.T_low)).float().to(self.device)
        mask_hr_first = leave_only_first_nonzero(mask_hr, dim=2)
        mask_hr = einops.rearrange(mask_hr_first, 'b T_low f w -> b T_low f 1 w')
        mask_lr = einops.reduce(mask_hr_first, 'b T_low f w -> b T_low 1 w', 'sum')

        kspc_hr = i2k_complex(imgs_hr, dims=-1)
        kspc_hr = einops.rearrange(kspc_hr, 'b (T_low f) h w -> b T_low f h w', T_low=int(self.T_low))
        kspc_lr = einops.reduce(kspc_hr * mask_hr, 'b T_low f h w -> b T_low h w', 'sum')
        # assert torch.all( (kspc_lr.imag.abs() > 0).float() == mask_lr )
        # print(f'R_25={R_25:.1f}, R_high={R_high:.1f}, R_low={R_low}')
        # print(f'R_high={1/mask_hr_first.mean():.1f}, R_low={1/mask_lr.float().mean():.1f}')  # larger: since only the first non-zero measurement is kept in each bin
       
        return kspc_lr, mask_lr


class CINEContDataset(Dataset):
    def __init__(self, params: dict, imgs: np.ndarray, disp_f: np.ndarray, disp_b: np.ndarray):
        """
        params: mode: str {train | val | test}, seed: {int | None}, train_val_split: float,
                T_high = 32, T_low = 8, T_query = 16, R_25 = 6

        imgs: (N, T, H, W); disp_f, disp_b: (N, T - 1, 2, H, W) where T = 25, H = W = 127 
        """
        super().__init__()
        self.params = params
        total_num_subjects = imgs.shape[0]

        # train-val split
        seed = self.params.get("seed", None)
        if seed is not None:
            np.random.seed(seed)
        indices = np.arange(total_num_subjects)
        np.random.shuffle(indices)
        if self.params["mode"] == "test":
            self.indices = indices  # return all images for evaluation without cropping in T-axis; callback: using the last image
        else:
            split_idx = int(total_num_subjects * self.params["train_val_split"])
            if self.params["mode"] == "train":
                self.indices = indices[:split_idx]
            elif self.params["mode"] == "val":
                self.indices = indices[split_idx:]
            else:
                raise KeyError("Invalid mode")
        
        imgs = imgs[self.indices, ...]  # (N', T, H, W)
        disp_f = disp_f[self.indices, ...]  # (N', T - 1, 2, H, W)
        disp_b = disp_b[self.indices, ...]
        self.t_anchors = torch.linspace(0., 1., self.params["T_query"])  # (T',)
        self.radius = 0.5 / (self.params["T_query"] - 1)
        self.unif = Uniform(-self.radius, self.radius)

        self.sampler = ContDataSampler(imgs, disp_f, disp_b, T_high=self.params["T_high"], T_low=self.params["T_low"])
    
    def __len__(self):
        return self.sampler.imgs.shape[0]
    
    def __getitem__(self, idx: int):
        taus = self.t_anchors.clone()
        if self.params["mode"] == "train":
            random_shift = self.unif.sample(self.t_anchors.shape)
            taus += random_shift

        taus = taus.clamp(0, 1)
        img = self.sampler.get_image_at_t([idx], taus).squeeze(0)  # (T', H', W')
        kspc, mask = self.sampler.sample_us_kspace([idx], self.params["R_25"])
        kspc = kspc.squeeze(0)  # (T_low, H', W')
        mask = mask.squeeze(0)  # (T_low, 1, W')

        out_dict = {
            IMAGE_KEY: img,  # (T', H', W')
            MEASUREMENT_KEY: kspc,  # (T_low, H', W')
            ZF_KEY: k2i_complex(kspc, dims=-1),  # (T_low, H', W')
            COORD_KEY: taus,  # (T',),
            MASK_KEY: mask  # (T_low, 1, W')
        }

        return out_dict


class CINEContDM(LightningDataModule):
    def __init__(self, params: dict):
        """
        params: see CINEContDataset
                    data_filename, batch_size: int, (test_batch_size: int), num_workers: int = 0
        """
        super().__init__()
        self.params = params
        data_filename = self.params.get("data_filename", DATA_PATH)
        with h5py.File(data_filename, "r") as rf:
            self.imgs = np.array(rf["imgs_r"]) + 1j * np.array(rf['imgs_i'])
            self.disp_f = np.array(rf["disp_f"]).astype(np.float32)
            self.disp_b = np.array(rf["disp_b"]).astype(np.float32)
        
        self.train_ds, self.val_ds, self.test_ds = None, None, None

    def prepare_data(self) -> None:
        return super().prepare_data()
    
    def setup(self, stage: Union[str, None] = None) -> None:
        params = self.params.copy()
        params["mode"] = "train"
        self.train_ds = CINEContDataset(params, self.imgs, self.disp_f, self.disp_b)

        params = self.params.copy()
        params["mode"] = "val"
        self.val_ds = CINEContDataset(params, self.imgs, self.disp_f, self.disp_b)

        params = self.params.copy()
        params["mode"] = "test"
        self.test_ds = CINEContDataset(params, self.imgs, self.disp_f, self.disp_b)
    
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
