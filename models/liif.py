import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import ImplicitNeuralRepr.utils.pytorch_utils as ptu

from monai.networks.nets import UNet
from ImplicitNeuralRepr.models.siren import SirenComplex
from collections import deque
from einops import rearrange
from tqdm import trange


class MLP(nn.Module):
    def __init__(self, params: dict):
        """
        params: in_features, out_features, hidden_features, hidden_layers
        """
        super().__init__()
        self.params = params
        self.net_list = [nn.Linear(self.params["in_features"], self.params["hidden_features"])]
        for _ in range(self.params["hidden_layers"]):
            self.net_list.extend([nn.ReLU(), nn.Linear(self.params["hidden_features"], self.params["hidden_features"])])
        self.net_list.append(nn.Linear(self.params["hidden_features"], self.params["out_features"]))
        self.net = nn.Sequential(*self.net_list)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x_out = self.net(x)

        return x_out


class LIIFParametric(nn.Module):
    def __init__(self, params: dict):
        """
        params: see 2d_time_liif_parametric.yml
        """
        super().__init__()
        self.params = params
        self.params["mlp"]["in_features"] = self.params["unet"]["out_channels"] + 1
        self.encoder_real = UNet(**self.params["unet"])
        self.encoder_imag = UNet(**self.params["unet"])
        self.mlp_real = MLP(self.params["mlp"])
        self.mlp_imag = MLP(self.params["mlp"])
    
    def __shared_forward(self, x_zf: torch.Tensor, encoder, mlp, t_coord=None) -> torch.Tensor:
        # x_zf: (B, T, H, W), t_coord: (T_any, )
        B, T, H, W = x_zf.shape
        if t_coord is None:
            t_coord = torch.linspace(-1, 1, T, device=x_zf.device)  # (T,)
        else:
            t_coord = t_coord.to(x_zf.device)
            T = t_coord.shape[0]
        x = encoder(x_zf)  # (B, C, H, W)
        x = torch.tanh(x)
        x = rearrange(x, "B C H W -> B H W C")
        t_coord = t_coord.reshape(1, T, 1, 1, 1)
        t_coord_expanded = t_coord.expand(B, T, H, W, 1)
        C = x.shape[-1]
        x_expanded = x.reshape(B, 1, H, W, C).expand(B, T, H, W, C)
        x_coord = torch.cat([x_expanded, t_coord_expanded], dim=-1)  # (B, T, H, W, C + 1)
        x_pred = mlp(x_coord)  # (B, T, H, W, 1)

        return x_pred

    def forward(self, x_zf: torch.Tensor, t_coord=None) -> torch.Tensor:
        x_pred_real = self.__shared_forward(torch.real(x_zf), self.encoder_real, self.mlp_real, t_coord)
        x_pred_imag = self.__shared_forward(torch.imag(x_zf), self.encoder_imag, self.mlp_imag, t_coord)
        x_pred = x_pred_real + 1j * x_pred_imag  # (B, T, H, W, 1)

        # (B, T, H, W)
        return x_pred.squeeze(-1)


class LIIFParametricComplexSiren(nn.Module):
    def __init__(self, params: dict):
        super().__init__()
        self.params = params
        self.params["mlp"]["in_features"] = self.params["unet"]["out_channels"] + 1
        self.encoder = UNet(**self.params["unet"])
        self.mlp = SirenComplex(self.params["mlp"])
    
    def forward(self, x_zf: torch.Tensor, t_coord=None):
        # x_zf: (B, T, H, W)
        B, T, H, W = x_zf.shape
        x_real = torch.real(x_zf)
        x_imag = torch.imag(x_zf)
        x = torch.cat([x_real, x_imag], dim=1)  # (B, 2 * T, H, W)
        if t_coord is None:
            t_coord = torch.linspace(-1, 1, T, device=x_zf.device)  # (T,)
        else:
            t_coord = t_coord.to(x_zf.device)
            T = t_coord.shape[0]
        x = self.encoder(x)  # (B, C, H, W)
        x = torch.tanh(x)
        x = rearrange(x, "B C H W -> B H W C")
        t_coord = t_coord.reshape(1, T, 1, 1, 1)
        t_coord_expanded = t_coord.expand(B, T, H, W, 1)
        C = x.shape[-1]
        x_expanded = x.reshape(B, 1, H, W, C).expand(B, T, H, W, C)
        x_coord = torch.cat([x_expanded, t_coord_expanded], dim=-1)  # (B, T, H, W, C + 1)
        x_pred = self.mlp(x_coord).squeeze(-1)  # (B, T, H, W, 1) -> (B, T, H, W)

        return x_pred

        
class LIIFNonParametric(nn.Module):
    def __init__(self, params: dict):
        """
        params: see 2d_time_liif_non_parametric.yml
        """
        super().__init__()
        self.params = params
        self.encoder_real = UNet(**self.params["unet"])
        self.encoder_imag = UNet(**self.params["unet"])
        self.num_heads = self.params["mlp"]["num_heads"]
        
    def __shared_forward(self, x_zf: torch.Tensor, encoder, t_coord=None) -> torch.Tensor:
        # x_zf: (B, T, H, W), t_coord: (T_any,)
        B, T, H, W = x_zf.shape
        if t_coord is None:
            t_coord = torch.linspace(-1, 1, T, device=x_zf.device)  # (T,)
        else:
            t_coord = t_coord.to(x_zf.device)
            T = t_coord.shape[0]
        x = encoder(x_zf)  # (B, C, H, W)
        x = rearrange(x, "B C H W -> B H W C")
        C = x.shape[-1]
        x = rearrange(x, "B H W (num_heads C1 C2) -> B H W num_heads C1 C2", num_heads=self.num_heads, C1=np.sqrt(C / self.num_heads).astype(int))
        C_head = x.shape[-1]
        t_coord_forward = t_coord.reshape(T, 1).expand(T, C_head)
        for idx_head in range(self.num_heads):
            weights_iter = x[..., idx_head, :, :]  # (B, H, W, C_head, C_head)
            if idx_head == 0:
                t_coord_forward = torch.einsum("BHWOI,TI->BTHWO", weights_iter, t_coord_forward)  # (B, T, H, W, C_head)
            else:
                t_coord_forward = torch.einsum("BHWOI,BTHWI->BTHWO", weights_iter, t_coord_forward)  # (B, T, H, W, C_head)
            if idx_head < self.params["mlp"]["num_heads"] - 1:
                t_coord_forward = torch.relu(t_coord_forward)
        
        x_pred = t_coord_forward[..., 0]  # (B, T, H, W, C_head) -> (B, T, H, W)

        return x_pred
    
    def forward(self, x_zf: torch.Tensor, t_coord=None) -> torch.Tensor:
        x_pred_real = self.__shared_forward(torch.real(x_zf), self.encoder_real, t_coord)
        x_pred_imag = self.__shared_forward(torch.imag(x_zf), self.encoder_imag, t_coord)
        x_pred = x_pred_real + 1j * x_pred_imag

        # (B, T, H, W)
        return x_pred


class LIIFParametric3DConv(nn.Module):
    def __init__(self, params: dict):
        super().__init__()
        self.params = params
        self.params["mlp"]["in_features"] = self.params["unet"]["out_channels"] + 1
        self.encoder = UNet(**self.params["unet"])
        self.mlp = SirenComplex(self.params["mlp"])
    
    def forward(self, x_zf: torch.Tensor, t_coord: torch.Tensor):
        # x_zf: (B, T, H, W), t_coord: (B, T'); T should be fixed to be standart input length
        B, T, H, W = x_zf.shape
        T_coord = t_coord.shape[-1]  # T'
        x_real = torch.real(x_zf)
        x_imag = torch.imag(x_zf)
        x = torch.stack([x_real, x_imag], dim=1)  # (B, 2, T, H, W)
        x_enc = self.encoder(x)  # (B, C, T, H, W)
        x_enc = rearrange(x_enc, "B C T H W -> B (C H W) T")  # (B, C', T)
        radius = 2 / (T - 1) / 2
        feat_coord = torch.linspace(-1, 1, T).to(x_zf.device).reshape(1, 1, T).expand(B, 1, T)  # (T,) -> (1, 1, T) -> (B, 1, T)
        
        preds, dists = deque(), deque()
        for v in [-1, 1]:
            t_coord_ = t_coord.clone()
            t_coord_ += radius * v + self.params["eps_shift"]
            t_coord_ = torch.clamp(t_coord_, -1 + self.params["eps_shift"], 1 - self.params["eps_shift"])
            t_coord_ = t_coord_.unsqueeze(-1)  # (B, T', 1)
            q_coord = ptu.grid_sample1D(feat_coord, t_coord_) .squeeze(1)  # (B, 1, T') -> (B, T')
            q_feats = ptu.grid_sample1D(x_enc, t_coord_)  # (B, C', T')
            q_feats = rearrange(q_feats, "B (C H W) T -> B T H W C", H=H, W=W)  # (B, T', H, W, C)
            rel_coord = (t_coord - q_coord) * (T - 1)
            rel_coord = rel_coord.reshape(B, T_coord, 1, 1, 1).expand(B, T_coord, H, W, 1)  # (B, T', H, W, 1)
            mlp_input = torch.cat([q_feats, rel_coord], dim=-1)  # (B, T', H, W, C + 1)
            pred = self.mlp(mlp_input).squeeze(-1)  # (B, T', H, W, 1) -> (B, T', H, W), complex-valued
            dist = torch.abs(rel_coord)
            preds.append(pred)
            dists.appendleft(dist)
        
        preds = torch.stack(list(preds), dim=0)  # [(B, T', H, W)...] -> (2, B, T', H, W)
        dists = torch.stack(list(dists), dim=0).reshape(-1, B, T_coord, H, W)  # [(B, T')...] -> (2, B, T') -> (2, B, T', 1, 1)
        pred = preds * dists / dists.sum(dim=0)  # (2, B, T', H, W)
        pred = pred.sum(dim=0)  

        # (B, T', H, W)
        return pred


@torch.no_grad()
def sliding_window_inference(x_zf: torch.Tensor, upsample_rate: float, roi_size: int, overlap: float, predictor: nn.Module) -> torch.Tensor:
    """
    x_zf: (B, T, H, W), roi_size: T0
    Twice: forward and backward to accomodate corners 
    """
    if_train = predictor.training
    predictor.eval()

    B, T, H, W = x_zf.shape
    T_out_per_window = int(roi_size * upsample_rate)  # T'
    t_coord = torch.linspace(-1, 1, T_out_per_window).unsqueeze(0).expand(B, -1)  # (B, T')
    stride = int(roi_size * overlap)
    T_out = int(T * upsample_rate)

    # forward
    x_out = torch.zeros((B, T_out, H, W), device=x_zf.device, dtype=x_zf.dtype)
    counter = torch.zeros((T_out,), device=x_zf.device)
    pbar = trange(0, T + 1 - roi_size, stride, desc="forward", leave=False)
    for idx in pbar:
        x_zf_in = x_zf[:, idx:idx + roi_size, ...]
        x_pred_iter = predictor(x_zf_in, t_coord)  # (B, T', H, W)
        tgt_idx = int(idx * upsample_rate)
        x_out[:, tgt_idx:tgt_idx + T_out_per_window, ...] += x_pred_iter
        counter[tgt_idx:tgt_idx + T_out_per_window] += 1
    x_out_forward = x_out / counter.reshape(-1, 1, 1)

    # backward
    x_out = torch.zeros((B, T_out, H, W), device=x_zf.device, dtype=x_zf.dtype)
    counter = torch.zeros((T_out,), device=x_zf.device)
    pbar = trange(0, T + 1 - roi_size, stride, desc="backward", leave=False)
    x_zf = x_zf.flip(1)  # (3, 2), (1, 0)
    for idx in pbar:
        x_zf_in = x_zf[:, idx:idx + roi_size, ...]  
        x_zf_in = x_zf_in.flip(1)  # (3, 2) -> (2, 3)
        x_pred_iter = predictor(x_zf_in, t_coord)  # (B, T', H, W)
        tgt_idx = int(idx * upsample_rate)
        x_out[:, tgt_idx:tgt_idx + T_out_per_window, ...] += x_pred_iter.flip(1)  # (2, 3) -> (3, 2), assuming upsample_rate == 1
        counter[tgt_idx:tgt_idx + T_out_per_window] += 1
    
    x_out_backward = x_out / counter.reshape(-1, 1, 1)  # (3, 2), (1, 0)
    x_out_backward = x_out_backward.flip(1)  # (0, 1), (2, 3)
    x_out = (x_out_forward + x_out_backward) / 2

    if if_train:
        predictor.train()
    
    # (B, T_out, H, W)
    return x_out
