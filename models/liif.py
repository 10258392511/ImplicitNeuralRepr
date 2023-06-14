import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import ImplicitNeuralRepr.utils.pytorch_utils as ptu

from monai.networks.nets import UNet
from .rdn import RDN
from monai.networks.blocks import Convolution
from ImplicitNeuralRepr.models.siren import SirenComplex
from ImplicitNeuralRepr.linear_transforms import i2k_complex, k2i_complex
from ImplicitNeuralRepr.configs import (
    IMAGE_KEY,
    MEASUREMENT_KEY,
    ZF_KEY,
    COORD_KEY,
    MASK_KEY
)
from collections import deque, defaultdict
from einops import rearrange
from tqdm import trange
from typing import Union


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
        self.encoder = None
        self.__load_encoder()
        self.mlp = SirenComplex(self.params["mlp"])
    
    def __load_encoder(self):
        enc_name = self.params["encoder"]["name"]
        enc_config = self.params[enc_name]
        self.params["mlp"]["in_features"] = enc_config["out_channels"] + 1
        if enc_name == "rdn":
            self.encoder = RDN(**enc_config)
        elif enc_name == "unet":
            self.encoder = UNet(**enc_config)
        else:
            raise NotImplementedError

    def forward(self, x_zf: torch.Tensor, t_coord: torch.Tensor, **kwargs):
        # x_zf: (B, T, H, W), t_coord: (B, T'); T should be fixed to be standard input length
        B, T, H, W = x_zf.shape
        T_coord = t_coord.shape[-1]  # T'
        x_real = torch.real(x_zf)
        x_imag = torch.imag(x_zf)
        x = torch.stack([x_real, x_imag], dim=1)  # (B, 2, T, H, W)
        x_enc = self.encoder(x)  # (B, C, T, H, W)
        # feat_coord = torch.linspace(-1, 1, T).to(x_zf.device).reshape(1, 1, T).expand(B, 1, T)  # (T,) -> (1, 1, T) -> (B, 1, T)
        feat_coord = torch.linspace(-1, 1, T).to(x_zf.device)  # (T,)

        d = 2 / (T - 1)
        t_coord = t_coord.clamp(-1, 1 - self.params["eps_shift"])
        t_inds = torch.floor((t_coord + 1) / d).long()  # (B, T')
        tau = t_coord - feat_coord[t_inds]  # (B, T')
        tau /= d
        x_enc = rearrange(x_enc, "B C T H W -> B T H W C")
        t_inds = t_inds.reshape(*t_inds.shape, 1, 1, 1).expand(-1, -1, *x_enc.shape[2:])  # (B, T', H, W, C)
        x_enc_floor = torch.gather(x_enc, 1, t_inds)  # (B, T', H, W, C)
        x_enc_ceil = torch.gather(x_enc, 1, t_inds + 1)  # (B, T', H, W, C)
        # taus = [tau, -(1 - tau)]
        preds = []

        t_coord = t_coord.reshape(*t_coord.shape, 1, 1, 1).expand(-1, -1, H, W, -1)  # (B, T', H, W, 1)
        for x_enc_iter in [x_enc_floor, x_enc_ceil]:
            mlp_in = torch.cat([x_enc_iter, t_coord], dim=-1)  # (B, T', H, W, C + 1)
            pred_iter = self.mlp(mlp_in).squeeze(-1)  # (B, T', H, W, 1) -> (B, T', H, W)
            preds.append(pred_iter)
        
        # for x_enc_iter, tau_iter in zip([x_enc_floor, x_enc_ceil], taus):
        #     tau_iter = tau_iter.reshape(*tau_iter.shape, 1, 1, 1).expand(-1, -1, H, W, -1)  # (B, T', H, W, C)
        #     mlp_in = torch.cat([x_enc_iter, tau_iter], dim=-1)  # (B, T', H, W, C + 1)
        #     pred_iter = self.mlp(mlp_in).squeeze(-1)  # (B, T', H, W, 1) -> (B, T', H, W)
        #     preds.append(pred_iter)

        tau = tau.reshape(*tau.shape, 1, 1)  # (B, T', 1, 1)
        x_out = preds[0] * (1 - tau) + preds[1] * tau

        # (B, T', H, W)
        return x_out
    
    def forward_bilinear(self, x_zf: torch.Tensor, t_coord: torch.Tensor, if_debug=False):
        # x_zf: (B, T, H, W), t_coord: (B, T'); T should be fixed to be standard input length
        B, T, H, W = x_zf.shape
        T_coord = t_coord.shape[-1]  # T'
        x_real = torch.real(x_zf)
        x_imag = torch.imag(x_zf)
        x = torch.stack([x_real, x_imag], dim=1)  # (B, 2, T, H, W)
        x_enc = self.encoder(x)  # (B, C, T, H, W)
        x_enc = rearrange(x_enc, "B C T H W -> B (C H W) T")  # (B, C', T)
        feat_coord = torch.linspace(-1, 1, T).to(x_zf.device).reshape(1, 1, T).expand(B, 1, T)  # (T,) -> (1, 1, T) -> (B, 1, T)
        
        if if_debug:
            intermediates_dict = defaultdict(deque)
            intermediates_dict["x_enc"].append(rearrange(x_enc, "B (C H W) T -> B C T H W", H=H, W=W))
            intermediates_dict["feat_coord"].append(feat_coord)

        t_coord_ = t_coord.unsqueeze(-1)  # (B, T', 1)
        q_coord = ptu.grid_sample1D(feat_coord, t_coord_, mode="bilinear").squeeze(1)  # (B, 1, T') -> (B, T')
        q_feats = ptu.grid_sample1D(x_enc, t_coord_, mode="bilinear")  # (B, C', T')
        q_feats = rearrange(q_feats, "B (C H W) T -> B T H W C", H=H, W=W)  # (B, T', H, W, C)
        rel_coord = t_coord
        rel_coord = rel_coord.reshape(B, T_coord, 1, 1, 1).expand(B, T_coord, H, W, 1)  # (B, T', H, W, 1)
        mlp_input = torch.cat([q_feats, rel_coord], dim=-1)  # (B, T', H, W, C + 1)
        pred = self.mlp(mlp_input).squeeze(-1)  # (B, T', H, W, 1) -> (B, T', H, W), complex-valued

        if if_debug:
            intermediates_dict["t_coord"].append(t_coord)
            intermediates_dict["t_coord_"].append(t_coord_)
            intermediates_dict["q_coord"].append(q_coord)
            intermediates_dict["q_feats"].append(rearrange(q_feats, "B T H W C -> B C T H W"))
            intermediates_dict["rel_coord"].append(rel_coord)
            intermediates_dict["pred"].append(pred)

        if if_debug:
            return pred, intermediates_dict

        # (B, T', H, W)
        return pred
    
    def forward_nearest(self, x_zf: torch.Tensor, t_coord: torch.Tensor, if_debug=False):
        # x_zf: (B, T, H, W), t_coord: (B, T'); T should be fixed to be standard input length
        B, T, H, W = x_zf.shape
        T_coord = t_coord.shape[-1]  # T'
        x_real = torch.real(x_zf)
        x_imag = torch.imag(x_zf)
        x = torch.stack([x_real, x_imag], dim=1)  # (B, 2, T, H, W)
        x_enc = self.encoder(x)  # (B, C, T, H, W)
        x_enc = rearrange(x_enc, "B C T H W -> B (C H W) T")  # (B, C', T)
        radius = 2 / (T - 1) / 2
        # radius = 2 / T / 2
        feat_coord = torch.linspace(-1, 1, T).to(x_zf.device).reshape(1, 1, T).expand(B, 1, T)  # (T,) -> (1, 1, T) -> (B, 1, T)
        
        preds, dists = deque(), deque()

        if if_debug:
            intermediates_dict = defaultdict(deque)
            intermediates_dict["x_enc"].append(rearrange(x_enc, "B (C H W) T -> B C T H W", H=H, W=W))
            intermediates_dict["feat_coord"].append(feat_coord)

        # jitter t_coord to prevent NaN in x_enc (heuristic)
        t_coord += self.params["eps_shift"] * (np.random.randint(0, 2) * 2 - 1)
        for v in [-1, 1]:
            t_coord_ = t_coord.clone()
            t_coord_ += radius * v + self.params["eps_shift"]
            t_coord_ = torch.clamp(t_coord_, -1 + self.params["eps_shift"], 1 - self.params["eps_shift"])
            t_coord_ = t_coord_.unsqueeze(-1)  # (B, T', 1)
            q_coord = ptu.grid_sample1D(feat_coord, t_coord_).squeeze(1)  # (B, 1, T') -> (B, T')
            q_feats = ptu.grid_sample1D(x_enc, t_coord_)  # (B, C', T')
            q_feats = rearrange(q_feats, "B (C H W) T -> B T H W C", H=H, W=W)  # (B, T', H, W, C)
            rel_coord = (t_coord - q_coord) * (T - 1)
            # rel_coord = (t_coord - q_coord) * T
            rel_coord = rel_coord.reshape(B, T_coord, 1, 1, 1).expand(B, T_coord, H, W, 1)  # (B, T', H, W, 1)
            mlp_input = torch.cat([q_feats, rel_coord], dim=-1)  # (B, T', H, W, C + 1)
            pred = self.mlp(mlp_input).squeeze(-1)  # (B, T', H, W, 1) -> (B, T', H, W), complex-valued
            dist = torch.abs(rel_coord)
            preds.append(pred)
            dists.appendleft(dist)

            if if_debug:
                intermediates_dict["t_coord"].append(t_coord)
                intermediates_dict["t_coord_"].append(t_coord_)
                intermediates_dict["q_coord"].append(q_coord)
                intermediates_dict["q_feats"].append(rearrange(q_feats, "B T H W C -> B C T H W"))
                intermediates_dict["rel_coord"].append(rel_coord)
                intermediates_dict["pred"].append(pred)
        
        preds = torch.stack(list(preds), dim=0)  # [(B, T', H, W)...] -> (2, B, T', H, W)
        dists = torch.stack(list(dists), dim=0).reshape(-1, B, T_coord, H, W)  # [(B, T', H, W)...] -> (2, B, T', H, W)
        pred = preds * dists / dists.sum(dim=0)  # (2, B, T', H, W)
        pred = pred.sum(dim=0)

        if if_debug:
            return pred, intermediates_dict

        # (B, T', H, W)
        return pred


@torch.no_grad()
def sliding_window_inference(x_zf: torch.Tensor, upsample_rate: float, roi_size: int, overlap: float, predictor: nn.Module) -> torch.Tensor:
    """
    x_zf: (B, T, H, W), roi_size: T0
    Twice: forward and backward to accomodate corners 
    """
    x_zf = x_zf.to(ptu.DEVICE)
    if_train = predictor.training
    predictor.to(x_zf.device)
    predictor.eval()

    B, T, H, W = x_zf.shape
    T_out_per_window = int(roi_size * upsample_rate)  # T'
    t_coord = torch.linspace(-1, 1, T_out_per_window).unsqueeze(0).expand(B, -1)  # (B, T')
    t_coord = t_coord.to(x_zf.device)
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
        assert not torch.any(torch.isnan(x_out)), f"idx = {idx}"
        counter[tgt_idx:tgt_idx + T_out_per_window] += 1
    
    # counter[counter == 0] = 1
    # x_out_forward = x_out / counter.reshape(-1, 1, 1)

    # # backward
    # x_out = torch.zeros((B, T_out, H, W), device=x_zf.device, dtype=x_zf.dtype)
    # counter = torch.zeros((T_out,), device=x_zf.device)
    # pbar = trange(T + 1 - roi_size, -1, stride, desc="backward", leave=False)
    # x_zf = x_zf.flip(1)  # (3, 2), (1, 0)
    # for idx in pbar:
    #     x_zf_in = x_zf[:, idx:idx + roi_size, ...]  
    #     x_zf_in = x_zf_in.flip(1)  # (3, 2) -> (2, 3)
    #     x_pred_iter = predictor(x_zf_in, t_coord)  # (B, T', H, W)
    #     tgt_idx = int(idx * upsample_rate)
    #     assert not torch.any(torch.isnan(x_out)), f"idx = {idx}"
    #     x_out[:, tgt_idx:tgt_idx + T_out_per_window, ...] += x_pred_iter.flip(1)  # (2, 3) -> (3, 2), assuming upsample_rate == 1
    #     counter[tgt_idx:tgt_idx + T_out_per_window] += 1
    
    # counter[counter == 0] = 1
    # x_out_backward = x_out / counter.reshape(-1, 1, 1)  # (3, 2), (1, 0)
    # x_out_backward = x_out_backward.flip(1)  # (0, 1), (2, 3)
    # x_out = (x_out_forward + x_out_backward) / 2

    # only add the last window
    if torch.any(counter < 1):
        x_zf_in = x_zf[:, -roi_size:, ...]
        x_pred_iter = predictor(x_zf_in, t_coord)  # (B, T', H, W)
        tgt_idx = int(idx * upsample_rate)
        mask = (counter < 1)
        num_zeros = mask.sum()
        x_out[:, mask, ...] += x_pred_iter[:, -num_zeros:, ...]
        assert not torch.any(torch.isnan(x_out)), f"idx = {idx}"
        counter[mask] += 1
    
    x_out /= counter.reshape(-1, 1, 1)

    if if_train:
        predictor.train()
    
    # (B, T_out, H, W)
    return x_out


class LIIFCascade(nn.Module):
    def __init__(self, params: dict):
        super().__init__()
        self.params = params
        self.enc_tuples = []
        if self.params["encoder_type"] == "conv":
            out_channels = self.params["conv"]["out_channels"]
            self.enc_tuples.append((self.__make_conv_block(2), self.__make_conv_block(out_channels + 2)))
            for _ in range(self.params["num_cascade_times"] - 1):
                self.enc_tuples.append((self.__make_conv_block(), self.__make_conv_block(out_channels + 2)))

        elif self.params["encoder_type"] == "unet":
            out_channels = self.params["unet"]["out_channels"]
            self.enc_tuples.append((self.__make_unet(2), self.__make_conv_block(out_channels + 2)))
            for _ in range(self.params["num_cascade_times"] - 1):
                self.enc_tuples.append((self.__make_unet(), self.__make_conv_block(out_channels + 2)))
        else:
            raise NotImplementedError
        
        self.params["mlp"]["in_features"] = self.params["unet"]["out_channels"] + 1
        self.mlp = SirenComplex(self.params["mlp"])
    
    def __make_conv_block(self, in_channels: Union[int, None] = None):
        conv_params = self.params["conv"].copy()
        if in_channels is not None:
            conv_params["in_channels"] = in_channels
        layers = [Convolution(**conv_params)]
        for _ in range(self.params["num_conv_layers"] - 1):
            conv_params = self.params["conv"].copy()
            layers.append(Convolution(**conv_params))
        
        conv_params = self.params["conv"].copy()
        conv_params["conv_only"] = True
        layers.append(Convolution(**conv_params))
        net = nn.Sequential(*layers)

        return net

    def __make_unet(self, in_channels: Union[int, None] = None):
        unet_params = self.params["unet"].copy()
        if in_channels is not None:
            unet_params["in_channels"] = in_channels
        net = UNet(**unet_params)

        return net
    
    def encode(self, enc_head: nn.Module, enc_tail: nn.Module, x_last_enc: torch.Tensor, ksp: torch.Tensor, mask: torch.Tensor):
        enc_head.to(x_last_enc.device)
        enc_tail.to(x_last_enc.device)
        B, C, T, H, W = x_last_enc.shape
        x = enc_head(x_last_enc)  # (B, C or 2, T, H, W) -> (B, C, T, H, W)
        feat_coord = torch.linspace(-1, 1, T).to(x_last_enc.device)  # (T,)
        feat_coord = feat_coord.reshape(1, 1, T, 1, 1).expand(B, -1, -1, H, W)  # (B, 1, T, H, W)
        mlp_in = torch.cat([x, feat_coord], dim=1)  # (B, C + 1, T, H, W)
        mlp_in = rearrange(mlp_in, "B C T H W -> B T H W C")
        x_recons = self.mlp(mlp_in).squeeze(-1)  # (B, T, H, W)
        # ksp: (B, T, H, W), mask: (B, T, 1, W)
        x_res = k2i_complex(mask * (i2k_complex(x_recons, dims=-1) - ksp), dims=-1)  # (B, T, H, W)
        x_res = torch.stack([torch.real(x_res), torch.imag(x_res)], dim=1)  # (B, 2, T, H, W)
        x = torch.cat([x, x_res], dim=1)  # (B, C + 2, T, H, W)
        x = enc_tail(x)  # (B, C, T, H, W)

        return x

    def forward(self, **data_dict):
        x = data_dict[IMAGE_KEY]  # (B, T, H, W)
        ksp = data_dict[MEASUREMENT_KEY]
        x_zf = data_dict[ZF_KEY]
        t_coord = data_dict[COORD_KEY]  # (B, T')
        mask = data_dict[MASK_KEY]  # (B, T, 1, W)

        B, T, H, W = x_zf.shape
        x_real = torch.real(x_zf)
        x_imag = torch.imag(x_zf)
        x_enc = torch.stack([x_real, x_imag], dim=1)  # (B, 2, T, H, W)
        for conv_head, conv_tail in self.enc_tuples:
            x_enc = self.encode(conv_head, conv_tail, x_enc, ksp, mask)
        
        # x_enc: (B, C, T, H, W)
        feat_coord = torch.linspace(-1, 1, T).to(x_zf.device)  # (T,)

        d = 2 / (T - 1)
        t_coord = t_coord.clamp(-1, 1 - self.params["eps_shift"])
        t_inds = torch.floor((t_coord + 1) / d).long()  # (B, T')
        tau = t_coord - feat_coord[t_inds]  # (B, T')
        tau /= d
        x_enc = rearrange(x_enc, "B C T H W -> B T H W C")
        t_inds = t_inds.reshape(*t_inds.shape, 1, 1, 1).expand(-1, -1, *x_enc.shape[2:])  # (B, T', H, W, C)
        x_enc_floor = torch.gather(x_enc, 1, t_inds)  # (B, T', H, W, C)
        x_enc_ceil = torch.gather(x_enc, 1, t_inds + 1)  # (B, T', H, W, C)
        preds = []

        t_coord = t_coord.reshape(*t_coord.shape, 1, 1, 1).expand(-1, -1, H, W, -1)  # (B, T', H, W, 1)
        for x_enc_iter in [x_enc_floor, x_enc_ceil]:
            mlp_in = torch.cat([x_enc_iter, t_coord], dim=-1)  # (B, T', H, W, C + 1)
            pred_iter = self.mlp(mlp_in).squeeze(-1)  # (B, T', H, W, 1) -> (B, T', H, W)
            preds.append(pred_iter)

        tau = tau.reshape(*tau.shape, 1, 1)  # (B, T', 1, 1)
        x_out = preds[0] * (1 - tau) + preds[1] * tau

        # (B, T', H, W)
        return x_out
           