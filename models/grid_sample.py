import torch
import torch.nn as nn
import torch.nn.functional as F


class GridSample(nn.Module):
    def __init__(self, params: dict):
        """
        params: seed, kernel_shape: (1, C, H0, W0) or (1, C, T0, H0, W0)
        """
        super().__init__()
        self.params = params
        seed = self.params.get("seed", None)
        if seed is not None:
            torch.manual_seed(seed)
        low_res = torch.randn(*self.params["kernel_shape"])
        self.low_res = nn.parameter.Parameter(low_res)
    
    def forward(self, x: torch.Tensor):
        # x: (H, W, 2) or (T, H, W, 3)
        x = x.unsqueeze(0)  # (1, H, W, 2) or (1, T, H, W, 3)
        x_out = F.grid_sample(self.low_res, x)  # (1, C, H, W) or (1, C, T, H, W)
        x_out = x_out.squeeze(0)  # (C, H, W) or (C, T, H, W)
        
        return x_out
