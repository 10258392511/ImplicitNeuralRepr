import torch
import torch.nn.functional as F

DEVICE = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")


def to_numpy(x: torch.Tensor):
    return x.detach().cpu().numpy()


def grid_sample1D(kernel: torch.Tensor, grid: torch.Tensor, mode: str = "nearest") -> torch.Tensor:
    """
    kernel: (B, C, T), grid: (B, T', 1)
    """
    B, C, T = kernel.shape
    kernel = kernel.unsqueeze(-1).expand(B, C, T, 3)  # (B, C, T, 1) -> (B, C, T, 3)
    grid = grid.unsqueeze(1)  # (B, 1, T', 1)
    grid = torch.cat([-torch.ones_like(grid), grid], dim=-1)  # (B, 1, T', 2)
    output = F.grid_sample(kernel, grid, mode, padding_mode="reflection", align_corners=True)  # (B, C, 1, T')
    output = output.squeeze(-2)  # (B, C, T')

    # print(f"kernel:\n{kernel[0, 0, :, 0]}")
    # print(f"grid:\n{grid[0, 0, :, 1]}")
    # print(f"output:\n{output[0, 0, :]}")
    # print("-" * 100)

    return output


def grid_sample_cplx(imgs, disps, **kwargs):
    im_r = F.grid_sample(imgs.real, disps, **kwargs)
    im_i = F.grid_sample(imgs.imag, disps, **kwargs)
    
    return torch.complex(im_r, im_i)

