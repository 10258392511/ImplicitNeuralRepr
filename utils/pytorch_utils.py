import torch
import torch.nn.functional as F

DEVICE = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")


def to_numpy(x: torch.Tensor):
    return x.detach().cpu().numpy()


def grid_sample1D(kernel: torch.Tensor, grid: torch.Tensor, mode: str = "nearest") -> torch.Tensor:
    """
    kernel: (B, C, T), grid: (B, T', 1)
    """
    kernel = kernel.unsqueeze(-1)  # (B, C, T, 1)
    grid = grid.unsqueeze(-1)  # (B, T', 1, 1)
    grid = torch.cat([-torch.ones_like(grid), grid], dim=-1)  # (B, T', 1, 2)
    output = F.grid_sample(kernel, grid, mode)  # (B, C, T', 1)
    output = output.squeeze(-1)  # (B, C, T')

    return output
