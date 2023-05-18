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
    grid = grid.unsqueeze(1)  # (B, 1, T', 1)
    grid = torch.cat([-torch.ones_like(grid), grid], dim=-1)  # (B, 1, T', 2)
    output = F.grid_sample(kernel, grid, mode)  # (B, C, 1, T')
    output = output.squeeze(-2)  # (B, C, T')

    return output
