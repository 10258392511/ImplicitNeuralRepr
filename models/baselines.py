import torch
import torch.nn as nn
import torch.nn.functional as F
import ImplicitNeuralRepr.utils.pytorch_utils as ptu

# from ImplicitNeuralRepr.models.reconstruction.networks.nets.varnet import VariationalNetworkModel
# from ImplicitNeuralRepr.models.reconstruction.networks.nets.complex_unet import ComplexUnet
# from ImplicitNeuralRepr.models.reconstruction.networks.nets.coil_sensitivity_model import CoilSensitivityModel
from ImplicitNeuralRepr.linear_transforms import FiniteDiff, i2k_complex
from ImplicitNeuralRepr.configs import (
    IMAGE_KEY,
    ZF_KEY,
    MASK_KEY,
    COORD_KEY,
    MEASUREMENT_KEY
)


# class VariationalNetworkUpsample(nn.Module):
#     def __init__(self, params: dict):
#         """
#         params: see 2d_time_VN.yml
#         """
#         super().__init__()
#         self.params = params
#         self.coil_sens_model = CoilSensitivityModel(spatial_dims=2, features=self.params["sensitivity_model_features"])
#         self.refinement_model = ComplexUnet(spatial_dims=2, features=self.params["features"])
#         self.vn = VariationalNetworkModel(self.coil_sens_model, self.refinement_model, self.params["num_cascades"]).to(ptu.DEVICE)
#         self.vn.load_state_dict(torch.load(self.params["ckpt_path"]))

#     def forward(self, batch: dict):
#         # B = 1
#         ksp = batch[MEASUREMENT_KEY]  # (B, T, H, W)
#         mask = batch[MASK_KEY]  # (B, T, 1, W)
#         img = batch[IMAGE_KEY]  # (B, T', H, W)
#         zf = batch[ZF_KEY]  # (B, T, H, W)
#         ksp = ksp.unsqueeze(1)  # (B, 1, T, H, W)
#         ksp = torch.view_as_real(ksp)  # (B, 1, T, H, W, 2)
#         mask = mask.unsqueeze(-1)  # (B, T, 1, W, 1)
#         recons = torch.zeros_like(zf)  # (B, T, H, W)
#         scale = img.shape[1] / zf.shape[1]

#         for t in range(ksp.shape[1]):
#             ksp_iter = ksp[:, :, t, ...]  # (B, 1, H, W, 2)
#             mask_iter = mask[:, t:t + 1, ...]  # (B = 1, 1, 1, W, 1)
#             recons[:, t, ...] = self.vn(ksp_iter, mask_iter)  # (B, H, W)
        
#         recons = recons.unsuqeeze(1)  # (B, 1, T, H, W)
#         recons = F.interpolate(recons, (scale, 1., 1.), mode="trilinear")  # (B, 1, T', H, W)
#         recons = recons.squeeze(1)  # (B, T', H, W)

#         return recons


class TemporalTV(nn.Module):
    def __init__(self, params: dict, batch: dict):
        super().__init__()
        self.params = params
        self.zf = batch[ZF_KEY]  # (T, H, W)
        self.ksp = batch[MEASUREMENT_KEY]  # (T, H, W)
        self.mask = batch[MASK_KEY]  # (T, 1, W)
        self.finite_diff_t = FiniteDiff(dim=0)
        
        self.kernel = nn.Parameter(self.zf, requires_grad=True)
        self.num_params = self.kernel.numel()
        self.mask = self.mask.to(ptu.DEVICE)
        self.ksp = self.ksp.to(ptu.DEVICE)

    def forward(self):
        recons = self.kernel  # (T, H, W)
        dc_loss = 0.5 * torch.norm(self.mask * i2k_complex(recons, dims=-1) - self.ksp) ** 2
        reg_loss = torch.abs(self.finite_diff_t(recons)).sum()
        loss =  dc_loss * + reg_loss * self.params["lambda"]

        recons = recons.unsqueeze(0)  # (1, T, H, W)

        return recons, loss
