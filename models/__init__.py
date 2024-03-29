import torch.nn as nn
import os

from .siren import Siren, SirenComplex
from .grid_sample import GridSample
from .liif import (
    LIIFParametric, 
    LIIFParametricComplexSiren, 
    LIIFNonParametric,
    LIIFParametric3DConv,
    LIIFCascade
)
from .rdn import RDN
# from .baselines import VariationalNetworkUpsample
from .baselines import TemporalTV
from ImplicitNeuralRepr.configs import load_config
from typing import Union

ROOT = os.path.abspath(__file__)
for _ in range(2):
    ROOT = os.path.dirname(ROOT)


MODEL_NAME_MAP = {
    "SirenComplex": SirenComplex,
    "LIIFParametric": LIIFParametric,
    "LIIFNonParametric": LIIFNonParametric,
    "LIIFParametricComplexSiren": LIIFParametricComplexSiren,
    "LIIFParametric3DConv": LIIFParametric3DConv,
    "LIIFCascade": LIIFCascade,
    # "VariationalNetworkUpsample": VariationalNetworkUpsample
    "TemporalTV": TemporalTV  # load it with a batch; DON'T use load_model(.)
}


# task names: see CONFIG_PATHS from configs/__init__.py
MODEL_RELOAD_PATHS = {
    "spatial": "spatial_logs/2023_03_17_21_52_38_197732",
    "2d+time": "2d_time_logs/repeat_temporal_samples/siren_1_grid_0_repeat_t_6_epochs_300/2023_04_22_00_47_11_233446",
    "2d+time+explicit_reg": "2d_time_explicit_reg_logs/width_vs_depth/depth_8_width_512/2023_04_27_13_03_32_323027",
    "2d+time_liif_param": "2d_time_liif_logs/tune_enc_out_channels/C_8/2023_05_10_22_56_21_913628/",
    "2d+time_liif_3d_conv": "2d_time_liif_3d_conv_logs/binning_num_unet_out_channels_data_aug_0_5/C_16/2023_05_24_11_45_22_922523",
    # "2d+time_liif_cont": "2d_time_liif_cont_logs/architecture_search/final_results/unet_R_3_T_low_4/2023_06_21_16_58_10_089934",
    # "2d+time_liif_cont": "2d_time_liif_cont_logs/architecture_search/final_results/unet_R_6_T_low_4/2023_06_21_16_56_48_044263",
    # "2d+time_liif_cont": "2d_time_liif_cont_logs/architecture_search/final_results/unet_R_6/2023_06_14_20_47_19_285334",
    # "2d+time_liif_cont": "2d_time_liif_cont_logs/architecture_search/final_results/unet_R_3/2023_06_14_10_35_45_212586",
    # "2d+time_liif_cont": "2d_time_liif_cont_logs/architecture_search/final_results/RDN_R_3_T_low_4/2023_06_20_21_08_45_334229",
    # "2d+time_liif_cont": "2d_time_liif_cont_logs/architecture_search/final_results/RDN_R_6_T_low_4/2023_06_20_20_59_18_492488",
    # "2d+time_liif_cont": "2d_time_liif_cont_logs/architecture_search/final_results/RDN_R_3/2023_06_14_10_35_05_497101",
    "2d+time_liif_cont": "2d_time_liif_cont_logs/architecture_search/final_results/RDN_R_6_T_low_8_new/2023_06_21_10_09_46_575451"
}


def load_model(task_name: str, config_dict: Union[dict, None] = None, **kwargs):
    if config_dict is None:
        config_dict = load_config(task_name)
    model_params = config_dict["model"]
    model_name = model_params.pop("name")
    model_ctor = MODEL_NAME_MAP[model_name]
    model = model_ctor(model_params)

    return model


def reload_model(task_name: str) -> str:
    ckpt_path = os.path.join(ROOT, MODEL_RELOAD_PATHS[task_name], "checkpoints", "last.ckpt")
    
    return ckpt_path


def gaussian_init(model: SirenComplex, gaussian_params: dict) -> SirenComplex:
    """
    The first layer use uniform initialization; the rest use Gaussian initialization.

    gaussian_params: {"mean": ..., "std": ...}
    """
    for i, layer_iter in enumerate(model.siren.net_list):
        # if i == 0 or i == len(model.siren.net_list) - 1:
        if i == 0:
            continue
        if isinstance(layer_iter, nn.Linear):
            nn.init.normal_(layer_iter.weight, gaussian_params["mean"][i], gaussian_params["std"][i])
        else:    
            nn.init.normal_(layer_iter.linear.weight, gaussian_params["mean"][i], gaussian_params["std"][i])
    
    return model
