import os

from .base import i2k_complex, k2i_complex, LinearTransform, Indentity
from .finite_diff import FiniteDiff
from .undersampling_fourier import RandomUndersamplingFourier, SENSE
from ImplicitNeuralRepr.configs import load_config
from typing import Union


ROOT = os.path.abspath(__file__)
for _ in range(2):
    ROOT = os.path.dirname(ROOT)


TFM_NAME_MAP = {
    "SENSE": SENSE,
    "FiniteDiff": FiniteDiff,
    "Identity": Indentity
}


def load_linear_transform(task_name: str, mode: str, config_dict: Union[dict, None] = None, **kwargs):
    assert mode in ["dc", "reg"]
    if config_dict is None:
        config_dict = load_config(task_name)
    lin_tfm_params = config_dict["transforms"][mode]
    lin_tfm_name = lin_tfm_params.pop("name")
    lin_tfm_ctor = TFM_NAME_MAP[lin_tfm_name]
    lin_tfm = lin_tfm_ctor(**lin_tfm_params)

    return lin_tfm
