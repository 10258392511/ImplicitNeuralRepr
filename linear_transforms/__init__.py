import os
import yaml

from .base import i2k_complex, k2i_complex, LinearTransform, Indentity
from .finite_diff import FiniteDiff
from .undersampling_fourier import RandomUndersamplingFourier, SENSE


ROOT = os.path.abspath(__file__)
for _ in range(2):
    ROOT = os.path.dirname(ROOT)

CONFIG_PATHS = {
    "spatial": os.path.join(ROOT, "configs", "spatial.yml")
}

TFM_NAME_MAP = {
    "SENSE": SENSE,
    "FiniteDiff": FiniteDiff,
    "Identity": Indentity
}


def load_linear_transform(task_name: str, mode: str, **kwargs):
    assert task_name in CONFIG_PATHS
    assert mode in ["dc", "reg"]
    with open(CONFIG_PATHS[task_name], "r") as rf:
        config_dict = yaml.load(rf, yaml.Loader)
    lin_tfm_params = config_dict["transforms"][mode]
    lin_tfm_name = lin_tfm_params.pop("name")
    lin_tfm_ctor = TFM_NAME_MAP[lin_tfm_name]
    lin_tfm = lin_tfm_ctor(**lin_tfm_params)

    return lin_tfm
