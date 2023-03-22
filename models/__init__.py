import os

from .siren import Siren, SirenComplex
from .grid_sample import GridSample
from ImplicitNeuralRepr.configs import load_config
from typing import Union

ROOT = os.path.abspath(__file__)
for _ in range(2):
    ROOT = os.path.dirname(ROOT)


MODEL_NAME_MAP = {
    "SirenComplex": SirenComplex
}


# task names: see CONFIG_PATHS from configs/__init__.py
MODEL_RELOAD_PATHS = {
    "spatial": "spatial_logs/2023_03_17_21_52_38_197732",
    "2d+time": None
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
