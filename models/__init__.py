import os
import yaml

from .siren import Siren, SirenComplex

ROOT = os.path.abspath(__file__)
for _ in range(2):
    ROOT = os.path.dirname(ROOT)

CONFIG_PATHS = {
    "spatial": os.path.join(ROOT, "configs", "spatial.yml")
}

MODEL_NAME_MAP = {
    "SirenComplex": SirenComplex
}


def load_model(task_name: str, **kwargs):
    assert task_name in CONFIG_PATHS
    config_filename = CONFIG_PATHS[task_name]
    with open(config_filename, "r") as rf:
        config_dict = yaml.load(rf, yaml.Loader)
    model_params = config_dict["model"]
    model_name = model_params.pop("name")
    model_ctor = MODEL_NAME_MAP[model_name]
    model = model_ctor(model_params)

    return model
