import torch
import os
import yaml

FIGSIZE_UNIT = 3.6

ROOT = os.path.abspath(__file__)
for _ in range(2):
    ROOT = os.path.dirname(ROOT)


CONFIG_PATHS = {
    "spatial": os.path.join(ROOT, "configs", "spatial.yml"),
    "2d+time": os.path.join(ROOT, "configs", "2d_time.yml"),
    "2d+time+reg": os.path.join(ROOT, "configs", "2d_time_reg.yml"),
    "2d+time+explicit_reg": os.path.join(ROOT, "configs", "2d_time_explicit_reg.yml"),
    "2d+time_liif_param": os.path.join(ROOT, "configs", "2d_time_liif_parametric.yml"),
    "2d+time_liif_non_param": os.path.join(ROOT, "configs", "2d_time_liif_non_parametric.yml")
}


OPTIMIZER_MAP = {
    "AdamW": torch.optim.AdamW,
    "Adam": torch.optim.Adam,
    "LBFGS": torch.optim.LBFGS
}


SCHEDULER_MAP = {
    "StepLR": torch.optim.lr_scheduler.StepLR
}


def load_config(task_name: str) -> dict:
    assert task_name in CONFIG_PATHS
    config_filename = CONFIG_PATHS[task_name]
    with open(config_filename, "r") as rf:
        config_dict = yaml.load(rf, yaml.Loader)
    
    return config_dict
