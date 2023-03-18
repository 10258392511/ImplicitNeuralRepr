import os
import yaml

ROOT = os.path.abspath(__file__)
for _ in range(2):
    ROOT = os.path.dirname(ROOT)


CONFIG_PATHS = {
    "spatial": os.path.join(ROOT, "configs", "spatial.yml")
}


def load_config(task_name: str) -> dict:
    assert task_name in CONFIG_PATHS
    config_filename = CONFIG_PATHS[task_name]
    with open(config_filename, "r") as rf:
        config_dict = yaml.load(rf, yaml.Loader)
    
    return config_dict
