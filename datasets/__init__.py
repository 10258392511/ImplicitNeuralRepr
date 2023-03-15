import os

from .cine import load_cine
from .coord import MetaCoordDM, CoordDataset


ROOT_DIR = os.path.abspath(__file__)
for _ in range(2):
    ROOT_DIR = os.path.dirname(ROOT_DIR)

DATA_PATHS = {
    "CINE64": os.path.join(ROOT_DIR, "data/cine_64"),
    "CINE127": os.path.join(ROOT_DIR, "data/cine_127"),
}

LOADER = {
    "CINE64": load_cine,
    "CINE127": load_cine
}


def load_data(ds_name: str, mode: str, **kwargs):
    assert ds_name in DATA_PATHS.keys()
    assert mode in ["train", "val", "test"]

    data_dir = DATA_PATHS[ds_name]
    loader = LOADER[ds_name]
    ds = loader(data_dir, mode=mode, **kwargs)

    return ds
