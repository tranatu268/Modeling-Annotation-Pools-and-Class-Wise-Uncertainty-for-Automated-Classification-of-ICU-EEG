# Authors: Tran Anh Tu, Huynh Hoang Nhut
from pathlib import Path

import yaml


def load_config(path="config.yaml"):
    with open(path, encoding="utf-8") as f:
        cfg = yaml.safe_load(f)
    for k in ("data_dir", "work_dir"):
        cfg[k] = Path(cfg[k])
    return cfg
