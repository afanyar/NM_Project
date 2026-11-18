import json
import os
import random
from pathlib import Path

import numpy as np
import torch
import yaml


def set_seed(seed, deterministic=False):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    torch.backends.cudnn.benchmark = not deterministic
    torch.backends.cudnn.deterministic = deterministic


def load_yaml(path):
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def save_json(obj, path):
    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f, indent=2, ensure_ascii=False)


def make_run_id(aug_name, frac, seed, equal_steps=False):
    suffix = "__eqsteps" if equal_steps else ""
    return f"{aug_name}__f{frac:g}__s{seed}{suffix}"


def run_dir(out, run_id):
    return Path(out) / "runs" / run_id
