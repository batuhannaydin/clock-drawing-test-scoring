"""Device + seed helpers shared across train / evaluate."""
from __future__ import annotations

import random
from typing import Union

import numpy as np
import torch


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def get_device() -> torch.device:
    if torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


def describe_device(d: Union[torch.device, str]) -> str:
    d = torch.device(d) if isinstance(d, str) else d
    if d.type == "cuda":
        return f"cuda ({torch.cuda.get_device_name(0)})"
    return "cpu"
