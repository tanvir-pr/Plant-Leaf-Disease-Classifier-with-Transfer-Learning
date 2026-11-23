"""Paths, config, and seeding."""

from __future__ import annotations

import os
import random
from pathlib import Path

import numpy as np
import yaml

def project_root() -> Path:
    return Path(__file__).resolve().parents[1]


def load_config(path: Path | None = None) -> dict:
    config_path = path or project_root() / "configs" / "plant.yaml"
    with config_path.open(encoding="utf-8") as handle:
        config = yaml.safe_load(handle)
    if not isinstance(config, dict):
        raise ValueError(f"Config at {config_path} must be a mapping.")
    return config


def set_seed(seed: int) -> None:
    """Fix Python and NumPy seeds. PyTorch is seeded when it is installed."""
    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    try:
        import torch
    except ImportError:
        return
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
