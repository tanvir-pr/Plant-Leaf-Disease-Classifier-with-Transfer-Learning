"""Load trained plant classifiers from disk."""

from __future__ import annotations

from pathlib import Path

import torch
from torch import nn

from src.models import build_model


def load_plant_checkpoint(
    path: Path,
    device: torch.device | str = "cpu",
    pretrained: bool = False,
) -> tuple[nn.Module, list[str], dict]:
    payload = torch.load(path, map_location="cpu", weights_only=False)
    class_names = list(payload["class_names"])
    config = payload.get("config") or {}
    train_cfg = config.get("train") or {}
    model_name = str(train_cfg.get("model", "efficientnet_b0"))
    strategy = str(train_cfg.get("strategy", "full"))
    model = build_model(
        model_name,
        num_classes=len(class_names),
        strategy=strategy,
        pretrained=pretrained,
    )
    model.load_state_dict(payload["model_state"])
    model.to(device)
    model.eval()
    return model, class_names, config
