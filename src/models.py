"""Backbones, classification heads, and freeze helpers."""

from __future__ import annotations

import torch
from torch import nn
from torchvision import models
from torchvision.models import ResNet50_Weights


class ScratchCNN(nn.Module):
    """Small CNN trained from random weights as a plant baseline."""

    def __init__(self, num_classes: int) -> None:
        super().__init__()
        if num_classes < 2:
            raise ValueError("num_classes must be at least 2.")
        self.features = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),
            nn.Conv2d(128, 256, kernel_size=3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.AdaptiveAvgPool2d(1),
        )
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Dropout(0.3),
            nn.Linear(256, num_classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.classifier(self.features(x))


def set_trainable(module: nn.Module, trainable: bool) -> None:
    for param in module.parameters():
        param.requires_grad = trainable


def count_trainable_parameters(model: nn.Module) -> int:
    return sum(param.numel() for param in model.parameters() if param.requires_grad)


def trainable_parameters(model: nn.Module):
    return [param for param in model.parameters() if param.requires_grad]


def apply_freeze_strategy(model: nn.Module, strategy: str) -> nn.Module:
    """Freeze pretrained layers for transfer learning.

    Strategies:
    - frozen: train only the final classifier
    - last_block: train layer4 and the classifier
    - full: train every parameter
    """
    key = strategy.lower().replace("-", "_")
    if key not in {"frozen", "last_block", "full"}:
        raise ValueError(f"Unknown freeze strategy: {strategy}")

    if not hasattr(model, "fc"):
        raise ValueError("Freeze helpers expect a model with an fc classifier.")

    set_trainable(model, True)
    if key == "full":
        return model

    set_trainable(model, False)
    set_trainable(model.fc, True)
    if key == "last_block":
        set_trainable(model.layer4, True)
    return model


def set_transfer_train_mode(model: nn.Module, strategy: str) -> None:
    """Keep frozen BatchNorm layers in eval so their running stats stay fixed."""
    key = strategy.lower().replace("-", "_")
    if key == "full":
        model.train()
        return
    model.train()
    if key == "frozen":
        model.eval()
        model.fc.train()
        return
    model.eval()
    model.layer4.train()
    model.fc.train()


def build_resnet50(
    num_classes: int,
    strategy: str = "frozen",
    pretrained: bool = True,
) -> nn.Module:
    if num_classes < 2:
        raise ValueError("num_classes must be at least 2.")
    weights = ResNet50_Weights.IMAGENET1K_V2 if pretrained else None
    model = models.resnet50(weights=weights)
    model.fc = nn.Linear(model.fc.in_features, num_classes)
    return apply_freeze_strategy(model, strategy)


def build_model(
    name: str,
    num_classes: int,
    strategy: str = "frozen",
    pretrained: bool = True,
) -> nn.Module:
    key = name.lower().replace("-", "_")
    if key in {"scratch", "scratch_cnn"}:
        return ScratchCNN(num_classes)
    if key in {"resnet50", "resnet_50"}:
        return build_resnet50(
            num_classes,
            strategy=strategy,
            pretrained=pretrained,
        )
    raise ValueError(f"Unknown model: {name}")
