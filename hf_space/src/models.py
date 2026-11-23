"""Backbones, classification heads, and freeze helpers."""

from __future__ import annotations

import torch
from torch import nn
from torchvision import models
from torchvision.models import (
    EfficientNet_B0_Weights,
    MobileNet_V2_Weights,
    ResNet50_Weights,
)


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


def classifier_prefix(model: nn.Module) -> str:
    if hasattr(model, "fc"):
        return "fc."
    if hasattr(model, "classifier"):
        return "classifier."
    raise ValueError("Model has no fc or classifier head.")


def classifier_module(model: nn.Module) -> nn.Module:
    if hasattr(model, "fc"):
        return model.fc
    if hasattr(model, "classifier"):
        return model.classifier
    raise ValueError("Model has no fc or classifier head.")


def last_block_module(model: nn.Module) -> nn.Module:
    if hasattr(model, "layer4"):
        return model.layer4
    if hasattr(model, "features"):
        # EfficientNet / MobileNet: unfreeze the final feature stage.
        return model.features[-1]
    raise ValueError("Model has no last-block module.")


def build_optimizer_param_groups(
    model: nn.Module,
    strategy: str,
    head_lr: float,
    backbone_lr: float | None = None,
) -> list[dict]:
    """Lower learning rate for pretrained layers, higher for the new head."""
    key = strategy.lower().replace("-", "_")
    backbone_lr = head_lr if backbone_lr is None else backbone_lr

    if isinstance(model, ScratchCNN) or key == "frozen":
        params = trainable_parameters(model)
        if not params:
            raise RuntimeError("Model has no trainable parameters.")
        return [{"params": params, "lr": head_lr}]

    prefix = classifier_prefix(model)
    head_params = [
        param
        for name, param in model.named_parameters()
        if param.requires_grad and name.startswith(prefix)
    ]
    if key == "last_block":
        block = last_block_module(model)
        backbone_params = [param for param in block.parameters() if param.requires_grad]
    else:
        backbone_params = [
            param
            for name, param in model.named_parameters()
            if param.requires_grad and not name.startswith(prefix)
        ]

    groups: list[dict] = []
    if backbone_params:
        groups.append({"params": backbone_params, "lr": backbone_lr})
    if head_params:
        groups.append({"params": head_params, "lr": head_lr})
    if not groups:
        raise RuntimeError("Model has no trainable parameters.")
    return groups


def apply_freeze_strategy(model: nn.Module, strategy: str) -> nn.Module:
    """Freeze pretrained layers for transfer learning.

    Strategies:
    - frozen: train only the final classifier
    - last_block: train the last feature block and the classifier
    - full: train every parameter
    """
    key = strategy.lower().replace("-", "_")
    if key not in {"frozen", "last_block", "full"}:
        raise ValueError(f"Unknown freeze strategy: {strategy}")

    head = classifier_module(model)
    set_trainable(model, True)
    if key == "full":
        return model

    set_trainable(model, False)
    set_trainable(head, True)
    if key == "last_block":
        set_trainable(last_block_module(model), True)
    return model


def set_transfer_train_mode(model: nn.Module, strategy: str) -> None:
    """Keep frozen BatchNorm layers in eval so their running stats stay fixed."""
    if isinstance(model, ScratchCNN):
        model.train()
        return

    key = strategy.lower().replace("-", "_")
    if key == "full":
        model.train()
        return

    model.train()
    head = classifier_module(model)
    if key == "frozen":
        model.eval()
        head.train()
        return

    model.eval()
    last_block_module(model).train()
    head.train()


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


def build_efficientnet_b0(
    num_classes: int,
    strategy: str = "full",
    pretrained: bool = True,
) -> nn.Module:
    if num_classes < 2:
        raise ValueError("num_classes must be at least 2.")
    weights = EfficientNet_B0_Weights.IMAGENET1K_V1 if pretrained else None
    model = models.efficientnet_b0(weights=weights)
    in_features = model.classifier[-1].in_features
    model.classifier[-1] = nn.Linear(in_features, num_classes)
    return apply_freeze_strategy(model, strategy)


def build_mobilenet_v2(
    num_classes: int,
    strategy: str = "full",
    pretrained: bool = True,
) -> nn.Module:
    if num_classes < 2:
        raise ValueError("num_classes must be at least 2.")
    weights = MobileNet_V2_Weights.IMAGENET1K_V2 if pretrained else None
    model = models.mobilenet_v2(weights=weights)
    in_features = model.classifier[-1].in_features
    model.classifier[-1] = nn.Linear(in_features, num_classes)
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
    if key in {"efficientnet_b0", "efficientnetb0"}:
        return build_efficientnet_b0(
            num_classes,
            strategy=strategy,
            pretrained=pretrained,
        )
    if key in {"mobilenet_v2", "mobilenetv2"}:
        return build_mobilenet_v2(
            num_classes,
            strategy=strategy,
            pretrained=pretrained,
        )
    raise ValueError(f"Unknown model: {name}")
