"""Grad-CAM, Grad-CAM++, and SmoothGrad saliency maps."""

from __future__ import annotations

from typing import Sequence

import cv2
import numpy as np
import torch
from captum.attr import NoiseTunnel, Saliency
from pytorch_grad_cam import GradCAM, GradCAMPlusPlus
from pytorch_grad_cam.utils.image import show_cam_on_image
from pytorch_grad_cam.utils.model_targets import ClassifierOutputTarget
from torch import nn

from src.transforms import IMAGENET_MEAN, IMAGENET_STD, denormalize


def cam_target_layers(model: nn.Module) -> list[nn.Module]:
    """Last convolutional stage used for Grad-CAM overlays."""
    if hasattr(model, "features"):
        return [model.features[-1]]
    if hasattr(model, "layer4"):
        return [model.layer4[-1]]
    raise ValueError("Unsupported backbone for Grad-CAM target layers.")


def tensor_to_rgb(image: torch.Tensor) -> np.ndarray:
    """Convert a normalised CHW tensor to an RGB float image in [0, 1]."""
    if image.ndim != 3:
        raise ValueError("Expected a single CHW image tensor.")
    rgb = denormalize(image.detach().cpu()).permute(1, 2, 0).numpy()
    return np.clip(rgb, 0.0, 1.0).astype(np.float32)


def overlay_cam(rgb: np.ndarray, grayscale_cam: np.ndarray) -> np.ndarray:
    heatmap = show_cam_on_image(rgb, grayscale_cam, use_rgb=True)
    return heatmap


def compute_gradcam(
    model: nn.Module,
    images: torch.Tensor,
    target_classes: Sequence[int],
    plus_plus: bool = False,
) -> np.ndarray:
    """Return Grad-CAM maps with shape (N, H, W)."""
    model.eval()
    targets = [ClassifierOutputTarget(int(cls)) for cls in target_classes]
    layers = cam_target_layers(model)
    cam_cls = GradCAMPlusPlus if plus_plus else GradCAM
    with cam_cls(model=model, target_layers=layers) as cam:
        return cam(input_tensor=images, targets=targets)


def compute_smoothgrad(
    model: nn.Module,
    images: torch.Tensor,
    target_classes: Sequence[int],
    n_samples: int = 25,
    stdevs: float = 0.15,
) -> np.ndarray:
    """SmoothGrad saliency maps aggregated over channels, shape (N, H, W)."""
    model.eval()
    images = images.clone().detach().requires_grad_(True)
    saliency = Saliency(model)
    tunnel = NoiseTunnel(saliency)
    attributions = tunnel.attribute(
        images,
        nt_type="smoothgrad",
        nt_samples=n_samples,
        stdevs=stdevs,
        target=torch.tensor(list(target_classes), device=images.device),
        abs=True,
    )
    maps = attributions.detach().cpu().numpy()
    maps = maps.mean(axis=1)
    maps = maps - maps.min(axis=(1, 2), keepdims=True)
    denom = maps.max(axis=(1, 2), keepdims=True)
    denom = np.where(denom < 1e-8, 1.0, denom)
    return (maps / denom).astype(np.float32)


def side_by_side(
    rgb: np.ndarray,
    gradcam: np.ndarray,
    gradcam_pp: np.ndarray,
    smoothgrad: np.ndarray,
) -> np.ndarray:
    """Stack original, Grad-CAM, Grad-CAM++, and SmoothGrad left to right."""
    panels = [
        (rgb * 255).astype(np.uint8),
        overlay_cam(rgb, gradcam),
        overlay_cam(rgb, gradcam_pp),
        overlay_cam(rgb, smoothgrad),
    ]
    return np.concatenate(panels, axis=1)
