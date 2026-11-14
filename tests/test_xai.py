"""Shape checks for Grad-CAM helpers."""

import torch

from src.models import build_model
from src.xai import cam_target_layers, compute_gradcam, compute_smoothgrad, tensor_to_rgb


def test_cam_target_layers_for_efficientnet():
    model = build_model(
        "efficientnet_b0",
        num_classes=38,
        strategy="full",
        pretrained=False,
    )
    layers = cam_target_layers(model)
    assert len(layers) == 1


def test_gradcam_and_smoothgrad_shapes():
    model = build_model(
        "efficientnet_b0",
        num_classes=38,
        strategy="full",
        pretrained=False,
    )
    model.eval()
    images = torch.randn(2, 3, 224, 224)
    targets = [0, 1]
    cam = compute_gradcam(model, images, targets, plus_plus=False)
    cam_pp = compute_gradcam(model, images, targets, plus_plus=True)
    smooth = compute_smoothgrad(model, images, targets, n_samples=3)
    assert cam.shape == (2, 224, 224)
    assert cam_pp.shape == (2, 224, 224)
    assert smooth.shape == (2, 224, 224)


def test_tensor_to_rgb_range():
    image = torch.zeros(3, 224, 224)
    rgb = tensor_to_rgb(image)
    assert rgb.shape == (224, 224, 3)
    assert rgb.min() >= 0.0
    assert rgb.max() <= 1.0
