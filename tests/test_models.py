"""Shape and freeze checks for plant classifiers."""

import torch

from src.metrics import classification_metrics
from src.models import (
    ScratchCNN,
    apply_freeze_strategy,
    build_model,
    count_trainable_parameters,
)


def test_scratch_cnn_output_shape():
    model = ScratchCNN(num_classes=38)
    logits = model(torch.randn(4, 3, 224, 224))
    assert logits.shape == (4, 38)


def test_build_model_scratch_alias():
    model = build_model("scratch_cnn", num_classes=38)
    assert isinstance(model, ScratchCNN)


def test_resnet50_frozen_trains_only_fc():
    model = build_model(
        "resnet50",
        num_classes=38,
        strategy="frozen",
        pretrained=False,
    )
    trainable = [name for name, param in model.named_parameters() if param.requires_grad]
    assert trainable
    assert all(name.startswith("fc.") for name in trainable)
    expected = model.fc.weight.numel() + model.fc.bias.numel()
    assert count_trainable_parameters(model) == expected


def test_resnet50_last_block_unfreezes_layer4():
    model = build_model(
        "resnet50",
        num_classes=38,
        strategy="last_block",
        pretrained=False,
    )
    trainable = {name for name, param in model.named_parameters() if param.requires_grad}
    assert any(name.startswith("layer4.") for name in trainable)
    assert any(name.startswith("fc.") for name in trainable)
    assert not any(name.startswith("layer3.") for name in trainable)


def test_resnet50_output_shape():
    model = build_model(
        "resnet50",
        num_classes=38,
        strategy="frozen",
        pretrained=False,
    )
    logits = model(torch.randn(2, 3, 224, 224))
    assert logits.shape == (2, 38)


def test_apply_freeze_rejects_unknown_strategy():
    model = build_model("resnet50", num_classes=38, strategy="full", pretrained=False)
    try:
        apply_freeze_strategy(model, "partial")
        raised = False
    except ValueError:
        raised = True
    assert raised


def test_classification_metrics_include_topk():
    y_true = [0, 1, 2, 3]
    y_pred = [0, 1, 1, 3]
    y_prob = [
        [0.9, 0.04, 0.03, 0.02, 0.005, 0.005],
        [0.1, 0.8, 0.05, 0.04, 0.005, 0.005],
        [0.2, 0.5, 0.2, 0.08, 0.01, 0.01],
        [0.05, 0.05, 0.1, 0.78, 0.01, 0.01],
    ]
    metrics = classification_metrics(
        y_true,
        y_pred,
        y_prob=y_prob,
        labels=[0, 1, 2, 3, 4, 5],
    )
    assert "accuracy" in metrics
    assert "macro_f1" in metrics
    assert "top3" in metrics
    assert "top5" in metrics
