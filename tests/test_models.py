"""Shape checks for the scratch plant classifier."""

import torch

from src.metrics import classification_metrics
from src.models import ScratchCNN, build_model


def test_scratch_cnn_output_shape():
    model = ScratchCNN(num_classes=38)
    logits = model(torch.randn(4, 3, 224, 224))
    assert logits.shape == (4, 38)


def test_build_model_scratch_alias():
    model = build_model("scratch_cnn", num_classes=38)
    assert isinstance(model, ScratchCNN)


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
