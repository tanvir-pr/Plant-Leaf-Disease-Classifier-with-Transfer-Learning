"""Metric helpers checked against scikit-learn."""

import numpy as np
from sklearn.metrics import accuracy_score, f1_score, top_k_accuracy_score

from src.metrics import classification_metrics


def test_classification_metrics_match_sklearn():
    y_true = np.asarray([0, 1, 2, 3, 1, 2], dtype=np.int64)
    y_pred = np.asarray([0, 1, 1, 3, 1, 2], dtype=np.int64)
    y_prob = np.asarray(
        [
            [0.80, 0.05, 0.05, 0.04, 0.03, 0.03],
            [0.10, 0.70, 0.08, 0.05, 0.04, 0.03],
            [0.15, 0.55, 0.20, 0.05, 0.03, 0.02],
            [0.05, 0.05, 0.10, 0.70, 0.05, 0.05],
            [0.08, 0.75, 0.07, 0.04, 0.03, 0.03],
            [0.10, 0.10, 0.65, 0.05, 0.05, 0.05],
        ],
        dtype=np.float64,
    )
    labels = [0, 1, 2, 3, 4, 5]
    metrics = classification_metrics(y_true, y_pred, y_prob=y_prob, labels=labels)

    assert metrics["accuracy"] == accuracy_score(y_true, y_pred)
    assert metrics["macro_f1"] == f1_score(
        y_true, y_pred, average="macro", zero_division=0
    )
    assert metrics["top3"] == top_k_accuracy_score(
        y_true, y_prob, k=3, labels=labels
    )
    assert metrics["top5"] == top_k_accuracy_score(
        y_true, y_prob, k=5, labels=labels
    )
