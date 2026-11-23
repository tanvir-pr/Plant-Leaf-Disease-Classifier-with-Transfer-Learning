"""Classification metrics used on the validation split."""

from __future__ import annotations

import numpy as np
from sklearn.metrics import accuracy_score, f1_score, top_k_accuracy_score


def classification_metrics(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    y_prob: np.ndarray | None = None,
    labels: list[int] | None = None,
) -> dict[str, float]:
    metrics = {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "macro_f1": float(f1_score(y_true, y_pred, average="macro", zero_division=0)),
    }
    if y_prob is not None and labels is not None and len(labels) >= 5:
        metrics["top3"] = float(
            top_k_accuracy_score(y_true, y_prob, k=3, labels=labels)
        )
        metrics["top5"] = float(
            top_k_accuracy_score(y_true, y_prob, k=5, labels=labels)
        )
    return metrics
