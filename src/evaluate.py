"""Validation metrics and a single held-out test evaluation."""

from __future__ import annotations

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader
from tqdm import tqdm

from src.metrics import classification_metrics


@torch.no_grad()
def evaluate_loader(
    model: nn.Module,
    loader: DataLoader,
    device: torch.device,
    num_classes: int,
) -> dict[str, float]:
    model.eval()
    losses: list[float] = []
    y_true: list[int] = []
    y_pred: list[int] = []
    y_prob: list[np.ndarray] = []
    criterion = nn.CrossEntropyLoss()

    for images, labels in tqdm(loader, desc="eval", leave=False):
        images = images.to(device)
        labels = labels.to(device)
        logits = model(images)
        loss = criterion(logits, labels)
        probs = torch.softmax(logits, dim=1)
        losses.append(float(loss.item()))
        y_true.extend(labels.cpu().tolist())
        y_pred.extend(probs.argmax(dim=1).cpu().tolist())
        y_prob.append(probs.cpu().numpy())

    true_arr = np.asarray(y_true, dtype=np.int64)
    pred_arr = np.asarray(y_pred, dtype=np.int64)
    prob_arr = np.concatenate(y_prob, axis=0)
    metrics = classification_metrics(
        true_arr,
        pred_arr,
        y_prob=prob_arr,
        labels=list(range(num_classes)),
    )
    metrics["loss"] = float(np.mean(losses)) if losses else float("nan")
    return metrics
