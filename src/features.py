"""Frozen-backbone feature extraction for linear probing."""

from __future__ import annotations

from pathlib import Path

import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset
from tqdm import tqdm


@torch.no_grad()
def extract_features(
    model: nn.Module,
    loader: DataLoader,
    device: torch.device,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Run images through a ResNet with the classifier replaced by Identity."""
    if not hasattr(model, "fc"):
        raise ValueError("Feature extraction expects a model with an fc attribute.")
    original_fc = model.fc
    model.fc = nn.Identity()
    model.eval()

    features: list[torch.Tensor] = []
    labels: list[torch.Tensor] = []
    try:
        for images, batch_labels in tqdm(loader, desc="features", leave=False):
            images = images.to(device, non_blocking=True)
            feats = model(images)
            features.append(feats.cpu())
            labels.append(batch_labels.cpu())
    finally:
        model.fc = original_fc

    return torch.cat(features, dim=0), torch.cat(labels, dim=0)


def save_feature_bank(
    path: Path,
    features: torch.Tensor,
    labels: torch.Tensor,
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save({"features": features, "labels": labels}, path)


def load_feature_bank(path: Path) -> tuple[torch.Tensor, torch.Tensor]:
    payload = torch.load(path, map_location="cpu", weights_only=True)
    return payload["features"], payload["labels"]


def feature_loader(
    features: torch.Tensor,
    labels: torch.Tensor,
    batch_size: int,
    shuffle: bool,
) -> DataLoader:
    dataset = TensorDataset(features, labels)
    return DataLoader(dataset, batch_size=batch_size, shuffle=shuffle)


def train_linear_probe(
    num_features: int,
    num_classes: int,
    train_loader: DataLoader,
    val_loader: DataLoader,
    device: torch.device,
    epochs: int,
    lr: float,
) -> tuple[nn.Linear, list[dict]]:
    head = nn.Linear(num_features, num_classes).to(device)
    optimizer = torch.optim.Adam(head.parameters(), lr=lr)
    criterion = nn.CrossEntropyLoss()
    history: list[dict] = []
    best_state = None
    best_macro = -1.0

    from src.evaluate import evaluate_loader

    for epoch in range(1, epochs + 1):
        head.train()
        running = 0.0
        count = 0
        for feats, labels in tqdm(train_loader, desc="probe", leave=False):
            feats = feats.to(device)
            labels = labels.to(device)
            optimizer.zero_grad(set_to_none=True)
            logits = head(feats)
            loss = criterion(logits, labels)
            loss.backward()
            optimizer.step()
            running += float(loss.item()) * labels.size(0)
            count += labels.size(0)

        val_metrics = evaluate_loader(head, val_loader, device, num_classes)
        row = {
            "epoch": epoch,
            "train_loss": running / max(count, 1),
            **val_metrics,
        }
        history.append(row)
        if val_metrics["macro_f1"] > best_macro:
            best_macro = val_metrics["macro_f1"]
            best_state = {key: value.detach().cpu().clone() for key, value in head.state_dict().items()}

    if best_state is None:
        raise RuntimeError("Linear probe did not produce a checkpoint.")
    head.load_state_dict(best_state)
    return head, history
