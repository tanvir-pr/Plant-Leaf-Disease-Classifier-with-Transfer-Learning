"""Config-driven training loop."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import pandas as pd
import torch
from torch import nn
from torch.utils.data import DataLoader
from tqdm import tqdm

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.dataset import PlantSplitDataset, build_class_mapping, load_split_frame
from src.evaluate import evaluate_loader
from src.models import build_model
from src.transforms import build_transforms
from src.utils import load_config, project_root, set_seed


def resolve_device(requested: str) -> torch.device:
    if requested == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(requested)


def make_loader(
    frame: pd.DataFrame,
    dataset_root: Path,
    class_to_idx: dict[str, int],
    image_size: int,
    batch_size: int,
    train: bool,
    num_workers: int,
) -> DataLoader:
    dataset = PlantSplitDataset(
        frame,
        dataset_root=dataset_root,
        class_to_idx=class_to_idx,
        transform=build_transforms(image_size, train=train),
    )
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=train,
        num_workers=num_workers,
        pin_memory=torch.cuda.is_available(),
    )


def overfit_one_batch(
    model: nn.Module,
    images: torch.Tensor,
    labels: torch.Tensor,
    device: torch.device,
    steps: int,
    lr: float,
) -> list[float]:
    """Sanity check: a trainable model should drive one batch loss near zero."""
    model.train()
    images = images.to(device)
    labels = labels.to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    criterion = nn.CrossEntropyLoss()
    losses: list[float] = []
    for _ in range(steps):
        optimizer.zero_grad(set_to_none=True)
        logits = model(images)
        loss = criterion(logits, labels)
        loss.backward()
        optimizer.step()
        losses.append(float(loss.item()))
    return losses


def train_one_epoch(
    model: nn.Module,
    loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
    amp: bool,
) -> float:
    model.train()
    criterion = nn.CrossEntropyLoss()
    scaler = torch.amp.GradScaler("cuda", enabled=amp)
    running = 0.0
    count = 0
    for images, labels in tqdm(loader, desc="train", leave=False):
        images = images.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)
        optimizer.zero_grad(set_to_none=True)
        with torch.amp.autocast("cuda", enabled=amp):
            logits = model(images)
            loss = criterion(logits, labels)
        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()
        running += float(loss.item()) * labels.size(0)
        count += labels.size(0)
    return running / max(count, 1)


def save_checkpoint(
    path: Path,
    model: nn.Module,
    class_names: list[str],
    config: dict,
    metrics: dict,
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "model_state": model.state_dict(),
            "class_names": class_names,
            "config": config,
            "metrics": metrics,
        },
        path,
    )


def run_training(config: dict) -> dict:
    set_seed(int(config["seed"]))
    root = project_root()
    train_cfg = config["train"]
    device = resolve_device(str(train_cfg.get("device", "auto")))
    amp = bool(train_cfg.get("amp", False)) and device.type == "cuda"

    split_csv = root / config["paths"]["split_csv"]
    dataset_root = root / config["plantvillage"]["local_dir"]
    full_frame = pd.read_csv(split_csv)
    class_names, class_to_idx = build_class_mapping(full_frame)
    if len(class_names) != int(config["expected_classes"]):
        raise RuntimeError(
            f"Expected {config['expected_classes']} classes, found {len(class_names)}"
        )

    train_frame = load_split_frame(split_csv, "train")
    val_frame = load_split_frame(split_csv, "val")
    image_size = int(config["image"]["size"])
    batch_size = int(train_cfg["batch_size"])
    num_workers = int(train_cfg.get("num_workers", 0))

    train_loader = make_loader(
        train_frame,
        dataset_root,
        class_to_idx,
        image_size,
        batch_size,
        train=True,
        num_workers=num_workers,
    )
    val_loader = make_loader(
        val_frame,
        dataset_root,
        class_to_idx,
        image_size,
        batch_size,
        train=False,
        num_workers=num_workers,
    )

    model = build_model(str(train_cfg["model"]), len(class_names)).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=float(train_cfg["lr"]))
    out_dir = root / train_cfg["output_dir"]
    out_dir.mkdir(parents=True, exist_ok=True)

    history: list[dict] = []
    best_macro = -1.0
    best_path = out_dir / "best.pt"

    for epoch in range(1, int(train_cfg["epochs"]) + 1):
        train_loss = train_one_epoch(model, train_loader, optimizer, device, amp)
        val_metrics = evaluate_loader(model, val_loader, device, len(class_names))
        row = {"epoch": epoch, "train_loss": train_loss, **val_metrics}
        history.append(row)
        print(json.dumps(row, indent=2))
        if val_metrics["macro_f1"] > best_macro:
            best_macro = val_metrics["macro_f1"]
            save_checkpoint(best_path, model, class_names, config, val_metrics)

    summary = {
        "best_checkpoint": str(best_path.relative_to(root).as_posix()),
        "best_macro_f1": best_macro,
        "history": history,
        "class_names": class_names,
        "device": str(device),
    }
    summary_path = out_dir / "summary.json"
    summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(f"Wrote {summary_path}")
    return summary


def run_overfit(config: dict) -> dict:
    set_seed(int(config["seed"]))
    root = project_root()
    train_cfg = config["train"]
    device = resolve_device(str(train_cfg.get("device", "auto")))
    split_csv = root / config["paths"]["split_csv"]
    dataset_root = root / config["plantvillage"]["local_dir"]
    full_frame = pd.read_csv(split_csv)
    class_names, class_to_idx = build_class_mapping(full_frame)
    train_frame = load_split_frame(split_csv, "train")
    loader = make_loader(
        train_frame,
        dataset_root,
        class_to_idx,
        int(config["image"]["size"]),
        int(train_cfg["batch_size"]),
        train=True,
        num_workers=0,
    )
    images, labels = next(iter(loader))
    model = build_model(str(train_cfg["model"]), len(class_names)).to(device)
    losses = overfit_one_batch(
        model,
        images,
        labels,
        device,
        steps=int(train_cfg.get("overfit_steps", 80)),
        lr=float(train_cfg.get("overfit_lr", 1e-3)),
    )
    result = {
        "first_loss": losses[0],
        "last_loss": losses[-1],
        "steps": len(losses),
        "batch_size": int(labels.size(0)),
        "unique_labels": sorted({int(x) for x in labels.tolist()}),
    }
    out_dir = root / train_cfg["output_dir"]
    out_dir.mkdir(parents=True, exist_ok=True)
    path = out_dir / "overfit.json"
    path.write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(json.dumps(result, indent=2))
    print(f"Wrote {path}")
    return result


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train a plant disease classifier.")
    parser.add_argument(
        "--config",
        type=Path,
        default=project_root() / "configs" / "plant.yaml",
    )
    parser.add_argument(
        "--overfit-batch",
        action="store_true",
        help="Overfit a single training batch as a sanity check.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config = load_config(args.config)
    if args.overfit_batch:
        run_overfit(config)
    else:
        run_training(config)


if __name__ == "__main__":
    main()
