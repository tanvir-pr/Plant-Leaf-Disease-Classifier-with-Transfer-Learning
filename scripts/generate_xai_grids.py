"""Build Grad-CAM and SmoothGrad grids for correct and wrong val predictions."""

from __future__ import annotations

import json
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.checkpoint import load_plant_checkpoint
from src.dataset import PlantSplitDataset, build_class_mapping, load_split_frame
from src.transforms import build_transforms
from src.utils import load_config, project_root, set_seed
from src.xai import (
    compute_gradcam,
    compute_smoothgrad,
    side_by_side,
    tensor_to_rgb,
)


def collect_predictions(
    model: torch.nn.Module,
    loader: DataLoader,
    device: torch.device,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    y_true: list[int] = []
    y_pred: list[int] = []
    y_prob: list[float] = []
    model.eval()
    with torch.no_grad():
        for images, labels in loader:
            images = images.to(device)
            logits = model(images)
            probs = torch.softmax(logits, dim=1)
            conf, pred = probs.max(dim=1)
            y_true.extend(labels.tolist())
            y_pred.extend(pred.cpu().tolist())
            y_prob.extend(conf.cpu().tolist())
    return (
        np.asarray(y_true, dtype=np.int64),
        np.asarray(y_pred, dtype=np.int64),
        np.asarray(y_prob, dtype=np.float32),
    )


def pick_indices(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    y_prob: np.ndarray,
    n_each: int,
) -> tuple[list[int], list[int]]:
    correct = np.where(y_true == y_pred)[0]
    wrong = np.where(y_true != y_pred)[0]
    correct = correct[np.argsort(-y_prob[correct])]
    wrong = wrong[np.argsort(-y_prob[wrong])]
    return correct[:n_each].tolist(), wrong[:n_each].tolist()


def render_row(
    model: torch.nn.Module,
    image: torch.Tensor,
    true_name: str,
    pred_name: str,
    device: torch.device,
) -> np.ndarray:
    import cv2

    batch = image.unsqueeze(0).to(device)
    with torch.no_grad():
        pred = int(model(batch).argmax(dim=1).item())
    cam = compute_gradcam(model, batch, [pred], plus_plus=False)[0]
    cam_pp = compute_gradcam(model, batch, [pred], plus_plus=True)[0]
    smooth = compute_smoothgrad(model, batch, [pred])[0]
    rgb = tensor_to_rgb(image)
    panel = side_by_side(rgb, cam, cam_pp, smooth)
    labeled = panel.copy()
    title = f"true={true_name} | pred={pred_name}"
    cv2.putText(
        labeled,
        title[:120],
        (8, 22),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.45,
        (255, 255, 255),
        1,
        cv2.LINE_AA,
    )
    return labeled


def save_grid(
    panels: list[np.ndarray],
    out_path: Path,
    column_titles: tuple[str, ...] = (
        "image",
        "Grad-CAM",
        "Grad-CAM++",
        "SmoothGrad",
    ),
) -> None:
    if not panels:
        raise ValueError("No panels to save.")
    # Each panel is already 4 images wide; stack rows.
    grid = np.concatenate(panels, axis=0)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig_h = max(len(panels) * 2.2, 4)
    fig_w = 12
    fig, ax = plt.subplots(figsize=(fig_w, fig_h))
    ax.imshow(grid)
    ax.axis("off")
    ax.set_title(" | ".join(column_titles), fontsize=11)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    set_seed(42)
    root = project_root()
    choice_path = root / "reports" / "tables" / "augmentation_choice.json"
    if choice_path.is_file():
        choice = json.loads(choice_path.read_text(encoding="utf-8"))
        ckpt_path = root / choice["best_checkpoint"]
    else:
        ckpt_path = root / "runs" / "aug_color_jitter" / "best.pt"

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model, class_names, config = load_plant_checkpoint(ckpt_path, device=device)
    plant_cfg = load_config(root / "configs" / "aug_color_jitter.yaml")
    split_csv = root / plant_cfg["paths"]["split_csv"]
    dataset_root = root / plant_cfg["plantvillage"]["local_dir"]
    full_frame = pd.read_csv(split_csv)
    _, class_to_idx = build_class_mapping(full_frame)
    val_frame = load_split_frame(split_csv, "val")
    dataset = PlantSplitDataset(
        val_frame,
        dataset_root=dataset_root,
        class_to_idx=class_to_idx,
        transform=build_transforms(
            int(plant_cfg["image"]["size"]),
            train=False,
            augmentation="none",
        ),
    )
    loader = DataLoader(dataset, batch_size=32, shuffle=False, num_workers=0)
    y_true, y_pred, y_prob = collect_predictions(model, loader, device)
    correct_idx, wrong_idx = pick_indices(y_true, y_pred, y_prob, n_each=6)

    correct_panels = []
    for index in correct_idx:
        image, label = dataset[index]
        true_name = class_names[int(label)]
        pred_name = class_names[int(y_pred[index])]
        correct_panels.append(
            render_row(model, image, true_name, pred_name, device)
        )

    wrong_panels = []
    for index in wrong_idx:
        image, label = dataset[index]
        true_name = class_names[int(label)]
        pred_name = class_names[int(y_pred[index])]
        wrong_panels.append(
            render_row(model, image, true_name, pred_name, device)
        )

    figure_dir = root / "reports" / "figures"
    save_grid(correct_panels, figure_dir / "xai_correct_grid.png")
    save_grid(wrong_panels, figure_dir / "xai_wrong_grid.png")

    meta = {
        "checkpoint": str(ckpt_path.relative_to(root).as_posix()),
        "device": str(device),
        "n_correct": len(correct_panels),
        "n_wrong": len(wrong_panels),
        "correct_indices": correct_idx,
        "wrong_indices": wrong_idx,
        "val_accuracy": float((y_true == y_pred).mean()),
    }
    meta_path = figure_dir / "xai_grid_meta.json"
    meta_path.write_text(json.dumps(meta, indent=2), encoding="utf-8")
    print(json.dumps(meta, indent=2))
    print(f"Wrote {figure_dir / 'xai_correct_grid.png'}")
    print(f"Wrote {figure_dir / 'xai_wrong_grid.png'}")
    print(f"Wrote {meta_path}")


if __name__ == "__main__":
    main()
