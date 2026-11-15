"""Compare validation metrics with lab background removed."""

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

from src.background import crop_to_leaf, mask_background
from src.checkpoint import load_plant_checkpoint
from src.dataset import PlantSplitDataset, build_class_mapping, load_split_frame
from src.evaluate import evaluate_loader
from src.transforms import build_transforms
from src.utils import load_config, project_root, set_seed


VARIANTS = (
    ("original", None),
    ("masked_black", mask_background),
    ("cropped_leaf", crop_to_leaf),
)


def make_val_loader(preprocess, plant_cfg, class_to_idx, dataset_root, split_csv):
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
        preprocess=preprocess,
    )
    return DataLoader(dataset, batch_size=32, shuffle=False, num_workers=0)


def save_example_grid(
    dataset_root: Path,
    frame: pd.DataFrame,
    out_path: Path,
    n: int = 4,
) -> None:
    sample = frame.sample(n=n, random_state=42)
    fig, axes = plt.subplots(n, 3, figsize=(9, 3 * n))
    if n == 1:
        axes = np.expand_dims(axes, 0)
    titles = ["original", "masked", "cropped"]
    for row_i, (_, row) in enumerate(sample.iterrows()):
        from PIL import Image

        original = Image.open(dataset_root / row["relative_path"]).convert("RGB")
        images = [original, mask_background(original), crop_to_leaf(original)]
        for col_i, image in enumerate(images):
            axes[row_i, col_i].imshow(image)
            axes[row_i, col_i].axis("off")
            if row_i == 0:
                axes[row_i, col_i].set_title(titles[col_i])
            if col_i == 0:
                axes[row_i, col_i].set_ylabel(row["class_name"], fontsize=7)
    fig.suptitle("Background shortcut transforms", fontsize=12)
    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
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
    model, class_names, _config = load_plant_checkpoint(ckpt_path, device=device)
    plant_cfg = load_config(root / "configs" / "aug_color_jitter.yaml")
    split_csv = root / plant_cfg["paths"]["split_csv"]
    dataset_root = root / plant_cfg["plantvillage"]["local_dir"]
    full_frame = pd.read_csv(split_csv)
    _, class_to_idx = build_class_mapping(full_frame)
    val_frame = load_split_frame(split_csv, "val")

    results = []
    for name, preprocess in VARIANTS:
        loader = make_val_loader(
            preprocess, plant_cfg, class_to_idx, dataset_root, split_csv
        )
        metrics = evaluate_loader(model, loader, device, len(class_names))
        row = {"variant": name, **metrics}
        results.append(row)
        print(json.dumps(row, indent=2))

    frame = pd.DataFrame(results)
    original = frame.loc[frame["variant"] == "original"].iloc[0]
    note_rows = []
    for _, row in frame.iterrows():
        if row["variant"] == "original":
            continue
        drop = float(original["macro_f1"] - row["macro_f1"])
        note_rows.append(
            {
                "variant": row["variant"],
                "macro_f1_drop": drop,
                "accuracy_drop": float(original["accuracy"] - row["accuracy"]),
            }
        )

    out_dir = root / "reports"
    table_dir = out_dir / "tables"
    figure_dir = out_dir / "figures"
    table_dir.mkdir(parents=True, exist_ok=True)
    csv_path = table_dir / "shortcut_test.csv"
    md_path = table_dir / "shortcut_test.md"
    json_path = table_dir / "shortcut_test.json"
    frame.to_csv(csv_path, index=False)

    headers = list(frame.columns)
    lines = [
        "| " + " | ".join(headers) + " |",
        "| " + " | ".join("---" for _ in headers) + " |",
    ]
    for row in frame.itertuples(index=False):
        cells = []
        for value in row:
            if isinstance(value, float):
                cells.append(f"{value:.4f}")
            else:
                cells.append(str(value))
        lines.append("| " + " | ".join(cells) + " |")
    md_path.write_text("\n".join(lines) + "\n", encoding="utf-8")

    mean_drop = float(np.mean([item["macro_f1_drop"] for item in note_rows]))
    finding = (
        "Removing the lab background hurts validation score. "
        "The model likely used background cues as a shortcut."
        if mean_drop > 0.02
        else "Removing the lab background barely changes validation score. "
        "The model appears to rely mainly on the leaf itself."
    )
    payload = {
        "checkpoint": str(ckpt_path.relative_to(root).as_posix()),
        "device": str(device),
        "results": results,
        "drops": note_rows,
        "finding": finding,
    }
    json_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")

    save_example_grid(
        dataset_root,
        val_frame,
        figure_dir / "shortcut_examples.png",
    )

    # Bar chart of macro-F1.
    fig, ax = plt.subplots(figsize=(7, 4))
    ax.bar(frame["variant"], frame["macro_f1"], color=["#1b4f72", "#b9770e", "#148f77"])
    ax.set_ylabel("macro-F1")
    ax.set_title("Background shortcut test on validation")
    ax.set_ylim(0.0, 1.05)
    for i, value in enumerate(frame["macro_f1"]):
        ax.text(i, value + 0.01, f"{value:.3f}", ha="center", fontsize=9)
    fig.tight_layout()
    chart_path = figure_dir / "shortcut_macro_f1.png"
    fig.savefig(chart_path, dpi=150)
    plt.close(fig)

    print(finding)
    print(f"Wrote {csv_path}")
    print(f"Wrote {md_path}")
    print(f"Wrote {json_path}")
    print(f"Wrote {figure_dir / 'shortcut_examples.png'}")
    print(f"Wrote {chart_path}")


if __name__ == "__main__":
    main()
