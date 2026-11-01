"""Build the leaf-grouped plant split and the class-distribution figure."""

from __future__ import annotations

import json
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.splits import index_color_images, load_leaf_map, stratified_group_split  # noqa: E402
from src.utils import load_config, project_root, set_seed  # noqa: E402


def plot_class_distribution(frame: pd.DataFrame, out_path: Path) -> None:
    counts = frame.groupby("class_name").size().sort_values(ascending=True)
    fig, ax = plt.subplots(figsize=(10, 12))
    ax.barh(counts.index, counts.to_numpy(), color="#2a6f4e")
    ax.set_xlabel("Images")
    ax.set_title("PlantVillage color set: images per class")
    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def plot_split_balance(frame: pd.DataFrame, out_path: Path) -> None:
    counts = (
        frame.groupby(["class_name", "split"])
        .size()
        .unstack(fill_value=0)
        .reindex(columns=["train", "val", "test"])
    )
    order = counts.sum(axis=1).sort_values(ascending=True).index
    counts = counts.loc[order]
    fig, ax = plt.subplots(figsize=(10, 12))
    left = None
    colors = {"train": "#1b4f72", "val": "#148f77", "test": "#b9770e"}
    for split_name in ("train", "val", "test"):
        values = counts[split_name].to_numpy()
        ax.barh(
            counts.index,
            values,
            left=left,
            label=split_name,
            color=colors[split_name],
        )
        left = values if left is None else left + values
    ax.set_xlabel("Images")
    ax.set_title("Leaf-grouped split: train / val / test")
    ax.legend(loc="lower right")
    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def _summary(frame: pd.DataFrame, config: dict) -> dict:
    grouped = frame.groupby("group_id")["split"].first()
    return {
        "seed": int(config["seed"]),
        "ratios": config["split"],
        "n_images": int(len(frame)),
        "n_classes": int(frame["class_name"].nunique()),
        "n_leaf_groups": int(frame["group_id"].nunique()),
        "images_per_split": frame["split"].value_counts().sort_index().to_dict(),
        "leaf_groups_per_split": grouped.value_counts().sort_index().to_dict(),
        "images_per_class": frame["class_name"].value_counts().sort_index().to_dict(),
        "fallback_leaf_ids": int(frame["leaf_id"].astype(str).str.startswith("fallback_").sum()),
    }


def prepare() -> Path:
    config = load_config()
    set_seed(int(config["seed"]))
    root = project_root()
    plant = config["plantvillage"]
    dataset_root = root / plant["local_dir"]
    color_root = dataset_root / plant["color_subdir"]
    if not color_root.is_dir():
        raise FileNotFoundError(
            f"Missing {color_root}. Run: python data/download_plantvillage.py"
        )

    leaf_map_path = dataset_root / plant["leaf_map"]
    if leaf_map_path.is_file():
        leaf_map = load_leaf_map(leaf_map_path)
    else:
        print(f"Leaf map not found at {leaf_map_path}; grouping by file name.")
        leaf_map = {}

    indexed = index_color_images(color_root, leaf_map, dataset_root)
    n_classes = int(indexed["class_name"].nunique())
    expected = int(config["expected_classes"])
    if n_classes != expected:
        raise RuntimeError(f"Expected {expected} classes, found {n_classes} under {color_root}")

    assigned = stratified_group_split(indexed, config["split"], int(config["seed"]))
    split_dir = root / "data" / "splits"
    figure_dir = root / "reports" / "figures"
    split_dir.mkdir(parents=True, exist_ok=True)
    csv_path = split_dir / "plant_split.csv"
    summary_path = split_dir / "plant_split_summary.json"
    assigned.sort_values(["split", "class_name", "relative_path"]).to_csv(csv_path, index=False)
    summary = _summary(assigned, config)
    summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")

    plot_class_distribution(assigned, figure_dir / "plant_class_distribution.png")
    plot_split_balance(assigned, figure_dir / "plant_split_balance.png")

    print(json.dumps(summary, indent=2))
    print(f"Wrote {csv_path}")
    print(f"Wrote {summary_path}")
    return csv_path


if __name__ == "__main__":
    prepare()
