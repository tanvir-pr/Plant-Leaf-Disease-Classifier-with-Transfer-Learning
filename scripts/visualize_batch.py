"""Save a labelled grid from one training batch."""

from __future__ import annotations

import sys
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
import torch
from torch.utils.data import DataLoader

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.dataset import PlantSplitDataset, build_class_mapping, load_split_frame
from src.transforms import build_transforms, denormalize
from src.utils import load_config, project_root, set_seed


def main() -> None:
    config = load_config()
    set_seed(int(config["seed"]))
    root = project_root()
    split_csv = root / config["paths"]["split_csv"]
    dataset_root = root / config["plantvillage"]["local_dir"]
    full_frame = pd.read_csv(split_csv)
    class_names, class_to_idx = build_class_mapping(full_frame)
    train_frame = load_split_frame(split_csv, "train")

    dataset = PlantSplitDataset(
        train_frame,
        dataset_root=dataset_root,
        class_to_idx=class_to_idx,
        transform=build_transforms(int(config["image"]["size"]), train=True),
    )
    loader = DataLoader(dataset, batch_size=16, shuffle=True, num_workers=0)
    images, labels = next(iter(loader))

    rows, cols = 4, 4
    fig, axes = plt.subplots(rows, cols, figsize=(12, 12))
    for index, ax in enumerate(axes.flat):
        image = denormalize(images[index]).permute(1, 2, 0).cpu().numpy()
        ax.imshow(image)
        ax.set_title(class_names[int(labels[index])], fontsize=8)
        ax.axis("off")
    fig.suptitle("Training batch with class labels", fontsize=12)
    fig.tight_layout()
    out_path = root / "reports" / "figures" / "plant_train_batch.png"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    print(f"Wrote {out_path}")


if __name__ == "__main__":
    main()
