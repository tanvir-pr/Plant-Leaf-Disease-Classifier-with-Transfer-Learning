"""PlantVillage images loaded from the leaf-grouped split CSV."""

from __future__ import annotations

from pathlib import Path

import pandas as pd
from PIL import Image
from torch.utils.data import Dataset


class PlantSplitDataset(Dataset):
    """One row of the split CSV becomes one labelled image."""

    def __init__(
        self,
        frame: pd.DataFrame,
        dataset_root: Path,
        class_to_idx: dict[str, int],
        transform=None,
    ) -> None:
        if frame.empty:
            raise ValueError("Split frame has no rows.")
        missing = {"relative_path", "class_name"}.difference(frame.columns)
        if missing:
            raise ValueError(f"Split frame is missing columns: {sorted(missing)}")
        self.frame = frame.reset_index(drop=True)
        self.dataset_root = Path(dataset_root)
        self.class_to_idx = class_to_idx
        self.transform = transform

    def __len__(self) -> int:
        return len(self.frame)

    def __getitem__(self, index: int):
        row = self.frame.iloc[index]
        path = self.dataset_root / row["relative_path"]
        image = Image.open(path).convert("RGB")
        label = self.class_to_idx[str(row["class_name"])]
        if self.transform is not None:
            image = self.transform(image)
        return image, label


def load_split_frame(split_csv: Path, split: str) -> pd.DataFrame:
    frame = pd.read_csv(split_csv)
    if "split" not in frame.columns:
        raise ValueError(f"{split_csv} has no split column.")
    subset = frame[frame["split"] == split].copy()
    if subset.empty:
        raise ValueError(f"No rows for split={split!r} in {split_csv}")
    return subset.reset_index(drop=True)


def build_class_mapping(frame: pd.DataFrame) -> tuple[list[str], dict[str, int]]:
    class_names = sorted(frame["class_name"].astype(str).unique())
    class_to_idx = {name: index for index, name in enumerate(class_names)}
    return class_names, class_to_idx
