"""Build field-style proxy photos when real own-leaf images are missing."""

from __future__ import annotations

import json
import sys
from pathlib import Path

import pandas as pd
from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.domain_shift import list_image_files, make_field_proxy
from src.utils import load_config, project_root, set_seed


def already_has_photos(own_dir: Path, min_count: int = 5) -> bool:
    labels = own_dir / "labels.csv"
    if labels.is_file():
        frame = pd.read_csv(labels)
        if "source" in frame.columns:
            real = frame[frame["source"].astype(str) != "proxy_field"]
            if len(real) >= min_count:
                return True
        elif len(frame) >= min_count:
            return True
    return len(list_image_files(own_dir)) >= min_count


def sample_train_rows(split_csv: Path, n: int, seed: int) -> pd.DataFrame:
    frame = pd.read_csv(split_csv)
    train = frame[frame["split"] == "train"].copy()
    # Prefer one image per class when possible, then fill.
    classes = sorted(train["class_name"].unique())
    picked = []
    rng = __import__("numpy").random.default_rng(seed)
    rng.shuffle(classes)
    for class_name in classes:
        subset = train[train["class_name"] == class_name]
        row = subset.sample(n=1, random_state=int(rng.integers(0, 1_000_000))).iloc[0]
        picked.append(row)
        if len(picked) >= n:
            break
    if len(picked) < n:
        remaining = train.drop(index=[r.name for r in picked], errors="ignore")
        extra = remaining.sample(
            n=min(n - len(picked), len(remaining)),
            random_state=seed,
        )
        picked.extend([row for _, row in extra.iterrows()])
    return pd.DataFrame(picked).reset_index(drop=True)


def main() -> None:
    set_seed(42)
    root = project_root()
    own_dir = root / "data" / "own_leaves"
    own_dir.mkdir(parents=True, exist_ok=True)

    if already_has_photos(own_dir):
        print(f"Found existing photos in {own_dir}; skipping proxy build.")
        return

    # Clear previous proxy files only.
    for path in list_image_files(own_dir):
        path.unlink()
    labels_path = own_dir / "labels.csv"
    if labels_path.is_file():
        labels_path.unlink()

    plant_cfg = load_config(root / "configs" / "aug_color_jitter.yaml")
    split_csv = root / plant_cfg["paths"]["split_csv"]
    dataset_root = root / plant_cfg["plantvillage"]["local_dir"]
    n = 24
    sample = sample_train_rows(split_csv, n=n, seed=42)

    rows = []
    for i, (_, row) in enumerate(sample.iterrows()):
        src = dataset_root / row["relative_path"]
        image = Image.open(src).convert("RGB")
        proxy = make_field_proxy(image, seed=42 + int(i))
        name = f"proxy_{i:02d}_{row['class_name'].replace('/', '-')}.jpg"
        out_path = own_dir / name
        proxy.save(out_path, quality=85)
        rows.append(
            {
                "relative_path": name,
                "class_name": row["class_name"],
                "source": "proxy_field",
                "source_relative_path": row["relative_path"],
            }
        )

    frame = pd.DataFrame(rows)
    frame.to_csv(labels_path, index=False)
    meta = {
        "n_images": len(frame),
        "source": "proxy_field",
        "note": (
            "Synthetic field-style proxies built from PlantVillage train leaves "
            "(outdoor backdrop + phone lighting/JPEG). Replace with real photos "
            "in data/own_leaves/ and update labels.csv when available."
        ),
        "labels_csv": str(labels_path.relative_to(root).as_posix()),
    }
    meta_path = own_dir / "proxy_meta.json"
    meta_path.write_text(json.dumps(meta, indent=2), encoding="utf-8")
    print(json.dumps(meta, indent=2))
    print(f"Wrote {len(frame)} proxy images to {own_dir}")


if __name__ == "__main__":
    main()
