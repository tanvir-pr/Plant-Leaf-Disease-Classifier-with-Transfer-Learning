"""Download the PlantVillage color images and the leaf-id map.

Only the color folder is checked out. Grayscale and segmented copies are left
on the remote so the download stays about one third of the full repository.
"""

from __future__ import annotations

import subprocess
import sys 
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.utils import load_config, project_root  # noqa: E402


def _color_ready(color_root: Path, expected_classes: int) -> bool:
    if not color_root.is_dir():
        return False
    class_dirs = [path for path in color_root.iterdir() if path.is_dir()]
    if len(class_dirs) < expected_classes:
        return False
    return any(class_dirs[0].glob("*.*"))


def download() -> Path:
    config = load_config()
    root = project_root()
    plant = config["plantvillage"]
    dest = root / plant["local_dir"]
    color_root = dest / plant["color_subdir"]
    expected = int(config["expected_classes"])
    if _color_ready(color_root, expected):
        print(f"PlantVillage color images already present at {color_root}")
        return dest

    dest.parent.mkdir(parents=True, exist_ok=True)
    if not (dest / ".git").exists():
        subprocess.run(
            [
                "git",
                "clone",
                "--depth",
                "1",
                "--filter=blob:none",
                "--sparse",
                plant["repo_url"],
                str(dest),
            ],
            check=True,
        )
    subprocess.run(
        [
            "git",
            "-C",
            str(dest),
            "sparse-checkout",
            "set",
            plant["color_subdir"],
            "leaf_grouping",
        ],
        check=True,
    )
    if not _color_ready(color_root, expected):
        raise RuntimeError(f"Download finished but class folders are missing under {color_root}")
    print(f"Downloaded PlantVillage color images to {color_root}")
    return dest


if __name__ == "__main__":
    download()
