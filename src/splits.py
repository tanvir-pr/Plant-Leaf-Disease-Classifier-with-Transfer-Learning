"""Index PlantVillage color images and build a leaf-grouped split.

Several photos often show the same physical leaf. An image-level random split
would put those near-duplicates on both sides of the split. Groups stay
together, and the groups are stratified by disease class.
"""

from __future__ import annotations

import json
import re
from pathlib import Path

import pandas as pd
from sklearn.model_selection import train_test_split

IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp"}
SPLIT_NAMES = ("train", "val", "test")
_COPY_RE = re.compile(r"copy", re.IGNORECASE)
_EXT_RE = re.compile(r"\.(jpg|jpeg|png|bmp)$", re.IGNORECASE)
_NUMBER_RE = re.compile(r"(\d+)")


def image_identifier(filename: str) -> str:
    """Match the PlantVillage leaf-map key derived from a file name."""
    name = filename.replace("_final_masked", "")
    if "___" in name:
        name = name.split("___")[-1]
    name = name.split("copy")[0]
    suffix = Path(name).suffix
    if suffix.lower() in IMAGE_EXTENSIONS:
        name = name[: -len(suffix)]
    return name.strip().lower()


def leaf_id_for(filename: str, class_name: str, leaf_map: dict) -> str:
    """Return the physical-leaf id, falling back to the file stem."""
    key = image_identifier(filename)
    suggestions = leaf_map.get(key)
    if not suggestions:
        return f"fallback_{key}"
    if isinstance(suggestions, str):
        suggestions = [suggestions]
    if len(suggestions) == 1:
        return str(suggestions[0])
    for suggestion in suggestions:
        if class_name in str(suggestion):
            return str(suggestion)
    return f"fallback_{key}"


def photo_number(filename: str) -> str | None:
    """First photo id in a leaf token, ignoring a trailing ``copy`` suffix."""
    name = filename.replace("_final_masked", "")
    if "___" in name:
        name = name.split("___")[-1]
    name = _COPY_RE.split(name, maxsplit=1)[0]
    name = _EXT_RE.sub("", name)
    match = _NUMBER_RE.search(name)
    if not match:
        return None
    return match.group(1).lstrip("0") or "0"


def build_photo_number_index(leaf_map: dict) -> dict[tuple[str, str], str]:
    """Map (class, photo number) to a leaf id when that pair is unambiguous.

    Some files use a different lab code than the leaf map (``Com.G_TgS_FL 8360``
    versus ``com.g_fl 8360``) but the same photo number. Numbers that point at
    two different leaves are left out.
    """
    found: dict[tuple[str, str], str] = {}
    ambiguous: set[tuple[str, str]] = set()
    for key, suggestions in leaf_map.items():
        number = photo_number(str(key))
        if number is None:
            continue
        if isinstance(suggestions, str):
            suggestions = [suggestions]
        for suggestion in suggestions:
            text = str(suggestion)
            class_name, separator, _leaf = text.partition(":::")
            if not separator:
                continue
            slot = (class_name, number)
            previous = found.get(slot)
            if previous is None:
                found[slot] = text
            elif previous != text:
                ambiguous.add(slot)
    for slot in ambiguous:
        found.pop(slot, None)
    return found


def resolve_leaf_id(
    filename: str,
    class_name: str,
    leaf_map: dict,
    number_index: dict[tuple[str, str], str],
) -> str:
    """Prefer an exact leaf-map key, then an unambiguous photo number."""
    direct = leaf_id_for(filename, class_name, leaf_map)
    if not direct.startswith("fallback_"):
        return direct
    number = photo_number(filename)
    if number is None:
        return direct
    return number_index.get((class_name, number), direct)


def load_leaf_map(path: Path) -> dict:
    with path.open(encoding="utf-8") as handle:
        payload = json.load(handle)
    if not isinstance(payload, dict):
        raise ValueError(f"Leaf map at {path} must be a JSON object.")
    return payload


def index_color_images(
    color_root: Path,
    leaf_map: dict,
    dataset_root: Path,
) -> pd.DataFrame:
    """Walk class folders and attach a group id that cannot cross classes."""
    number_index = build_photo_number_index(leaf_map)
    rows: list[dict] = []
    class_dirs = sorted(path for path in color_root.iterdir() if path.is_dir())
    for class_dir in class_dirs:
        class_name = class_dir.name
        crop, separator, disease = class_name.partition("___")
        if not separator:
            disease = "unknown"
        for image_path in sorted(class_dir.iterdir()):
            if not image_path.is_file():
                continue
            if image_path.suffix.lower() not in IMAGE_EXTENSIONS:
                continue
            leaf_id = resolve_leaf_id(image_path.name, class_name, leaf_map, number_index)
            relative = image_path.relative_to(dataset_root).as_posix()
            rows.append(
                {
                    "relative_path": relative,
                    "class_name": class_name,
                    "crop": crop,
                    "disease": disease,
                    "leaf_id": leaf_id,
                    "group_id": f"{class_name}/{leaf_id}",
                }
            )
    if not rows:
        raise FileNotFoundError(f"No images found under {color_root}")
    return pd.DataFrame(rows)


def _check_ratios(ratios: dict[str, float]) -> tuple[float, float, float]:
    missing = [name for name in SPLIT_NAMES if name not in ratios]
    if missing:
        raise ValueError(f"Split ratios missing {missing}")
    train_ratio = float(ratios["train"])
    val_ratio = float(ratios["val"])
    test_ratio = float(ratios["test"])
    total = train_ratio + val_ratio + test_ratio
    if abs(total - 1.0) > 1e-6:
        raise ValueError(f"Split ratios must sum to 1, got {total}")
    if min(train_ratio, val_ratio, test_ratio) <= 0:
        raise ValueError("Train, val, and test ratios must all be positive.")
    return train_ratio, val_ratio, test_ratio


def stratified_group_split(
    frame: pd.DataFrame,
    ratios: dict[str, float],
    seed: int,
) -> pd.DataFrame:
    """Assign every image in a leaf group to the same split."""
    required = {"relative_path", "class_name", "group_id"}
    missing = required.difference(frame.columns)
    if missing:
        raise ValueError(f"Frame is missing columns: {sorted(missing)}")
    if frame.empty:
        raise ValueError("Cannot split an empty frame.")
    if frame["relative_path"].duplicated().any():
        raise ValueError("relative_path values must be unique.")

    label_counts = frame.groupby("group_id")["class_name"].nunique()
    mixed = label_counts[label_counts > 1]
    if not mixed.empty:
        raise ValueError(f"Groups span more than one class: {list(mixed.index[:5])}")

    _, val_ratio, test_ratio = _check_ratios(ratios)
    groups = (
        frame.groupby("group_id", sort=False)["class_name"]
        .first()
        .rename("label")
        .reset_index()
    )
    class_counts = groups["label"].value_counts()
    too_small = class_counts[class_counts < 3]
    if not too_small.empty:
        detail = too_small.to_dict()
        raise ValueError(
            "Each class needs at least 3 leaf groups for train/val/test. "
            f"Too small: {detail}"
        )

    holdout = val_ratio + test_ratio
    train_groups, temp_groups = train_test_split(
        groups,
        test_size=holdout,
        random_state=seed,
        stratify=groups["label"],
    )
    val_groups, test_groups = train_test_split(
        temp_groups,
        test_size=test_ratio / holdout,
        random_state=seed,
        stratify=temp_groups["label"],
    )

    assignment: dict[str, str] = {}
    for split_name, split_groups in (
        ("train", train_groups),
        ("val", val_groups),
        ("test", test_groups),
    ):
        for group_id in split_groups["group_id"]:
            assignment[group_id] = split_name

    assigned = frame.copy()
    assigned["split"] = assigned["group_id"].map(assignment)
    if assigned["split"].isna().any():
        raise RuntimeError("Some images were not assigned a split.")
    return assigned.reset_index(drop=True)
