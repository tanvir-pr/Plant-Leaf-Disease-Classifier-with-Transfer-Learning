"""Heuristics for plant validation failure taxonomy."""

from __future__ import annotations

from collections import Counter

import cv2
import numpy as np
from PIL import Image

from src.background import leaf_mask_rgb


TAXONOMY_LABELS = (
    "shortcut",
    "confused_disease",
    "rare_class",
    "bad_image",
    "other",
)


def crop_name(class_name: str) -> str:
    """PlantVillage class = Crop___Disease; return the crop token."""
    return str(class_name).split("___", 1)[0]


def rare_class_names(
    class_counts: dict[str, int],
    quantile: float = 0.25,
) -> set[str]:
    """Classes at or below the given train-count quantile."""
    if not class_counts:
        return set()
    values = np.asarray(list(class_counts.values()), dtype=np.float64)
    threshold = float(np.quantile(values, quantile))
    return {name for name, count in class_counts.items() if count <= threshold}


def image_quality_flags(image: Image.Image) -> dict[str, float | bool]:
    """Cheap quality scores used to flag bad photos."""
    rgb = np.asarray(image.convert("RGB"))
    gray = cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY).astype(np.float32)
    mean = float(gray.mean())
    std = float(gray.std())
    lap_var = float(cv2.Laplacian(gray, cv2.CV_32F).var())
    mask = leaf_mask_rgb(rgb)
    leaf_frac = float(mask.mean()) if mask.size else 0.0
    return {
        "mean_luma": mean,
        "std_luma": std,
        "laplacian_var": lap_var,
        "leaf_frac": leaf_frac,
        "too_dark": mean < 45.0,
        "too_bright": mean > 210.0,
        "low_contrast": std < 18.0,
        "blurry": lap_var < 40.0,
        "tiny_leaf": leaf_frac < 0.12,
    }


def is_bad_image(image: Image.Image) -> bool:
    flags = image_quality_flags(image)
    return bool(
        flags["too_dark"]
        or flags["too_bright"]
        or flags["low_contrast"]
        or flags["blurry"]
        or flags["tiny_leaf"]
    )


def assign_taxonomy(
    true_name: str,
    pred_name: str,
    *,
    original_pred: int,
    masked_pred: int,
    true_label: int,
    rare_classes: set[str],
    image: Image.Image,
) -> str:
    """Pick one primary failure label for a wrong prediction.

    Priority: shortcut → confused_disease → rare_class → bad_image → other.
    """
    if true_name == pred_name:
        raise ValueError("assign_taxonomy expects a misclassification.")

    # Background removal flips the answer toward the true class, or changes
    # the wrong guess — both point at lab-background cues.
    if masked_pred != original_pred and (
        masked_pred == true_label or original_pred != true_label
    ):
        return "shortcut"

    if crop_name(true_name) == crop_name(pred_name):
        return "confused_disease"

    if true_name in rare_classes:
        return "rare_class"

    if is_bad_image(image):
        return "bad_image"

    return "other"


def select_error_indices(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    y_prob: np.ndarray,
    n: int = 30,
) -> np.ndarray:
    """Highest-confidence mistakes first, capped at n."""
    wrong = np.where(y_true != y_pred)[0]
    if wrong.size == 0:
        return wrong
    order = wrong[np.argsort(-y_prob[wrong])]
    return order[:n]


def taxonomy_counts(labels: list[str]) -> dict[str, int]:
    counter = Counter(labels)
    return {name: int(counter.get(name, 0)) for name in TAXONOMY_LABELS}
