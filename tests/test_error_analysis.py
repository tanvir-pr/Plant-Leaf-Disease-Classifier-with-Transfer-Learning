"""Failure taxonomy heuristics."""

import numpy as np
from PIL import Image

from src.error_analysis import (
    assign_taxonomy,
    crop_name,
    image_quality_flags,
    is_bad_image,
    rare_class_names,
    select_error_indices,
    taxonomy_counts,
)


def _leaf_image(fill_bg=(170, 170, 170), fill_leaf=(40, 140, 50)) -> Image.Image:
    rgb = np.full((128, 128, 3), fill_bg, dtype=np.uint8)
    rgb[30:100, 40:90] = fill_leaf
    return Image.fromarray(rgb)


def test_crop_name_splits_plantvillage_label():
    assert crop_name("Tomato___Early_blight") == "Tomato"
    assert crop_name("Apple___healthy") == "Apple"


def test_rare_class_names_bottom_quartile():
    counts = {
        "a": 100,
        "b": 200,
        "c": 300,
        "d": 400,
        "e": 500,
        "f": 600,
        "g": 700,
        "h": 800,
    }
    rare = rare_class_names(counts, quantile=0.25)
    assert "a" in rare
    assert "h" not in rare


def test_assign_taxonomy_priority():
    image = _leaf_image()
    rare = {"Potato___healthy"}

    assert (
        assign_taxonomy(
            "Tomato___Early_blight",
            "Tomato___Late_blight",
            original_pred=1,
            masked_pred=0,
            true_label=0,
            rare_classes=rare,
            image=image,
        )
        == "shortcut"
    )

    assert (
        assign_taxonomy(
            "Tomato___Early_blight",
            "Tomato___Late_blight",
            original_pred=1,
            masked_pred=1,
            true_label=0,
            rare_classes=rare,
            image=image,
        )
        == "confused_disease"
    )

    assert (
        assign_taxonomy(
            "Potato___healthy",
            "Tomato___healthy",
            original_pred=1,
            masked_pred=1,
            true_label=0,
            rare_classes=rare,
            image=image,
        )
        == "rare_class"
    )


def test_bad_image_flags_flat_gray():
    flat = Image.fromarray(np.full((64, 64, 3), 128, dtype=np.uint8))
    flags = image_quality_flags(flat)
    assert flags["low_contrast"] or flags["tiny_leaf"]
    assert is_bad_image(flat)


def test_select_error_indices_prefers_confident_mistakes():
    y_true = np.asarray([0, 0, 1, 1], dtype=np.int64)
    y_pred = np.asarray([0, 1, 0, 1], dtype=np.int64)
    y_prob = np.asarray([0.9, 0.95, 0.8, 0.7], dtype=np.float32)
    chosen = select_error_indices(y_true, y_pred, y_prob, n=2)
    assert chosen.tolist() == [1, 2]


def test_taxonomy_counts_fills_missing_labels():
    counts = taxonomy_counts(["confused_disease", "confused_disease", "other"])
    assert counts["confused_disease"] == 2
    assert counts["shortcut"] == 0
    assert counts["other"] == 1
