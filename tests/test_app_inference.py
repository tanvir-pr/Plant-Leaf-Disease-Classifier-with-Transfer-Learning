"""Gradio inference helpers (no server launch)."""

import numpy as np
from PIL import Image

from app.inference import (
    DISCLAIMER,
    friendly_label,
    preprocess_leaf,
    uncertainty_flags,
)
from app.main import format_result_text


def test_friendly_label():
    assert friendly_label("Tomato___Early_blight") == "Tomato — Early blight"


def test_disclaimer_is_educational():
    text = DISCLAIMER.lower()
    assert "educational" in text
    assert "not for field diagnosis" in text


def test_uncertainty_flags_low_confidence_and_small_gap():
    top = [
        {"label": "a", "confidence": 0.60},
        {"label": "b", "confidence": 0.21},
    ]
    flags = uncertainty_flags(top)
    assert flags["uncertain"]
    assert flags["margin"] == 0.39


def test_preprocess_crop_returns_square_rgb():
    rgb = np.full((120, 160, 3), 170, dtype=np.uint8)
    rgb[20:100, 40:120] = (40, 140, 50)
    out = preprocess_leaf(Image.fromarray(rgb), mode="crop_to_leaf")
    assert out.mode == "RGB"
    assert out.size[0] == out.size[1]


def test_format_result_text_includes_disclaimer():
    result = {
        "top_label": "Tomato — healthy",
        "confidence": 0.91,
        "preprocess": "crop_to_leaf",
        "margin": 0.87,
        "uncertain": False,
        "uncertainty_reasons": [],
        "top_k": [
            {"label": "Tomato — healthy", "confidence": 0.91},
            {"label": "Potato — healthy", "confidence": 0.04},
        ],
        "disclaimer": DISCLAIMER,
        "photo_tips": "tip",
    }
    text = format_result_text(result)
    assert "Tomato — healthy" in text
    assert "91.0%" in text
    assert DISCLAIMER in text
    assert "Uncertain" not in text


def test_format_result_text_marks_uncertain():
    result = {
        "top_label": "Pepper, bell — healthy",
        "confidence": 0.60,
        "preprocess": "none",
        "margin": 0.20,
        "uncertain": True,
        "uncertainty_reasons": ["confidence below 70%"],
        "top_k": [
            {"label": "Pepper, bell — healthy", "confidence": 0.60},
            {"label": "Corn — Common rust", "confidence": 0.40},
        ],
        "disclaimer": DISCLAIMER,
        "photo_tips": "tip",
    }
    text = format_result_text(result)
    assert "Uncertain" in text
