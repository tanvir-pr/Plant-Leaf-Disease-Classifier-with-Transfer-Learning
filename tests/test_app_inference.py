"""Gradio inference helpers (no server launch)."""

from app.inference import DISCLAIMER, friendly_label
from app.main import format_result_text


def test_friendly_label():
    assert friendly_label("Tomato___Early_blight") == "Tomato — Early blight"


def test_disclaimer_is_educational():
    text = DISCLAIMER.lower()
    assert "educational" in text
    assert "not for field diagnosis" in text


def test_format_result_text_includes_disclaimer():
    result = {
        "top_label": "Tomato — healthy",
        "confidence": 0.91,
        "top_k": [
            {"label": "Tomato — healthy", "confidence": 0.91},
            {"label": "Potato — healthy", "confidence": 0.04},
        ],
        "disclaimer": DISCLAIMER,
    }
    text = format_result_text(result)
    assert "Tomato — healthy" in text
    assert "91.0%" in text
    assert DISCLAIMER in text
