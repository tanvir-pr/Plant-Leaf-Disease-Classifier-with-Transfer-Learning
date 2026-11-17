"""Field-style domain-shift helpers."""

import numpy as np
from PIL import Image

from src.domain_shift import (
    composite_leaf_on_background,
    make_field_proxy,
    outdoor_background,
    phone_style_corrupt,
)


def _lab_leaf() -> Image.Image:
    rgb = np.full((128, 128, 3), 170, dtype=np.uint8)
    rgb[30:100, 40:90] = (40, 140, 50)
    return Image.fromarray(rgb)


def test_outdoor_background_shape():
    bg = outdoor_background((64, 80), np.random.default_rng(0))
    assert bg.shape == (64, 80, 3)


def test_composite_changes_corners():
    original = np.asarray(_lab_leaf())
    composed = np.asarray(composite_leaf_on_background(_lab_leaf(), np.random.default_rng(1)))
    assert composed.shape == original.shape
    # Lab grey corners should no longer match.
    assert not np.allclose(composed[2, 2], original[2, 2], atol=5)


def test_phone_style_keeps_size():
    image = _lab_leaf()
    out = phone_style_corrupt(image, np.random.default_rng(2))
    assert out.size == image.size


def test_make_field_proxy_is_rgb():
    out = make_field_proxy(_lab_leaf(), seed=3)
    assert out.mode == "RGB"
    assert out.size == (128, 128)
