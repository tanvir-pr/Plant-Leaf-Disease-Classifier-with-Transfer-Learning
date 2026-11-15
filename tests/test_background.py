"""Leaf mask and crop helpers."""

import numpy as np
from PIL import Image

from src.background import crop_to_leaf, leaf_mask_rgb, mask_background


def _synthetic_leaf() -> Image.Image:
    rgb = np.full((128, 128, 3), 170, dtype=np.uint8)
    rgb[30:100, 40:90] = (40, 140, 50)
    return Image.fromarray(rgb)


def test_leaf_mask_keeps_green_blob():
    image = np.asarray(_synthetic_leaf())
    mask = leaf_mask_rgb(image)
    assert mask.shape == (128, 128)
    assert mask[60, 60]
    assert not mask[5, 5]


def test_mask_background_fills_corners():
    masked = np.asarray(mask_background(_synthetic_leaf(), fill=(0, 0, 0)))
    assert tuple(masked[5, 5].tolist()) == (0, 0, 0)
    assert masked[60, 60, 1] > 100


def test_crop_to_leaf_is_square():
    cropped = crop_to_leaf(_synthetic_leaf())
    assert cropped.size[0] == cropped.size[1]
    assert cropped.size[0] > 0
