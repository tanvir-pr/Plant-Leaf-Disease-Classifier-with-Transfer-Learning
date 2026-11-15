"""Remove lab backgrounds from PlantVillage leaf photos."""

from __future__ import annotations

import cv2
import numpy as np
from PIL import Image


def _largest_component(mask: np.ndarray) -> np.ndarray:
    num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(
        mask.astype(np.uint8), connectivity=8
    )
    if num_labels <= 1:
        return mask.astype(bool)
    # Label 0 is background.
    areas = stats[1:, cv2.CC_STAT_AREA]
    best = 1 + int(np.argmax(areas))
    return labels == best


def leaf_mask_rgb(rgb: np.ndarray) -> np.ndarray:
    """Binary leaf mask. True where the leaf is likely present.

    PlantVillage photos usually sit on a flat grey lab backdrop. The mask keeps
    pixels that are either greener than the corners or far from the corner colour,
    then retains the largest connected blob.
    """
    if rgb.ndim != 3 or rgb.shape[2] != 3:
        raise ValueError("Expected an HxWx3 RGB image.")
    image = rgb.astype(np.uint8)
    h, w, _ = image.shape
    corners = np.stack(
        [
            image[0, 0],
            image[0, w - 1],
            image[h - 1, 0],
            image[h - 1, w - 1],
        ],
        axis=0,
    ).astype(np.float32)
    bg = corners.mean(axis=0)
    dist = np.linalg.norm(image.astype(np.float32) - bg, axis=2)
    diff_mask = dist > 28.0

    hsv = cv2.cvtColor(image, cv2.COLOR_RGB2HSV)
    green = (
        (hsv[:, :, 0] >= 25)
        & (hsv[:, :, 0] <= 95)
        & (hsv[:, :, 1] >= 25)
        & (hsv[:, :, 2] >= 25)
    )
    mask = (diff_mask | green).astype(np.uint8) * 255
    kernel = np.ones((5, 5), np.uint8)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel, iterations=2)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel, iterations=1)
    return _largest_component(mask)


def mask_background(image: Image.Image, fill: tuple[int, int, int] = (0, 0, 0)) -> Image.Image:
    """Keep the leaf and fill the lab background with a solid colour."""
    rgb = np.asarray(image.convert("RGB"))
    mask = leaf_mask_rgb(rgb)
    out = rgb.copy()
    out[~mask] = np.asarray(fill, dtype=np.uint8)
    return Image.fromarray(out)


def crop_to_leaf(
    image: Image.Image,
    fill: tuple[int, int, int] = (0, 0, 0),
    pad: int = 8,
) -> Image.Image:
    """Mask the background, then crop to the leaf bounding box and pad square."""
    rgb = np.asarray(image.convert("RGB"))
    mask = leaf_mask_rgb(rgb)
    if not mask.any():
        return Image.fromarray(np.full_like(rgb, fill))

    ys, xs = np.where(mask)
    y0 = max(int(ys.min()) - pad, 0)
    y1 = min(int(ys.max()) + pad + 1, rgb.shape[0])
    x0 = max(int(xs.min()) - pad, 0)
    x1 = min(int(xs.max()) + pad + 1, rgb.shape[1])
    cropped = rgb[y0:y1, x0:x1].copy()
    crop_mask = mask[y0:y1, x0:x1]
    cropped[~crop_mask] = np.asarray(fill, dtype=np.uint8)

    side = max(cropped.shape[0], cropped.shape[1])
    canvas = np.full((side, side, 3), fill, dtype=np.uint8)
    oy = (side - cropped.shape[0]) // 2
    ox = (side - cropped.shape[1]) // 2
    canvas[oy : oy + cropped.shape[0], ox : ox + cropped.shape[1]] = cropped
    return Image.fromarray(canvas)
