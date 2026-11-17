"""Own-leaf / field-style domain-shift helpers."""

from __future__ import annotations

import io
from pathlib import Path

import cv2
import numpy as np
import pandas as pd
from PIL import Image, ImageEnhance, ImageFilter

from src.background import leaf_mask_rgb

IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png", ".webp", ".bmp"}


def list_image_files(directory: Path) -> list[Path]:
    if not directory.is_dir():
        return []
    files = [
        path
        for path in sorted(directory.iterdir())
        if path.is_file() and path.suffix.lower() in IMAGE_SUFFIXES
    ]
    return files


def load_own_leaves_frame(own_dir: Path) -> pd.DataFrame:
    """Load labels.csv if present, else invent rows from loose image files."""
    own_dir = Path(own_dir)
    labels_path = own_dir / "labels.csv"
    if labels_path.is_file():
        frame = pd.read_csv(labels_path)
        missing = {"relative_path", "class_name"}.difference(frame.columns)
        if missing:
            raise ValueError(f"{labels_path} missing columns: {sorted(missing)}")
        return frame.reset_index(drop=True)

    rows = []
    for path in list_image_files(own_dir):
        rows.append(
            {
                "relative_path": path.name,
                "class_name": "",
                "source": "unlabeled",
            }
        )
    return pd.DataFrame(rows)


def outdoor_background(size: tuple[int, int], rng: np.random.Generator) -> np.ndarray:
    """Cheap grass/soil-like backdrop (not a lab grey)."""
    h, w = size
    yy, xx = np.mgrid[0:h, 0:w]
    green = 40 + (yy / max(h - 1, 1)) * 50
    red = 30 + (xx / max(w - 1, 1)) * 35
    blue = 20 + ((xx + yy) / max(h + w - 2, 1)) * 25
    bg = np.stack([red, green, blue], axis=-1).astype(np.float32)
    noise = rng.normal(0.0, 12.0, size=bg.shape).astype(np.float32)
    mottled = np.clip(bg + noise, 0, 255).astype(np.uint8)
    # Soft blur so it reads as distant foliage/soil.
    mottled = cv2.GaussianBlur(mottled, (0, 0), sigmaX=2.5)
    return mottled


def composite_leaf_on_background(
    image: Image.Image,
    rng: np.random.Generator | None = None,
) -> Image.Image:
    """Cut the leaf out of a lab photo and paste it onto a field-like backdrop."""
    rng = rng or np.random.default_rng(0)
    rgb = np.asarray(image.convert("RGB"))
    mask = leaf_mask_rgb(rgb)
    bg = outdoor_background(rgb.shape[:2], rng)
    out = bg.copy()
    out[mask] = rgb[mask]
    return Image.fromarray(out)


def phone_style_corrupt(
    image: Image.Image,
    rng: np.random.Generator | None = None,
) -> Image.Image:
    """Lighting, focus, and JPEG artifacts typical of phone snaps."""
    rng = rng or np.random.default_rng(0)
    out = image.convert("RGB")

    # Random zoom/crop toward the leaf.
    w, h = out.size
    scale = float(rng.uniform(0.78, 0.95))
    nw, nh = int(w * scale), int(h * scale)
    left = int(rng.integers(0, max(w - nw, 1)))
    top = int(rng.integers(0, max(h - nh, 1)))
    out = out.crop((left, top, left + nw, top + nh)).resize((w, h), Image.BICUBIC)

    out = ImageEnhance.Brightness(out).enhance(float(rng.uniform(0.75, 1.25)))
    out = ImageEnhance.Contrast(out).enhance(float(rng.uniform(0.8, 1.35)))
    out = ImageEnhance.Color(out).enhance(float(rng.uniform(0.75, 1.3)))

    if float(rng.random()) < 0.45:
        out = out.filter(ImageFilter.GaussianBlur(radius=float(rng.uniform(0.4, 1.2))))

    # Warm or cool cast.
    arr = np.asarray(out).astype(np.float32)
    cast = np.asarray(
        [
            float(rng.uniform(-18, 22)),
            float(rng.uniform(-12, 12)),
            float(rng.uniform(-22, 10)),
        ],
        dtype=np.float32,
    )
    arr = np.clip(arr + cast, 0, 255).astype(np.uint8)
    out = Image.fromarray(arr)

    quality = int(rng.integers(35, 75))
    buffer = io.BytesIO()
    out.save(buffer, format="JPEG", quality=quality)
    buffer.seek(0)
    return Image.open(buffer).convert("RGB")


def make_field_proxy(
    image: Image.Image,
    seed: int,
) -> Image.Image:
    rng = np.random.default_rng(seed)
    composed = composite_leaf_on_background(image, rng)
    return phone_style_corrupt(composed, rng)
