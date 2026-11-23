"""Load the best plant checkpoint and run Grad-CAM inference."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import torch
from PIL import Image

from src.background import crop_to_leaf, mask_background
from src.checkpoint import load_plant_checkpoint
from src.transforms import build_transforms
from src.utils import project_root
from src.xai import compute_gradcam, overlay_cam, tensor_to_rgb

DISCLAIMER = (
    "Educational demo only — not for field diagnosis or agricultural advice. "
    "The model was trained on PlantVillage lab photos and can fail on phone "
    "photos, new crops, or unfamiliar lighting."
)

PHOTO_TIPS = (
    "Best results: one leaf filling the frame, plain background, sharp focus. "
    "Field / bush / fruit photos often confuse this lab-trained model."
)

PREPROCESS_MODES = ("none", "mask_background", "crop_to_leaf")


def resolve_checkpoint(root: Path | None = None) -> Path:
    root = root or project_root()
    choice_path = root / "reports" / "tables" / "augmentation_choice.json"
    if choice_path.is_file():
        choice = json.loads(choice_path.read_text(encoding="utf-8"))
        return root / choice["best_checkpoint"]
    return root / "runs" / "aug_color_jitter" / "best.pt"


def friendly_label(class_name: str) -> str:
    return class_name.replace("___", " — ").replace("_", " ")


def preprocess_leaf(image: Image.Image, mode: str = "crop_to_leaf") -> Image.Image:
    """Bring phone/field photos a bit closer to PlantVillage-style inputs."""
    key = (mode or "none").strip().lower()
    rgb = image.convert("RGB")
    if key in {"none", "off", "original"}:
        return rgb
    if key in {"mask_background", "mask"}:
        return mask_background(rgb)
    if key in {"crop_to_leaf", "crop"}:
        return crop_to_leaf(rgb)
    raise ValueError(f"Unknown preprocess mode={mode!r}. Choose from {PREPROCESS_MODES}.")


def uncertainty_flags(
    top_k: list[dict],
    *,
    low_conf: float = 0.70,
    small_gap: float = 0.25,
) -> dict:
    """Mark predictions that should not be trusted as a firm diagnosis."""
    if not top_k:
        return {
            "uncertain": True,
            "reasons": ["no predictions"],
            "margin": 0.0,
        }
    top1 = float(top_k[0]["confidence"])
    top2 = float(top_k[1]["confidence"]) if len(top_k) > 1 else 0.0
    margin = top1 - top2
    reasons: list[str] = []
    if top1 < low_conf:
        reasons.append(f"confidence below {low_conf:.0%}")
    if margin < small_gap:
        reasons.append(f"top-1 and top-2 are close (gap {margin:.1%})")
    return {
        "uncertain": bool(reasons),
        "reasons": reasons,
        "margin": margin,
    }


class PlantPredictor:
    """Lazy-loaded classifier used by the Gradio demo."""

    def __init__(
        self,
        checkpoint: Path | None = None,
        device: str | torch.device | None = None,
        image_size: int = 224,
    ) -> None:
        root = project_root()
        self.checkpoint = Path(checkpoint) if checkpoint else resolve_checkpoint(root)
        if device is None:
            device = "cuda" if torch.cuda.is_available() else "cpu"
        self.device = torch.device(device)
        self.image_size = int(image_size)
        self.model, self.class_names, _config = load_plant_checkpoint(
            self.checkpoint,
            device=self.device,
        )
        self.transform = build_transforms(
            self.image_size,
            train=False,
            augmentation="none",
        )

    @torch.inference_mode()
    def predict_proba(self, image: Image.Image) -> np.ndarray:
        tensor = self.transform(image.convert("RGB")).unsqueeze(0).to(self.device)
        probs = torch.softmax(self.model(tensor), dim=1)[0]
        return probs.detach().cpu().numpy()

    def explain(self, image: Image.Image, class_index: int) -> np.ndarray:
        tensor = self.transform(image.convert("RGB")).unsqueeze(0).to(self.device)
        cam = compute_gradcam(self.model, tensor, [int(class_index)], plus_plus=False)[0]
        rgb = tensor_to_rgb(tensor[0])
        return overlay_cam(rgb, cam)

    def predict(
        self,
        image: Image.Image,
        top_k: int = 5,
        preprocess: str = "crop_to_leaf",
    ) -> dict:
        if image is None:
            raise ValueError("No image provided.")
        processed = preprocess_leaf(image, mode=preprocess)
        probs = self.predict_proba(processed)
        order = np.argsort(-probs)[: int(top_k)]
        top = [
            {
                "class_name": self.class_names[int(i)],
                "label": friendly_label(self.class_names[int(i)]),
                "confidence": float(probs[int(i)]),
            }
            for i in order
        ]
        pred_idx = int(order[0])
        overlay = self.explain(processed, pred_idx)
        flags = uncertainty_flags(top)
        label_to_conf = {item["label"]: item["confidence"] for item in top}
        return {
            "top_label": top[0]["label"],
            "top_class_name": top[0]["class_name"],
            "confidence": top[0]["confidence"],
            "top_k": top,
            "label_confidences": label_to_conf,
            "overlay": overlay,
            "processed_image": processed,
            "preprocess": preprocess,
            "uncertain": flags["uncertain"],
            "uncertainty_reasons": flags["reasons"],
            "margin": flags["margin"],
            "disclaimer": DISCLAIMER,
            "photo_tips": PHOTO_TIPS,
        }
