"""Load the best plant checkpoint and run Grad-CAM inference."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import torch
from PIL import Image

from src.checkpoint import load_plant_checkpoint
from src.transforms import build_transforms
from src.utils import project_root
from src.xai import compute_gradcam, overlay_cam, tensor_to_rgb

DISCLAIMER = (
    "Educational demo only — not for field diagnosis or agricultural advice. "
    "The model was trained on PlantVillage lab photos and can fail on phone "
    "photos, new crops, or unfamiliar lighting."
)


def resolve_checkpoint(root: Path | None = None) -> Path:
    root = root or project_root()
    choice_path = root / "reports" / "tables" / "augmentation_choice.json"
    if choice_path.is_file():
        choice = json.loads(choice_path.read_text(encoding="utf-8"))
        return root / choice["best_checkpoint"]
    return root / "runs" / "aug_color_jitter" / "best.pt"


def friendly_label(class_name: str) -> str:
    return class_name.replace("___", " — ").replace("_", " ")


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
    ) -> dict:
        if image is None:
            raise ValueError("No image provided.")
        probs = self.predict_proba(image)
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
        overlay = self.explain(image, pred_idx)
        label_to_conf = {item["label"]: item["confidence"] for item in top}
        return {
            "top_label": top[0]["label"],
            "top_class_name": top[0]["class_name"],
            "confidence": top[0]["confidence"],
            "top_k": top,
            "label_confidences": label_to_conf,
            "overlay": overlay,
            "disclaimer": DISCLAIMER,
        }
