"""Gradio demo: leaf photo → class, confidence, Grad-CAM overlay."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import gradio as gr
import numpy as np
from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from app.inference import DISCLAIMER, PlantPredictor


_PREDICTOR: PlantPredictor | None = None


def get_predictor() -> PlantPredictor:
    global _PREDICTOR
    if _PREDICTOR is None:
        _PREDICTOR = PlantPredictor()
    return _PREDICTOR


def format_result_text(result: dict) -> str:
    lines = [
        f"**Prediction:** {result['top_label']}",
        f"**Confidence:** {result['confidence']:.1%}",
        "",
        "Top-5:",
    ]
    for item in result["top_k"]:
        lines.append(f"- {item['label']}: {item['confidence']:.1%}")
    lines.extend(["", f"_{result['disclaimer']}_"])
    return "\n".join(lines)


def run_inference(image: Image.Image | np.ndarray | None):
    if image is None:
        raise gr.Error("Upload a leaf photo first.")
    if isinstance(image, np.ndarray):
        image = Image.fromarray(image.astype(np.uint8)).convert("RGB")
    else:
        image = image.convert("RGB")

    result = get_predictor().predict(image, top_k=5)
    overlay = result["overlay"]
    text = format_result_text(result)
    return overlay, result["label_confidences"], text


def example_paths(root: Path) -> list[str]:
    own = root / "data" / "own_leaves"
    paths = sorted(own.glob("proxy_*.jpg"))[:4]
    return [str(path) for path in paths]


def build_demo() -> gr.Blocks:
    examples = example_paths(ROOT)
    with gr.Blocks(title="Plant Leaf Disease Demo") as demo:
        gr.Markdown(
            """
# Plant Leaf Disease Classifier

Upload a leaf photo to see the predicted class, confidence, and a Grad-CAM overlay.

"""
            + f"**Disclaimer:** {DISCLAIMER}"
        )
        with gr.Row():
            with gr.Column():
                image_in = gr.Image(type="pil", label="Leaf photo")
                run_btn = gr.Button("Classify", variant="primary")
            with gr.Column():
                overlay_out = gr.Image(type="numpy", label="Grad-CAM overlay")
                conf_out = gr.Label(label="Top predictions", num_top_classes=5)
                text_out = gr.Markdown()

        run_btn.click(
            fn=run_inference,
            inputs=image_in,
            outputs=[overlay_out, conf_out, text_out],
        )
        image_in.change(
            fn=run_inference,
            inputs=image_in,
            outputs=[overlay_out, conf_out, text_out],
        )
        if examples:
            gr.Examples(examples=examples, inputs=image_in, label="Example proxies")
        gr.Markdown(
            "Trained on PlantVillage color photos with a leaf-grouped split. "
            "Best validation checkpoint: EfficientNet-B0 + color jitter."
        )
    return demo


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=7860)
    parser.add_argument("--share", action="store_true")
    args = parser.parse_args()

    # Warm the model before opening the UI.
    predictor = get_predictor()
    print(f"Loaded {predictor.checkpoint} on {predictor.device}", flush=True)
    print(f"Classes: {len(predictor.class_names)}", flush=True)

    demo = build_demo()
    demo.launch(server_name=args.host, server_port=args.port, share=args.share)


if __name__ == "__main__":
    main()
