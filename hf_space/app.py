"""Hugging Face Spaces entrypoint."""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from app.main import build_demo, get_predictor

# Warm the checkpoint once at startup.
_predictor = get_predictor()
print(f"Loaded {_predictor.checkpoint} on {_predictor.device}", flush=True)

demo = build_demo()

if __name__ == "__main__":
    demo.launch()
