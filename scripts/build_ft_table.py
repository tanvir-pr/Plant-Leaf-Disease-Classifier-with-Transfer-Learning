"""Build the ResNet-50 fine-tuning comparison table from run summaries."""

from __future__ import annotations

import json
import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.utils import project_root


RUNS = [
    ("scratch_cnn", "runs/scratch_cnn/summary.json"),
    ("resnet50_frozen", "runs/resnet50_frozen/summary.json"),
    ("resnet50_last_block", "runs/resnet50_last_block/summary.json"),
    ("resnet50_full", "runs/resnet50_full/summary.json"),
]


def _row(name: str, summary: dict) -> dict:
    metrics = summary.get("best_metrics") or {}
    if not metrics and summary.get("history"):
        metrics = max(summary["history"], key=lambda item: item.get("macro_f1", -1))
    return {
        "run": name,
        "strategy": summary.get("strategy", "full"),
        "accuracy": metrics.get("accuracy"),
        "macro_f1": metrics.get("macro_f1", summary.get("best_macro_f1")),
        "top3": metrics.get("top3"),
        "top5": metrics.get("top5"),
        "trainable_parameters": summary.get("trainable_parameters"),
        "head_lr": summary.get("head_lr"),
        "backbone_lr": summary.get("backbone_lr"),
        "best_checkpoint": summary.get("best_checkpoint"),
    }


def main() -> None:
    root = project_root()
    rows: list[dict] = []
    missing: list[str] = []
    for name, relative in RUNS:
        path = root / relative
        if not path.is_file():
            missing.append(relative)
            continue
        summary = json.loads(path.read_text(encoding="utf-8"))
        rows.append(_row(name, summary))

    if not rows:
        raise FileNotFoundError("No run summaries found.")

    frame = pd.DataFrame(rows)
    out_dir = root / "reports" / "tables"
    out_dir.mkdir(parents=True, exist_ok=True)
    csv_path = out_dir / "resnet50_finetune_table.csv"
    md_path = out_dir / "resnet50_finetune_table.md"
    frame.to_csv(csv_path, index=False)
    headers = list(frame.columns)
    lines = [
        "| " + " | ".join(headers) + " |",
        "| " + " | ".join("---" for _ in headers) + " |",
    ]
    for row in frame.itertuples(index=False):
        cells: list[str] = []
        for value in row:
            if isinstance(value, float):
                cells.append(f"{value:.4f}")
            elif value is None or (isinstance(value, float) and pd.isna(value)):
                cells.append("")
            else:
                cells.append(str(value))
        lines.append("| " + " | ".join(cells) + " |")
    md_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(frame.to_string(index=False))
    print(f"Wrote {csv_path}")
    print(f"Wrote {md_path}")
    if missing:
        print("Missing summaries:")
        for item in missing:
            print(f"  - {item}")


if __name__ == "__main__":
    main()
