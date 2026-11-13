"""Build the augmentation study table from run summaries."""

from __future__ import annotations

import json
import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.utils import project_root

# flip baseline reuses the Day-5 EfficientNet-B0 run (same transform).
RUNS = [
    ("none", "runs/aug_none/summary.json"),
    ("flip", "runs/efficientnet_b0/summary.json"),
    ("flip_rotate", "runs/aug_flip_rotate/summary.json"),
    ("color_jitter", "runs/aug_color_jitter/summary.json"),
    ("randaugment", "runs/aug_randaugment/summary.json"),
]


def _row(aug_name: str, summary: dict) -> dict:
    metrics = summary.get("best_metrics") or {}
    if not metrics and summary.get("history"):
        metrics = max(summary["history"], key=lambda item: item.get("macro_f1", -1))
    return {
        "augmentation": summary.get("augmentation", aug_name),
        "model": summary.get("model", "efficientnet_b0"),
        "accuracy": metrics.get("accuracy"),
        "macro_f1": metrics.get("macro_f1", summary.get("best_macro_f1")),
        "top3": metrics.get("top3"),
        "top5": metrics.get("top5"),
        "best_checkpoint": summary.get("best_checkpoint"),
    }


def main() -> None:
    root = project_root()
    rows: list[dict] = []
    missing: list[str] = []
    for aug_name, relative in RUNS:
        path = root / relative
        if not path.is_file():
            missing.append(relative)
            continue
        summary = json.loads(path.read_text(encoding="utf-8"))
        if "augmentation" not in summary:
            summary = dict(summary)
            summary["augmentation"] = aug_name
        rows.append(_row(aug_name, summary))

    if not rows:
        raise FileNotFoundError("No augmentation run summaries found.")

    frame = pd.DataFrame(rows).sort_values("macro_f1", ascending=False)
    out_dir = root / "reports" / "tables"
    out_dir.mkdir(parents=True, exist_ok=True)
    csv_path = out_dir / "augmentation_study.csv"
    md_path = out_dir / "augmentation_study.md"
    frame.to_csv(csv_path, index=False)

    best = frame.iloc[0]
    choice_path = out_dir / "augmentation_choice.json"
    choice = {
        "best_augmentation": best["augmentation"],
        "macro_f1": float(best["macro_f1"]),
        "accuracy": float(best["accuracy"]),
        "best_checkpoint": best["best_checkpoint"],
    }
    choice_path.write_text(json.dumps(choice, indent=2), encoding="utf-8")

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
    print(json.dumps(choice, indent=2))
    print(f"Wrote {csv_path}")
    print(f"Wrote {md_path}")
    print(f"Wrote {choice_path}")
    if missing:
        print("Missing summaries:")
        for item in missing:
            print(f"  - {item}")


if __name__ == "__main__":
    main()
