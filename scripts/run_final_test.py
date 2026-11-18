"""Score the held-out PlantVillage test split exactly once."""

from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd
import torch
from torch.utils.data import DataLoader

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.checkpoint import load_plant_checkpoint
from src.dataset import PlantSplitDataset, build_class_mapping, load_split_frame
from src.evaluate import HeldOutAlreadyEvaluatedError, assert_test_unlocked, evaluate_loader
from src.transforms import build_transforms
from src.utils import load_config, project_root, set_seed


def markdown_table(frame: pd.DataFrame) -> str:
    headers = list(frame.columns)
    lines = [
        "| " + " | ".join(headers) + " |",
        "| " + " | ".join("---" for _ in headers) + " |",
    ]
    for row in frame.itertuples(index=False):
        cells = []
        for value in row:
            if isinstance(value, float):
                cells.append(f"{value:.4f}")
            else:
                cells.append(str(value))
        lines.append("| " + " | ".join(cells) + " |")
    return "\n".join(lines) + "\n"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--force",
        action="store_true",
        help="Overwrite an existing locked test result (avoid unless intentional).",
    )
    args = parser.parse_args()

    set_seed(42)
    root = project_root()
    table_dir = root / "reports" / "tables"
    table_dir.mkdir(parents=True, exist_ok=True)
    lock_path = table_dir / "final_test.json"

    try:
        assert_test_unlocked(lock_path, force=args.force)
    except HeldOutAlreadyEvaluatedError as exc:
        print(str(exc))
        print("Existing locked metrics:")
        print(lock_path.read_text(encoding="utf-8"))
        raise SystemExit(1) from exc

    choice_path = table_dir / "augmentation_choice.json"
    if choice_path.is_file():
        choice = json.loads(choice_path.read_text(encoding="utf-8"))
        ckpt_path = root / choice["best_checkpoint"]
        val_accuracy = float(choice.get("accuracy", float("nan")))
        val_macro_f1 = float(choice.get("macro_f1", float("nan")))
    else:
        ckpt_path = root / "runs" / "aug_color_jitter" / "best.pt"
        val_accuracy = float("nan")
        val_macro_f1 = float("nan")
        summary_path = root / "runs" / "aug_color_jitter" / "summary.json"
        if summary_path.is_file():
            summary = json.loads(summary_path.read_text(encoding="utf-8"))
            best = summary.get("best_metrics") or {}
            val_accuracy = float(best.get("accuracy", float("nan")))
            val_macro_f1 = float(best.get("macro_f1", float("nan")))

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model, class_names, _config = load_plant_checkpoint(ckpt_path, device=device)
    plant_cfg = load_config(root / "configs" / "aug_color_jitter.yaml")
    split_csv = root / plant_cfg["paths"]["split_csv"]
    dataset_root = root / plant_cfg["plantvillage"]["local_dir"]
    full_frame = pd.read_csv(split_csv)
    _, class_to_idx = build_class_mapping(full_frame)
    test_frame = load_split_frame(split_csv, "test")

    dataset = PlantSplitDataset(
        test_frame,
        dataset_root=dataset_root,
        class_to_idx=class_to_idx,
        transform=build_transforms(
            int(plant_cfg["image"]["size"]),
            train=False,
            augmentation="none",
        ),
    )
    loader = DataLoader(dataset, batch_size=32, shuffle=False, num_workers=0)
    metrics = evaluate_loader(model, loader, device, len(class_names))

    payload = {
        "locked": True,
        "evaluated_at_utc": datetime.now(timezone.utc).isoformat(),
        "checkpoint": str(ckpt_path.relative_to(root).as_posix()),
        "device": str(device),
        "split": "test",
        "n_images": int(len(test_frame)),
        "n_classes": int(len(class_names)),
        "seed": 42,
        "metrics": {
            "accuracy": float(metrics["accuracy"]),
            "macro_f1": float(metrics["macro_f1"]),
            "top3": float(metrics["top3"]),
            "top5": float(metrics["top5"]),
            "loss": float(metrics["loss"]),
        },
        "val_reference": {
            "accuracy": val_accuracy,
            "macro_f1": val_macro_f1,
        },
        "note": (
            "Held-out test scored once. Do not re-run to tune the model. "
            "These numbers are the locked plant-chapter result."
        ),
    }

    compare = pd.DataFrame(
        [
            {
                "split": "val",
                "accuracy": val_accuracy,
                "macro_f1": val_macro_f1,
                "top3": float("nan"),
                "top5": float("nan"),
                "loss": float("nan"),
                "n_images": int((full_frame["split"] == "val").sum()),
            },
            {
                "split": "test",
                "accuracy": float(metrics["accuracy"]),
                "macro_f1": float(metrics["macro_f1"]),
                "top3": float(metrics["top3"]),
                "top5": float(metrics["top5"]),
                "loss": float(metrics["loss"]),
                "n_images": int(len(test_frame)),
            },
        ]
    )

    csv_path = table_dir / "final_test.csv"
    md_path = table_dir / "final_test.md"
    compare.to_csv(csv_path, index=False)
    lock_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")

    md_lines = [
        "# Locked held-out test result",
        "",
        f"Checkpoint: `{payload['checkpoint']}`",
        f"Device: `{payload['device']}`",
        f"Evaluated (UTC): `{payload['evaluated_at_utc']}`",
        "",
        "This split was scored **once**. Do not change training and re-measure test.",
        "",
        markdown_table(compare).rstrip(),
        "",
        "## Locked test metrics",
        "",
        f"- accuracy: `{payload['metrics']['accuracy']:.4f}`",
        f"- macro-F1: `{payload['metrics']['macro_f1']:.4f}`",
        f"- top-3: `{payload['metrics']['top3']:.4f}`",
        f"- top-5: `{payload['metrics']['top5']:.4f}`",
        f"- loss: `{payload['metrics']['loss']:.4f}`",
        "",
    ]
    md_path.write_text("\n".join(md_lines) + "\n", encoding="utf-8")

    print(json.dumps(payload, indent=2))
    print(f"Wrote {csv_path}")
    print(f"Wrote {md_path}")
    print(f"Wrote {lock_path}")


if __name__ == "__main__":
    main()
