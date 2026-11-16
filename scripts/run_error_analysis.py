"""Validation confusion matrix and a 30-error failure taxonomy."""

from __future__ import annotations

import json
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import torch
from PIL import Image
from sklearn.metrics import classification_report, confusion_matrix
from torch.utils.data import DataLoader

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.background import mask_background
from src.checkpoint import load_plant_checkpoint
from src.dataset import PlantSplitDataset, build_class_mapping, load_split_frame
from src.error_analysis import (
    TAXONOMY_LABELS,
    assign_taxonomy,
    image_quality_flags,
    rare_class_names,
    select_error_indices,
    taxonomy_counts,
)
from src.transforms import build_transforms
from src.utils import load_config, project_root, set_seed


def collect_predictions(
    model: torch.nn.Module,
    loader: DataLoader,
    device: torch.device,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    y_true: list[int] = []
    y_pred: list[int] = []
    y_prob: list[float] = []
    model.eval()
    with torch.no_grad():
        for images, labels in loader:
            images = images.to(device)
            logits = model(images)
            probs = torch.softmax(logits, dim=1)
            conf, pred = probs.max(dim=1)
            y_true.extend(labels.tolist())
            y_pred.extend(pred.cpu().tolist())
            y_prob.extend(conf.cpu().tolist())
    return (
        np.asarray(y_true, dtype=np.int64),
        np.asarray(y_pred, dtype=np.int64),
        np.asarray(y_prob, dtype=np.float32),
    )


def predict_one(
    model: torch.nn.Module,
    image: Image.Image,
    transform,
    device: torch.device,
) -> tuple[int, float]:
    tensor = transform(image).unsqueeze(0).to(device)
    with torch.no_grad():
        probs = torch.softmax(model(tensor), dim=1)[0]
    conf, pred = probs.max(dim=0)
    return int(pred.item()), float(conf.item())


def short_label(name: str, max_len: int = 28) -> str:
    text = name.replace("___", " / ")
    if len(text) <= max_len:
        return text
    return text[: max_len - 1] + "…"


def save_confusion_matrix(
    cm: np.ndarray,
    class_names: list[str],
    out_path: Path,
) -> None:
    labels = [short_label(name, 22) for name in class_names]
    fig, ax = plt.subplots(figsize=(14, 12))
    sns.heatmap(
        cm,
        ax=ax,
        cmap="Blues",
        square=True,
        cbar_kws={"shrink": 0.6},
        xticklabels=labels,
        yticklabels=labels,
    )
    ax.set_xlabel("Predicted")
    ax.set_ylabel("True")
    ax.set_title("Validation confusion matrix")
    plt.setp(ax.get_xticklabels(), rotation=90, fontsize=6)
    plt.setp(ax.get_yticklabels(), rotation=0, fontsize=6)
    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=160, bbox_inches="tight")
    plt.close(fig)


def save_error_grid(
    rows: list[dict],
    dataset_root: Path,
    out_path: Path,
    cols: int = 5,
) -> None:
    n = len(rows)
    if n == 0:
        return
    rows_n = int(np.ceil(n / cols))
    fig, axes = plt.subplots(rows_n, cols, figsize=(cols * 2.4, rows_n * 2.8))
    axes = np.atleast_1d(axes).ravel()
    for ax in axes:
        ax.axis("off")
    for i, row in enumerate(rows):
        image = Image.open(dataset_root / row["relative_path"]).convert("RGB")
        axes[i].imshow(image)
        axes[i].set_title(
            f"{row['taxonomy']}\n"
            f"T: {short_label(row['true_name'], 24)}\n"
            f"P: {short_label(row['pred_name'], 24)}\n"
            f"p={row['confidence']:.2f}",
            fontsize=6,
        )
        axes[i].axis("off")
    fig.suptitle("Top-30 high-confidence validation errors", fontsize=12)
    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


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
    set_seed(42)
    root = project_root()
    choice_path = root / "reports" / "tables" / "augmentation_choice.json"
    if choice_path.is_file():
        choice = json.loads(choice_path.read_text(encoding="utf-8"))
        ckpt_path = root / choice["best_checkpoint"]
    else:
        ckpt_path = root / "runs" / "aug_color_jitter" / "best.pt"

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model, class_names, _config = load_plant_checkpoint(ckpt_path, device=device)
    plant_cfg = load_config(root / "configs" / "aug_color_jitter.yaml")
    split_csv = root / plant_cfg["paths"]["split_csv"]
    dataset_root = root / plant_cfg["plantvillage"]["local_dir"]
    summary = json.loads(
        (root / "data" / "splits" / "plant_split_summary.json").read_text(
            encoding="utf-8"
        )
    )
    class_counts = {
        str(k): int(v) for k, v in summary["images_per_class"].items()
    }
    rare = rare_class_names(class_counts, quantile=0.25)

    full_frame = pd.read_csv(split_csv)
    _, class_to_idx = build_class_mapping(full_frame)
    val_frame = load_split_frame(split_csv, "val")
    transform = build_transforms(
        int(plant_cfg["image"]["size"]),
        train=False,
        augmentation="none",
    )
    dataset = PlantSplitDataset(
        val_frame,
        dataset_root=dataset_root,
        class_to_idx=class_to_idx,
        transform=transform,
    )
    loader = DataLoader(dataset, batch_size=32, shuffle=False, num_workers=0)
    y_true, y_pred, y_prob = collect_predictions(model, loader, device)

    labels = list(range(len(class_names)))
    cm = confusion_matrix(y_true, y_pred, labels=labels)
    report = classification_report(
        y_true,
        y_pred,
        labels=labels,
        target_names=class_names,
        output_dict=True,
        zero_division=0,
    )

    per_class_rows = []
    for name in class_names:
        stats = report[name]
        per_class_rows.append(
            {
                "class_name": name,
                "precision": float(stats["precision"]),
                "recall": float(stats["recall"]),
                "f1": float(stats["f1-score"]),
                "support": int(stats["support"]),
                "train_count": int(class_counts.get(name, 0)),
                "is_rare": name in rare,
            }
        )
    per_class = pd.DataFrame(per_class_rows).sort_values(
        "f1", ascending=True
    )

    # Off-diagonal pairs sorted by count.
    pair_rows = []
    for i, true_name in enumerate(class_names):
        for j, pred_name in enumerate(class_names):
            if i == j:
                continue
            count = int(cm[i, j])
            if count <= 0:
                continue
            pair_rows.append(
                {
                    "true_name": true_name,
                    "pred_name": pred_name,
                    "count": count,
                    "same_crop": true_name.split("___", 1)[0]
                    == pred_name.split("___", 1)[0],
                }
            )
    pairs = pd.DataFrame(pair_rows).sort_values(
        ["count", "true_name"], ascending=[False, True]
    )

    error_idx = select_error_indices(y_true, y_pred, y_prob, n=30)
    error_rows: list[dict] = []
    for rank, index in enumerate(error_idx.tolist(), start=1):
        row = val_frame.iloc[int(index)]
        path = dataset_root / row["relative_path"]
        pil = Image.open(path).convert("RGB")
        true_label = int(y_true[index])
        pred_label = int(y_pred[index])
        true_name = class_names[true_label]
        pred_name = class_names[pred_label]
        masked_pred, masked_conf = predict_one(
            model, mask_background(pil), transform, device
        )
        taxonomy = assign_taxonomy(
            true_name,
            pred_name,
            original_pred=pred_label,
            masked_pred=masked_pred,
            true_label=true_label,
            rare_classes=rare,
            image=pil,
        )
        quality = image_quality_flags(pil)
        error_rows.append(
            {
                "rank": rank,
                "val_index": int(index),
                "relative_path": str(row["relative_path"]),
                "true_name": true_name,
                "pred_name": pred_name,
                "confidence": float(y_prob[index]),
                "masked_pred_name": class_names[masked_pred],
                "masked_confidence": masked_conf,
                "taxonomy": taxonomy,
                "mean_luma": quality["mean_luma"],
                "std_luma": quality["std_luma"],
                "laplacian_var": quality["laplacian_var"],
                "leaf_frac": quality["leaf_frac"],
            }
        )

    errors = pd.DataFrame(error_rows)
    counts = taxonomy_counts(errors["taxonomy"].tolist() if not errors.empty else [])

    table_dir = root / "reports" / "tables"
    figure_dir = root / "reports" / "figures"
    table_dir.mkdir(parents=True, exist_ok=True)
    figure_dir.mkdir(parents=True, exist_ok=True)

    save_confusion_matrix(cm, class_names, figure_dir / "confusion_matrix_val.png")
    save_error_grid(error_rows, dataset_root, figure_dir / "error_taxonomy_grid.png")

    per_class_path = table_dir / "per_class_val.csv"
    pairs_path = table_dir / "confused_pairs_val.csv"
    errors_path = table_dir / "error_taxonomy_30.csv"
    per_class.to_csv(per_class_path, index=False)
    pairs.to_csv(pairs_path, index=False)
    errors.to_csv(errors_path, index=False)

    counts_frame = pd.DataFrame(
        [{"taxonomy": k, "count": v} for k, v in counts.items()]
    )
    counts_path = table_dir / "error_taxonomy_counts.csv"
    counts_frame.to_csv(counts_path, index=False)

    top_pairs = pairs.head(15)
    worst_classes = per_class.head(10)
    findings = []
    if not pairs.empty:
        top = pairs.iloc[0]
        findings.append(
            f"Largest confusion: {top['true_name']} → {top['pred_name']} "
            f"({int(top['count'])} val images)."
        )
    same_crop_share = (
        float(pairs["same_crop"].mean()) if not pairs.empty else 0.0
    )
    findings.append(
        f"{same_crop_share:.0%} of all off-diagonal confusions stay inside the same crop."
    )
    dominant = max(counts, key=counts.get) if errors.shape[0] else "other"
    findings.append(
        f"Among the {len(error_rows)} highest-confidence errors, "
        f"the dominant taxonomy label is `{dominant}` "
        f"({counts.get(dominant, 0)})."
    )
    if counts.get("shortcut", 0) == 0:
        findings.append(
            "No sampled error flipped after background masking, "
            "consistent with the Day-8 shortcut test."
        )

    note_lines = [
        "# Validation failure taxonomy",
        "",
        f"Checkpoint: `{ckpt_path.relative_to(root).as_posix()}`",
        f"Device: `{device}`",
        f"Val accuracy: `{(y_true == y_pred).mean():.4f}`",
        f"Val errors: `{int((y_true != y_pred).sum())}` / `{len(y_true)}`",
        "",
        "## Taxonomy counts (top-30 high-confidence errors)",
        "",
        markdown_table(counts_frame).rstrip(),
        "",
        "## Labels",
        "",
        "- `shortcut` — prediction changes after lab-background masking",
        "- `confused_disease` — same crop, different disease label",
        "- `rare_class` — true class is in the bottom quartile of dataset size",
        "- `bad_image` — dark / bright / low contrast / blurry / tiny leaf",
        "- `other` — none of the above",
        "",
        "## Findings",
        "",
    ]
    for item in findings:
        note_lines.append(f"- {item}")
    note_lines.extend(
        [
            "",
            "## Worst classes by F1",
            "",
            markdown_table(
                worst_classes[
                    ["class_name", "precision", "recall", "f1", "support", "train_count"]
                ]
            ).rstrip(),
            "",
            "## Top confused pairs",
            "",
            markdown_table(
                top_pairs[["true_name", "pred_name", "count", "same_crop"]]
            ).rstrip(),
            "",
        ]
    )
    note_path = table_dir / "error_taxonomy.md"
    note_path.write_text("\n".join(note_lines) + "\n", encoding="utf-8")

    payload = {
        "checkpoint": str(ckpt_path.relative_to(root).as_posix()),
        "device": str(device),
        "val_accuracy": float((y_true == y_pred).mean()),
        "n_val": int(len(y_true)),
        "n_errors": int((y_true != y_pred).sum()),
        "n_sampled": len(error_rows),
        "taxonomy_counts": counts,
        "rare_classes": sorted(rare),
        "findings": findings,
        "errors": error_rows,
    }
    json_path = table_dir / "error_taxonomy.json"
    json_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")

    print(json.dumps({"taxonomy_counts": counts, "findings": findings}, indent=2))
    print(f"Wrote {figure_dir / 'confusion_matrix_val.png'}")
    print(f"Wrote {figure_dir / 'error_taxonomy_grid.png'}")
    print(f"Wrote {per_class_path}")
    print(f"Wrote {pairs_path}")
    print(f"Wrote {errors_path}")
    print(f"Wrote {counts_path}")
    print(f"Wrote {note_path}")
    print(f"Wrote {json_path}")


if __name__ == "__main__":
    main()
