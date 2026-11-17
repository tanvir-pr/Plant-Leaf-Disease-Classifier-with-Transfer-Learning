"""Compare lab validation metrics against own-leaf / field-style photos."""

from __future__ import annotations

import json
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from PIL import Image
from torch.utils.data import DataLoader, Dataset

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.checkpoint import load_plant_checkpoint
from src.dataset import PlantSplitDataset, build_class_mapping, load_split_frame
from src.domain_shift import load_own_leaves_frame
from src.evaluate import evaluate_loader
from src.metrics import classification_metrics
from src.transforms import build_transforms
from src.utils import load_config, project_root, set_seed


class OwnLeavesDataset(Dataset):
    def __init__(
        self,
        frame: pd.DataFrame,
        image_root: Path,
        class_to_idx: dict[str, int],
        transform,
    ) -> None:
        self.frame = frame.reset_index(drop=True)
        self.image_root = Path(image_root)
        self.class_to_idx = class_to_idx
        self.transform = transform
        unknown = sorted(
            {
                str(name)
                for name in self.frame["class_name"].tolist()
                if str(name) and str(name) not in class_to_idx
            }
        )
        if unknown:
            raise ValueError(f"Unknown class labels in own leaves: {unknown}")

    def __len__(self) -> int:
        return len(self.frame)

    def __getitem__(self, index: int):
        row = self.frame.iloc[index]
        path = self.image_root / row["relative_path"]
        image = Image.open(path).convert("RGB")
        label = self.class_to_idx[str(row["class_name"])]
        if self.transform is not None:
            image = self.transform(image)
        return image, label


def short_label(name: str, max_len: int = 26) -> str:
    text = name.replace("___", " / ")
    return text if len(text) <= max_len else text[: max_len - 1] + "…"


def save_prediction_grid(
    frame: pd.DataFrame,
    image_root: Path,
    out_path: Path,
    cols: int = 4,
) -> None:
    n = len(frame)
    if n == 0:
        return
    rows_n = int(np.ceil(n / cols))
    fig, axes = plt.subplots(rows_n, cols, figsize=(cols * 2.6, rows_n * 2.9))
    axes = np.atleast_1d(axes).ravel()
    for ax in axes:
        ax.axis("off")
    for i, (_, row) in enumerate(frame.iterrows()):
        image = Image.open(image_root / row["relative_path"]).convert("RGB")
        axes[i].imshow(image)
        ok = "OK" if row["correct"] else "ERR"
        axes[i].set_title(
            f"{ok} p={row['confidence']:.2f}\n"
            f"T: {short_label(str(row['class_name']))}\n"
            f"P: {short_label(str(row['pred_name']))}",
            fontsize=6,
            color="#1a7f37" if row["correct"] else "#b00020",
        )
        axes[i].axis("off")
    fig.suptitle("Own-leaf / field-style domain-shift predictions", fontsize=12)
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


def ensure_proxy(root: Path) -> None:
    own_dir = root / "data" / "own_leaves"
    labels = own_dir / "labels.csv"
    from src.domain_shift import list_image_files

    has_labels = labels.is_file() and len(pd.read_csv(labels)) >= 5
    has_images = len(list_image_files(own_dir)) >= 5
    if has_labels and has_images:
        return
    import subprocess

    script = root / "scripts" / "prepare_own_leaves_proxy.py"
    subprocess.check_call([sys.executable, str(script)], cwd=str(root))


def main() -> None:
    set_seed(42)
    root = project_root()
    ensure_proxy(root)

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
    full_frame = pd.read_csv(split_csv)
    _, class_to_idx = build_class_mapping(full_frame)
    transform = build_transforms(
        int(plant_cfg["image"]["size"]),
        train=False,
        augmentation="none",
    )

    # Lab validation baseline (same checkpoint).
    val_frame = load_split_frame(split_csv, "val")
    val_ds = PlantSplitDataset(
        val_frame,
        dataset_root=dataset_root,
        class_to_idx=class_to_idx,
        transform=transform,
    )
    val_loader = DataLoader(val_ds, batch_size=32, shuffle=False, num_workers=0)
    lab_metrics = evaluate_loader(model, val_loader, device, len(class_names))

    own_dir = root / "data" / "own_leaves"
    own_frame = load_own_leaves_frame(own_dir)
    if own_frame.empty:
        raise SystemExit(f"No images found under {own_dir}")
    if own_frame["class_name"].astype(str).str.len().eq(0).any():
        raise SystemExit(
            "Own-leaf images need class labels in data/own_leaves/labels.csv"
        )

    own_ds = OwnLeavesDataset(own_frame, own_dir, class_to_idx, transform)
    own_loader = DataLoader(own_ds, batch_size=16, shuffle=False, num_workers=0)

    y_true: list[int] = []
    y_pred: list[int] = []
    y_prob: list[float] = []
    model.eval()
    with torch.no_grad():
        for images, labels in own_loader:
            images = images.to(device)
            probs = torch.softmax(model(images), dim=1)
            conf, pred = probs.max(dim=1)
            y_true.extend(labels.tolist())
            y_pred.extend(pred.cpu().tolist())
            y_prob.extend(conf.cpu().tolist())

    true_arr = np.asarray(y_true, dtype=np.int64)
    pred_arr = np.asarray(y_pred, dtype=np.int64)
    own_metrics = classification_metrics(
        true_arr,
        pred_arr,
        y_prob=None,
        labels=list(range(len(class_names))),
    )

    detail_rows = []
    own_frame = own_frame.reset_index(drop=True)
    for i, row in own_frame.iterrows():
        true_name = str(row["class_name"])
        pred_name = class_names[int(pred_arr[i])]
        detail_rows.append(
            {
                "relative_path": row["relative_path"],
                "class_name": true_name,
                "pred_name": pred_name,
                "confidence": float(y_prob[i]),
                "correct": true_name == pred_name,
                "source": row.get("source", "own"),
            }
        )
    details = pd.DataFrame(detail_rows)

    compare = pd.DataFrame(
        [
            {
                "domain": "lab_val",
                "n_images": len(val_frame),
                "accuracy": float(lab_metrics["accuracy"]),
                "macro_f1": float(lab_metrics["macro_f1"]),
            },
            {
                "domain": "own_leaves",
                "n_images": len(own_frame),
                "accuracy": float(own_metrics["accuracy"]),
                "macro_f1": float(own_metrics["macro_f1"]),
            },
        ]
    )
    compare["accuracy_drop"] = (
        float(lab_metrics["accuracy"]) - compare["accuracy"]
    )
    compare["macro_f1_drop"] = (
        float(lab_metrics["macro_f1"]) - compare["macro_f1"]
    )
    # Lab row drop is zero by definition for display clarity.
    compare.loc[compare["domain"] == "lab_val", ["accuracy_drop", "macro_f1_drop"]] = 0.0

    source = "proxy_field"
    if "source" in own_frame.columns:
        sources = sorted(own_frame["source"].astype(str).unique())
        source = ",".join(sources)

    drop_f1 = float(lab_metrics["macro_f1"] - own_metrics["macro_f1"])
    drop_acc = float(lab_metrics["accuracy"] - own_metrics["accuracy"])
    finding = (
        f"Field-style / own-leaf macro-F1 drops by {drop_f1:.3f} "
        f"(acc drop {drop_acc:.3f}) versus lab validation. "
        "Lighting, background, and phone artifacts break the lab-trained model."
        if drop_f1 > 0.05
        else (
            f"Own-leaf / field-style macro-F1 drop is only {drop_f1:.3f}. "
            "Domain shift is mild on this sample."
        )
    )
    if "proxy_field" in source:
        finding += (
            " Images used here are synthetic field-style proxies from train leaves; "
            "replace data/own_leaves/ with real phone photos for the final write-up."
        )

    table_dir = root / "reports" / "tables"
    figure_dir = root / "reports" / "figures"
    table_dir.mkdir(parents=True, exist_ok=True)
    figure_dir.mkdir(parents=True, exist_ok=True)

    compare_path = table_dir / "domain_shift.csv"
    details_path = table_dir / "domain_shift_predictions.csv"
    md_path = table_dir / "domain_shift.md"
    json_path = table_dir / "domain_shift.json"
    compare.to_csv(compare_path, index=False)
    details.to_csv(details_path, index=False)

    # Bar chart.
    fig, ax = plt.subplots(figsize=(6.5, 4))
    x = np.arange(2)
    width = 0.35
    ax.bar(x - width / 2, compare["accuracy"], width, label="accuracy", color="#1b4f72")
    ax.bar(x + width / 2, compare["macro_f1"], width, label="macro-F1", color="#148f77")
    ax.set_xticks(x)
    ax.set_xticklabels(compare["domain"])
    ax.set_ylim(0, 1.05)
    ax.set_ylabel("score")
    ax.set_title("Lab validation vs own-leaf / field-style")
    ax.legend()
    for i, row in compare.iterrows():
        ax.text(i - width / 2, row["accuracy"] + 0.02, f"{row['accuracy']:.2f}", ha="center", fontsize=8)
        ax.text(i + width / 2, row["macro_f1"] + 0.02, f"{row['macro_f1']:.2f}", ha="center", fontsize=8)
    fig.tight_layout()
    chart_path = figure_dir / "domain_shift_metrics.png"
    fig.savefig(chart_path, dpi=150)
    plt.close(fig)

    save_prediction_grid(
        details,
        own_dir,
        figure_dir / "domain_shift_grid.png",
        cols=4,
    )

    md = [
        "# Domain-shift evaluation",
        "",
        f"Checkpoint: `{ckpt_path.relative_to(root).as_posix()}`",
        f"Device: `{device}`",
        f"Own-leaf source: `{source}`",
        "",
        "## Lab vs own leaves",
        "",
        markdown_table(compare).rstrip(),
        "",
        "## Finding",
        "",
        finding,
        "",
        f"Correct: `{int(details['correct'].sum())}` / `{len(details)}`",
        "",
    ]
    md_path.write_text("\n".join(md) + "\n", encoding="utf-8")

    payload = {
        "checkpoint": str(ckpt_path.relative_to(root).as_posix()),
        "device": str(device),
        "source": source,
        "lab_metrics": {
            "accuracy": float(lab_metrics["accuracy"]),
            "macro_f1": float(lab_metrics["macro_f1"]),
            "n_images": int(len(val_frame)),
        },
        "own_metrics": {
            "accuracy": float(own_metrics["accuracy"]),
            "macro_f1": float(own_metrics["macro_f1"]),
            "n_images": int(len(own_frame)),
        },
        "accuracy_drop": drop_acc,
        "macro_f1_drop": drop_f1,
        "finding": finding,
    }
    json_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")

    print(json.dumps(payload, indent=2))
    print(f"Wrote {compare_path}")
    print(f"Wrote {details_path}")
    print(f"Wrote {md_path}")
    print(f"Wrote {json_path}")
    print(f"Wrote {chart_path}")
    print(f"Wrote {figure_dir / 'domain_shift_grid.png'}")


if __name__ == "__main__":
    main()
