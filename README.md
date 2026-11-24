# Plant Leaf Disease Classifier

**Course P11 · Computer Vision · Transfer Learning + Explainability**

Upload a leaf photo → get a disease class, confidence, and a Grad-CAM overlay that shows *where* the model looked.

> Educational demo only — **not** for field diagnosis or agricultural advice.

---

## Table of contents

1. [Overview](#overview)
2. [What we used (tech stack)](#what-we-used-tech-stack)
3. [Dataset](#dataset)
4. [What photos to test with](#what-photos-to-test-with)
5. [Highlights](#highlights)
6. [Results](#results-at-a-glance)
7. [Figures](#figures)
8. [Demo](#demo-gradio)
9. [Setup & reproduce](#setup)
10. [Limitations](#limitations)

---

## Overview

This project trains and audits CNN classifiers on **PlantVillage** color leaf photos. The goal is not only a high lab score, but an honest pipeline:

| Stage | What we did |
| --- | --- |
| **Data** | Leaf-grouped 70 / 15 / 15 split so the same physical leaf never leaks into test |
| **Models** | Scratch CNN → ResNet-50 (freeze / last-block / full) → EfficientNet-B0 & MobileNetV2 |
| **Augmentation** | Controlled study on EfficientNet-B0; **color jitter** won |
| **Explain** | Grad-CAM / Grad-CAM++ / SmoothGrad on correct vs wrong cases |
| **Stress** | Background shortcut test, 30-error taxonomy, field-style domain shift |
| **Lock** | Held-out test scored **once** — no retuning on test |
| **Ship** | Gradio app with Grad-CAM, optional leaf crop, and uncertainty flags |

**Best model:** EfficientNet-B0 + color jitter → locked test **accuracy 0.992 · macro-F1 0.987**.

---

## What we used (tech stack)

| Area | Tools |
| --- | --- |
| **Language** | Python 3 |
| **Deep learning** | PyTorch, torchvision (ResNet-50, EfficientNet-B0, MobileNetV2, custom Scratch CNN) |
| **Training** | YAML configs, TensorBoard, AdamW / discriminative LR, ImageNet-pretrained weights |
| **Data** | Pillow, OpenCV, Albumentations / torchvision transforms, pandas |
| **Metrics** | scikit-learn (accuracy, macro-F1, top-k), custom evaluation lock |
| **Explainability** | pytorch-grad-cam, Captum (Grad-CAM / Grad-CAM++ / SmoothGrad) |
| **Demo** | Gradio (upload → class + confidence + CAM + uncertainty) |
| **Tests** | pytest |
| **Deploy (optional)** | Hugging Face Hub / Space package in `hf_space/` |

Core idea: **transfer learning** on lab leaf photos, then check whether the model looks at the leaf (not the background) and how hard it fails on phone-style images.

---

## Dataset

| Item | Detail |
| --- | --- |
| **Name** | [PlantVillage Dataset](https://github.com/spMohanty/PlantVillage-Dataset) (`raw/color`) |
| **Paper** | Mohanty, Hughes & Salathé, *Using Deep Learning for Image-Based Plant Disease Detection*, Frontiers in Plant Science, 2016 |
| **Images** | **54,305** RGB leaf photos |
| **Classes** | **38** (healthy + disease labels) |
| **Crops** | **14** plant types |
| **Look** | Mostly single leaf, plain / grey lab background, controlled lighting |
| **Split** | **70% train / 15% val / 15% test**, seed **42**, grouped by **leaf id** (no near-duplicate leak) |
| **Counts** | train 38,132 · val 7,939 · test 8,234 |

Images are **not** committed. Download and split:

```bash
python data/download_plantvillage.py
python data/prepare_plant_split.py
```

### Crops covered (14)

Apple · Blueberry · Cherry · Corn (maize) · Grape · Orange · Peach · Bell pepper · Potato · Raspberry · Soybean · Squash · Strawberry · Tomato

### Example classes (38 total)

| Crop | Example labels |
| --- | --- |
| Apple | Apple scab, Black rot, Cedar apple rust, healthy |
| Corn | Gray leaf spot, Common rust, Northern leaf blight, healthy |
| Potato | Early blight, Late blight, healthy |
| Tomato | Bacterial spot, Early/Late blight, Leaf mold, Septoria, Spider mites, Target spot, YLCV, Mosaic virus, healthy |
| Others | Blueberry healthy, Grape black rot / Esca / blight, Orange HLB, Peach bacterial spot, Pepper bacterial spot, Soybean healthy, Squash powdery mildew, Strawberry leaf scorch, … |

Full label list: `data/splits/plant_split.csv` (column `class_name`).

---

## What photos to test with

The model was trained on **PlantVillage-style** images. Demo and domain-shift results depend heavily on photo type.

### Best (high confidence expected)

| Do this | Why |
| --- | --- |
| **One leaf** filling most of the frame | Matches training distribution |
| **Plain background** (paper, desk, wall) | Less background shortcut / clutter |
| **Sharp focus**, leaf flat / facing camera | Disease spots stay visible |
| **Crop from PlantVillage** (tomato, corn, potato, apple, …) | Class exists in the 38 labels |
| Use Gradio preprocess **Crop to leaf** for phone shots | Brings field photos closer to lab style |

### Risky / often wrong (domain shift)

| Avoid or treat as uncertain | Why |
| --- | --- |
| Whole plant / bush / fruit / multiple leaves | Model expects a single leaf |
| Soil, grass, sky, hands, pots in frame | Out-of-distribution background |
| Dark / yellow indoor light, heavy blur, JPEG noise | Lighting ≠ lab set |
| Crops **not** in PlantVillage (mango, rice, rose, …) | No trained class → forced wrong guess |
| Mixed diseases on one leaf | Labels assume one class per image |

### How to test properly

1. **Lab-style check** — take a few images from the held-out idea: clean single leaf of a known class (or use demo_test images). Expect high confidence.
2. **Phone / own leaves** — put 20–30 real photos in `data/own_leaves/` + `labels.csv` (see `data/own_leaves/README.md`). Run `scripts/run_domain_shift.py`.
3. **Gradio** — upload → read Top-5 + uncertainty flag. If Top-1 looks wrong but Top-2/3 show the right crop+disease, treat it as domain shift, not a broken app.
4. **Do not** retune on the locked test split after peeking at scores.

```text
# own_leaves/labels.csv example
relative_path,class_name,source
my_tomato_01.jpg,Tomato___healthy,own
my_apple_02.jpg,Apple___Apple_scab,own
```

---

## Highlights

| | |
| --- | --- |
| **Dataset** | PlantVillage `raw/color` · 54,305 images · 38 classes · 14 crops |
| **Split** | 70 / 15 / 15 · seed 42 · leaf-id grouped |
| **Winner** | `runs/aug_color_jitter/best.pt` (EfficientNet-B0 + color jitter) |
| **Locked test** | Acc **0.992** · macro-F1 **0.987** · top-3/5 **1.000** · *n* = 8,234 |
| **Demo** | `python -m app.main` → http://127.0.0.1:7860 |

---

## Results at a glance

### Backbone comparison (validation)

| Run | Accuracy | Macro-F1 | Top-3 | Top-5 |
| --- | ---: | ---: | ---: | ---: |
| Scratch CNN | 0.900 | 0.868 | 0.982 | 0.993 |
| ResNet-50 frozen | 0.974 | 0.966 | 0.998 | 1.000 |
| ResNet-50 last block | 0.985 | 0.978 | 0.999 | 1.000 |
| ResNet-50 full | 0.984 | 0.979 | 0.996 | 0.999 |
| **EfficientNet-B0** | **0.990** | **0.986** | **0.999** | **1.000** |
| MobileNetV2 | 0.984 | 0.981 | 0.999 | 1.000 |

### Augmentation study (EfficientNet-B0, validation)

| Augmentation | Accuracy | Macro-F1 |
| --- | ---: | ---: |
| **color_jitter** ★ | 0.990 | **0.986** |
| none | 0.990 | 0.986 |
| flip | 0.990 | 0.986 |
| randaugment | 0.989 | 0.986 |
| flip_rotate | 0.987 | 0.981 |

### Locked held-out test

| Split | Accuracy | Macro-F1 | Top-3 | Top-5 | *n* |
| --- | ---: | ---: | ---: | ---: | ---: |
| Val | 0.990 | 0.986 | — | — | 7,939 |
| **Test (locked)** | **0.992** | **0.987** | **1.000** | **1.000** | **8,234** |

Re-running `scripts/run_final_test.py` refuses unless you pass `--force`.

### Audits beyond accuracy

| Check | Finding |
| --- | --- |
| Background shortcut | Mask/crop drops val macro-F1 by ~0.012 → mostly leaf-focused |
| Failure taxonomy (top-30) | 24/30 are same-crop disease confusions |
| Field-style proxies | macro-F1 0.986 → 0.889 (−0.097) |

---

## Figures

<p align="center">
  <img src="reports/figures/plant_train_batch.png" width="720" alt="Training batch sample" />
  <br/><em>Sample training batch — what the model “expects”</em>
</p>

<p align="center">
  <img src="reports/figures/confusion_matrix_val.png" width="720" alt="Validation confusion matrix" />
  <br/><em>Validation confusion matrix</em>
</p>

<p align="center">
  <img src="reports/figures/xai_correct_grid.png" width="48%" alt="Grad-CAM correct" />
  <img src="reports/figures/xai_wrong_grid.png" width="48%" alt="Grad-CAM wrong" />
  <br/><em>Grad-CAM on correct (left) vs wrong (right) predictions</em>
</p>

<p align="center">
  <img src="reports/figures/shortcut_macro_f1.png" width="48%" alt="Shortcut test" />
  <img src="reports/figures/domain_shift_metrics.png" width="48%" alt="Domain shift" />
  <br/><em>Background shortcut · Domain-shift drop on field-style images</em>
</p>

---

## Why this model won

EfficientNet-B0 with light **color jitter** beat the scratch CNN by a large margin and edged ResNet-50 / MobileNetV2 on validation macro-F1 while staying smaller than ResNet-50. On this clean lab set, mild color jitter helped more than aggressive RandAugment or heavy rotation. Last-block ResNet fine-tuning was nearly as strong as full fine-tuning — early ImageNet features were already useful.

---

## What the pipeline taught

- **Freeze vs fine-tune** — a frozen ResNet head already reaches ~0.97 macro-F1; unfreezing the last block closes most of the gap.
- **Grad-CAM ≠ truth** — overlays show gradient focus for the *predicted* class, not biological proof.
- **Lab shortcuts are mild here** — grey backdrop removal barely hurts; outdoor phones hurt a lot more.
- **Errors are intra-crop** — corn blight ↔ gray leaf spot and tomato look-alikes dominate, not random cross-crop mistakes.
- **Lab score ≠ phone score** — domain shift is the honest stress test for a course demo.

---

## Demo (Gradio)

```bash
python -m app.main
```

Open **http://127.0.0.1:7860**.

| Output | Meaning |
| --- | --- |
| Top-1 + confidence | Best guess among 38 classes |
| Top-5 | Alternatives (check these if uncertain) |
| Grad-CAM overlay | Where the model attended |
| Preprocess: crop / mask / none | Helps phone photos look more like lab leaves |
| Uncertainty flag | Low confidence or top-1 ≈ top-2 → do not trust alone |

Temporary public link (laptop must stay on):

```bash
python -m app.main --share
```

Optional Hub weights: `tanvir-pr/plant-leaf-disease`. Always-on free Space usually needs **ZeroGPU** (CPU Gradio Spaces often need PRO).

---

## Setup

```bash
python -m venv .venv
.venv\Scripts\activate          # Windows
# source .venv/bin/activate     # macOS / Linux

pip install -r requirements.txt
```

### How to reproduce

```bash
# Data
python data/download_plantvillage.py
python data/prepare_plant_split.py

# Train / compare
python -m src.train --config configs/resnet50_frozen.yaml
python -m src.train --config configs/efficientnet_b0.yaml
python -m src.train --config configs/aug_color_jitter.yaml

# Explainability & audits
python scripts/generate_xai_grids.py
python scripts/run_shortcut_test.py
python scripts/run_error_analysis.py
python scripts/prepare_own_leaves_proxy.py
python scripts/run_domain_shift.py

# Locked test (once)
python scripts/run_final_test.py

# Demo + unit tests
python -m app.main
python -m pytest -q
```

---

## Project layout

```text
Project2/
├── app/                  # Gradio demo + inference (crop, Grad-CAM, uncertainty)
├── configs/              # YAML training & augmentation configs
├── data/                 # download, split, own_leaves helpers
├── hf_space/             # Hugging Face Space package
├── reports/
│   ├── figures/          # confusion, Grad-CAM, shortcut, domain shift
│   └── tables/           # metrics, locks, taxonomies
├── runs/                 # checkpoints (best.pt per experiment)
├── scripts/              # XAI, shortcut, errors, domain shift, final test
├── src/                  # dataset, models, train, metrics, xai, …
├── tests/                # pytest
└── README.md
```

---

## Limitations

- Trained on PlantVillage **lab** photos — phones, soil, new crops, and mixed diseases are OOD.
- Own-leaf numbers use synthetic field-style proxies unless you add real labeled photos.
- Grad-CAM can highlight healthy tissue or edges when the model is wrong.
- Labels assume a **single** disease class per image.
- Not a farm decision tool and not a medical device.

---

## License

See [LICENSE](LICENSE). PlantVillage data remains under its upstream terms.
