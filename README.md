# Plant Leaf Disease Classifier with Transfer Learning

Classify color photographs of plant leaves into healthy and disease classes.

## Setup

```bash
python -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt
```

Preparing the plant images uses NumPy, pandas, matplotlib, scikit-learn, and PyYAML. Training uses PyTorch and torchvision.

## PlantVillage

Source: [spMohanty/PlantVillage-Dataset](https://github.com/spMohanty/PlantVillage-Dataset). 54,305 color images, 38 classes, 14 crops. Cite Mohanty, Hughes, and Salathé, *Using Deep Learning for Image-Based Plant Disease Detection*, Frontiers in Plant Science, 2016.

Images are not committed. The download checks out only `raw/color` plus the leaf-id map.

```bash
python data/download_plantvillage.py
python data/prepare_plant_split.py
```

Outputs:

- `data/splits/plant_split.csv` — path, class, leaf id, split
- `data/splits/plant_split_summary.json`
- `reports/figures/plant_class_distribution.png`
- `reports/figures/plant_split_balance.png`

The split is 70% train, 15% val, 15% test, seed 42. Photos of the same physical leaf stay in one split. Groups are stratified by disease class, so a near-duplicate cannot leak into test. The test split is held out for a single evaluation.

The published leaf map covers 31 of the 38 classes. When a file uses a different lab code than the map but the same photo number, that number is used if it belongs to exactly one leaf of that class. Corn, squash, healthy grape, and tomato mosaic virus have no leaf ids in the map, so those photos are grouped by their own file id. A `copy` file stays with the original.

## Scratch CNN baseline

Inspect one training batch, overfit that batch as a sanity check, then train the small CNN from random weights. Metrics are reported on the validation split only.

```bash
python scripts/visualize_batch.py
python -m src.train --overfit-batch
python -m src.train
```

Outputs:

- `reports/figures/plant_train_batch.png`
- `runs/scratch_cnn/overfit.json`
- `runs/scratch_cnn/best.pt`
- `runs/scratch_cnn/summary.json`

## Frozen ResNet-50

ImageNet-pretrained ResNet-50 with a frozen backbone and a new linear head. Features are extracted once, then the head is trained on the validation split metrics only.

```bash
python -m src.train --config configs/resnet50_frozen.yaml
```

Outputs:

- `runs/resnet50_frozen/train_features.pt`
- `runs/resnet50_frozen/val_features.pt`
- `runs/resnet50_frozen/best.pt`
- `runs/resnet50_frozen/summary.json`

## Tests

```bash
python -m pytest -q
```
