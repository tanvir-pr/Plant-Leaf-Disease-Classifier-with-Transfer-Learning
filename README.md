# Plant Leaf Disease Classifier with Transfer Learning

Classify color photographs of plant leaves into healthy and disease classes.

## Setup

```bash
python -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt
```

Preparing the plant images uses NumPy, pandas, matplotlib, scikit-learn, and PyYAML.

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

## Tests

```bash
python -m pytest -q
```
