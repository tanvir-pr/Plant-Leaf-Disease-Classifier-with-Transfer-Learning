.PHONY: download-plant eda visualize-batch overfit train-scratch train-resnet-frozen test

download-plant:
	python data/download_plantvillage.py

eda:
	python data/prepare_plant_split.py

visualize-batch:
	python scripts/visualize_batch.py

overfit:
	python -m src.train --overfit-batch

train-scratch:
	python -m src.train

train-resnet-frozen:
	python -m src.train --config configs/resnet50_frozen.yaml

test:
	python -m pytest -q
