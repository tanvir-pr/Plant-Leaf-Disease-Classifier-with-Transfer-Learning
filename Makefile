.PHONY: download-plant eda test

download-plant:
	python data/download_plantvillage.py

eda:
	python data/prepare_plant_split.py

test:
	python -m pytest -q
