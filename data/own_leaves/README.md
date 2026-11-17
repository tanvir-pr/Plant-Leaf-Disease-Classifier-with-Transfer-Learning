Leaf photographs kept separate from the PlantVillage folders. Training does not read this directory.

Put 20–30 real phone photos here for the domain-shift test, plus a `labels.csv`:

```text
relative_path,class_name,source
my_tomato_01.jpg,Tomato___healthy,own
my_apple_02.jpg,Apple___Apple_scab,own
```

Class names must match PlantVillage labels (see `data/splits/plant_split.csv`).

If this folder is empty, `scripts/prepare_own_leaves_proxy.py` builds synthetic field-style proxies from train leaves so the Day-10 comparison can still run. Replace those proxies with real photos before the final write-up.
