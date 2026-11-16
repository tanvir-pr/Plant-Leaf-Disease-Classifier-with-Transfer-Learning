# Validation failure taxonomy

Checkpoint: `runs/aug_color_jitter/best.pt`
Device: `cuda`
Val accuracy: `0.9895`
Val errors: `83` / `7939`

## Taxonomy counts (top-30 high-confidence errors)

| taxonomy | count |
| --- | --- |
| shortcut | 1 |
| confused_disease | 24 |
| rare_class | 0 |
| bad_image | 0 |
| other | 5 |

## Labels

- `shortcut` — prediction changes after lab-background masking
- `confused_disease` — same crop, different disease label
- `rare_class` — true class is in the bottom quartile of dataset size
- `bad_image` — dark / bright / low contrast / blurry / tiny leaf
- `other` — none of the above

## Findings

- Largest confusion: Corn_(maize)___Northern_Leaf_Blight → Corn_(maize)___Cercospora_leaf_spot Gray_leaf_spot (8 val images).
- 78% of all off-diagonal confusions stay inside the same crop.
- Among the 30 highest-confidence errors, the dominant taxonomy label is `confused_disease` (24).

## Worst classes by F1

| class_name | precision | recall | f1 | support | train_count |
| --- | --- | --- | --- | --- | --- |
| Corn_(maize)___Cercospora_leaf_spot Gray_leaf_spot | 0.8961 | 0.8961 | 0.8961 | 77 | 513 |
| Corn_(maize)___Northern_Leaf_Blight | 0.9549 | 0.9270 | 0.9407 | 137 | 985 |
| Tomato___Early_blight | 0.9583 | 0.9324 | 0.9452 | 148 | 1000 |
| Tomato___Target_Spot | 0.9799 | 0.9198 | 0.9489 | 212 | 1404 |
| Apple___Apple_scab | 0.9770 | 0.9551 | 0.9659 | 89 | 630 |
| Tomato___Septoria_leaf_spot | 0.9549 | 0.9790 | 0.9668 | 238 | 1771 |
| Potato___Late_blight | 0.9613 | 0.9803 | 0.9707 | 152 | 1000 |
| Tomato___Late_blight | 0.9812 | 0.9720 | 0.9766 | 322 | 1909 |
| Potato___healthy | 0.9600 | 1.0000 | 0.9796 | 24 | 152 |
| Tomato___Bacterial_spot | 0.9726 | 1.0000 | 0.9861 | 320 | 2127 |

## Top confused pairs

| true_name | pred_name | count | same_crop |
| --- | --- | --- | --- |
| Corn_(maize)___Northern_Leaf_Blight | Corn_(maize)___Cercospora_leaf_spot Gray_leaf_spot | 8 | True |
| Tomato___Target_Spot | Tomato___Septoria_leaf_spot | 7 | True |
| Corn_(maize)___Cercospora_leaf_spot Gray_leaf_spot | Corn_(maize)___Northern_Leaf_Blight | 6 | True |
| Tomato___Late_blight | Potato___Late_blight | 6 | False |
| Tomato___Septoria_leaf_spot | Tomato___Bacterial_spot | 5 | True |
| Apple___Apple_scab | Apple___healthy | 4 | True |
| Tomato___Early_blight | Tomato___Late_blight | 4 | True |
| Tomato___Target_Spot | Tomato___Early_blight | 4 | True |
| Grape___Esca_(Black_Measles) | Grape___Black_rot | 3 | True |
| Pepper,_bell___Bacterial_spot | Pepper,_bell___healthy | 3 | True |
| Tomato___Spider_mites Two-spotted_spider_mite | Tomato___Target_Spot | 3 | True |
| Tomato___Target_Spot | Tomato___Bacterial_spot | 3 | True |
| Tomato___Target_Spot | Tomato___Spider_mites Two-spotted_spider_mite | 3 | True |
| Corn_(maize)___Cercospora_leaf_spot Gray_leaf_spot | Corn_(maize)___Common_rust_ | 2 | True |
| Corn_(maize)___Northern_Leaf_Blight | Corn_(maize)___Common_rust_ | 2 | True |

