# Domain-shift evaluation

Checkpoint: `runs/aug_color_jitter/best.pt`
Device: `cuda`
Own-leaf source: `proxy_field`

## Lab vs own leaves

| domain | n_images | accuracy | macro_f1 | accuracy_drop | macro_f1_drop |
| --- | --- | --- | --- | --- | --- |
| lab_val | 7939 | 0.9895 | 0.9861 | 0.0000 | 0.0000 |
| own_leaves | 24 | 0.9167 | 0.8889 | 0.0729 | 0.0972 |

## Finding

Field-style / own-leaf macro-F1 drops by 0.097 (acc drop 0.073) versus lab validation. Lighting, background, and phone artifacts break the lab-trained model. Images used here are synthetic field-style proxies from train leaves; replace data/own_leaves/ with real phone photos for the final write-up.

Correct: `22` / `24`

