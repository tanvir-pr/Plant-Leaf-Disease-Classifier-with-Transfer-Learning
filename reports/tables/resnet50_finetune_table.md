| run | strategy | accuracy | macro_f1 | top3 | top5 | trainable_parameters | head_lr | backbone_lr | best_checkpoint |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| scratch_cnn | full | 0.9001 | 0.8680 | 0.9816 | 0.9933 | nan | nan | nan | runs/scratch_cnn/best.pt |
| resnet50_frozen | frozen | 0.9739 | 0.9662 | 0.9982 | 0.9996 | 77862.0000 | nan | nan | runs/resnet50_frozen/best.pt |
| resnet50_last_block | last_block | 0.9851 | 0.9780 | 0.9994 | 0.9999 | 15042598.0000 | 0.0010 | 0.0001 | runs/resnet50_last_block/best.pt |
| resnet50_full | full | 0.9843 | 0.9790 | 0.9965 | 0.9995 | 23585894.0000 | 0.0010 | 0.0001 | runs/resnet50_full/best.pt |
