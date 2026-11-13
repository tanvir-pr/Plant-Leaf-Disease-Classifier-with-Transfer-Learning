"""Checks for named augmentation presets."""

from src.transforms import AUGMENTATION_PRESETS, build_transforms


def test_all_augmentation_presets_build():
    for name in AUGMENTATION_PRESETS:
        train_tfm = build_transforms(224, train=True, augmentation=name)
        eval_tfm = build_transforms(224, train=False, augmentation=name)
        assert train_tfm is not None
        assert eval_tfm is not None


def test_unknown_augmentation_raises():
    try:
        build_transforms(224, train=True, augmentation="cutmix")
        raised = False
    except ValueError:
        raised = True
    assert raised
