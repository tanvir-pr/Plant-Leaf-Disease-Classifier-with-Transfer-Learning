"""Image transforms for plant classification."""

from __future__ import annotations

from torchvision import transforms
from torchvision.transforms import autoaugment

IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)

AUGMENTATION_PRESETS = (
    "none",
    "flip",
    "flip_rotate",
    "color_jitter",
    "randaugment",
)


def build_transforms(
    image_size: int,
    train: bool,
    augmentation: str = "flip",
):
    """Build train/eval transforms. Augmentations apply only when train=True."""
    key = augmentation.lower().replace("-", "_")
    if key not in AUGMENTATION_PRESETS:
        raise ValueError(
            f"Unknown augmentation={augmentation!r}. "
            f"Choose from {AUGMENTATION_PRESETS}."
        )

    eval_tfms = [
        transforms.Resize((image_size, image_size)),
        transforms.ToTensor(),
        transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
    ]
    if not train:
        return transforms.Compose(eval_tfms)

    train_ops: list = [transforms.Resize((image_size, image_size))]
    if key == "none":
        pass
    elif key == "flip":
        train_ops.append(transforms.RandomHorizontalFlip())
    elif key == "flip_rotate":
        train_ops.extend(
            [
                transforms.RandomHorizontalFlip(),
                transforms.RandomRotation(degrees=25),
            ]
        )
    elif key == "color_jitter":
        train_ops.extend(
            [
                transforms.RandomHorizontalFlip(),
                transforms.ColorJitter(
                    brightness=0.3,
                    contrast=0.3,
                    saturation=0.3,
                    hue=0.05,
                ),
            ]
        )
    elif key == "randaugment":
        train_ops.extend(
            [
                transforms.RandomHorizontalFlip(),
                autoaugment.RandAugment(num_ops=2, magnitude=9),
            ]
        )

    train_ops.extend(
        [
            transforms.ToTensor(),
            transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
        ]
    )
    return transforms.Compose(train_ops)


def denormalize(tensor):
    """Undo ImageNet normalisation for plotting."""
    mean = tensor.new_tensor(IMAGENET_MEAN)[:, None, None]
    std = tensor.new_tensor(IMAGENET_STD)[:, None, None]
    return (tensor * std + mean).clamp(0.0, 1.0)
