"""Leaf-grouped split keeps one physical leaf on one side of the split."""

from pathlib import Path

import pandas as pd
import pytest

from src.splits import image_identifier, index_color_images, leaf_id_for, stratified_group_split

RATIOS = {"train": 0.70, "val": 0.15, "test": 0.15}


def test_image_identifier_strips_copy_and_extension():
    assert image_identifier("Leaf12copy.JPG") == "leaf12"
    assert image_identifier("Apple___healthy___photo.PNG") == "photo"


def test_leaf_id_picks_the_suggestion_for_this_class():
    leaf_map = {"img1": ["Apple___healthy:::1", "Tomato___healthy:::9"]}
    assert leaf_id_for("img1.JPG", "Tomato___healthy", leaf_map) == "Tomato___healthy:::9"
    assert leaf_id_for("missing.JPG", "Tomato___healthy", leaf_map) == "fallback_missing"


def test_photo_number_matches_a_renamed_lab_code():
    from src.splits import build_photo_number_index, resolve_leaf_id

    leaf_map = {"com.g_fl 8360": ["Tomato___Target_Spot:::255.0"]}
    number_index = build_photo_number_index(leaf_map)
    filename = "002213fb-b620-4593-b9ac-6a6cc119b100___Com.G_TgS_FL 8360.JPG"
    assert (
        resolve_leaf_id(filename, "Tomato___Target_Spot", leaf_map, number_index)
        == "Tomato___Target_Spot:::255.0"
    )


def test_ambiguous_photo_number_stays_a_fallback():
    from src.splits import build_photo_number_index, resolve_leaf_id

    leaf_map = {
        "code_a 10": ["Tomato___healthy:::1"],
        "code_b 10": ["Tomato___healthy:::2"],
    }
    number_index = build_photo_number_index(leaf_map)
    assert number_index == {}
    assert resolve_leaf_id("shot___Other 10.JPG", "Tomato___healthy", leaf_map, number_index).startswith(
        "fallback_"
    )


def _write_image(directory: Path, name: str) -> None:
    directory.mkdir(parents=True, exist_ok=True)
    (directory / name).write_bytes(b"\xff\xd8\xff")


def _tiny_dataset(root: Path) -> pd.DataFrame:
    classes = ("Apple___healthy", "Tomato___Late_blight")
    leaf_map = {}
    for class_name in classes:
        for leaf_number in range(12):
            for copy_index in range(2):
                filename = f"leaf{leaf_number}" if copy_index == 0 else f"leaf{leaf_number}copy"
                _write_image(root / "raw" / "color" / class_name, f"{filename}.JPG")
            leaf_map[f"leaf{leaf_number}"] = [f"{class_name}:::{leaf_number}"]
    return index_color_images(root / "raw" / "color", leaf_map, root)


def test_group_split_has_no_leaf_leakage_and_is_repeatable(tmp_path: Path):
    frame = _tiny_dataset(tmp_path)
    first = stratified_group_split(frame, RATIOS, seed=42)
    second = stratified_group_split(frame, RATIOS, seed=42)
    pd.testing.assert_frame_equal(first, second)

    split_per_group = first.groupby("group_id")["split"].nunique()
    assert (split_per_group == 1).all()
    assert set(first["split"]) == {"train", "val", "test"}
    for class_name, class_frame in first.groupby("class_name"):
        assert set(class_frame["split"]) == {"train", "val", "test"}, class_name

    group_splits = first.groupby("group_id")["split"].first()
    train_share = (group_splits == "train").mean()
    assert train_share == pytest.approx(0.70, abs=0.15)


def test_split_rejects_ratios_that_do_not_sum_to_one():
    frame = pd.DataFrame(
        {
            "relative_path": ["a.jpg", "b.jpg", "c.jpg"],
            "class_name": ["one", "one", "one"],
            "group_id": ["g1", "g2", "g3"],
        }
    )
    with pytest.raises(ValueError, match="sum to 1"):
        stratified_group_split(frame, {"train": 0.5, "val": 0.5, "test": 0.5}, seed=1)
