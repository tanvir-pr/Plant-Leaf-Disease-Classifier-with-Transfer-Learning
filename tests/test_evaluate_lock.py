"""Held-out test lock guard."""

from pathlib import Path

import pytest

from src.evaluate import HeldOutAlreadyEvaluatedError, assert_test_unlocked


def test_assert_test_unlocked_allows_missing(tmp_path: Path):
    assert_test_unlocked(tmp_path / "final_test.json", force=False)


def test_assert_test_unlocked_blocks_existing(tmp_path: Path):
    lock = tmp_path / "final_test.json"
    lock.write_text("{}", encoding="utf-8")
    with pytest.raises(HeldOutAlreadyEvaluatedError):
        assert_test_unlocked(lock, force=False)


def test_assert_test_unlocked_force_overrides(tmp_path: Path):
    lock = tmp_path / "final_test.json"
    lock.write_text("{}", encoding="utf-8")
    assert_test_unlocked(lock, force=True)
