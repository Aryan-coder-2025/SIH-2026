import numpy as np
import pytest

from src.eval.unseen_attack import (
    UnseenAttackError,
    assert_no_category_leakage,
    unseen_attack_split,
)


def _categories():
    return np.array(
        ["BENIGN", "BENIGN", "DoS", "DoS", "Infiltration", "Infiltration", "BruteForce"]
    )


def test_held_out_category_excluded_from_train():
    split = unseen_attack_split(_categories(), held_out_category="Infiltration")

    assert_no_category_leakage(split, _categories())  # must not raise
    assert not np.any(_categories()[split.train_mask] == "Infiltration")


def test_held_out_category_is_the_only_thing_in_held_out_mask():
    categories = _categories()
    split = unseen_attack_split(categories, held_out_category="DoS")

    assert set(categories[split.held_out_mask]) == {"DoS"}


def test_train_includes_benign_and_other_attacks():
    categories = _categories()
    split = unseen_attack_split(categories, held_out_category="Infiltration")

    train_categories = set(categories[split.train_mask])
    assert "BENIGN" in train_categories
    assert "DoS" in train_categories
    assert "BruteForce" in train_categories
    assert "Infiltration" not in train_categories


def test_masks_are_complementary_and_cover_all_rows():
    categories = _categories()
    split = unseen_attack_split(categories, held_out_category="DoS")

    assert np.array_equal(split.train_mask, ~split.held_out_mask)
    assert (split.train_mask | split.held_out_mask).all()


def test_held_out_category_not_present_rejected():
    with pytest.raises(UnseenAttackError):
        unseen_attack_split(_categories(), held_out_category="NotARealCategory")


def test_all_rows_being_held_out_rejected():
    categories = np.array(["DoS", "DoS", "DoS"])
    with pytest.raises(UnseenAttackError):
        unseen_attack_split(categories, held_out_category="DoS")


def test_empty_categories_rejected():
    with pytest.raises(UnseenAttackError):
        unseen_attack_split(np.array([]), held_out_category="DoS")


def test_leakage_detection_catches_a_corrupted_mask():
    categories = _categories()
    split = unseen_attack_split(categories, held_out_category="Infiltration")

    from dataclasses import replace

    # Corrupt: force one Infiltration row into the train mask.
    bad_train_mask = split.train_mask.copy()
    infiltration_idx = np.flatnonzero(categories == "Infiltration")[0]
    bad_train_mask[infiltration_idx] = True
    corrupted = replace(split, train_mask=bad_train_mask)

    with pytest.raises(UnseenAttackError):
        assert_no_category_leakage(corrupted, categories)
