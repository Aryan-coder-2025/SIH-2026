import numpy as np
import pytest

from src.eval.temporal_split import (
    TemporalSplitError,
    apply_split,
    assert_no_temporal_leakage,
    temporal_train_validation_test_split,
)


def test_split_produces_correct_time_ordering():
    # timestamps deliberately shuffled to prove no reliance on row order
    timestamps = [50, 10, 90, 30, 70, 20, 80, 40, 60]

    split = temporal_train_validation_test_split(timestamps, train_end=40, validation_end=70)

    ts = np.asarray(timestamps)
    assert np.all(ts[split.train_idx] < 40)
    assert np.all((ts[split.validation_idx] >= 40) & (ts[split.validation_idx] < 70))
    assert np.all(ts[split.test_idx] >= 70)


def test_training_earlier_than_validation_earlier_than_test():
    timestamps = list(range(100))
    split = temporal_train_validation_test_split(timestamps, train_end=40, validation_end=70)
    assert_no_temporal_leakage(split, timestamps)  # must not raise


def test_no_overlap_between_splits():
    timestamps = list(range(50))
    split = temporal_train_validation_test_split(timestamps, train_end=20, validation_end=35)

    all_idx = np.concatenate([split.train_idx, split.validation_idx, split.test_idx])
    assert len(all_idx) == len(set(all_idx.tolist()))
    assert len(all_idx) == len(timestamps)


def test_leakage_detected_if_split_indices_are_corrupted():
    timestamps = list(range(30))
    split = temporal_train_validation_test_split(timestamps, train_end=10, validation_end=20)

    # Manually corrupt: merge one validation index into train to simulate a bug.
    from dataclasses import replace

    corrupted = replace(split, train_idx=np.append(split.train_idx, split.validation_idx[0]))
    with pytest.raises(TemporalSplitError):
        assert_no_temporal_leakage(corrupted, timestamps)


def test_no_shuffle_row_order_preserved_within_each_split():
    timestamps = [5, 1, 3, 2, 4]  # unsorted on purpose
    split = temporal_train_validation_test_split(timestamps, train_end=3, validation_end=5)

    # train_idx must preserve the ORIGINAL positions of qualifying rows,
    # in their original order -- not resorted.
    ts = np.asarray(timestamps)
    assert list(ts[split.train_idx]) == [1, 2]  # positions 1, 3 in original order


def test_empty_timestamps_rejected():
    with pytest.raises(TemporalSplitError):
        temporal_train_validation_test_split([], train_end=1, validation_end=2)


def test_validation_end_before_train_end_rejected():
    with pytest.raises(TemporalSplitError):
        temporal_train_validation_test_split(list(range(10)), train_end=8, validation_end=3)


def test_empty_resulting_split_rejected():
    timestamps = list(range(10))  # all in [0, 9]
    with pytest.raises(TemporalSplitError):
        # validation_end far beyond all data -> empty test split
        temporal_train_validation_test_split(timestamps, train_end=5, validation_end=1000)


def test_apply_split_indexes_parallel_arrays_consistently():
    timestamps = list(range(10))
    X = np.arange(10).reshape(-1, 1) * 10
    y = np.array([0, 1] * 5)

    split = temporal_train_validation_test_split(timestamps, train_end=4, validation_end=7)
    X_train, X_val, X_test, y_train, y_val, y_test = apply_split(split, X, y)

    assert X_train.shape[0] == split.train_idx.shape[0]
    assert y_test.shape[0] == split.test_idx.shape[0]
    # values line up correctly between X and y for the same rows
    assert list(X_train.flatten()) == list(np.arange(10)[split.train_idx] * 10)


def test_apply_split_rejects_mismatched_array_length():
    timestamps = list(range(10))
    split = temporal_train_validation_test_split(timestamps, train_end=4, validation_end=7)

    wrong_length_array = np.arange(5)
    with pytest.raises(TemporalSplitError):
        apply_split(split, wrong_length_array)


def test_no_random_shuffle_is_used_split_is_deterministic():
    timestamps = list(range(20))
    split_a = temporal_train_validation_test_split(timestamps, train_end=8, validation_end=14)
    split_b = temporal_train_validation_test_split(timestamps, train_end=8, validation_end=14)

    assert np.array_equal(split_a.train_idx, split_b.train_idx)
    assert np.array_equal(split_a.validation_idx, split_b.validation_idx)
    assert np.array_equal(split_a.test_idx, split_b.test_idx)
