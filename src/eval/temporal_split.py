from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence, Tuple, Union

import numpy as np

ArrayLike = Union[Sequence[float], np.ndarray]


class TemporalSplitError(ValueError):
    """Raised when a temporal split configuration or input is invalid."""


@dataclass(frozen=True)
class TemporalSplit:
    """
    Index sets for a TRAIN -> VALIDATION -> FINAL TEST temporal split.

    Every index array refers to positions in the `timestamps` array the
    split was built from. No row is ever moved or copied here -- callers
    index their own X/y/timestamps arrays with these to avoid duplicating
    the underlying data.
    """

    train_idx: np.ndarray
    validation_idx: np.ndarray
    test_idx: np.ndarray
    train_end: float
    validation_end: float


def temporal_train_validation_test_split(
    timestamps: ArrayLike,
    train_end: float,
    validation_end: float,
) -> TemporalSplit:
    """
    Split samples into TRAIN / VALIDATION / FINAL TEST by a global
    timestamp cutoff -- never by shuffling.

        TRAIN:      timestamp <  train_end
        VALIDATION: train_end <= timestamp < validation_end
        FINAL TEST: timestamp >= validation_end

    `train_end` and `validation_end` are a single global cutoff applied
    across every entity (source_host) -- not a per-host split, and not a
    fraction. This matches the project's requirement that training data
    represent strictly earlier time than validation, which in turn is
    strictly earlier than the final test period, with no leakage.

    The row order of `timestamps` is preserved inside each returned
    index array (this function only filters by value, it never
    reorders), so downstream code that expects temporally-ordered
    batches is not disturbed.

    Raises TemporalSplitError if:
      - `timestamps` is empty
      - `validation_end <= train_end`
      - any resulting split (train/validation/test) is empty
    """
    ts = np.asarray(timestamps, dtype=float)

    if ts.ndim != 1:
        raise TemporalSplitError("'timestamps' must be a one-dimensional sequence.")

    if ts.size == 0:
        raise TemporalSplitError("'timestamps' is empty. At least one sample is required.")

    if not np.all(np.isfinite(ts)):
        raise TemporalSplitError("'timestamps' contains NaN or infinite values.")

    if not np.isfinite(train_end) or not np.isfinite(validation_end):
        raise TemporalSplitError("'train_end' and 'validation_end' must be finite numbers.")

    if validation_end <= train_end:
        raise TemporalSplitError(
            "'validation_end' must be strictly after 'train_end' "
            f"(got train_end={train_end}, validation_end={validation_end})."
        )

    train_idx = np.flatnonzero(ts < train_end)
    validation_idx = np.flatnonzero((ts >= train_end) & (ts < validation_end))
    test_idx = np.flatnonzero(ts >= validation_end)

    empty_splits = [
        name
        for name, idx in (
            ("train", train_idx),
            ("validation", validation_idx),
            ("test", test_idx),
        )
        if idx.size == 0
    ]
    if empty_splits:
        raise TemporalSplitError(
            f"The following split(s) would be empty with train_end={train_end}, "
            f"validation_end={validation_end}: {empty_splits}. Adjust the cutoffs."
        )

    return TemporalSplit(
        train_idx=train_idx,
        validation_idx=validation_idx,
        test_idx=test_idx,
        train_end=float(train_end),
        validation_end=float(validation_end),
    )


def apply_split(
    split: TemporalSplit, *arrays: ArrayLike
) -> Tuple[np.ndarray, ...]:
    """
    Index one or more parallel arrays (e.g. X, y) by `split`, returning
    (train, validation, test) for each input array in order:

        X_train, X_val, X_test, y_train, y_val, y_test = apply_split(split, X, y)
    """
    result: list = []
    for array in arrays:
        arr = np.asarray(array)
        if arr.shape[0] != split.train_idx.shape[0] + split.validation_idx.shape[0] + split.test_idx.shape[0]:
            raise TemporalSplitError(
                "Array length does not match the number of timestamps the split was built from "
                f"(got {arr.shape[0]} rows)."
            )
        result.extend([arr[split.train_idx], arr[split.validation_idx], arr[split.test_idx]])
    return tuple(result)


def assert_no_temporal_leakage(split: TemporalSplit, timestamps: ArrayLike) -> None:
    """
    Verify, from the actual timestamp values, that:
      - every train timestamp < every validation timestamp
      - every validation timestamp < every test timestamp
      - no index appears in more than one split

    Intended for tests/QA, not the hot path. Raises TemporalSplitError on
    any violation, naming the offending boundary.
    """
    ts = np.asarray(timestamps, dtype=float)

    train_ts = ts[split.train_idx]
    val_ts = ts[split.validation_idx]
    test_ts = ts[split.test_idx]

    if train_ts.size and val_ts.size and train_ts.max() >= val_ts.min():
        raise TemporalSplitError(
            "Leakage: a train timestamp is not strictly before every validation timestamp "
            f"(train max={train_ts.max()}, validation min={val_ts.min()})."
        )

    if val_ts.size and test_ts.size and val_ts.max() >= test_ts.min():
        raise TemporalSplitError(
            "Leakage: a validation timestamp is not strictly before every test timestamp "
            f"(validation max={val_ts.max()}, test min={test_ts.min()})."
        )

    all_idx = np.concatenate([split.train_idx, split.validation_idx, split.test_idx])
    if np.unique(all_idx).size != all_idx.size:
        raise TemporalSplitError("Leakage: the same sample index appears in more than one split.")
