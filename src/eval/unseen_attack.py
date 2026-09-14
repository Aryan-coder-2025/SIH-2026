from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence, Union

import numpy as np

ArrayLike = Union[Sequence[str], np.ndarray]


class UnseenAttackError(ValueError):
    """Raised when an unseen-attack split configuration or input is invalid."""


@dataclass(frozen=True)
class UnseenAttackSplit:
    """
    Boolean masks separating rows by attack category for an unseen-attack
    generalization protocol:

        TRAIN: every category EXCEPT `held_out_category`
                (includes benign rows and every other attack category)
        HELD-OUT TEST: only `held_out_category`

    IMPORTANT SCIENTIFIC FRAMING: a model scoring well on the held-out
    test set demonstrates "future-risk detection/generalization on an
    attack category not seen during training." It does NOT demonstrate
    "unseen attack classification" -- the model was never asked to name
    the category, only to flag elevated risk. Report results using the
    former phrasing; the latter is a stronger, unsupported claim (see
    docs/07_EVALUATION_PROTOCOL.md).
    """

    train_mask: np.ndarray
    held_out_mask: np.ndarray
    held_out_category: str


def unseen_attack_split(
    attack_category: ArrayLike,
    held_out_category: str,
) -> UnseenAttackSplit:
    """
    Build train / held-out-test masks by attack CATEGORY LABEL, never by
    random row sampling. `attack_category` is a 1D array-like of category
    labels (e.g. "BENIGN", "DoS", "Infiltration", ...), one per sample.

    Raises UnseenAttackError if `held_out_category` does not appear in
    `attack_category` at all, or if excluding it would leave no training
    rows (e.g. every row belongs to the held-out category).
    """
    categories = np.asarray(attack_category, dtype=object)

    if categories.ndim != 1:
        raise UnseenAttackError("'attack_category' must be a one-dimensional sequence.")

    if categories.size == 0:
        raise UnseenAttackError("'attack_category' is empty. At least one sample is required.")

    if not np.any(categories == held_out_category):
        raise UnseenAttackError(
            f"held_out_category={held_out_category!r} does not appear anywhere in "
            "'attack_category'; nothing would be held out."
        )

    held_out_mask = categories == held_out_category
    train_mask = ~held_out_mask

    if not np.any(train_mask):
        raise UnseenAttackError(
            "Excluding held_out_category would leave zero training rows -- every "
            "sample belongs to the held-out category."
        )

    return UnseenAttackSplit(
        train_mask=train_mask,
        held_out_mask=held_out_mask,
        held_out_category=held_out_category,
    )


def assert_no_category_leakage(split: UnseenAttackSplit, attack_category: ArrayLike) -> None:
    """
    Verify, from the actual category labels, that `held_out_category`
    never appears among the rows selected by `train_mask`. Intended for
    tests/QA. Raises UnseenAttackError on violation.
    """
    categories = np.asarray(attack_category, dtype=object)
    train_categories = categories[split.train_mask]

    if np.any(train_categories == split.held_out_category):
        raise UnseenAttackError(
            f"Leakage: held_out_category={split.held_out_category!r} appears in the "
            "training subset."
        )
