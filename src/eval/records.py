from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, List, Sequence, Set, Tuple

import numpy as np

# ----------------------------------------------------------------------
# Canonical supported forecast horizons (seconds).
#
# This is the SINGLE source of truth for which horizons the evaluation
# layer accepts. Nothing else in src/eval or src/baseline should
# hard-code the literals 10/20/30 for this purpose -- import
# SUPPORTED_HORIZONS from here instead.
# ----------------------------------------------------------------------
SUPPORTED_HORIZONS: Tuple[int, ...] = (10, 20, 30)

_TIME_TOLERANCE_SECONDS = 1e-6


class RecordValidationError(ValueError):
    """Raised when one or more prediction records are malformed."""


# ----------------------------------------------------------------------
# Canonical prediction record
# ----------------------------------------------------------------------
@dataclass(frozen=True)
class PredictionRecord:
    """
    One forecast prediction, fully self-describing.

    Fields (per the project's evaluation contract):
        source_host:      canonical host identifier (the forecasting
                           entity). Must be a non-empty string.
        prediction_time:  the time (seconds, any consistent epoch) at
                           which the forecast was issued.
        forecast_horizon: seconds ahead being forecast; must be one of
                           SUPPORTED_HORIZONS (10, 20, or 30).
        target_time:      the time the forecast concerns. Must equal
                           prediction_time + forecast_horizon.
        y_true:            ground-truth binary label (0/1) at target_time.
        risk:              predicted risk/probability score in [0, 1] for
                           target_time.

    The (source_host, prediction_time, forecast_horizon) triple is this
    record's unique key -- see `find_duplicate_keys`.
    """

    source_host: str
    prediction_time: float
    forecast_horizon: int
    target_time: float
    y_true: int
    risk: float

    def key(self) -> Tuple[str, float, int]:
        """The unique (source_host, prediction_time, forecast_horizon) key."""
        return (self.source_host, self.prediction_time, self.forecast_horizon)


# ----------------------------------------------------------------------
# Field-level validation
# ----------------------------------------------------------------------
def _validate_one_record(record: PredictionRecord, index: int) -> List[str]:
    """Return a list of human-readable problems with `record` (empty if valid)."""
    problems: List[str] = []

    if not isinstance(record.source_host, str) or record.source_host.strip() == "":
        problems.append(f"record[{index}]: source_host must be a non-empty string.")

    for field_name in ("prediction_time", "target_time"):
        value = getattr(record, field_name)
        if isinstance(value, bool) or not isinstance(value, (int, float)):
            problems.append(f"record[{index}]: {field_name} must be numeric.")
        elif not np.isfinite(value):
            problems.append(f"record[{index}]: {field_name} must be finite.")

    if record.forecast_horizon not in SUPPORTED_HORIZONS:
        problems.append(
            f"record[{index}]: forecast_horizon={record.forecast_horizon!r} is not "
            f"supported (must be one of {SUPPORTED_HORIZONS})."
        )

    # Only check target_time = prediction_time + horizon if the inputs
    # above were themselves well-formed numbers.
    if not problems:
        expected_target = record.prediction_time + record.forecast_horizon
        if abs(record.target_time - expected_target) > _TIME_TOLERANCE_SECONDS:
            problems.append(
                f"record[{index}]: target_time ({record.target_time}) does not equal "
                f"prediction_time + forecast_horizon ({expected_target})."
            )

    if isinstance(record.y_true, bool) or record.y_true not in (0, 1):
        problems.append(
            f"record[{index}]: y_true must be a binary label (0 or 1), "
            f"got {record.y_true!r}."
        )

    if isinstance(record.risk, bool) or not isinstance(record.risk, (int, float)):
        problems.append(f"record[{index}]: risk must be numeric.")
    else:
        if not np.isfinite(record.risk):
            problems.append(f"record[{index}]: risk must be finite.")
        elif not (0.0 <= record.risk <= 1.0):
            problems.append(
                f"record[{index}]: risk must lie within [0, 1], got {record.risk}."
            )

    return problems


def find_duplicate_keys(
    records: Sequence[PredictionRecord],
) -> List[Tuple[str, float, int]]:
    """
    Return the list of (source_host, prediction_time, forecast_horizon)
    keys that appear more than once in `records`. Empty if there are no
    duplicates. Duplicates are never silently dropped -- callers must
    inspect this (or use `validate_prediction_records`, which raises).
    """
    seen: Set[Tuple[str, float, int]] = set()
    duplicates: List[Tuple[str, float, int]] = []
    for record in records:
        key = record.key()
        if key in seen and key not in duplicates:
            duplicates.append(key)
        seen.add(key)
    return duplicates


def find_missing_prediction_keys(
    records: Sequence[PredictionRecord],
    expected_keys: Iterable[Tuple[str, float, int]],
) -> List[Tuple[str, float, int]]:
    """
    Given the set of (source_host, prediction_time, forecast_horizon)
    keys that were EXPECTED to have a prediction, return the ones with
    no matching record. Empty if nothing is missing. This never drops
    missing entries silently -- it is the caller's job to decide what to
    do with a non-empty report (e.g. fail the run, or evaluate on the
    reduced sample with the gap explicitly logged).
    """
    present = {record.key() for record in records}
    return [key for key in expected_keys if key not in present]


def validate_prediction_records(
    records: Sequence[PredictionRecord],
    expected_keys: Iterable[Tuple[str, float, int]] | None = None,
) -> None:
    """
    Validate a batch of PredictionRecords. Raises RecordValidationError
    (listing every problem found, not just the first) if any record is
    malformed or if duplicate keys are present. If `expected_keys` is
    given, also raises if any expected (host, prediction_time, horizon)
    combination has no matching record.
    """
    if len(records) == 0:
        raise RecordValidationError(
            "No prediction records were given. At least one record is required."
        )

    problems: List[str] = []
    for index, record in enumerate(records):
        problems.extend(_validate_one_record(record, index))

    duplicates = find_duplicate_keys(records)
    if duplicates:
        problems.append(
            "Duplicate prediction keys (source_host, prediction_time, "
            f"forecast_horizon) found: {duplicates}."
        )

    if expected_keys is not None:
        missing = find_missing_prediction_keys(records, expected_keys)
        if missing:
            problems.append(f"Missing prediction keys: {missing}.")

    if problems:
        raise RecordValidationError(
            f"{len(problems)} problem(s) found in prediction records:\n"
            + "\n".join(f"  - {p}" for p in problems)
        )
