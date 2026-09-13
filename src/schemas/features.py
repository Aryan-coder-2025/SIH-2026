from __future__ import annotations

from typing import Sequence


# Fields that represent raw identifiers, ground-truth targets, or post-event stage outputs.
# These MUST NEVER enter the model as input features.
FORBIDDEN_FEATURE_NAMES: frozenset[str] = frozenset({
    "src_ip",
    "dst_ip",
    "src_port",
    "dst_port",
    "flow_id",
    "host_id",
    "source_host",
    "timestamp",
    "window_start",
    "window_end",
    "window_id",
    "risk_score",
    "future_malicious",
    "future_malicious_risk",
    "label",
    "attack_type",
    "y_true",
    "target",
    "ground_truth",
})


class FeatureOrderError(ValueError):
    """Raised when feature columns do not match the expected canonical order."""


class FeatureLeakageError(ValueError):
    """Raised when forbidden identifier or target-derived columns enter feature selection."""


def validate_feature_names(
    feature_names: Sequence[str],
    expected_order: Sequence[str] | None = None,
) -> list[str]:
    """
    Validate that feature names are unique, contain no forbidden target-derived
    or raw identifier columns, and strictly conform to expected ordering.
    """
    if not feature_names:
        raise ValueError("feature_names cannot be empty")

    cleaned_names: list[str] = []
    seen = set()

    for idx, name in enumerate(feature_names):
        if not isinstance(name, str) or not name.strip():
            raise ValueError(f"Feature name at index {idx} must be a non-empty string")

        clean = name.strip()
        lower_clean = clean.lower()

        # Reject forbidden target-derived or identifier fields
        if lower_clean in FORBIDDEN_FEATURE_NAMES or lower_clean.startswith("target_"):
            raise FeatureLeakageError(
                f"Forbidden feature '{clean}' rejected: target-derived or identifier leakage"
            )

        # Reject duplicates
        if clean in seen:
            raise ValueError(f"Duplicate feature name detected: '{clean}'")

        seen.add(clean)
        cleaned_names.append(clean)

    # Validate exact ordering if expected_order is enforced
    if expected_order is not None:
        expected_list = [e.strip() for e in expected_order]
        if cleaned_names != expected_list:
            raise FeatureOrderError(
                f"Feature ordering mismatch: expected {expected_list}, got {cleaned_names}"
            )

    return cleaned_names
