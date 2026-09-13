from __future__ import annotations

import math
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Any, Sequence


def normalize_to_utc(ts: Any) -> datetime:
    """
    Ensure timestamp is a valid datetime normalized to UTC.
    If naive, localizes to UTC. If timezone-aware, converts to UTC.
    """
    if not isinstance(ts, datetime):
        raise ValueError(f"timestamp must be a valid datetime instance, got {type(ts).__name__}: {ts}")
    if ts.tzinfo is None:
        return ts.replace(tzinfo=timezone.utc)
    return ts.astimezone(timezone.utc)


@dataclass(frozen=True)
class TrafficWindow:
    """
    Canonical representation of network activity for one source host
    during a discrete temporal window.
    """

    timestamp: datetime
    source_host: str
    packet_count: int
    byte_count: int
    window_id: str | None = None
    features: tuple[float, ...] | None = None
    label: float | int | None = None

    def __post_init__(self) -> None:
        # Validate and normalize timestamp
        norm_ts = normalize_to_utc(self.timestamp)
        object.__setattr__(self, "timestamp", norm_ts)

        # Validate source_host
        if not isinstance(self.source_host, str) or not self.source_host.strip():
            raise ValueError("source_host cannot be empty")
        object.__setattr__(self, "source_host", self.source_host.strip())

        # Validate packet_count: must be a non-negative integer quantity.
        # Packets in network traffic are discrete countable events.
        # If float is supplied (e.g., from upstream dataframe column before casting),
        # it must represent an exact integer value (e.g., 100.0 -> 100).
        # Fractional quantities (e.g., 10.5 packets) are physically invalid and rejected.
        if isinstance(self.packet_count, bool) or not isinstance(
            self.packet_count, (int, float)
        ):
            raise ValueError("packet_count must be an integer")
        if not math.isfinite(self.packet_count):
            raise ValueError("packet_count must be a finite number")
        if isinstance(self.packet_count, float):
            if not self.packet_count.is_integer():
                raise ValueError(
                    f"packet_count must be an exact integer quantity, got fractional float {self.packet_count}"
                )
            object.__setattr__(self, "packet_count", int(self.packet_count))
        if self.packet_count < 0:
            raise ValueError("packet_count cannot be negative")

        # Validate byte_count: must be a non-negative integer quantity.
        if isinstance(self.byte_count, bool) or not isinstance(
            self.byte_count, (int, float)
        ):
            raise ValueError("byte_count must be an integer")
        if not math.isfinite(self.byte_count):
            raise ValueError("byte_count must be a finite number")
        if isinstance(self.byte_count, float):
            if not self.byte_count.is_integer():
                raise ValueError(
                    f"byte_count must be an exact integer quantity, got fractional float {self.byte_count}"
                )
            object.__setattr__(self, "byte_count", int(self.byte_count))
        if self.byte_count < 0:
            raise ValueError("byte_count cannot be negative")

        # Validate and enforce deep immutability for optional features vector.
        # A frozen dataclass only prevents reassigning top-level attributes;
        # nested mutable structures like lists remain internally mutable.
        # We explicitly convert any passed sequence to an immutable tuple of floats.
        if self.features is not None:
            if not isinstance(self.features, (tuple, list)):
                raise ValueError("features must be a sequence of finite floats")
            normalized_features: list[float] = []
            for idx, feat in enumerate(self.features):
                if (
                    isinstance(feat, bool)
                    or not isinstance(feat, (int, float))
                    or not math.isfinite(feat)
                ):
                    raise ValueError(
                        f"feature at index {idx} must be a finite number, got {feat}"
                    )
                normalized_features.append(float(feat))
            object.__setattr__(self, "features", tuple(normalized_features))

        # Validate optional label
        if self.label is not None:
            if (
                isinstance(self.label, bool)
                or not isinstance(self.label, (int, float))
                or not math.isfinite(self.label)
            ):
                raise ValueError("label must be a finite number")

        # Validate optional window_id
        if self.window_id is not None and (
            not isinstance(self.window_id, str) or not self.window_id.strip()
        ):
            raise ValueError("window_id must be a non-empty string if provided")

    @property
    def canonical_window_id(self) -> str:
        """Deterministic window identifier: {source_host}_{timestamp_epoch}."""
        if self.window_id:
            return self.window_id
        return f"{self.source_host}_{int(self.timestamp.timestamp())}"