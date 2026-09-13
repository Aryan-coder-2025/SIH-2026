from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from typing import Any, Sequence

import numpy as np

from src.config import TemporalConfig, get_temporal_config
from src.schemas.traffic import TrafficWindow, normalize_to_utc


@dataclass(frozen=True)
class TemporalSequenceBatch:
    """
    Container for extracted temporal sequences, future targets, and metadata.

    Fields:
    - X: 3D NumPy array of shape (N, history_length, num_features).
    - y: 2D NumPy array of shape (N, forecast_horizon_windows).
    - hosts: List of source host identifiers corresponding to each sequence.
    - prediction_times: Physical timestamp at which the forecast is made (end of history).
    - target_times: List of physical timestamps corresponding to each forecast horizon.
    - forecast_steps: Future window step offsets (e.g. (1, 2, 3)).
    - offsets_seconds: Future horizon offsets in physical seconds (e.g. (10, 20, 30)).
    """

    X: np.ndarray
    y: np.ndarray
    hosts: list[str]
    prediction_times: list[datetime]
    target_times: list[list[datetime]]
    forecast_steps: tuple[int, ...] = (1, 2, 3)
    offsets_seconds: tuple[int, ...] = (10, 20, 30)

    def __iter__(self):
        """Allow seamless unpacking as (X, y) for backward compatibility."""
        return iter((self.X, self.y))

    def to_prediction_records(
        self,
        predicted_risks: np.ndarray,
        offsets_seconds: Sequence[int] | None = None,
    ) -> list[Any]:
        """
        Convert batch and model predictions to validated PredictionRecord instances.

        CRITICAL EVALUATION INVARIANTS:
        1. predicted_risks is strictly required. Evaluation records must NEVER silently
           manufacture predictions or default missing predictions to zeros.
        2. Horizon Alignment: Evaluation horizon offsets must strictly match the horizons
           established during sequence construction. If offsets_seconds is passed, it must
           equal the batch's actual offsets_seconds. If omitted, batch.offsets_seconds is used.
        """
        from src.evaluation.records import PredictionRecord

        if predicted_risks is None:
            raise ValueError(
                "predicted_risks must be explicitly supplied. "
                "Evaluation records must never silently manufacture predictions or default to zeros."
            )

        pred_arr = np.asarray(predicted_risks, dtype=float)
        if pred_arr.shape != self.y.shape:
            raise ValueError(
                f"predicted_risks shape {pred_arr.shape} does not match targets shape {self.y.shape}"
            )

        # Resolve and validate evaluation horizons
        if offsets_seconds is not None:
            supplied_offsets = tuple(int(s) for s in offsets_seconds)
            if len(supplied_offsets) != self.y.shape[1]:
                raise ValueError(
                    f"Supplied offsets_seconds length ({len(supplied_offsets)}) does not match "
                    f"target horizon count ({self.y.shape[1]})."
                )
            if supplied_offsets != self.offsets_seconds:
                raise ValueError(
                    f"Supplied offsets_seconds {supplied_offsets} does not match sequence batch "
                    f"offsets_seconds {self.offsets_seconds}. Horizon semantics must be consistent."
                )
            active_offsets = supplied_offsets
        else:
            active_offsets = self.offsets_seconds

        records = []
        num_seqs = len(self.X)
        for i in range(num_seqs):
            host = self.hosts[i]
            pred_time = self.prediction_times[i]
            for h_idx, offset_sec in enumerate(active_offsets):
                targ_time = self.target_times[i][h_idx]
                y_true_val = float(self.y[i, h_idx])
                pred_risk_val = float(pred_arr[i, h_idx])
                records.append(
                    PredictionRecord(
                        source_host=host,
                        prediction_time=pred_time,
                        forecast_horizon=offset_sec,
                        target_time=targ_time,
                        y_true=y_true_val,
                        predicted_risk=pred_risk_val,
                    )
                )
        return records



def build_sequences(
    features: np.ndarray,
    targets: np.ndarray,
    history_length: int | None = None,
    forecast_horizons: Sequence[int] | None = None,
    *,
    timestamps: Sequence[datetime] | np.ndarray | None = None,
    source_hosts: Sequence[str] | np.ndarray | None = None,
    window_seconds: int | None = None,
    config: dict[str, Any] | TemporalConfig | None = None,
    strict_continuity: bool = False,
    allow_synthetic_fallbacks: bool = False,
    return_metadata: bool = False,
) -> tuple[np.ndarray, np.ndarray] | TemporalSequenceBatch:
    """
    Build chronological, host-isolated input sequences and multi-step future targets.

    Invariants Enforced:
    1. Host Isolation: Observations from different source hosts are never mixed.
    2. Chronological Ordering: Records are sorted strictly in time.
    3. Temporal Continuity: Physical time between consecutive windows must match
       window_seconds. Missing windows (gaps) are never silently compressed.
    4. Duplicate Rejection: Duplicate (source_host, timestamp) windows are rejected.
    5. Anti-Leakage: Historical sequences contain strictly past observations.
    6. Numerical Integrity: NaN and Inf values in features or targets are rejected.
    7. No Silent Fallbacks: Real source_host and timestamps are mandatory in production
       (allow_synthetic_fallbacks=False). Synthetic fallbacks are restricted to toy tests.
    """
    # ------------------------------------------------------------------
    # 1. Resolve configuration parameters
    # ------------------------------------------------------------------
    temporal_cfg = get_temporal_config(config)

    win_sec = window_seconds if window_seconds is not None else temporal_cfg.window_seconds
    hist_len = history_length if history_length is not None else temporal_cfg.history_length
    horizons = tuple(
        forecast_horizons
        if forecast_horizons is not None
        else temporal_cfg.forecast_horizon_steps
    )
    offsets_sec = tuple(int(h * win_sec) for h in horizons)

    if win_sec <= 0:
        raise ValueError("window_seconds must be positive")
    if hist_len <= 0:
        raise ValueError("history_length must be positive")
    if not horizons:
        raise ValueError("forecast_horizons cannot be empty")
    if any(horizon <= 0 for horizon in horizons):
        raise ValueError("forecast horizons must be positive")

    # ------------------------------------------------------------------
    # 2. Input validation
    # ------------------------------------------------------------------
    if not isinstance(features, np.ndarray):
        features = np.asarray(features)
    if not isinstance(targets, np.ndarray):
        targets = np.asarray(targets)

    if features.ndim != 2:
        raise ValueError("features must be a 2D array")
    if targets.ndim != 1:
        raise ValueError("targets must be a 1D array")
    if len(features) != len(targets):
        raise ValueError("features and targets must have the same length")

    num_samples = len(features)
    num_features = features.shape[1]

    # Reject NaN / Inf
    if num_samples > 0:
        if np.isnan(features).any() or np.isinf(features).any():
            raise ValueError("features contain NaN or Inf values")
        if np.isnan(targets).any() or np.isinf(targets).any():
            raise ValueError("targets contain NaN or Inf values")

    # Fast path for empty input
    if num_samples == 0:
        empty_X = np.empty((0, hist_len, num_features), dtype=features.dtype)
        empty_y = np.empty((0, len(horizons)), dtype=targets.dtype)
        if return_metadata:
            return TemporalSequenceBatch(
                X=empty_X,
                y=empty_y,
                hosts=[],
                prediction_times=[],
                target_times=[],
                forecast_steps=horizons,
                offsets_seconds=offsets_sec,
            )
        return empty_X, empty_y

    # ------------------------------------------------------------------
    # 3. Host and timestamp alignment
    # ------------------------------------------------------------------
    # PRODUCTION VS TEST TEMPORAL METADATA INVARIANT:
    # In production and scientific pipelines, every network observation must possess
    # an explicit source_host and timestamp. Fabricating synthetic temporal identity
    # in production risks masking missing physical time and mixing real host traffic.
    # Therefore, allow_synthetic_fallbacks=False is the default.
    # Synthetic generation is strictly permitted when allow_synthetic_fallbacks=True
    # is explicitly enabled for isolated toy array tests.
    if not allow_synthetic_fallbacks:
        if source_hosts is None:
            raise ValueError(
                "source_hosts must be explicitly provided in scientific pipelines. "
                "Synthetic host identity is prohibited unless allow_synthetic_fallbacks=True is set for isolated unit tests."
            )
        if timestamps is None:
            raise ValueError(
                "timestamps must be explicitly provided in scientific pipelines. "
                "Synthetic temporal identity is prohibited unless allow_synthetic_fallbacks=True is set for isolated unit tests."
            )

    if source_hosts is None:
        hosts_list = ["default_host"] * num_samples
    else:
        if len(source_hosts) != num_samples:
            raise ValueError("source_hosts must have the same length as features")
        hosts_list = [str(h) for h in source_hosts]

    if timestamps is None:
        # Generate contiguous synthetic timestamps for backward compatibility (isolated toy tests only)
        base_time = datetime(2026, 1, 1, 0, 0, 0, tzinfo=timezone.utc)
        step_delta = timedelta(seconds=win_sec)
        times_list = [base_time + (step_delta * i) for i in range(num_samples)]
    else:
        if len(timestamps) != num_samples:
            raise ValueError("timestamps must have the same length as features")
        times_list = [normalize_to_utc(t) for t in timestamps]

    # ------------------------------------------------------------------
    # 4. Group by host and sort chronologically
    # ------------------------------------------------------------------
    host_records: dict[str, list[tuple[datetime, np.ndarray, float | int]]] = (
        defaultdict(list)
    )

    for i in range(num_samples):
        h = hosts_list[i]
        t = times_list[i]
        f = features[i]
        tg = targets[i]
        host_records[h].append((t, f, tg))

    all_X: list[np.ndarray] = []
    all_y: list[list[float | int]] = []
    out_hosts: list[str] = []
    out_prediction_times: list[datetime] = []
    out_target_times: list[list[datetime]] = []

    max_horizon = max(horizons)
    step_delta = timedelta(seconds=win_sec)

    # Deterministic processing order by sorting host identifiers
    for host in sorted(host_records.keys()):
        records = host_records[host]
        # Sort chronologically by timestamp
        records.sort(key=lambda r: r[0])

        # Validate duplicates and partition into contiguous segments
        contiguous_segments: list[list[tuple[datetime, np.ndarray, float | int]]] = []
        current_segment: list[tuple[datetime, np.ndarray, float | int]] = []

        for idx, rec in enumerate(records):
            if idx == 0:
                current_segment.append(rec)
                continue

            prev_time = records[idx - 1][0]
            curr_time = rec[0]

            if curr_time == prev_time:
                raise ValueError(
                    f"Duplicate window detected for host '{host}' at {curr_time.isoformat()}"
                )

            time_diff = curr_time - prev_time

            if time_diff == step_delta:
                # Contiguous step
                current_segment.append(rec)
            else:
                # Discontinuity / gap detected
                if strict_continuity:
                    raise ValueError(
                        f"Temporal discontinuity detected for host '{host}': "
                        f"expected {prev_time + step_delta}, got {curr_time} (gap of {time_diff.total_seconds()}s)"
                    )
                # Break current segment and start a new contiguous segment
                if current_segment:
                    contiguous_segments.append(current_segment)
                current_segment = [rec]

        if current_segment:
            contiguous_segments.append(current_segment)

        # Build sequences strictly within each contiguous segment
        for segment in contiguous_segments:
            seg_len = len(segment)
            # Need at least (history_length + max_horizon) windows to form 1 valid sequence
            if seg_len < (hist_len + max_horizon):
                continue

            seg_times = [r[0] for r in segment]
            seg_feats = np.array([r[1] for r in segment])
            seg_targs = np.array([r[2] for r in segment])

            for end_index in range(hist_len - 1, seg_len - max_horizon):
                start_index = end_index - hist_len + 1

                # Historical features: strictly past up to end_index
                seq_X = seg_feats[start_index : end_index + 1]
                all_X.append(seq_X)

                # Future targets: strictly future horizons
                seq_y = [seg_targs[end_index + h] for h in horizons]
                all_y.append(seq_y)

                # Metadata tracking
                pred_time = seg_times[end_index]
                targ_times = [seg_times[end_index + h] for h in horizons]

                out_hosts.append(host)
                out_prediction_times.append(pred_time)
                out_target_times.append(targ_times)

    if not all_X:
        empty_X = np.empty((0, hist_len, num_features), dtype=features.dtype)
        empty_y = np.empty((0, len(horizons)), dtype=targets.dtype)
        if return_metadata:
            return TemporalSequenceBatch(
                X=empty_X,
                y=empty_y,
                hosts=[],
                prediction_times=[],
                target_times=[],
                forecast_steps=horizons,
                offsets_seconds=offsets_sec,
            )
        return empty_X, empty_y

    X_arr = np.asarray(all_X)
    y_arr = np.asarray(all_y)

    if return_metadata:
        return TemporalSequenceBatch(
            X=X_arr,
            y=y_arr,
            hosts=out_hosts,
            prediction_times=out_prediction_times,
            target_times=out_target_times,
            forecast_steps=horizons,
            offsets_seconds=offsets_sec,
        )

    return X_arr, y_arr


def build_sequences_from_windows(
    windows: Sequence[TrafficWindow],
    *,
    config: dict[str, Any] | TemporalConfig | None = None,
    strict_continuity: bool = False,
    return_metadata: bool = True,
) -> TemporalSequenceBatch | tuple[np.ndarray, np.ndarray]:
    """
    Convenience wrapper to build temporal sequences directly from TrafficWindow objects.
    """
    if not windows:
        temporal_cfg = get_temporal_config(config)
        empty_X = np.empty((0, temporal_cfg.history_length, 2), dtype=np.float64)
        empty_y = np.empty((0, temporal_cfg.forecast_horizon_windows), dtype=np.float64)
        if return_metadata:
            return TemporalSequenceBatch(
                X=empty_X,
                y=empty_y,
                hosts=[],
                prediction_times=[],
                target_times=[],
                forecast_steps=temporal_cfg.forecast_horizon_steps,
                offsets_seconds=temporal_cfg.forecast_offsets_seconds,
            )
        return empty_X, empty_y

    timestamps = [w.timestamp for w in windows]
    source_hosts = [w.source_host for w in windows]

    # Extract features: if w.features is provided use it, otherwise [packet_count, byte_count]
    feat_list = []
    targ_list = []

    for w in windows:
        if w.features is not None:
            feat_list.append(list(w.features))
        else:
            feat_list.append([float(w.packet_count), float(w.byte_count)])

        # CRITICAL GROUND-TRUTH INVARIANT:
        # Do not treat a missing label (None) as benign (0.0).
        # Missing ground truth means the state was unobserved or unverified;
        # converting None -> 0.0 would fabricate false benign labels in the training set.
        if w.label is None:
            raise ValueError(
                f"Window at {w.timestamp.isoformat()} for host '{w.source_host}' has missing label (None). "
                "Ground truth cannot be silently assumed to be benign (0.0). "
                "All windows used for supervised target construction must have verified labels."
            )
        targ_list.append(float(w.label))

    features = np.asarray(feat_list, dtype=np.float64)
    targets = np.asarray(targ_list, dtype=np.float64)

    return build_sequences(
        features=features,
        targets=targets,
        timestamps=timestamps,
        source_hosts=source_hosts,
        config=config,
        strict_continuity=strict_continuity,
        return_metadata=return_metadata,
    )