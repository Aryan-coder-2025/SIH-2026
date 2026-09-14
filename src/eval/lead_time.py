from __future__ import annotations

from dataclasses import dataclass
from typing import List, Optional, Sequence

import numpy as np

from src.eval.episodes import AttackEpisode
from src.eval.forecast import RawPrediction


class LeadTimeError(ValueError):
    """Raised when lead-time inputs are invalid."""


@dataclass(frozen=True)
class LeadTimeResult:
    """Lead time for one attack episode, or `None` if it was never warned about."""

    episode: AttackEpisode
    lead_time_seconds: Optional[float]
    warning_prediction_time: Optional[float]


@dataclass(frozen=True)
class LeadTimeSummary:
    """Aggregate lead-time statistics across a set of episodes."""

    per_episode: List[LeadTimeResult]
    n_episodes: int
    n_warned: int
    mean_lead_time_seconds: Optional[float]
    median_lead_time_seconds: Optional[float]


def compute_lead_time(
    episode: AttackEpisode,
    predictions: Sequence[RawPrediction],
    threshold: float = 0.70,
) -> LeadTimeResult:
    """
    Lead time for one attack episode.

    DEFINITION
    ----------
    A prediction is a "valid warning" for `episode` if:
      - it was issued for the SAME host (source_host match -- no
        cross-host leakage),
      - its `prediction_time` is at or before the episode's onset
        (`prediction_time <= episode.onset_time`), i.e. the alarm was
        raised no later than the attack actually starting, and
      - its `risk >= threshold`.

    Among all valid warnings, the EARLIEST one (smallest `prediction_time`)
    is used. Lead time is then:

        lead_time_seconds = episode.onset_time - warning.prediction_time

    A lead time of 0 means the warning fired exactly at onset (still a
    valid, if minimal, warning). If no prediction satisfies the above,
    `lead_time_seconds` is `None` -- the episode was never warned about
    (this is reported explicitly, not silently coerced to 0 or dropped).

    `threshold` is a normal, configurable argument -- 0.70 is only the
    module's experimental default, not a claim that it is calibrated.
    """
    if threshold < 0.0 or threshold > 1.0:
        raise LeadTimeError(f"'threshold' must be within [0, 1] (got {threshold}).")

    candidates = [
        p
        for p in predictions
        if p.source_host == episode.source_host
        and p.prediction_time <= episode.onset_time
        and p.risk >= threshold
    ]

    if not candidates:
        return LeadTimeResult(episode=episode, lead_time_seconds=None, warning_prediction_time=None)

    earliest = min(candidates, key=lambda p: p.prediction_time)
    lead_time = episode.onset_time - earliest.prediction_time

    return LeadTimeResult(
        episode=episode,
        lead_time_seconds=float(lead_time),
        warning_prediction_time=float(earliest.prediction_time),
    )


def summarize_lead_times(
    episodes: Sequence[AttackEpisode],
    predictions: Sequence[RawPrediction],
    threshold: float = 0.70,
) -> LeadTimeSummary:
    """
    Compute lead time for every episode and summarize mean/median across
    the ones that WERE warned about. Episodes never warned about count
    toward `n_episodes` (the denominator for early-warning rate lives in
    src/eval/early_warning.py) but are excluded from the mean/median --
    averaging in `None` would silently misrepresent "never warned" as a
    lead time of zero.
    """
    per_episode = [compute_lead_time(ep, predictions, threshold=threshold) for ep in episodes]

    warned_lead_times = [
        r.lead_time_seconds for r in per_episode if r.lead_time_seconds is not None
    ]

    return LeadTimeSummary(
        per_episode=per_episode,
        n_episodes=len(episodes),
        n_warned=len(warned_lead_times),
        mean_lead_time_seconds=float(np.mean(warned_lead_times)) if warned_lead_times else None,
        median_lead_time_seconds=float(np.median(warned_lead_times)) if warned_lead_times else None,
    )
