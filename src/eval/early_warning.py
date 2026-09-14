from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

from src.eval.episodes import AttackEpisode
from src.eval.forecast import RawPrediction
from src.eval.lead_time import summarize_lead_times


@dataclass(frozen=True)
class EarlyWarningResult:
    """
    Early Warning Rate: the fraction of attack episodes that received a
    valid warning before (or exactly at) onset.

        rate = n_warned / n_episodes

    DEFINITIONS (shared with src/eval/lead_time.py -- both modules must
    agree, so this reuses `summarize_lead_times` rather than
    reimplementing the warning rule):

      - "attack episode": one continuous malicious run for one host, as
        produced by `src.eval.episodes.group_into_episodes`. Multiple
        consecutive malicious windows belonging to the same continuous
        attack are ONE episode, not several.
      - "valid warning": a same-host prediction issued at or before the
        episode's onset with risk >= `threshold`. See
        `src.eval.lead_time.compute_lead_time` for the exact rule.
      - "warning threshold": configurable via `threshold`; 0.70 is only
        the module default, not a claimed calibrated value.
      - False warnings (a prediction crossing threshold with no
        corresponding attack) are NOT penalized by this metric -- they
        are a false-positive-rate concern, already covered by
        `src/eval/metrics.py::false_positive_rate`. Early Warning Rate
        answers a different question ("were real attacks warned about
        in time?"), not "how many false alarms were raised?".
      - An episode with zero matching predictions of any kind (not just
        none crossing threshold) still counts as "not warned" -- there
        is no separate "no data" bucket; report that gap upstream via
        prediction-completeness checks (src/eval/records.py) if it
        matters for your run.

    `rate` is `0.0` if there are zero episodes (documented zero-episode
    policy, matching the zero-division convention used throughout
    src/eval/metrics.py) -- callers should check `n_episodes` before
    treating a `0.0` rate as "the system failed every attack".
    """

    rate: float
    n_episodes: int
    n_warned: int


def early_warning_rate(
    episodes: Sequence[AttackEpisode],
    predictions: Sequence[RawPrediction],
    threshold: float = 0.70,
) -> EarlyWarningResult:
    """Compute the Early Warning Rate for a set of attack episodes."""
    summary = summarize_lead_times(episodes, predictions, threshold=threshold)

    rate = summary.n_warned / summary.n_episodes if summary.n_episodes else 0.0

    return EarlyWarningResult(
        rate=float(rate),
        n_episodes=summary.n_episodes,
        n_warned=summary.n_warned,
    )
