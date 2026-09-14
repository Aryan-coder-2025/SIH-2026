from __future__ import annotations

from dataclasses import dataclass
from typing import List, Sequence, Tuple

import numpy as np


class EpisodeError(ValueError):
    """Raised when episode grouping inputs are invalid."""


@dataclass(frozen=True)
class AttackEpisode:
    """
    One continuous malicious episode for a single host.

    `onset_time` is the timestamp of the first malicious (y_true=1)
    observation in the run; `offset_time` is the timestamp of the last
    one before it returns to benign. A single attack that spans several
    consecutive malicious windows is ONE episode, not one per window --
    see `group_into_episodes`.
    """

    source_host: str
    onset_time: float
    offset_time: float


def group_into_episodes(
    observations: Sequence[Tuple[str, float, int]],
    max_gap_seconds: float = 10.0,
) -> List[AttackEpisode]:
    """
    Group raw (source_host, time, y_true) ground-truth observations into
    continuous attack episodes, per host.

    Consecutive malicious (y_true=1) observations for the SAME host are
    merged into a single episode as long as the gap between them is at
    most `max_gap_seconds` (default: one project window, 10 seconds).
    A larger gap starts a new episode -- this is what prevents one
    ongoing attack from being double-counted as several separate attacks
    just because it spans multiple 10-second windows, while still
    treating two genuinely distinct attacks (with a quiet interval
    between them) as separate episodes.

    `observations` may be given in any order; they are sorted internally
    by (source_host, time) before grouping -- this function never relies
    on the caller's row order.

    Raises EpisodeError on empty input or a non-binary y_true value.
    """
    if len(observations) == 0:
        raise EpisodeError("'observations' is empty. At least one observation is required.")

    for host, time, y_true in observations:
        if y_true not in (0, 1):
            raise EpisodeError(
                f"y_true must be binary (0 or 1); got {y_true!r} for host={host!r}, time={time!r}."
            )
        if not np.isfinite(time):
            raise EpisodeError(f"time must be finite; got {time!r} for host={host!r}.")

    ordered = sorted(observations, key=lambda row: (row[0], row[1]))

    episodes: List[AttackEpisode] = []
    current_host: str | None = None
    current_onset: float | None = None
    current_offset: float | None = None

    def _flush() -> None:
        if current_host is not None:
            episodes.append(
                AttackEpisode(
                    source_host=current_host,
                    onset_time=current_onset,
                    offset_time=current_offset,
                )
            )

    for host, time, y_true in ordered:
        if y_true == 0:
            _flush()
            current_host = None
            current_onset = None
            current_offset = None
            continue

        if (
            current_host == host
            and current_offset is not None
            and (time - current_offset) <= max_gap_seconds
        ):
            current_offset = time
        else:
            _flush()
            current_host = host
            current_onset = time
            current_offset = time

    _flush()

    return episodes
