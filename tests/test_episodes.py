import pytest

from src.eval.episodes import EpisodeError, group_into_episodes


def test_single_continuous_episode_is_one_episode_not_many():
    # Four consecutive 10s-apart malicious windows must merge into ONE episode.
    observations = [
        ("hostA", 100.0, 1),
        ("hostA", 110.0, 1),
        ("hostA", 120.0, 1),
        ("hostA", 130.0, 1),
    ]
    episodes = group_into_episodes(observations, max_gap_seconds=10.0)

    assert len(episodes) == 1
    assert episodes[0].onset_time == 100.0
    assert episodes[0].offset_time == 130.0


def test_two_episodes_separated_by_benign_gap():
    observations = [
        ("hostA", 100.0, 1),
        ("hostA", 110.0, 1),
        ("hostA", 120.0, 0),  # benign gap
        ("hostA", 130.0, 0),
        ("hostA", 140.0, 1),
        ("hostA", 150.0, 1),
    ]
    episodes = group_into_episodes(observations, max_gap_seconds=10.0)

    assert len(episodes) == 2
    assert (episodes[0].onset_time, episodes[0].offset_time) == (100.0, 110.0)
    assert (episodes[1].onset_time, episodes[1].offset_time) == (140.0, 150.0)


def test_large_time_gap_without_explicit_benign_marker_starts_new_episode():
    # No explicit "0" rows -- but the gap exceeds max_gap_seconds, so this
    # must still be treated as two distinct episodes.
    observations = [
        ("hostA", 100.0, 1),
        ("hostA", 110.0, 1),
        ("hostA", 500.0, 1),  # far later -- different attack
    ]
    episodes = group_into_episodes(observations, max_gap_seconds=10.0)
    assert len(episodes) == 2


def test_episodes_are_isolated_per_host():
    observations = [
        ("hostA", 100.0, 1),
        ("hostA", 110.0, 1),
        ("hostB", 100.0, 1),
        ("hostB", 110.0, 1),
    ]
    episodes = group_into_episodes(observations, max_gap_seconds=10.0)

    assert len(episodes) == 2
    hosts = {ep.source_host for ep in episodes}
    assert hosts == {"hostA", "hostB"}


def test_input_order_does_not_matter():
    # Deliberately out of order and interleaved across hosts.
    observations = [
        ("hostB", 110.0, 1),
        ("hostA", 100.0, 1),
        ("hostA", 110.0, 1),
        ("hostB", 100.0, 1),
    ]
    episodes = group_into_episodes(observations, max_gap_seconds=10.0)
    assert len(episodes) == 2


def test_all_benign_observations_yield_no_episodes():
    observations = [("hostA", 100.0, 0), ("hostA", 110.0, 0)]
    episodes = group_into_episodes(observations)
    assert episodes == []


def test_empty_observations_rejected():
    with pytest.raises(EpisodeError):
        group_into_episodes([])


def test_invalid_label_rejected():
    with pytest.raises(EpisodeError):
        group_into_episodes([("hostA", 100.0, 2)])
