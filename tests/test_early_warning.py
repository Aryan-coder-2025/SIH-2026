import pytest

from src.eval.early_warning import early_warning_rate
from src.eval.episodes import AttackEpisode
from src.eval.forecast import RawPrediction


def test_early_warning_rate_basic():
    episodes = [
        AttackEpisode("hostA", onset_time=100.0, offset_time=100.0),  # warned
        AttackEpisode("hostB", onset_time=200.0, offset_time=200.0),  # not warned
    ]
    predictions = [RawPrediction("hostA", prediction_time=80.0, forecast_horizon=10, risk=0.9)]

    result = early_warning_rate(episodes, predictions, threshold=0.70)

    assert result.n_episodes == 2
    assert result.n_warned == 1
    assert result.rate == pytest.approx(0.5)


def test_early_warning_rate_all_warned():
    episodes = [
        AttackEpisode("hostA", onset_time=100.0, offset_time=100.0),
        AttackEpisode("hostB", onset_time=200.0, offset_time=200.0),
    ]
    predictions = [
        RawPrediction("hostA", prediction_time=80.0, forecast_horizon=10, risk=0.9),
        RawPrediction("hostB", prediction_time=180.0, forecast_horizon=10, risk=0.9),
    ]

    result = early_warning_rate(episodes, predictions, threshold=0.70)
    assert result.rate == 1.0


def test_early_warning_rate_none_warned():
    episodes = [AttackEpisode("hostA", onset_time=100.0, offset_time=100.0)]
    result = early_warning_rate(episodes, predictions=[], threshold=0.70)
    assert result.rate == 0.0
    assert result.n_warned == 0


def test_early_warning_rate_zero_episodes_documented_policy():
    result = early_warning_rate(episodes=[], predictions=[], threshold=0.70)
    assert result.rate == 0.0
    assert result.n_episodes == 0


def test_false_warnings_do_not_affect_early_warning_rate():
    """A prediction crossing threshold with NO corresponding attack must
    not inflate or deflate the early warning rate -- that's an FPR concern."""
    episodes = [AttackEpisode("hostA", onset_time=100.0, offset_time=100.0)]
    predictions = [
        RawPrediction("hostA", prediction_time=80.0, forecast_horizon=10, risk=0.9),  # valid warning
        RawPrediction("hostB", prediction_time=500.0, forecast_horizon=10, risk=0.99),  # false alarm, different host/episode
    ]

    result = early_warning_rate(episodes, predictions, threshold=0.70)
    assert result.rate == 1.0  # unaffected by the unrelated false alarm


def test_continuous_episode_counts_once_not_per_window():
    from src.eval.episodes import group_into_episodes

    observations = [
        ("hostA", 100.0, 1),
        ("hostA", 110.0, 1),
        ("hostA", 120.0, 1),
    ]
    episodes = group_into_episodes(observations, max_gap_seconds=10.0)
    assert len(episodes) == 1  # one continuous attack, not three

    predictions = [RawPrediction("hostA", prediction_time=90.0, forecast_horizon=10, risk=0.9)]
    result = early_warning_rate(episodes, predictions, threshold=0.70)

    assert result.n_episodes == 1
    assert result.rate == 1.0
