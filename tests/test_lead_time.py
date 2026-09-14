import pytest

from src.eval.episodes import AttackEpisode
from src.eval.forecast import RawPrediction
from src.eval.lead_time import LeadTimeError, compute_lead_time, summarize_lead_times


def test_warning_before_attack_gives_positive_lead_time():
    episode = AttackEpisode(source_host="hostA", onset_time=100.0, offset_time=130.0)
    predictions = [RawPrediction("hostA", prediction_time=60.0, forecast_horizon=10, risk=0.9)]

    result = compute_lead_time(episode, predictions, threshold=0.70)

    assert result.lead_time_seconds == pytest.approx(40.0)
    assert result.warning_prediction_time == 60.0


def test_warning_exactly_at_onset_gives_zero_lead_time():
    episode = AttackEpisode(source_host="hostA", onset_time=100.0, offset_time=100.0)
    predictions = [RawPrediction("hostA", prediction_time=100.0, forecast_horizon=10, risk=0.9)]

    result = compute_lead_time(episode, predictions, threshold=0.70)

    assert result.lead_time_seconds == 0.0


def test_no_warning_returns_none_not_zero():
    episode = AttackEpisode(source_host="hostA", onset_time=100.0, offset_time=100.0)
    predictions = [RawPrediction("hostA", prediction_time=60.0, forecast_horizon=10, risk=0.2)]  # below threshold

    result = compute_lead_time(episode, predictions, threshold=0.70)

    assert result.lead_time_seconds is None
    assert result.warning_prediction_time is None


def test_multiple_warnings_uses_the_earliest():
    episode = AttackEpisode(source_host="hostA", onset_time=100.0, offset_time=100.0)
    predictions = [
        RawPrediction("hostA", prediction_time=90.0, forecast_horizon=10, risk=0.75),
        RawPrediction("hostA", prediction_time=50.0, forecast_horizon=10, risk=0.80),  # earliest
        RawPrediction("hostA", prediction_time=95.0, forecast_horizon=10, risk=0.99),
    ]

    result = compute_lead_time(episode, predictions, threshold=0.70)

    assert result.warning_prediction_time == 50.0
    assert result.lead_time_seconds == pytest.approx(50.0)


def test_warning_after_onset_does_not_count():
    episode = AttackEpisode(source_host="hostA", onset_time=100.0, offset_time=100.0)
    predictions = [RawPrediction("hostA", prediction_time=101.0, forecast_horizon=10, risk=0.95)]

    result = compute_lead_time(episode, predictions, threshold=0.70)
    assert result.lead_time_seconds is None


def test_warning_from_different_host_does_not_count():
    episode = AttackEpisode(source_host="hostA", onset_time=100.0, offset_time=100.0)
    predictions = [RawPrediction("hostB", prediction_time=50.0, forecast_horizon=10, risk=0.99)]

    result = compute_lead_time(episode, predictions, threshold=0.70)
    assert result.lead_time_seconds is None


def test_invalid_threshold_rejected():
    episode = AttackEpisode(source_host="hostA", onset_time=100.0, offset_time=100.0)
    with pytest.raises(LeadTimeError):
        compute_lead_time(episode, [], threshold=1.5)


# ----------------------------------------------------------------------
# Multiple episodes / aggregate statistics
# ----------------------------------------------------------------------
def test_summarize_lead_times_across_multiple_episodes():
    episodes = [
        AttackEpisode("hostA", onset_time=100.0, offset_time=100.0),
        AttackEpisode("hostB", onset_time=200.0, offset_time=200.0),
        AttackEpisode("hostC", onset_time=300.0, offset_time=300.0),  # never warned
    ]
    predictions = [
        RawPrediction("hostA", prediction_time=80.0, forecast_horizon=10, risk=0.9),  # lead=20
        RawPrediction("hostB", prediction_time=160.0, forecast_horizon=10, risk=0.9),  # lead=40
        # hostC gets no qualifying prediction at all
    ]

    summary = summarize_lead_times(episodes, predictions, threshold=0.70)

    assert summary.n_episodes == 3
    assert summary.n_warned == 2
    assert summary.mean_lead_time_seconds == pytest.approx(30.0)
    assert summary.median_lead_time_seconds == pytest.approx(30.0)


def test_summarize_lead_times_all_missed_gives_none_stats():
    episodes = [AttackEpisode("hostA", onset_time=100.0, offset_time=100.0)]
    predictions = []

    summary = summarize_lead_times(episodes, predictions, threshold=0.70)

    assert summary.n_warned == 0
    assert summary.mean_lead_time_seconds is None
    assert summary.median_lead_time_seconds is None
