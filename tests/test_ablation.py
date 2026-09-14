import pytest

from src.eval.ablation import (
    AblationConfig,
    AblationConfigError,
    SUPPORTED_FEATURE_GROUPS,
    SUPPORTED_TEMPORAL_MODES,
    standard_ablation_grid,
)


# ----------------------------------------------------------------------
# AblationConfig validation
# ----------------------------------------------------------------------
def test_valid_config_constructs_cleanly():
    config = AblationConfig(
        name="rf_flow_only_h10",
        feature_group="flow",
        temporal_mode="temporal",
        history_length=10,
        model_name="random_forest",
        forecast_horizon=10,
    )
    assert config.feature_group == "flow"
    assert config.forecast_horizon == 10


def test_static_mode_requires_history_length_one():
    with pytest.raises(AblationConfigError, match="static"):
        AblationConfig(
            name="bad",
            feature_group="flow",
            temporal_mode="static",
            history_length=10,  # invalid for static
            model_name="logistic_regression",
            forecast_horizon=10,
        )


def test_static_mode_with_history_length_one_is_valid():
    config = AblationConfig(
        name="ok",
        feature_group="flow+packet",
        temporal_mode="static",
        history_length=1,
        model_name="logistic_regression",
        forecast_horizon=20,
    )
    assert config.history_length == 1


def test_temporal_mode_requires_at_least_two_windows():
    with pytest.raises(AblationConfigError, match="temporal"):
        AblationConfig(
            name="bad",
            feature_group="flow",
            temporal_mode="temporal",
            history_length=1,  # invalid: not a "history"
            model_name="random_forest",
            forecast_horizon=10,
        )


def test_invalid_feature_group_rejected():
    with pytest.raises(AblationConfigError, match="feature_group"):
        AblationConfig(
            name="bad",
            feature_group="pcap_only",  # not supported
            temporal_mode="temporal",
            history_length=10,
            model_name="random_forest",
            forecast_horizon=10,
        )


def test_invalid_temporal_mode_rejected():
    with pytest.raises(AblationConfigError, match="temporal_mode"):
        AblationConfig(
            name="bad",
            feature_group="flow",
            temporal_mode="dynamic",  # not supported
            history_length=10,
            model_name="random_forest",
            forecast_horizon=10,
        )


def test_invalid_forecast_horizon_rejected():
    with pytest.raises(AblationConfigError, match="forecast_horizon"):
        AblationConfig(
            name="bad",
            feature_group="flow",
            temporal_mode="temporal",
            history_length=10,
            model_name="random_forest",
            forecast_horizon=15,  # not one of 10/20/30
        )


def test_empty_name_rejected():
    with pytest.raises(AblationConfigError, match="name"):
        AblationConfig(
            name="",
            feature_group="flow",
            temporal_mode="temporal",
            history_length=10,
            model_name="random_forest",
            forecast_horizon=10,
        )


def test_empty_model_name_rejected():
    with pytest.raises(AblationConfigError, match="model_name"):
        AblationConfig(
            name="x",
            feature_group="flow",
            temporal_mode="temporal",
            history_length=10,
            model_name="",
            forecast_horizon=10,
        )


def test_non_positive_history_length_rejected():
    with pytest.raises(AblationConfigError, match="history_length"):
        AblationConfig(
            name="x",
            feature_group="flow",
            temporal_mode="temporal",
            history_length=0,
            model_name="random_forest",
            forecast_horizon=10,
        )


def test_config_is_json_serializable_via_to_dict():
    config = AblationConfig(
        name="x",
        feature_group="packet",
        temporal_mode="temporal",
        history_length=10,
        model_name="random_forest",
        forecast_horizon=30,
    )
    d = config.to_dict()
    assert d == {
        "name": "x",
        "feature_group": "packet",
        "temporal_mode": "temporal",
        "history_length": 10,
        "model_name": "random_forest",
        "forecast_horizon": 30,
    }
    import json

    json.dumps(d)  # must not raise


# ----------------------------------------------------------------------
# standard_ablation_grid
# ----------------------------------------------------------------------
def test_standard_grid_covers_required_comparisons():
    grid = standard_ablation_grid(model_name="random_forest", forecast_horizon=10)

    feature_groups_seen = {c.feature_group for c in grid}
    assert feature_groups_seen == set(SUPPORTED_FEATURE_GROUPS)

    temporal_modes_seen = {c.temporal_mode for c in grid}
    assert temporal_modes_seen == set(SUPPORTED_TEMPORAL_MODES)


def test_standard_grid_has_exactly_five_configs():
    grid = standard_ablation_grid(model_name="logistic_regression", forecast_horizon=20)
    assert len(grid) == 5


def test_standard_grid_all_configs_valid_and_share_model_and_horizon():
    grid = standard_ablation_grid(model_name="persistence", forecast_horizon=30, history_length=10)

    for config in grid:
        assert config.model_name == "persistence"
        assert config.forecast_horizon == 30
        # __post_init__ already validated each config at construction time;
        # re-accessing to_dict() exercises it end-to-end without raising.
        config.to_dict()


def test_standard_grid_names_are_unique():
    grid = standard_ablation_grid(model_name="random_forest", forecast_horizon=10)
    names = [c.name for c in grid]
    assert len(names) == len(set(names))


def test_standard_grid_static_and_temporal_isolate_feature_group():
    """The static-vs-temporal comparison (D vs E) must hold feature_group
    fixed at flow+packet so only the temporal dimension varies."""
    grid = standard_ablation_grid(model_name="random_forest", forecast_horizon=10)

    static_config = next(c for c in grid if c.temporal_mode == "static")
    temporal_only_config = next(
        c for c in grid if c.temporal_mode == "temporal" and c.feature_group == "flow+packet"
    )
    assert static_config.feature_group == "flow+packet"
    assert temporal_only_config.feature_group == "flow+packet"
    assert static_config.history_length == 1
    assert temporal_only_config.history_length == 10
