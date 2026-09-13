from pathlib import Path

import pytest
import yaml

from src.config import ConfigError, load_config


CONFIG_PATH = Path("configs/project.yaml")


def test_valid_configuration():
    config = load_config(CONFIG_PATH)

    assert config["project"]["id"] == "SIH26153"
    assert config["data"]["dataset"] == "CSE-CIC-IDS2018"

    assert config["temporal"]["window_seconds"] == 10
    assert config["temporal"]["sequence_length_windows"] == 10
    assert config["temporal"]["forecast_horizon_windows"] == 3

    assert set(config["features"]["required_levels"]) == {
        "flow",
        "packet",
    }


def test_missing_configuration_is_rejected():
    missing_path = Path("configs") / "DOES_NOT_EXIST.yaml"

    with pytest.raises(ConfigError):
        load_config(missing_path)


def _write_modified_config(tmp_path, modify):
    with CONFIG_PATH.open("r", encoding="utf-8") as file:
        config = yaml.safe_load(file)

    modify(config)

    test_config = tmp_path / "project.yaml"

    with test_config.open("w", encoding="utf-8") as file:
        yaml.safe_dump(config, file)

    return test_config


def test_wrong_project_id_is_rejected(tmp_path):
    path = _write_modified_config(
        tmp_path,
        lambda config: config["project"].update(
            {"id": "WRONG_PROJECT"}
        ),
    )

    with pytest.raises(ConfigError):
        load_config(path)


def test_wrong_window_size_is_rejected(tmp_path):
    path = _write_modified_config(
        tmp_path,
        lambda config: config["temporal"].update(
            {"window_seconds": 5}
        ),
    )

    with pytest.raises(ConfigError):
        load_config(path)


def test_wrong_sequence_length_is_rejected(tmp_path):
    path = _write_modified_config(
        tmp_path,
        lambda config: config["temporal"].update(
            {"sequence_length_windows": 5}
        ),
    )

    with pytest.raises(ConfigError):
        load_config(path)


def test_missing_packet_features_are_rejected(tmp_path):
    def remove_packet(config):
        config["features"]["required_levels"] = ["flow"]

    path = _write_modified_config(tmp_path, remove_packet)

    with pytest.raises(ConfigError):
        load_config(path)


def test_wrong_forecast_offsets_are_rejected(tmp_path):
    path = _write_modified_config(
        tmp_path,
        lambda config: config["prediction"].update(
            {"forecast_offsets_seconds": [10, 20, 40]}
        ),
    )

    with pytest.raises(ConfigError):
        load_config(path)


def test_future_leakage_protection_cannot_be_disabled(tmp_path):
    path = _write_modified_config(
        tmp_path,
        lambda config: config["evaluation"].update(
            {"prevent_future_information_leakage": False}
        ),
    )

    with pytest.raises(ConfigError):
        load_config(path)

def test_temporal_split_cannot_be_disabled(tmp_path):
    path = _write_modified_config(
        tmp_path,
        lambda config: config["evaluation"].update(
            {"temporal_split": False}
        ),
    )

    with pytest.raises(ConfigError):
        load_config(path)


def test_split_before_sequence_construction_cannot_be_disabled(tmp_path):
    path = _write_modified_config(
        tmp_path,
        lambda config: config["evaluation"].update(
            {"split_before_sequence_construction": False}
        ),
    )

    with pytest.raises(ConfigError):
        load_config(path)


def test_invalid_feature_level_entries_are_rejected(tmp_path):
    path = _write_modified_config(
        tmp_path,
        lambda config: config["features"].update(
            {"required_levels": ["flow", "packet", {}]}
        ),
    )

    with pytest.raises(ConfigError):
        load_config(path)


def test_get_temporal_config_rejects_partial_or_ad_hoc_dict_bypass():
    """
    CRITICAL AUDIT INVARIANT (Issue 1):
    get_temporal_config() must never accept an ad-hoc or partial dictionary
    that bypasses full project configuration validation.
    """
    from src.config import get_temporal_config

    # Missing project, entity, features, data, evaluation sections
    with pytest.raises(ConfigError, match="project"):
        get_temporal_config({"temporal": {"window_seconds": 10}})

    with pytest.raises(ConfigError):
        get_temporal_config({
            "temporal": {
                "window_seconds": 10,
                "sequence_length_windows": 10,
                "forecast_horizon_windows": 3,
            },
            "prediction": {"forecast_offsets_seconds": [10, 20, 30]},
        })


def test_get_temporal_config_accepts_valid_full_project_dict():
    """Valid full project dictionary is accepted and yields matching TemporalConfig."""
    from src.config import get_temporal_config

    full_cfg = load_config(CONFIG_PATH)
    t_cfg = get_temporal_config(full_cfg)
    assert t_cfg.window_seconds == 10
    assert t_cfg.sequence_length_windows == 10
    assert t_cfg.forecast_horizon_windows == 3
    assert t_cfg.forecast_offsets_seconds == (10, 20, 30)
    assert t_cfg.forecast_horizon_steps == (1, 2, 3)


def test_temporal_config_horizon_semantics_and_mismatch_rejection():
    """
    CRITICAL AUDIT INVARIANT (Issue 2):
    Verify that TemporalConfig exposes unambiguous forecast_horizon_steps (window steps)
    and forecast_offsets_seconds (physical time in seconds), enforcing offset = step * window_seconds.
    """
    from src.config import TemporalConfig

    # Valid config
    cfg = TemporalConfig(
        window_seconds=10,
        sequence_length_windows=10,
        forecast_horizon_windows=3,
        forecast_offsets_seconds=(10, 20, 30),
    )
    assert cfg.forecast_horizon_steps == (1, 2, 3)
    assert cfg.forecast_offsets_seconds == (10, 20, 30)
    assert cfg.forecast_horizons == (1, 2, 3)  # Backward compatible alias

    # Mismatched offsets (e.g. 25s instead of 20s for step 2) must be rejected
    with pytest.raises(ConfigError, match="forecast_offsets_seconds"):
        TemporalConfig(
            window_seconds=10,
            sequence_length_windows=10,
            forecast_horizon_windows=3,
            forecast_offsets_seconds=(10, 25, 30),
        )
