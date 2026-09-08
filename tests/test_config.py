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