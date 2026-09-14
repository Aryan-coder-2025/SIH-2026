import os
import json
import pytest
from src.dashboard.app import load_artifacts

def test_load_artifacts_with_valid_file(tmp_path):
    # Create a temporary valid artifact
    valid_data = [{"experiment_name": "test", "forecast_horizon": 10}]
    file_path = tmp_path / "valid_artifact.json"
    with open(file_path, "w") as f:
        json.dump(valid_data, f)
    
    loaded = load_artifacts(str(file_path))
    assert isinstance(loaded, list)
    assert len(loaded) == 1
    assert loaded[0]["experiment_name"] == "test"

def test_load_artifacts_with_missing_file():
    loaded = load_artifacts("non_existent_file.json")
    assert isinstance(loaded, list)
    assert len(loaded) == 0

def test_load_artifacts_with_malformed_json(tmp_path):
    file_path = tmp_path / "malformed.json"
    with open(file_path, "w") as f:
        f.write("{malformed json")
    
    loaded = load_artifacts(str(file_path))
    assert isinstance(loaded, list)
    assert len(loaded) == 0
