import json
import numpy as np
from src.qa_mock_data import generate_mock_data
from src.qa_pipeline import run_qa_pipeline

def test_mock_data_structure():
    """Verify the expected structure/shape of generated data."""
    features, timestamps, hosts, targets = generate_mock_data(
        n_samples=50, n_features=3, n_windows=10
    )
    assert features.shape == (50, 10, 3)
    assert timestamps.shape == (50,)
    assert len(hosts) == 50
    assert list(targets.keys()) == [10, 20, 30]
    for h in [10, 20, 30]:
        assert targets[h].shape == (50,)
        
def test_mock_data_deterministic():
    """Verify that deterministic mock data generation works with seeds."""
    f1, t1, h1, tg1 = generate_mock_data(random_seed=123)
    f2, t2, h2, tg2 = generate_mock_data(random_seed=123)
    
    np.testing.assert_array_equal(f1, f2)
    np.testing.assert_array_equal(t1, t2)
    assert h1 == h2
    for h in [10, 20, 30]:
        np.testing.assert_array_equal(tg1[h], tg2[h])

def test_mock_data_chronological_ordering():
    """Verify chronological/host ordering and no obvious leakage."""
    features, timestamps, hosts, targets = generate_mock_data(n_samples=100)
    
    # Timestamps should be strictly increasing overall
    diffs = np.diff(timestamps)
    assert np.all(diffs > 0), "Timestamps are not strictly chronological"
    
    # Hosts should alternate
    assert hosts[0] == "host_A"
    assert hosts[1] == "host_B"
    assert hosts[2] == "host_A"

def test_pipeline_end_to_end():
    """Verify successful end-to-end pipeline execution."""
    results = run_qa_pipeline()
    
    assert isinstance(results, list)
    assert len(results) == 3

def test_pipeline_horizons_represented():
    """Verify forecast horizons 10/20/30 are represented."""
    results = run_qa_pipeline()
    
    horizons = [r["forecast_horizon"] for r in results]
    assert sorted(horizons) == [10, 20, 30]

def test_pipeline_json_serializable():
    """Verify final result is JSON serializable."""
    results = run_qa_pipeline()
    
    # Should not raise exception
    json_str = json.dumps(results)
    assert isinstance(json_str, str)
    assert len(json_str) > 0
    
    # Reload and verify
    reloaded = json.loads(json_str)
    assert len(reloaded) == 3
    assert reloaded[0]["model_name"] == "random_forest"
