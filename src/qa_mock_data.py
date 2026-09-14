import numpy as np
from typing import Dict, List, Tuple

def generate_mock_data(
    n_samples: int = 100,
    n_features: int = 5,
    n_windows: int = 10,
    random_seed: int = 42
) -> Tuple[np.ndarray, np.ndarray, List[str], Dict[int, np.ndarray]]:
    """
    Generate mock structured network data for the QA pipeline.
    Ensures timestamps are chronological and host-aware.

    Returns:
        features: (N, n_windows, n_features) numeric array
        timestamps: (N,) float array (strictly increasing)
        hosts: (N,) list of strings
        targets: Dict mapping horizon (10, 20, 30) to (N,) binary label arrays
    """
    rng = np.random.default_rng(random_seed)
    
    # Chronological timestamps
    hosts = ["host_A" if i % 2 == 0 else "host_B" for i in range(n_samples)]
    base_time = 1700000000.0
    timestamps = np.array([base_time + float(i * 10) for i in range(n_samples)])
    
    features = rng.random((n_samples, n_windows, n_features))
    
    targets = {
        10: rng.integers(0, 2, size=n_samples),
        20: rng.integers(0, 2, size=n_samples),
        30: rng.integers(0, 2, size=n_samples),
    }
    
    return features, timestamps, hosts, targets
