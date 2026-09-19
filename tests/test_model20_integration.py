import numpy as np
import pandas as pd
import pytest
from src.schemas.model20 import (
    CANONICAL_MODEL_20_FEATURES,
    validate_model_20_features,
    validate_feature_order,
    extract_model_20_features,
    get_schema_info,
)


def _valid_model_20_row():
    return {
        "src_ip": "10.0.0.1",
        "window_id": 0,
        "timestamp": 1000.0,
        "flow_count": 2.0,
        "bytes_total": 1500.0,
        "bytes_mean": 750.0,
        "bytes_std": 50.0,
        "packets_total": 10.0,
        "packets_mean": 5.0,
        "duration_mean": 2.5,
        "duration_std": 0.5,
        "iat_mean": 0.25,
        "iat_std": 0.05,
        "iat_max": 0.8,
        "syn_count": 2.0,
        "ack_count": 8.0,
        "fin_count": 1.0,
        "rst_count": 0.0,
        "psh_count": 3.0,
        "urg_count": 0.0,
        "tcp_flow_ratio": 1.0,
        "udp_flow_ratio": 0.0,
        "bidirectional_ratio": 0.5,
    }


def test_schema_info():
    info = get_schema_info()
    assert info["feature_count"] == 20
    assert len(info["feature_order"]) == 20
    assert info["schema_version"] == "2.0"
    assert "is_malicious" in info["forbidden_columns"]


def test_valid_model_20_extraction():
    df = pd.DataFrame([_valid_model_20_row()])
    validate_model_20_features(df)
    extracted = extract_model_20_features(df)
    assert len(extracted) == 1
    assert list(extracted.columns[-20:]) == CANONICAL_MODEL_20_FEATURES


def test_missing_feature_rejection():
    row = _valid_model_20_row()
    del row["syn_count"]
    df = pd.DataFrame([row])
    with pytest.raises(ValueError, match="Missing canonical model-20 features"):
        validate_model_20_features(df)


def test_nan_or_inf_rejection():
    row = _valid_model_20_row()
    row["bytes_total"] = np.nan
    df = pd.DataFrame([row])
    with pytest.raises(ValueError, match="contains NaN or infinite values"):
        validate_model_20_features(df)

    row["bytes_total"] = np.inf
    df = pd.DataFrame([row])
    with pytest.raises(ValueError, match="contains NaN or infinite values"):
        validate_model_20_features(df)


def test_negative_counts_rejection():
    row = _valid_model_20_row()
    row["flow_count"] = -1.0
    df = pd.DataFrame([row])
    with pytest.raises(ValueError, match="contains negative values"):
        validate_model_20_features(df)


def test_feature_order_verification():
    validate_feature_order(list(CANONICAL_MODEL_20_FEATURES))

    reversed_order = list(reversed(CANONICAL_MODEL_20_FEATURES))
    with pytest.raises(ValueError, match="Feature order mismatch"):
        validate_feature_order(reversed_order)
