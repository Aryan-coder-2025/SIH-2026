import numpy as np
import pandas as pd
import pytest

from src.baseline._validation import BaselineError
from src.baseline.logistic import LogisticRegressionBaseline
from src.baseline.random_forest import RandomForestBaseline
from src.eval.feature_schema import FeatureSchema, FeatureSchemaError


def test_schema_validates_array_column_count():
    schema = FeatureSchema(feature_names=["syn_count", "bytes_total", "ttl_mean"])
    good = np.zeros((5, 3))
    schema.validate_array(good)  # must not raise


def test_schema_rejects_wrong_column_count():
    schema = FeatureSchema(feature_names=["a", "b", "c"])
    bad = np.zeros((5, 2))
    with pytest.raises(FeatureSchemaError):
        schema.validate_array(bad)


def test_schema_validates_dataframe_columns_and_order():
    schema = FeatureSchema(feature_names=["syn_count", "bytes_total"])
    df = pd.DataFrame({"syn_count": [1, 2], "bytes_total": [100, 200]})
    schema.validate_dataframe_columns(df)  # must not raise


def test_schema_rejects_reordered_dataframe_columns():
    schema = FeatureSchema(feature_names=["syn_count", "bytes_total"])
    df = pd.DataFrame({"bytes_total": [100, 200], "syn_count": [1, 2]})  # swapped order
    with pytest.raises(FeatureSchemaError):
        schema.validate_dataframe_columns(df)


def test_schema_rejects_mismatched_dataframe_column_names():
    schema = FeatureSchema(feature_names=["syn_count", "bytes_total"])
    df = pd.DataFrame({"syn_count": [1, 2], "packet_count": [3, 4]})  # wrong name
    with pytest.raises(FeatureSchemaError):
        schema.validate_dataframe_columns(df)


def test_schema_rejects_empty_feature_names():
    with pytest.raises(FeatureSchemaError):
        FeatureSchema(feature_names=[])


def test_schema_rejects_duplicate_feature_names():
    with pytest.raises(FeatureSchemaError):
        FeatureSchema(feature_names=["a", "a", "b"])


# ----------------------------------------------------------------------
# Integration with baselines
# ----------------------------------------------------------------------
def test_logistic_baseline_rejects_mismatched_dataframe_schema():
    schema = FeatureSchema(feature_names=["syn_count", "bytes_total"])
    baseline = LogisticRegressionBaseline(random_state=42, feature_schema=schema)

    good_df = pd.DataFrame(
        {"syn_count": [1.0, 2.0, 3.0, 4.0], "bytes_total": [10.0, 20.0, 30.0, 40.0]}
    )
    y = [0, 1, 0, 1]
    baseline.fit(good_df, y)  # must not raise

    bad_df = pd.DataFrame(
        {"bytes_total": [10.0, 20.0], "syn_count": [1.0, 2.0]}
    )  # reordered columns
    with pytest.raises(BaselineError):
        baseline.predict_risk(bad_df)


def test_random_forest_baseline_rejects_mismatched_dataframe_schema():
    schema = FeatureSchema(feature_names=["a", "b"])
    baseline = RandomForestBaseline(random_state=42, feature_schema=schema)

    good_df = pd.DataFrame({"a": [1.0, 2.0, 3.0, 4.0], "b": [1.0, 0.0, 1.0, 0.0]})
    y = [0, 1, 0, 1]
    baseline.fit(good_df, y)

    wrong_name_df = pd.DataFrame({"a": [1.0, 2.0], "c": [1.0, 0.0]})
    with pytest.raises(BaselineError):
        baseline.predict_risk(wrong_name_df)


def test_baselines_without_schema_still_work_unchanged():
    """Backward compatibility: omitting feature_schema (the default)
    preserves the original plain-array behaviour."""
    X = np.array([[1.0, 2.0], [2.0, 1.0], [3.0, 3.0], [1.0, 4.0]])
    y = [0, 1, 0, 1]

    baseline = LogisticRegressionBaseline(random_state=42)
    baseline.fit(X, y)
    risk = baseline.predict_risk(X)
    assert risk.shape == (4,)
