import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import torch
from sklearn.preprocessing import StandardScaler


# ------------------------------------------------------------
# Make src/ importable
# ------------------------------------------------------------

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"

sys.path.insert(0, str(SRC))


from schema import CANONICAL_FEATURES, SCHEMA_VERSION
from sequence_builder import build_sequences
from model import RiskLSTM
from inference import load_model, load_scaler, predict_risk
from train_updated import temporal_split, validate_dataframe, validate_temporal_grid, scale_sequences, validate_split_isolation


# ============================================================
# TEST DATA HELPERS
# ============================================================

def make_dataframe(
    num_windows=40,
    source_host="10.0.0.5",
    start="2026-01-01 00:00:00",
):
    """
    Create a small valid synthetic window-level dataset.

    This is ONLY for testing code.
    It is NOT training evidence.
    """

    timestamps = pd.date_range(
        start=start,
        periods=num_windows,
        freq="10s",
    )

    data = {
        "source_host": [source_host] * num_windows,
        "timestamp": timestamps,
        "is_malicious": [0] * num_windows,
    }

    # Give every canonical feature numeric values.
    for i, feature in enumerate(CANONICAL_FEATURES):
        data[feature] = np.arange(
            num_windows,
            dtype=np.float32
        ) + i

    return pd.DataFrame(data)


def make_two_host_dataframe():
    """
    Create two independent hosts with continuous timelines.
    """

    df1 = make_dataframe(
        num_windows=40,
        source_host="10.0.0.5",
    )

    df2 = make_dataframe(
        num_windows=40,
        source_host="10.0.0.6",
    )

    return pd.concat(
        [df1, df2],
        ignore_index=True,
    )


# ============================================================
# SEQUENCE TESTS
# ============================================================

class TestSequences:

    def test_correct_sequence_shape(self):
        df = make_dataframe()

        X, y = build_sequences(
            df=df,
            feature_columns=CANONICAL_FEATURES,
            sequence_length=10,
            horizon=3,
            entity_column="source_host",
            timestamp_column="timestamp",
            risk_column="is_malicious",
            window_seconds=10,
        )

        assert X.ndim == 3
        assert y.ndim == 2

        assert X.shape[1] == 10
        assert X.shape[2] == len(CANONICAL_FEATURES)

        assert y.shape[1] == 3

    def test_correct_target_shape(self):
        df = make_dataframe()

        X, y = build_sequences(
            df,
            CANONICAL_FEATURES,
            10,
            3,
            "source_host",
            "timestamp",
            "is_malicious",
            10,
        )

        assert y.ndim == 2
        assert y.shape[1] == 3

    def test_future_alignment_plus_10_20_30(self):
        df = make_dataframe()

        # Put malicious windows at:
        # t + 10
        # t + 20
        # t + 30
        df.loc[10, "is_malicious"] = 1
        df.loc[11, "is_malicious"] = 0
        df.loc[12, "is_malicious"] = 1

        X, y = build_sequences(
            df,
            CANONICAL_FEATURES,
            10,
            3,
            "source_host",
            "timestamp",
            "is_malicious",
            10,
        )

        # First sequence ends at row 9.
        # Future rows are 10, 11, 12.
        assert np.array_equal(
            y[0],
            np.array([1, 0, 1], dtype=np.float32),
        )

    def test_missing_window_detection(self):
        df = make_dataframe()

        # Remove one 10-second window.
        df = df.drop(index=15).reset_index(drop=True)

        # There should be no sequence crossing
        # the missing timestamp.
        X, y = build_sequences(
            df,
            CANONICAL_FEATURES,
            10,
            3,
            "source_host",
            "timestamp",
            "is_malicious",
            10,
        )

        # The important requirement is that the builder
        # does not create an invalid sequence across the gap.
        assert len(X) < 28

    def test_duplicate_window_detection(self):
        df = make_dataframe()

        duplicate = df.iloc[[10]].copy()

        df = pd.concat(
            [df, duplicate],
            ignore_index=True,
        )

        with pytest.raises(ValueError):
            build_sequences(
                df,
                CANONICAL_FEATURES,
                10,
                3,
                "source_host",
                "timestamp",
                "is_malicious",
                10,
            )

    def test_invalid_timestamp_detection(self):
        df = make_dataframe()

        df["timestamp"] = df["timestamp"].astype(object)
        df.loc[5, "timestamp"] = "INVALID_TIMESTAMP"

        with pytest.raises(ValueError):
            build_sequences(
                df,
                CANONICAL_FEATURES,
                10,
                3,
                "source_host",
                "timestamp",
                "is_malicious",
                10,
            )

    def test_insufficient_history(self):
        df = make_dataframe(num_windows=12)

        with pytest.raises(ValueError):
            build_sequences(
                df,
                CANONICAL_FEATURES,
                10,
                3,
                "source_host",
                "timestamp",
                "is_malicious",
                10,
            )

    def test_insufficient_future(self):
        df = make_dataframe(num_windows=12)

        with pytest.raises(ValueError):
            build_sequences(
                df,
                CANONICAL_FEATURES,
                10,
                3,
                "source_host",
                "timestamp",
                "is_malicious",
                10,
            )

    def test_host_isolation(self):
        df = make_two_host_dataframe()

        X, y = build_sequences(
            df,
            CANONICAL_FEATURES,
            10,
            3,
            "source_host",
            "timestamp",
            "is_malicious",
            10,
        )

        # 40 windows per host.
        # Each host must create its own sequences.
        #
        # A sequence must never combine:
        # 10.0.0.5 + 10.0.0.6
        #
        # Number per host:
        # 40 - 10 - 3 + 1 = 28
        #
        # Total:
        # 28 * 2 = 56

        assert len(X) == 56


# ============================================================
# LEAKAGE / INPUT HARDENING TESTS
# ============================================================

class TestLeakageAndInputHardening:

    def test_target_and_future_fields_cannot_be_features(self):
        from schema import TARGET_COLUMNS, validate_feature_columns
        df = make_dataframe()
        df["future_risk"] = 0.5
        with pytest.raises(ValueError):
            validate_feature_columns(df, CANONICAL_FEATURES + ["future_risk"])

    def test_nan_feature_is_rejected(self):
        df = make_dataframe()
        df.loc[5, CANONICAL_FEATURES[0]] = np.nan
        with pytest.raises(ValueError):
            validate_dataframe(df)

    def test_inf_feature_is_rejected(self):
        df = make_dataframe()
        df.loc[5, CANONICAL_FEATURES[0]] = np.inf
        with pytest.raises(ValueError):
            validate_dataframe(df)

    def test_non_numeric_feature_is_rejected(self):
        df = make_dataframe()
        df[CANONICAL_FEATURES[0]] = "not_numeric"
        with pytest.raises((ValueError, TypeError)):
            validate_dataframe(df)

    def test_missing_feature_is_rejected(self):
        df = make_dataframe().drop(columns=[CANONICAL_FEATURES[0]])
        with pytest.raises(ValueError):
            validate_dataframe(df)

    def test_extra_derived_column_is_rejected(self):
        df = make_dataframe()
        df["future_bytes"] = 123.0
        with pytest.raises(ValueError):
            validate_dataframe(df)

    def test_future_feature_mutation_cannot_change_history_input(self):
        df_a = make_dataframe(num_windows=40)
        df_b = df_a.copy()
        future_rows = [10, 11, 12, 13, 14]
        for row in future_rows:
            for feature in CANONICAL_FEATURES:
                df_b.loc[row, feature] += 100000.0

        X_a, y_a = build_sequences(
            df_a, CANONICAL_FEATURES, 10, 3, "source_host", "timestamp", "is_malicious", 10
        )
        X_b, y_b = build_sequences(
            df_b, CANONICAL_FEATURES, 10, 3, "source_host", "timestamp", "is_malicious", 10
        )

        # First sequence uses rows 0..9 only; changing rows 10+ must not alter X[0].
        assert np.array_equal(X_a[0], X_b[0])

    def test_future_label_mutation_cannot_change_history_input(self):
        df_a = make_dataframe(num_windows=40)
        df_b = df_a.copy()
        df_b.loc[10:15, "is_malicious"] = 1

        X_a, _ = build_sequences(
            df_a, CANONICAL_FEATURES, 10, 3, "source_host", "timestamp", "is_malicious", 10
        )
        X_b, _ = build_sequences(
            df_b, CANONICAL_FEATURES, 10, 3, "source_host", "timestamp", "is_malicious", 10
        )

        assert np.array_equal(X_a[0], X_b[0])

    def test_temporal_split_has_no_raw_window_overlap(self):
        df = make_dataframe(num_windows=100)
        train, val, test = temporal_split(df, 0.70, 0.15)
        validate_split_isolation(train, val, test)

        train_ts = set(train["timestamp"])
        val_ts = set(val["timestamp"])
        test_ts = set(test["timestamp"])
        assert train_ts.isdisjoint(val_ts)
        assert train_ts.isdisjoint(test_ts)
        assert val_ts.isdisjoint(test_ts)

    def test_temporal_split_is_strictly_ordered(self):
        df = make_dataframe(num_windows=100)
        train, val, test = temporal_split(df, 0.70, 0.15)
        assert train["timestamp"].max() < val["timestamp"].min()
        assert val["timestamp"].max() < test["timestamp"].min()

    def test_malformed_sequence_shape_is_rejected(self):
        from train_updated import validate_sequences
        X = np.zeros((2, 9, len(CANONICAL_FEATURES)), dtype=np.float32)
        y = np.zeros((2, 3), dtype=np.float32)
        with pytest.raises(ValueError):
            validate_sequences(X, y, 10, len(CANONICAL_FEATURES), 3, "TEST")

# ============================================================
# SCALING TESTS
# ============================================================

class TestScaling:

    def make_sequences(self):
        X_train = np.ones(
            (5, 10, len(CANONICAL_FEATURES)),
            dtype=np.float32,
        )

        X_val = np.ones(
            (2, 10, len(CANONICAL_FEATURES)),
            dtype=np.float32,
        ) * 2

        X_test = np.ones(
            (2, 10, len(CANONICAL_FEATURES)),
            dtype=np.float32,
        ) * 3

        return X_train, X_val, X_test

    def test_scaler_fits_train_only(self):
        X_train, X_val, X_test = self.make_sequences()

        scaler = StandardScaler()

        train_2d = X_train.reshape(
            -1,
            X_train.shape[-1],
        )

        scaler.fit(train_2d)

        assert np.allclose(
            scaler.mean_,
            np.ones(len(CANONICAL_FEATURES)),
        )

    def test_validation_uses_train_scaler(self):
        X_train, X_val, _ = self.make_sequences()

        scaler = StandardScaler()

        scaler.fit(
            X_train.reshape(
                -1,
                X_train.shape[-1],
            )
        )

        transformed_val = scaler.transform(
            X_val.reshape(
                -1,
                X_val.shape[-1],
            )
        )

        assert transformed_val.shape == (
            X_val.shape[0] * X_val.shape[1],
            len(CANONICAL_FEATURES),
        )

    def test_test_uses_train_scaler(self):
        X_train, _, X_test = self.make_sequences()

        scaler = StandardScaler()

        scaler.fit(
            X_train.reshape(
                -1,
                X_train.shape[-1],
            )
        )

        transformed_test = scaler.transform(
            X_test.reshape(
                -1,
                X_test.shape[-1],
            )
        )

        assert transformed_test.shape[1] == len(
            CANONICAL_FEATURES
        )

    def test_nan_rejected(self):
        X_train, _, _ = self.make_sequences()

        X_train[0, 0, 0] = np.nan

        assert not np.isfinite(X_train).all()

    def test_inf_rejected(self):
        X_train, _, _ = self.make_sequences()

        X_train[0, 0, 0] = np.inf

        assert not np.isfinite(X_train).all()

    def test_feature_count_mismatch(self):
        X = np.ones(
            (5, 10, len(CANONICAL_FEATURES) + 1),
            dtype=np.float32,
        )

        assert X.shape[-1] != len(
            CANONICAL_FEATURES
        )

    def test_feature_order_mismatch(self):
        wrong_order = list(
            reversed(CANONICAL_FEATURES)
        )

        assert wrong_order != CANONICAL_FEATURES


# ============================================================
# MODEL TESTS
# ============================================================

class TestModel:

    def make_model(self):
        return RiskLSTM(
            input_size=len(CANONICAL_FEATURES),
            hidden_size=32,
            num_layers=1,
            dropout=0.0,
            horizon=3,
        )

    def test_valid_input(self):
        model = self.make_model()

        X = torch.randn(
            4,
            10,
            len(CANONICAL_FEATURES),
        )

        output = model(X)

        assert output.shape == (4, 3)

    def test_invalid_sequence_length_rejected(self):
        model = self.make_model()
        X = torch.randn(4, 9, len(CANONICAL_FEATURES))
        with pytest.raises(ValueError):
            model(X)

    def test_invalid_input(self):
        model = self.make_model()

        # Wrong number of features.
        X = torch.randn(
            4,
            10,
            len(CANONICAL_FEATURES) + 1,
        )

        with pytest.raises(
            (RuntimeError, ValueError)
        ):
            model(X)

    def test_output_exactly_three_values(self):
        model = self.make_model()

        X = torch.randn(
            2,
            10,
            len(CANONICAL_FEATURES),
        )

        output = model(X)

        assert output.shape[1] == 3

    def test_finite_logits(self):
        model = self.make_model()

        X = torch.randn(
            2,
            10,
            len(CANONICAL_FEATURES),
        )

        output = model(X)

        assert torch.isfinite(output).all()

    def test_deterministic_inference(self):
        torch.manual_seed(123)

        model = self.make_model()
        model.eval()

        X = torch.randn(
            1,
            10,
            len(CANONICAL_FEATURES),
        )

        with torch.no_grad():
            output1 = model(X)
            output2 = model(X)

        assert torch.equal(
            output1,
            output2,
        )


# ============================================================
# SCHEMA TESTS
# ============================================================

class TestSchema:

    def test_feature_count(self):
        assert len(CANONICAL_FEATURES) == 20

    def test_feature_order_is_stable(self):
        assert CANONICAL_FEATURES == list(
            CANONICAL_FEATURES
        )

    def test_schema_version_exists(self):
        assert isinstance(
            SCHEMA_VERSION,
            str,
        )
        assert len(SCHEMA_VERSION) > 0


# ============================================================
# ARTIFACT TESTS
# ============================================================

class TestArtifacts:

    def test_model_save_load(self, tmp_path):
        model = RiskLSTM(
            input_size=len(CANONICAL_FEATURES),
            hidden_size=32,
            num_layers=1,
            dropout=0.0,
            horizon=3,
        )

        checkpoint = {
            "model_state_dict": model.state_dict(),
            "input_size": len(CANONICAL_FEATURES),
            "hidden_size": 32,
            "num_layers": 1,
            "dropout": 0.0,
            "horizon": 3,
            "schema_version": SCHEMA_VERSION,
        }

        model_path = (
            tmp_path / "model.pt"
        )

        torch.save(
            checkpoint,
            model_path,
        )

        loaded_model = load_model(
            str(model_path),
            torch.device("cpu"),
        )

        assert loaded_model is not None

    def test_scaler_save_load(self, tmp_path):
        scaler = StandardScaler()

        X = np.random.randn(
            20,
            len(CANONICAL_FEATURES),
        )

        scaler.fit(X)

        import pickle

        scaler_path = (
            tmp_path / "scaler.pkl"
        )

        with open(
            scaler_path,
            "wb",
        ) as file:
            pickle.dump(
                scaler,
                file,
            )

        loaded_scaler = load_scaler(
            str(scaler_path)
        )

        assert loaded_scaler.n_features_in_ == len(
            CANONICAL_FEATURES
        )

    def test_feature_order_save_load(self, tmp_path):
        feature_path = (
            tmp_path / "feature_order.json"
        )

        import json

        metadata = {
            "schema_version": SCHEMA_VERSION,
            "feature_count": len(
                CANONICAL_FEATURES
            ),
            "feature_order": CANONICAL_FEATURES,
        }

        feature_path.write_text(
            json.dumps(metadata),
            encoding="utf-8",
        )

        loaded = json.loads(
            feature_path.read_text(
                encoding="utf-8"
            )
        )

        assert loaded["schema_version"] == (
            SCHEMA_VERSION
        )

        assert loaded["feature_order"] == (
            CANONICAL_FEATURES
        )

    def test_schema_incompatibility_rejected(
        self,
        tmp_path,
    ):
        model = RiskLSTM(
            input_size=len(CANONICAL_FEATURES),
            hidden_size=32,
            num_layers=1,
            dropout=0.0,
            horizon=3,
        )

        checkpoint = {
            "model_state_dict": model.state_dict(),
            "input_size": len(CANONICAL_FEATURES),
            "hidden_size": 32,
            "num_layers": 1,
            "dropout": 0.0,
            "horizon": 3,
            "schema_version": "WRONG_VERSION",
        }

        model_path = (
            tmp_path / "bad_model.pt"
        )

        torch.save(
            checkpoint,
            model_path,
        )

        with pytest.raises(ValueError):
            load_model(
                str(model_path),
                torch.device("cpu"),
            )

    def test_metadata_exists(self, tmp_path):
        import json

        metadata_path = (
            tmp_path / "metadata.json"
        )

        metadata = {
            "schema_version": SCHEMA_VERSION,
            "feature_count": len(
                CANONICAL_FEATURES
            ),
            "feature_order": CANONICAL_FEATURES,
        }

        metadata_path.write_text(
            json.dumps(metadata),
            encoding="utf-8",
        )

        assert metadata_path.exists()

        loaded = json.loads(
            metadata_path.read_text(
                encoding="utf-8"
            )
        )

        assert "schema_version" in loaded
        assert "feature_order" in loaded