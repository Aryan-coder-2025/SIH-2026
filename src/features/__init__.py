"""
Feature extraction, time-window aggregation, labeling, and sequence creation.
"""
from src.features.labels import classify_label
from src.features.flow_features import aggregate_flow_features
from src.features.windowing import add_time_columns, add_window_labels
from src.features.sequence_builder import build_sequences, FEATURE_COLUMNS, HISTORY_WINDOWS
from src.features.generate_synthetic_data import generate as generate_synthetic_data
from src.features.build_dataset import build_dataset

__all__ = [
    "classify_label",
    "aggregate_flow_features",
    "FEATURE_COLUMNS",
    "add_time_columns",
    "add_window_labels",
    "build_sequences",
    "HISTORY_WINDOWS",
    "generate_synthetic_data",
    "build_dataset",
]
