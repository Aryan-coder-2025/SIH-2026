"""
Backward-compatibility alias for src.inference (retaining original teammate file name).
"""
from src.inference import (
    ARTIFACT_DIR,
    ENTITY_COLUMN,
    HORIZON,
    SEQUENCE_LENGTH,
    TIMESTAMP_COLUMN,
    forecast,
    load_artifacts,
    prepare_input,
)

__all__ = [
    "ARTIFACT_DIR",
    "ENTITY_COLUMN",
    "HORIZON",
    "SEQUENCE_LENGTH",
    "TIMESTAMP_COLUMN",
    "forecast",
    "load_artifacts",
    "prepare_input",
]