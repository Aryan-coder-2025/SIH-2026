"""
Canonical Schema Module (Compatibility Adapter).
"""
from src.schemas.model20 import (
    CANONICAL_MODEL_20_FEATURES as CANONICAL_FEATURES,
    SCHEMA_VERSION,
    METADATA_COLUMNS,
    FORBIDDEN_COLUMNS as TARGET_COLUMNS,
    validate_model_20_features,
    validate_feature_order,
    extract_model_20_features as get_features,
    get_schema_info,
)

__all__ = [
    "CANONICAL_FEATURES",
    "SCHEMA_VERSION",
    "METADATA_COLUMNS",
    "TARGET_COLUMNS",
    "validate_model_20_features",
    "validate_feature_order",
    "get_features",
    "get_schema_info",
]
