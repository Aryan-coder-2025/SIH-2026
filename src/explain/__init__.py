"""
Explainability & Feature Attribution package.

DISCLOSURE & TERMINOLOGY:
- Integrated Gradients attribution is implemented (IntegratedGradientsExplainer).
- SHAP computation is not implemented/validated in the current environment due to shap package absence.
"""

from src.explain.integrated_gradients import (
    IntegratedGradientsAttributor,
    IntegratedGradientsExplainer,
)
from src.explain.shap_explain import (
    ExplainedFeature,
    FEATURE_DISPLAY_MAP,
    HorizonExplanation,
    ShapExplainer,
)

__all__ = [
    "ExplainedFeature",
    "FEATURE_DISPLAY_MAP",
    "HorizonExplanation",
    "IntegratedGradientsAttributor",
    "IntegratedGradientsExplainer",
    "ShapExplainer",
]

