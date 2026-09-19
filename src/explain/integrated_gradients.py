"""
Path-Integrated Gradients Feature Attribution Engine.
Project: SIH26153 - AI-Based Network Attack Forecasting from Network Traffic Data

EXPLAINABILITY PROTOCOL & HONEST TERMINOLOGY:
    This module implements axiomatic Path-Integrated Gradients (Sundararajan et al., 2017)
    directly on PyTorch LSTM model weights across discrete forecast horizons (+10s, +20s, +30s).
    It evaluates gradient paths between a neutral baseline reference and the input sequence:

        Attribution_i = (x_i - x'_i) * integral_{alpha=0}^1 (dF(x' + alpha*(x - x')) / dx_i) d_alpha

    Completeness/Conservation Axiom:
        sum_i(Attribution_i) ~= F(x) - F(x') within numerical Riemann sum tolerance.

    DISCLOSURE:
    This is Integrated Gradients, NOT Kernel SHAP or Tree SHAP.
    The external Python 'shap' library is uninstalled in this environment;
    full SHAP sampling is NOT VALIDATED / NOT IMPLEMENTED.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, Iterator, List, Optional, Tuple, Union
import numpy as np
import torch

from src.config import get_temporal_config
from src.schemas.features import CANONICAL_MODEL_FEATURE_NAMES, validate_feature_names


@dataclass
class IntegratedGradientsResult:
    """
    Result container for Path-Integrated Gradients attribution.
    Supports attribute access as well as tuple-unpacking (attributions, delta_target, attribution_sum).
    """
    attributions: np.ndarray  # Shape: (10, 41)
    delta_target: float       # F(x) - F(x')
    attribution_sum: float    # sum(attributions)
    feature_importance: Dict[str, float]
    top_features: Dict[str, float]
    completeness_error: float

    def __iter__(self) -> Iterator[Any]:
        """Support unpacking: attr, delta, attr_sum = result"""
        yield self.attributions
        yield self.delta_target
        yield self.attribution_sum


class IntegratedGradientsExplainer:
    """
    Computes Path-Integrated Gradients attributions for the 41-feature temporal LSTM.
    Preserves exact (batch, 10, 41) shape, canonical feature order, and per-horizon attribution.
    """

    def __init__(
        self,
        model: torch.nn.Module,
        feature_names: Optional[List[str]] = None,
        m_steps: int = 50,
        steps: Optional[int] = None,
    ):
        self.model = model
        self.feature_names = feature_names or list(CANONICAL_MODEL_FEATURE_NAMES)
        validate_feature_names(self.feature_names, expected_order=CANONICAL_MODEL_FEATURE_NAMES)
        effective_steps = steps if steps is not None else m_steps
        self.m_steps = max(10, effective_steps)
        self.temporal_cfg = get_temporal_config()

    def attribute(
        self,
        X_scaled: Optional[Union[np.ndarray, torch.Tensor]] = None,
        horizon_idx: int = 0,
        baseline: Optional[Union[np.ndarray, torch.Tensor]] = None,
        x_input: Optional[Union[np.ndarray, torch.Tensor]] = None,
        feature_names: Optional[List[str]] = None,
    ) -> IntegratedGradientsResult:
        """
        Compute Integrated Gradients attributions for a single sequence and horizon.
        """
        input_data = x_input if x_input is not None else X_scaled
        if input_data is None:
            raise ValueError("Must provide either X_scaled or x_input")

        if isinstance(input_data, torch.Tensor):
            X_arr = input_data.detach().cpu().numpy().astype(np.float32)
        else:
            X_arr = np.asarray(input_data, dtype=np.float32)

        if X_arr.ndim == 2:
            X_arr = np.expand_dims(X_arr, axis=0)

        names = feature_names or self.feature_names
        if X_arr.shape != (1, self.temporal_cfg.history_length, len(names)):
            raise ValueError(
                f"Input shape mismatch: expected (1, {self.temporal_cfg.history_length}, {len(names)}), "
                f"got {X_arr.shape}"
            )

        if not (0 <= horizon_idx < self.temporal_cfg.forecast_horizon_windows):
            raise ValueError(
                f"horizon_idx must be in range [0, {self.temporal_cfg.forecast_horizon_windows - 1}], got {horizon_idx}"
            )

        self.model.eval()
        device = next(self.model.parameters()).device

        # Baseline array
        if baseline is None:
            baseline_arr = np.zeros_like(X_arr)
        elif isinstance(baseline, torch.Tensor):
            baseline_arr = baseline.detach().cpu().numpy().astype(np.float32)
            if baseline_arr.ndim == 2:
                baseline_arr = np.expand_dims(baseline_arr, axis=0)
        else:
            baseline_arr = np.asarray(baseline, dtype=np.float32)
            if baseline_arr.ndim == 2:
                baseline_arr = np.expand_dims(baseline_arr, axis=0)

        x_torch = torch.tensor(X_arr, dtype=torch.float32, device=device)
        b_torch = torch.tensor(baseline_arr, dtype=torch.float32, device=device)

        # Baseline and input target outputs
        with torch.no_grad():
            f_base_logits, _ = self.model(b_torch)
            f_input_logits, _ = self.model(x_torch)
            f_base_val = float(f_base_logits[0, horizon_idx].item())
            f_input_val = float(f_input_logits[0, horizon_idx].item())

        delta_target = f_input_val - f_base_val

        # Generate m_steps interpolated path points
        alphas = np.linspace(0.0, 1.0, num=self.m_steps, endpoint=True)
        accumulated_grads = np.zeros_like(X_arr[0], dtype=np.float32)

        for alpha in alphas:
            x_step = b_torch + alpha * (x_torch - b_torch)
            x_step = x_step.clone().detach().requires_grad_(True)

            risk_logits, _ = self.model(x_step)
            target = risk_logits[0, horizon_idx]

            self.model.zero_grad()
            target.backward()

            if x_step.grad is not None:
                accumulated_grads += x_step.grad.detach().cpu().numpy()[0]

        avg_grads = accumulated_grads / float(self.m_steps)
        input_diff = (X_arr[0] - baseline_arr[0])
        attributions = input_diff * avg_grads

        attribution_sum = float(np.sum(attributions))
        completeness_error = abs(delta_target - attribution_sum)

        # Feature importance aggregated over the 10 temporal windows (sum across time)
        feat_imp_vec = np.sum(attributions, axis=0)
        feature_importance = {name: float(feat_imp_vec[i]) for i, name in enumerate(names)}
        top_sorted = dict(sorted(feature_importance.items(), key=lambda item: abs(item[1]), reverse=True))

        return IntegratedGradientsResult(
            attributions=attributions,
            delta_target=delta_target,
            attribution_sum=attribution_sum,
            feature_importance=feature_importance,
            top_features=top_sorted,
            completeness_error=completeness_error,
        )

    def check_completeness(
        self,
        delta_target: float,
        attribution_sum: float,
        relative_tolerance: float = 0.20,
    ) -> bool:
        """
        Axiomatic Completeness check:
        Verifies whether sum(Attribution) ~= F(x) - F(x') within numerical Riemann tolerance.
        """
        abs_diff = abs(delta_target - attribution_sum)
        denom = max(abs(delta_target), abs(attribution_sum), 1e-6)
        rel_diff = abs_diff / denom
        return rel_diff <= relative_tolerance

    def attribute_all_horizons(
        self,
        X_scaled: Union[np.ndarray, torch.Tensor],
        baseline: Optional[Union[np.ndarray, torch.Tensor]] = None,
    ) -> np.ndarray:
        """
        Compute Integrated Gradients attributions across all configured forecast horizons.
        Returns:
            attributions: Array of shape (horizon_cnt, sequence_length, feature_count)
        """
        horizon_cnt = self.temporal_cfg.forecast_horizon_windows
        all_attr = []
        for h in range(horizon_cnt):
            res = self.attribute(X_scaled, horizon_idx=h, baseline=baseline)
            all_attr.append(res.attributions)
        return np.array(all_attr)


# Backward-compatible and convenience alias
IntegratedGradientsAttributor = IntegratedGradientsExplainer
