from __future__ import annotations

from typing import Sequence
import numpy as np


class PersistenceBaseline:
    """
    Persistence baseline for attack forecasting.
    Predicts that future risk will equal the current (most recently observed) state.

    Invariant: Uses strictly the last observation in history X[:, -1] or last known state.
    Has zero access to future information.
    """

    def __init__(self, forecast_horizons: Sequence[int] = (1, 2, 3)) -> None:
        if not forecast_horizons:
            raise ValueError("forecast_horizons cannot be empty")
        self.forecast_horizons = tuple(forecast_horizons)

    def predict(self, current_states: np.ndarray) -> np.ndarray:
        """
        Predict future risk for each sample across configured horizons.
        Parameters:
            current_states: 1D array of shape (N,) representing current state at time t.
        Returns:
            predictions: 2D array of shape (N, len(forecast_horizons)).
        """
        states = np.asarray(current_states, dtype=float)
        if states.ndim != 1:
            raise ValueError("current_states must be a 1D array")

        num_horizons = len(self.forecast_horizons)
        # Repeat current state across all future horizons
        return np.repeat(states[:, np.newaxis], num_horizons, axis=1)
