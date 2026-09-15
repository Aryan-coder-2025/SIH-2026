import datetime
from typing import Sequence, List
import numpy as np

from src.eval.forecast import RawPrediction
from src.eval.records import SUPPORTED_HORIZONS


def convert_lstm_forecast_to_raw_predictions(
    source_host: str,
    prediction_timestamp: datetime.datetime,
    risk_array: np.ndarray,
    horizons: Sequence[int] = SUPPORTED_HORIZONS,
) -> List[RawPrediction]:
    """
    Adapter to convert the integrated LSTM model's forecast output into
    RawPrediction records required by the evaluation framework.
    
    Args:
        source_host: Canonical host identifier.
        prediction_timestamp: The time the forecast was issued (Pandas or Python datetime).
        risk_array: NumPy array of risk probabilities. Must match the length of horizons.
        horizons: The forecast horizons in seconds (defaults to SUPPORTED_HORIZONS).
        
    Returns:
        A list of RawPrediction objects, one for each horizon.
    """
    if not isinstance(prediction_timestamp, datetime.datetime):
        raise TypeError(
            f"prediction_timestamp must be a datetime.datetime (or pd.Timestamp), "
            f"got {type(prediction_timestamp).__name__}"
        )

    if risk_array.shape != (len(horizons),):
        raise ValueError(
            f"Expected risk_array to have shape ({len(horizons)},), "
            f"got {risk_array.shape}"
        )

    pred_time_float = prediction_timestamp.timestamp()

    records = []
    for i, horizon in enumerate(horizons):
        records.append(
            RawPrediction(
                source_host=source_host,
                prediction_time=pred_time_float,
                forecast_horizon=horizon,
                risk=float(risk_array[i]),
            )
        )

    return records
