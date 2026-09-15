import datetime
import numpy as np
import pandas as pd
import pytest

from src.eval.forecast import RawPrediction
from src.integration_adapter import convert_lstm_forecast_to_raw_predictions

def test_convert_lstm_forecast_python_datetime():
    dt = datetime.datetime(2026, 9, 15, 12, 0, 0, tzinfo=datetime.timezone.utc)
    risk_array = np.array([0.1, 0.5, 0.9])
    
    predictions = convert_lstm_forecast_to_raw_predictions(
        source_host="hostA",
        prediction_timestamp=dt,
        risk_array=risk_array
    )
    
    assert len(predictions) == 3
    
    assert predictions[0].source_host == "hostA"
    assert predictions[0].prediction_time == dt.timestamp()
    assert predictions[0].forecast_horizon == 10
    assert predictions[0].risk == 0.1
    
    assert predictions[1].forecast_horizon == 20
    assert predictions[1].risk == 0.5
    
    assert predictions[2].forecast_horizon == 30
    assert predictions[2].risk == 0.9

def test_convert_lstm_forecast_pandas_timestamp():
    dt = pd.Timestamp("2026-09-15 12:00:00+00:00")
    risk_array = np.array([0.2, 0.4, 0.6])
    
    predictions = convert_lstm_forecast_to_raw_predictions(
        source_host="hostB",
        prediction_timestamp=dt,
        risk_array=risk_array
    )
    
    assert len(predictions) == 3
    assert predictions[0].prediction_time == dt.timestamp()
    assert predictions[0].risk == 0.2

def test_convert_lstm_forecast_invalid_shape():
    dt = datetime.datetime.now(datetime.timezone.utc)
    risk_array = np.array([0.1, 0.5])  # Only 2 values
    
    with pytest.raises(ValueError, match=r"Expected risk_array to have shape \(3,\)"):
        convert_lstm_forecast_to_raw_predictions(
            source_host="hostA",
            prediction_timestamp=dt,
            risk_array=risk_array
        )

def test_convert_lstm_forecast_invalid_type():
    risk_array = np.array([0.1, 0.5, 0.9])
    with pytest.raises(TypeError, match="prediction_timestamp must be"):
        convert_lstm_forecast_to_raw_predictions(
            source_host="hostA",
            prediction_timestamp=1700000000.0, # Float instead of datetime
            risk_array=risk_array
        )
