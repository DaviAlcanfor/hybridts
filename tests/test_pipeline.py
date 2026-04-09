import pytest
import pandas as pd
import numpy as np
from hybridts.pipeline import HybridForecaster


def test_predict_before_fit_raises(fake_primary, fake_secondary):
    forecaster = HybridForecaster(fake_primary, fake_secondary)
    
    with pytest.raises(RuntimeError, match="not fitted"):
        forecaster.predict(horizon=7)


def test_fit_returns_self(sample_timeseries, fake_primary, fake_secondary):
    forecaster = HybridForecaster(fake_primary, fake_secondary)
    result = forecaster.fit(sample_timeseries)
    
    assert result is forecaster


def test_fit_marks_as_fitted(sample_timeseries, fake_primary, fake_secondary):
    forecaster = HybridForecaster(fake_primary, fake_secondary)
    forecaster.fit(sample_timeseries)
    
    assert forecaster._is_fitted is True


def test_predict_output_shape(sample_timeseries, fake_primary, fake_secondary):
    forecaster = HybridForecaster(fake_primary, fake_secondary)
    forecaster.fit(sample_timeseries)
    result = forecaster.predict(horizon=14)
    
    assert len(result) == 14


def test_predict_output_columns(sample_timeseries, fake_primary, fake_secondary):
    forecaster = HybridForecaster(fake_primary, fake_secondary)
    forecaster.fit(sample_timeseries)
    result = forecaster.predict(horizon=7)
    
    assert {"data", "forecast_primary_base", "residual_correction", "forecast_final"}.issubset(result.columns)


def test_fit_with_holidays_none(sample_timeseries, fake_primary, fake_secondary):
    forecaster = HybridForecaster(fake_primary, fake_secondary)
    forecaster.fit(sample_timeseries, holidays=None)
    
    assert forecaster._is_fitted


def test_fit_with_features_none(sample_timeseries, fake_primary, fake_secondary):
    forecaster = HybridForecaster(fake_primary, fake_secondary)
    forecaster.fit(sample_timeseries, features=None)
    
    assert forecaster._is_fitted


def test_evaluate_returns_metrics_and_arrays(sample_timeseries, fake_primary, fake_secondary):
    forecaster = HybridForecaster(fake_primary, fake_secondary, test_size=30)
    metrics, y_true, y_pred = forecaster.evaluate(sample_timeseries)
    
    assert {"MAPE", "RMSE", "MAE"}.issubset(metrics.keys())
    assert len(y_true) == 30
    assert len(y_pred) == 30


def test_evaluate_stores_results_on_instance(sample_timeseries, fake_primary, fake_secondary):
    forecaster = HybridForecaster(fake_primary, fake_secondary, test_size=30)
    forecaster.evaluate(sample_timeseries)
    
    assert hasattr(forecaster, "metrics_")
    assert hasattr(forecaster, "y_true_")
    assert hasattr(forecaster, "y_pred_")


def test_evaluate_and_fit_returns_forecaster_and_metrics(sample_timeseries, fake_primary, fake_secondary):
    forecaster = HybridForecaster(fake_primary, fake_secondary, test_size=30)
    result_forecaster, metrics = forecaster.evaluate_and_fit(sample_timeseries)
    
    assert result_forecaster is forecaster
    assert "MAPE" in metrics
    assert forecaster._is_fitted
