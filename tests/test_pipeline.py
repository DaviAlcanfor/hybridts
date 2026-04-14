from unittest.mock import MagicMock, patch

import numpy as np
import pandas as pd
import pytest

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


def test_predict_stores_state(sample_timeseries, fake_primary, fake_secondary):
    forecaster = HybridForecaster(fake_primary, fake_secondary)
    forecaster.fit(sample_timeseries)
    forecaster.predict(horizon=7)

    assert hasattr(forecaster, "forecast_")
    assert hasattr(forecaster, "forecast_plot_df_")
    assert hasattr(forecaster, "primary_plot_df_")


def test_predict_with_start_date(sample_timeseries, fake_primary, fake_secondary):
    forecaster = HybridForecaster(fake_primary, fake_secondary)
    forecaster.fit(sample_timeseries)
    start = pd.Timestamp("2024-01-01")
    result = forecaster.predict(horizon=7, start_date=start)

    assert result["data"].iloc[0] == pd.Timestamp("2024-01-02")
    assert result["data"].iloc[-1] == pd.Timestamp("2024-01-08")


def test_forecast_plot_df_has_only_ds_and_yhat(sample_timeseries, fake_primary, fake_secondary):
    forecaster = HybridForecaster(fake_primary, fake_secondary)
    forecaster.fit(sample_timeseries)
    forecaster.predict(horizon=7)

    assert list(forecaster.forecast_plot_df_.columns) == ["ds", "yhat"]
    assert list(forecaster.primary_plot_df_.columns) == ["ds", "yhat"]


def test_evaluate_stores_df_test_and_eval_forecast(sample_timeseries, fake_primary, fake_secondary):
    forecaster = HybridForecaster(fake_primary, fake_secondary, test_size=30)
    forecaster.evaluate(sample_timeseries)

    assert hasattr(forecaster, "df_test_")
    assert hasattr(forecaster, "eval_forecast_df_")
    assert len(forecaster.df_test_) == 30
    assert list(forecaster.eval_forecast_df_.columns) == ["ds", "yhat"]


def test_plot_forecast_raises_before_predict(sample_timeseries, fake_primary, fake_secondary):
    forecaster = HybridForecaster(fake_primary, fake_secondary)
    forecaster.fit(sample_timeseries)

    with pytest.raises(RuntimeError, match="No forecast found"):
        forecaster.plot_forecast(sample_timeseries)


def test_plot_evaluation_raises_before_evaluate(sample_timeseries, fake_primary, fake_secondary):
    forecaster = HybridForecaster(fake_primary, fake_secondary)

    with pytest.raises(RuntimeError, match="No evaluation results found"):
        forecaster.plot_evaluation()


def test_plot_forecast_calls_plot_function(sample_timeseries, fake_primary, fake_secondary):
    forecaster = HybridForecaster(fake_primary, fake_secondary)
    forecaster.fit(sample_timeseries)
    forecaster.predict(horizon=7)

    with patch("hybridts.plotting.plot_forecast", return_value=(MagicMock(), MagicMock())) as mock_plot:
        forecaster.plot_forecast(sample_timeseries)
        mock_plot.assert_called_once()


def test_plot_evaluation_calls_plot_function(sample_timeseries, fake_primary, fake_secondary):
    forecaster = HybridForecaster(fake_primary, fake_secondary, test_size=30)
    forecaster.evaluate(sample_timeseries)

    with patch("hybridts.plotting.plot_forecast", return_value=(MagicMock(), MagicMock())) as mock_plot:
        forecaster.plot_evaluation()
        mock_plot.assert_called_once()
