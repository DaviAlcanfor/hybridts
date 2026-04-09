import numpy as np
import pytest
from hybridts.src.metrics.forecast import ForecastMetrics


@pytest.fixture
def perfect_forecast():
    y = np.array([100.0, 200.0, 300.0, 400.0, 500.0])
    return ForecastMetrics(y_true=y, y_pred=y)


@pytest.fixture
def sample_forecast():
    y_true = np.array([100.0, 200.0, 300.0, 400.0, 500.0])
    y_pred = np.array([110.0, 190.0, 310.0, 380.0, 520.0])
    return ForecastMetrics(y_true=y_true, y_pred=y_pred)


# ------------------------------------------------------------------
# Sanity: perfect forecast
# ------------------------------------------------------------------

def test_perfect_forecast_mae(perfect_forecast):
    assert perfect_forecast.mae == 0.0


def test_perfect_forecast_mse(perfect_forecast):
    assert perfect_forecast.mse == 0.0


def test_perfect_forecast_rmse(perfect_forecast):
    assert perfect_forecast.rmse == 0.0


def test_perfect_forecast_mape(perfect_forecast):
    assert perfect_forecast.mape == 0.0


def test_perfect_forecast_smape(perfect_forecast):
    assert perfect_forecast.smape == 0.0


def test_perfect_forecast_r_squared(perfect_forecast):
    assert perfect_forecast.r_squared == pytest.approx(1.0)


def test_perfect_forecast_bias(perfect_forecast):
    assert perfect_forecast.bias == 0.0


# ------------------------------------------------------------------
# Metric values
# ------------------------------------------------------------------

def test_mae_value(sample_forecast):
    expected = np.mean(np.abs(np.array([100, 200, 300, 400, 500]) - np.array([110, 190, 310, 380, 520])))
    assert sample_forecast.mae == pytest.approx(expected)


def test_mse_value(sample_forecast):
    expected = np.mean((np.array([100, 200, 300, 400, 500]) - np.array([110, 190, 310, 380, 520])) ** 2)
    assert sample_forecast.mse == pytest.approx(expected)


def test_rmse_is_sqrt_of_mse(sample_forecast):
    assert sample_forecast.rmse == pytest.approx(np.sqrt(sample_forecast.mse))


def test_bias_sign_underestimation():
    y_true = np.array([100.0, 200.0, 300.0])
    y_pred = np.array([90.0, 180.0, 270.0])   # always below → positive bias
    report = ForecastMetrics(y_true, y_pred)
    assert report.bias > 0


def test_bias_sign_overestimation():
    y_true = np.array([100.0, 200.0, 300.0])
    y_pred = np.array([110.0, 220.0, 330.0])  # always above → negative bias
    report = ForecastMetrics(y_true, y_pred)
    assert report.bias < 0


# ------------------------------------------------------------------
# Edge cases
# ------------------------------------------------------------------

def test_mape_ignores_zero_y_true():
    y_true = np.array([0.0, 100.0, 200.0])
    y_pred = np.array([10.0, 110.0, 210.0])
    report = ForecastMetrics(y_true, y_pred)
    assert np.isfinite(report.mape)


def test_smape_ignores_zero_denominator():
    y_true = np.array([0.0, 100.0, 200.0])
    y_pred = np.array([0.0, 110.0, 210.0])
    report = ForecastMetrics(y_true, y_pred)
    assert np.isfinite(report.smape)


# ------------------------------------------------------------------
# all_metrics / summary
# ------------------------------------------------------------------

def test_all_metrics_keys(sample_forecast):
    keys = sample_forecast.all_metrics().keys()
    assert {"MAE", "MSE", "RMSE", "MAPE", "sMAPE", "R-squared", "Bias"} == set(keys)


def test_all_metrics_values_match_attributes(sample_forecast):
    m = sample_forecast.all_metrics()
    assert m["MAE"]  == sample_forecast.mae
    assert m["RMSE"] == sample_forecast.rmse
    assert m["MAPE"] == sample_forecast.mape
    assert m["Bias"] == sample_forecast.bias


def test_summary_contains_metric_names(sample_forecast):
    s = sample_forecast.summary()
    for key in ["MAE", "MSE", "RMSE", "MAPE", "sMAPE", "R-squared", "Bias"]:
        assert key in s


def test_methods_return_same_as_attributes(sample_forecast):
    assert sample_forecast.mean_absolute_error() == sample_forecast.mae
    assert sample_forecast.mean_squared_error() == sample_forecast.mse
    assert sample_forecast.root_mean_squared_error() == sample_forecast.rmse
    assert sample_forecast.mean_absolute_percentage_error() == sample_forecast.mape
    assert sample_forecast.symmetric_mean_absolute_percentage_error() == sample_forecast.smape
