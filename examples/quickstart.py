"""
Quickstart Example - HybridTS Forecasting Library
==================================================

Demonstrates the main usage modes for HybridTS.
"""

import numpy as np
import pandas as pd


def generate_sample_data(start_date="2023-01-01", periods=400):
    """Generate synthetic time series data for testing."""
    dates = pd.date_range(start=start_date, periods=periods, freq="D")
    trend = np.linspace(10000, 15000, periods)
    seasonality = 2000 * np.sin(2 * np.pi * np.arange(periods) / 7)
    noise = np.random.normal(0, 500, periods)
    values = np.maximum(trend + seasonality + noise, 0)
    return pd.DataFrame({"ds": dates, "y": values})


def example_xgboost():
    """Prophet + XGBoost with cross-validation hyperparameter tuning."""
    print("=" * 70)
    print("Example 1: Prophet + XGBoost (CV tuning)")
    print("=" * 70)

    from hybridts import HybridForecaster, ProphetModel, XGBoostModel

    df = generate_sample_data(periods=400)

    primary = ProphetModel(
        param_grid={
            "changepoint_prior_scale": [0.05],
            "seasonality_prior_scale": [5.0],
            "seasonality_mode": ["multiplicative"],
        },
        cv_params={
            "initial": "350 days",
            "period": "30 days",
            "horizon": "30 days",
            "parallel": "threads",
        },
    )

    secondary = XGBoostModel(
        param_grid={"window_length": [21], "estimator__max_depth": [5]},
        static_params={"max_depth": 5, "learning_rate": 0.05, "n_estimators": 200},
        regressor_params={"random_state": 42},
        cv_initial_window=300,
        cv_step_length=30,
        window_length=21,
        fh=30,
        strategy="recursive",
    )

    forecaster = HybridForecaster(
        primary_model=primary,
        secondary_model=secondary,
        test_size=30,
    )

    print("Training...")
    metrics, y_true, y_pred = forecaster.evaluate_and_fit(df)

    print(f"MAPE: {metrics['MAPE']:.2f}%")
    print(f"MAE:  {metrics['MAE']:.0f}")

    forecast = forecaster.predict(horizon=30)
    print(f"\nForecast (first 5 rows):\n{forecast[['data', 'forecast_final']].head()}")


def example_lightgbm():
    """Prophet + LightGBM with cross-validation hyperparameter tuning."""
    print("\n" + "=" * 70)
    print("Example 2: Prophet + LightGBM (CV tuning)")
    print("=" * 70)

    from hybridts import HybridForecaster, LightGBMModel, ProphetModel

    df = generate_sample_data(periods=400)

    primary = ProphetModel(
        param_grid={
            "changepoint_prior_scale": [0.05],
            "seasonality_mode": ["multiplicative"],
        },
        cv_params={
            "initial": "350 days",
            "period": "30 days",
            "horizon": "30 days",
            "parallel": "threads",
        },
    )

    secondary = LightGBMModel(
        lgbm_regressor_params={
            "strategy": "recursive",
            "n_estimators": 200,
            "learning_rate": 0.05,
            "num_leaves": 31,
            "random_state": 42,
        },
        param_grid={"window_length": [21, 30]},
        fh=30,
        initial_window=300,
        step_length=30,
        window_length=21,
    )

    forecaster = HybridForecaster(
        primary_model=primary,
        secondary_model=secondary,
        test_size=30,
    )

    print("Training...")
    metrics, _, _ = forecaster.evaluate_and_fit(df)

    print(f"MAPE: {metrics['MAPE']:.2f}%")
    print(f"MAE:  {metrics['MAE']:.0f}")


def example_custom_holidays():
    """Using custom holidays for a non-Brazilian market."""
    print("\n" + "=" * 70)
    print("Example 3: Custom Holidays (US Market)")
    print("=" * 70)

    from hybridts import HybridForecaster, ProphetModel, XGBoostModel

    df = generate_sample_data(periods=400)

    primary = ProphetModel(
        param_grid={"changepoint_prior_scale": [0.05]},
        cv_params={"initial": "350 days", "period": "30 days", "horizon": "30 days"},
    )

    secondary = XGBoostModel(
        param_grid={"window_length": [21]},
        static_params={"max_depth": 5},
        regressor_params={"random_state": 42},
        cv_initial_window=300,
        cv_step_length=30,
        window_length=21,
        fh=30,
        strategy="recursive",
    )

    forecaster = HybridForecaster(
        primary_model=primary,
        secondary_model=secondary,
        test_size=30,
        holidays_country="US",
        holidays_state=None,
    )

    print("Training with US holidays...")
    forecaster.fit(df)

    forecast = forecaster.predict(horizon=30)
    print(f"Total forecast: {forecast['forecast_final'].sum():,.0f}")


def example_metrics():
    """Direct usage of ForecastMetrics."""
    print("\n" + "=" * 70)
    print("Example 4: ForecastMetrics")
    print("=" * 70)

    from hybridts import ForecastMetrics

    y_true = np.array([100, 150, 200, 250, 300])
    y_pred = np.array([110, 145, 195, 260, 290])

    report = ForecastMetrics(y_true, y_pred)
    print(report.summary())


if __name__ == "__main__":
    example_xgboost()
    example_lightgbm()
    example_custom_holidays()
    example_metrics()
