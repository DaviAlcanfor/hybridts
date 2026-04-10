# HybridTS

![HybridTS Logo](docs/logo_v1.png)

Hybrid Time Series Forecasting for Python

Combines Prophet's trend and seasonality modeling with gradient boosting (XGBoost / LightGBM) to correct residuals — delivering more accurate forecasts than either model alone.

[![Python Version](https://img.shields.io/badge/python-3.8%2B-blue)](https://www.python.org/downloads/)
[![License](https://img.shields.io/badge/license-MIT-green)](LICENSE)
[![Version](https://img.shields.io/badge/version-0.4.0-orange)](https://github.com/DaviAlcanfor/hybridts)
[![Tests](https://github.com/DaviAlcanfor/hybridts/actions/workflows/tests.yml/badge.svg)](https://github.com/DaviAlcanfor/hybridts/actions/workflows/tests.yml)

---

## How It Works

1. **Prophet** fits trend, seasonality, and holiday effects
2. **XGBoost / LightGBM** learns from the residuals using engineered features
3. **Final forecast** = Prophet baseline + ML residual correction

---

## Installation

```bash
pip install hybridts
```

---

## Quick Start

```python
from hybridts import HybridForecaster, ProphetModel, XGBoostModel
import pandas as pd

prophet = ProphetModel(
    param_grid={"changepoint_prior_scale": [0.05, 0.1]},
    cv_params={"initial": "300 days", "period": "30 days", "horizon": "30 days"},
)
xgb = XGBoostModel(
    param_grid={"window_length": [21], "estimator__max_depth": [5, 7]},
    static_params={"n_estimators": 200, "max_depth": 5},
    regressor_params={"random_state": 42},
    cv_initial_window=270,
    cv_step_length=30,
    window_length=21,
    fh=30,
    strategy="recursive",
)

forecaster = HybridForecaster(primary_model=prophet, secondary_model=xgb)

df = pd.read_csv("data.csv", parse_dates=["ds"])  # columns: ds, y
forecaster.fit(df)
forecast = forecaster.predict(horizon=30)
print(forecast)
```

---

## Data Format

A pandas DataFrame with exactly two columns:

| Column | Type     | Description          |
|--------|----------|----------------------|
| `ds`   | datetime | Date of observation  |
| `y`    | float    | Value to forecast    |

---

## Core API

### `HybridForecaster`

The main pipeline. Orchestrates the primary and secondary models end-to-end.

```python
HybridForecaster(
    primary_model,           # ProphetModel instance
    secondary_model,         # XGBoostModel or LightGBMModel instance
    test_size=30,            # holdout size for evaluate()
    paydays_set=None,        # set of payday Timestamps (auto-generated if None)
    holidays_country="BR",   # country code for auto-generated holidays
    holidays_state=None,     # state/subdivision code
)
```

#### `fit(df, holidays=<auto>, features=<auto>)`

Trains both models on the provided data.

By default, holidays and features are auto-generated from the data range. Pass `None` to disable them, or a DataFrame to use your own.

```python
forecaster.fit(df)                         # auto holidays + auto features
forecaster.fit(df, holidays=None)          # no holidays passed to Prophet
forecaster.fit(df, holidays=my_holidays)   # custom holidays DataFrame
forecaster.fit(df, features=None)          # secondary model trains without exogenous features
forecaster.fit(df, features=my_features)   # custom feature DataFrame
```

#### `predict(horizon, features=<auto>)`

Generates a forecast for the next N days.

```python
forecast = forecaster.predict(horizon=30)
```

Returns a DataFrame with one row per forecasted day:

| Column                  | Description                        |
|-------------------------|------------------------------------|
| `data`                  | Forecast date                      |
| `forecast_primary_base` | Prophet baseline                   |
| `residual_correction`   | ML residual adjustment             |
| `forecast_final`        | Final hybrid forecast (int)        |

#### `evaluate(df, test_size=None)`

Evaluates model accuracy on a holdout set without data leakage. The model is trained on `df[:-test_size]` and tested on the last `test_size` days.

```python
metrics, y_true, y_pred = forecaster.evaluate(df)

# Full metric reports available after evaluate()
forecaster.metrics_report_          # ForecastMetrics for the hybrid forecast
forecaster.primary_metrics_report_  # ForecastMetrics for the primary model alone
```

#### `evaluate_and_fit(df, test_size=None)`

Evaluates on holdout, then retrains on the full dataset. Use this when you want both a reliable accuracy estimate and a production-ready model.

```python
forecaster, metrics = forecaster.evaluate_and_fit(df)
```

---

## Data Preprocessing

`TimeSeriesProcessor` validates and prepares raw data before training.

```python
from hybridts import TimeSeriesProcessor

processor = TimeSeriesProcessor()

# Pass a DataFrame directly
df = processor.prepare_data(df=raw_df)

# Or pass a loader callable
df = processor.prepare_data(data_loader=lambda: pd.read_parquet("data.parquet"))
```

`prepare_data` validates columns, converts types, sorts by date, and fills missing calendar days with zero.

---

## Models

### `ProphetModel`

Wraps Prophet with automated hyperparameter tuning via cross-validation.

```python
ProphetModel(
    param_grid={"changepoint_prior_scale": [0.05, 0.1], ...},
    cv_params={"initial": "300 days", "period": "30 days", "horizon": "30 days"},
    yearly_seasonality=True,
    weekly_seasonality=True,
    daily_seasonality=False,
    static_params={...},  # used by fit_static() — skips CV
)
```

| Method                      | Description                                           |
|-----------------------------|-------------------------------------------------------|
| `fit(df, holidays)`         | Tunes hyperparameters via CV, then fits on full data  |
| `fit_static(df, holidays)`  | Fits with `static_params`, no CV — faster             |
| `predict(df)`               | Returns Prophet forecast DataFrame                    |

### `XGBoostModel`

XGBoost residual forecaster using sktime's `make_reduction` with grid search and expanding window CV.

```python
XGBoostModel(
    param_grid={"window_length": [21], "estimator__max_depth": [5, 7]},
    static_params={"n_estimators": 200, "max_depth": 5},
    regressor_params={"random_state": 42},
    cv_initial_window=270,
    cv_step_length=30,
    window_length=21,
    fh=30,
    strategy="recursive",
)
```

### `LightGBMModel`

Same interface as `XGBoostModel`, using LightGBM as the estimator.

```python
LightGBMModel(
    param_grid={"window_length": [21, 28], ...},
    static_params={...},
    lgbm_regressor_params={"n_estimators": 200, ...},
    cv_initial_window=270,
    cv_step_length=30,
    window_length=21,
    fh=30,
    strategy="recursive",
)
```

Both models expose `fit()`, `fit_static()`, and `predict()`.

---

## Feature Engineering

```python
from hybridts import create_features, get_brazilian_paydays, create_holidays_prophet

# Generate Brazilian payday dates
paydays = get_brazilian_paydays(start_year=2022, end_year=2025)

# Create exogenous features
features = create_features(
    df_dates=df[["ds"]],
    paydays_set=paydays,       # optional — all payday features zeroed if None
    min_year=2022,
    max_year=2025,
    holidays_country="BR",
    holidays_state="SP",
)
```

Generated features:

| Feature                | Description                             |
|------------------------|-----------------------------------------|
| `is_weekend`           | 1 if Saturday or Sunday                 |
| `is_month_start`       | 1 if day <= 9                           |
| `is_month_end`         | 1 if last day of month                  |
| `day_of_week`          | 0 (Mon) to 6 (Sun)                      |
| `day_of_month`         | 1–31                                    |
| `is_payday`            | 1 if date is a payday                   |
| `is_salary_advance`    | 1 if salary advance day (around day 20) |
| `is_payday_friday`     | 1 if Friday and payday                  |
| `days_since_payday`    | Days elapsed since last payday          |
| `is_holiday`           | 1 if public holiday                     |
| `is_holiday_eve`       | 1 if day before a holiday               |
| `is_post_holiday`      | 1 if day after a holiday                |

---

## Metrics

`ForecastMetrics` computes accuracy metrics for any `y_true` / `y_pred` pair.

```python
from hybridts import ForecastMetrics

report = ForecastMetrics(y_true, y_pred)

report.mae        # Mean Absolute Error
report.mse        # Mean Squared Error
report.rmse       # Root Mean Squared Error
report.mape       # Mean Absolute Percentage Error (%)
report.smape      # Symmetric MAPE (%)
report.r_squared  # Coefficient of determination
report.bias       # Mean signed error

print(report.summary())
report.all_metrics()  # returns a dict with all metrics
```

After `evaluate()`, two reports are available on the forecaster:

```python
forecaster.metrics_report_          # hybrid forecast metrics
forecaster.primary_metrics_report_  # primary model metrics (before residual correction)
```

---

## Logging

HybridTS uses [loguru](https://github.com/Delgan/loguru) with logging disabled by default. To enable:

```python
from loguru import logger
logger.enable("hybridts")
```

---

## Project Structure

```text
hybridts/
├── pipeline.py                      # HybridForecaster
├── exceptions.py                    # ModelTrainingException, ModelPredictionException
├── features/
│   ├── engineering.py               # create_features
│   └── holidays.py                  # create_holidays_prophet, get_brazilian_paydays
├── models/
│   ├── primary/
│   │   └── prophet.py               # ProphetModel
│   └── secondary/
│       ├── xgboost_model.py         # XGBoostModel
│       └── lightgbm_model.py        # LightGBMModel
├── metrics/
│   └── forecast.py                  # ForecastMetrics
└── preprocessing/
    └── processor.py                 # TimeSeriesProcessor
```

---

## License

MIT — see [LICENSE](LICENSE)

---

**Davi Franco** — [GitHub](https://github.com/DaviAlcanfor)
