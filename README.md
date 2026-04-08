# HybridTS

![HybridTS Logo](docs/logo_v1.png)

Hybrid Time Series Forecasting for Python

Combines Prophet's trend and seasonality detection with gradient boosting (XGBoost / LightGBM) to correct residuals — delivering more accurate forecasts than either model alone.

[![Python Version](https://img.shields.io/badge/python-3.8%2B-blue)](https://www.python.org/downloads/)
[![License](https://img.shields.io/badge/license-MIT-green)](LICENSE)
[![Version](https://img.shields.io/badge/version-0.2.0-orange)](https://github.com/DaviAlcanfor/hybridts)
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
from hybridts import HybridForecaster, ProphetModel, XGBoostTuner
import pandas as pd

prophet = ProphetModel()
xgb = XGBoostTuner()

forecaster = HybridForecaster(primary_model=prophet, secondary_model=xgb)

df = pd.read_csv("data.csv", parse_dates=["ds"])  # columns: ds, y
forecaster.fit(df)
forecast = forecaster.predict(horizon=30)
print(forecast)
```

---

## Data Format

A pandas DataFrame with exactly two columns:

| Column | Type | Description |
|--------|------|-------------|
| `ds` | datetime | Date of observation |
| `y` | float | Value to forecast |

---

## Core API

### `HybridForecaster`

```python
HybridForecaster(
    primary_model,           # ProphetModel instance
    secondary_model,         # XGBoostTuner or LightGBMTuner instance
    test_size=30,            # holdout size for validate()
    paydays_set=None,        # set of payday Timestamps (auto-generated if None)
    holidays_country="BR",   # country code for auto-generated holidays
    holidays_state=None,     # state/subdivision code
)
```

#### `fit(df, holidays=<auto>, features=<auto>)`

Trains both models. By default, holidays and features are auto-generated from the data.

```python
forecaster.fit(df)                        # auto holidays + auto features
forecaster.fit(df, holidays=None)         # no holidays passed to Prophet
forecaster.fit(df, holidays=my_holidays)  # custom holidays DataFrame
forecaster.fit(df, features=None)         # secondary model trains without exogenous features
```

#### `predict(horizon, features=<auto>)`

Returns a DataFrame with `horizon` rows:

| Column                  | Description            |
| ----------------------- | ---------------------- |
| `data`                  | Forecast date          |
| `forecast_primary_base` | Prophet baseline       |
| `residual_correction`   | ML adjustment          |
| `forecast_final`        | Final hybrid forecast  |

#### `validate(df, test_size=None)`

Evaluates on a holdout set without data leakage. Returns `(metrics, y_true, y_pred)`.

```python
metrics, y_true, y_pred = forecaster.validate(df)
# metrics: {"mape", "rmse", "mae", "mdape"}
```

#### `validate_and_fit(df, test_size=None)`

Validates on holdout, then retrains on the full dataset.

```python
forecaster, metrics = forecaster.validate_and_fit(df)
```

---

## Models

### `ProphetModel`

Wraps Prophet with automated hyperparameter tuning via cross-validation.

```python
ProphetModel(
    param_grid={"changepoint_prior_scale": [0.05, 0.1], ...},  # grid search params
    cv_params={"initial": "300 days", "period": "30 days", "horizon": "30 days"},
    yearly_seasonality=True,
    weekly_seasonality=True,
    daily_seasonality=False,
    static_params={...},   # used by fit_static() — skips CV
)
```

| Method                     | Description                                          |
| -------------------------- | ---------------------------------------------------- |
| `fit(df, holidays)`        | Tunes hyperparameters via CV, then fits on full data |
| `fit_static(df, holidays)` | Fits with `static_params`, no CV                     |
| `predict(df)`              | Returns Prophet forecast DataFrame                   |

### `XGBoostTuner`

Wraps XGBoost via sktime's `make_reduction` with grid search and expanding window CV.

```python
XGBoostTuner(
    test_size=30,
    param_grid={"window_length": [21], "estimator__max_depth": [5, 7], ...},
    static_params={"n_estimators": 200, "max_depth": 5, ...},
    cv_initial_window=270,
    cv_step_length=30,
    window_length=21,
    fh=30,
    strategy="recursive",
    regressor_params={"random_state": 42},
)
```

### `LightGBMTuner`

Same interface as `XGBoostTuner`, using LightGBM as the estimator.

```python
LightGBMTuner(
    lgbm_regressor_params={"strategy": "recursive", "n_estimators": 200, ...},
    test_size=30,
    initial_window=270,
    step_length=30,
    window_length=21,
    param_grid={"window_length": [21, 28], ...},
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

| Feature                  | Description                    |
| ------------------------ | ------------------------------ |
| `is_weekend`             | 1 if Saturday or Sunday        |
| `is_month_start`         | 1 if day <= 9                  |
| `is_month_end`           | 1 if last day of month         |
| `day_of_week`            | 0 (Mon) to 6 (Sun)             |
| `day_of_month`           | 1-31                           |
| `is_payday`              | 1 if date is a payday          |
| `is_adiantamento`        | 1 if salary advance day        |
| `sextou_com_dinheiro`    | 1 if Friday and payday         |
| `dias_desde_pagamento`   | Days since last payday         |
| `is_holiday`             | 1 if public holiday            |
| `is_holiday_eve`         | 1 if day before holiday        |
| `is_post_holiday`        | 1 if day after holiday         |

---

## Logging

HybridTS uses [loguru](https://github.com/Delgan/loguru) with logging disabled by default (library-safe). To enable:

```python
from loguru import logger
logger.enable("hybridts")
```

---

## Project Structure

```text
hybridts/
├── __init__.py                  # public API
└── src/
    ├── features/
    │   ├── data_processor.py    # TimeSeriesProcessor
    │   ├── engineering.py       # create_features
    │   └── holidays.py          # create_holidays_prophet, get_brazilian_paydays
    ├── models/
    │   ├── primary/
    │   │   └── prophet.py       # ProphetModel
    │   └── secondary/
    │       ├── xgboost.py       # XGBoostTuner
    │       └── lightgbm.py      # LightGBMTuner
    ├── pipeline/
    │   └── pipeline.py          # HybridForecaster
    └── exception/
        ├── model_exception.py   # ModelTrainingException, ModelPredictionException
        └── dataframe_exception.py
```

---

## License

MIT — see [LICENSE](LICENSE)

---

**Davi Franco** — [GitHub](https://github.com/DaviAlcanfor)
