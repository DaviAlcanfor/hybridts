
<div align="center">

## *Hybrid Time Series Forecasting with Prophet and Gradient Boosting*

[![PyPI](https://img.shields.io/pypi/v/hybridts)](https://pypi.org/project/hybridts/)
[![Python](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/)
[![Tests](https://github.com/DaviAlcanfor/hybridts/actions/workflows/tests.yml/badge.svg)](https://github.com/DaviAlcanfor/hybridts/actions/workflows/tests.yml)
[![License](https://img.shields.io/badge/license-MIT-green.svg)](LICENSE)

HybridTS combines Prophet's trend and seasonality modeling with gradient boosting (XGBoost / LightGBM) to correct residuals.

[Quick Start](#quick-start) • [Installation](#installation) • [Examples](#examples) • [Contributing](#contributing)

</div>

---

## Features

- **Hybrid forecasting**: Prophet captures trend and seasonality; XGBoost or LightGBM corrects what Prophet misses.
- **sklearn-style API**: `fit()`, `predict()`, and `evaluate()` — no new paradigms to learn.
- **Dependency injection**: pass any configured model instance — no subclassing required.
- **Built-in evaluation**: holdout-based evaluation with MAE, RMSE, MAPE, sMAPE, R² and bias, accessible as attributes after `evaluate()`.
- **Integrated plotting**: visualize forecasts and evaluation results with a single method call (requires matplotlib).
- **Auto feature engineering**: holiday calendars, payday indicators, and calendar features generated automatically from the data range.

---

## How It Works

```text
┌────────────────────────────┐
│      Input Data (ds, y)    │
└──────────────┬─────────────┘
               │
       ┌───────▼─────────┐
       │  Primary Model  │
       │   (Prophet)     │
       │                 │
       │  trend +        │
       │  seasonality +  │
       │  holidays       │
       └───────┬─────────┘
               │
       ┌───────▼─────────────────┐
       │   Residual Calculation  │
       │   actual − ŷ_prophet    │
       └───────┬─────────────────┘
               │
       ┌───────▼─────────────────┐
       │     Residual Model      │
       │  (XGBoost / LightGBM)   │
       │                         │
       │  calendar + payday +    │
       │  holiday features       │
       └───────┬─────────────────┘
               │
       ┌───────▼─────────────────────┐
       │  Final Forecast             │
       │  ŷ_prophet  +  ŷ_residual   │
       └─────────────────────────────┘
```

---

## Installation

```bash
pip install hybridts
```

**Optional — plotting support:**

```bash
pip install hybridts[plotting]
```

---

## Quick Start

```python
import pandas as pd
from hybridts import HybridForecaster, ProphetModel, XGBoostModel

# Configure models
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

# Evaluate on holdout, then retrain on full data
forecaster = HybridForecaster(primary_model=prophet, secondary_model=xgb)
df = pd.read_csv("data.csv", parse_dates=["ds"])  # columns: ds, y

forecaster, metrics = forecaster.evaluate_and_fit(df)

# Forecast the next 30 days
forecast = forecaster.predict(horizon=30)
print(forecast)

# Visualize
forecaster.plot_forecast(df)
forecaster.plot_evaluation()
```

**Forecast output:**

| Column | Description |
| --- | --- |
| `data` | Forecast date |
| `forecast_primary_base` | Prophet baseline |
| `residual_correction` | Gradient boosting adjustment |
| `forecast_final` | Final hybrid forecast |

---

## Data Format

A pandas DataFrame with exactly two columns:

| Column | Type | Description |
| --- | --- | --- |
| `ds` | datetime | Date of observation |
| `y` | float | Value to forecast |

---

## Evaluation

```python
metrics, y_true, y_pred = forecaster.evaluate(df)

# Metric reports available after evaluate()
forecaster.metrics_report_          # hybrid forecast
forecaster.primary_metrics_report_  # Prophet baseline alone
```

Available metrics: `MAE`, `MSE`, `RMSE`, `MAPE`, `sMAPE`, `R-squared`, `Bias`.

---

## Examples

See the [`examples/`](examples/) folder for notebooks covering common use cases.

---

## Contributing

See [CONTRIBUTING.md](CONTRIBUTING.md) for development setup and contribution guidelines.

---

## License

MIT — see [LICENSE](LICENSE)

---

<div align="center">

**Davi Franco** — [GitHub](https://github.com/DaviAlcanfor)

[⬆ Back to top](#hybridts)

</div>
