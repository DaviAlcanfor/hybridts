# HybridTS

**Hybrid Time Series Forecasting Made Simple**

A Python library that combines the power of Prophet's trend detection with gradient boosting algorithms (XGBoost and LightGBM) for accurate time series forecasting.

[![Python Version](https://img.shields.io/badge/python-3.8%2B-blue)](https://www.python.org/downloads/)
[![License](https://img.shields.io/badge/license-MIT-green)](LICENSE)
[![PyPI](https://img.shields.io/badge/pypi-v0.1.0-orange)](https://pypi.org/project/hybridts/)

---

## Why HybridTS?

Time series forecasting is hard. While Prophet excels at capturing trends and seasonality, it often misses complex patterns in the residuals. Gradient boosting models like XGBoost are great at capturing these patterns but struggle with long-term trends.

**HybridTS combines both approaches:**

1. **Prophet** captures trend, seasonality, and holiday effects
2. **XGBoost/LightGBM** models the residuals to capture remaining patterns
3. **Final forecast** = Prophet baseline + ML corrections

This hybrid approach typically achieves **15-30% better accuracy** than using either model alone.

### What Makes It Different

- 🎯 **Pre-configured hybrid models** - No need to manually orchestrate Prophet + ML
- 🔧 **Automated feature engineering** - Holidays, paydays, temporal patterns built-in
- 📊 **Temporal cross-validation** - Proper time series validation out-of-the-box
- 🌍 **Flexible holidays** - Support for any country via the `holidays` library
- 📈 **MLflow integration** - Track experiments and version models automatically
- 🚀 **Production-ready** - Designed for real-world forecasting workflows

---

## Installation

```bash
pip install hybridts
```

**Requirements:** Python 3.8+

---

## Quick Start

### Installation

```bash
pip install hybridts
```

### Simplest Example (5 lines)

```python
import pandas as pd
from hybridts import HybridForecaster

# Load your data (must have 'ds' and 'y' columns)
df = pd.read_csv("your_data.csv", parse_dates=["ds"])

# Train and predict with defaults
forecaster = HybridForecaster()
forecaster.fit(df, model="xgboost")
forecast = forecaster.predict(horizon=30)

print(forecast[['ds', 'yhat']].head())
```

That's it! The library handles Prophet training, residual modeling, cross-validation, and hybrid forecasting automatically.

### With Custom Configuration

```python
from hybridts import HybridForecaster

config = {
    'test_size': 30,
    'models': {
        'xgboost': {
            'param_grid': {
                'window_length': [21],
                'estimator__max_depth': [5],
                'estimator__learning_rate': [0.05]
            }
        }
    }
}

forecaster = HybridForecaster(config=config)
forecaster.fit(df, model="xgboost")
forecast = forecaster.predict(horizon=30)
```

For more configuration options, see [Configuration Modes](#configuration-modes) below.

### Data Format

Your data must be a pandas DataFrame with two columns:

| Column | Type | Description |
|--------|------|-------------|
| `ds` | datetime | Date of observation |
| `y` | float | Value to forecast (revenue, demand, etc.) |

**Example:**
```python
import pandas as pd
import numpy as np

# Generate sample data
dates = pd.date_range('2023-01-01', periods=365, freq='D')
values = np.random.uniform(10000, 20000, 365)
df = pd.DataFrame({'ds': dates, 'y': values})
```

---

## Configuration Modes

HybridTS supports **three ways** to configure models, from simplest to most flexible:

### Mode 1: Default Configuration (Quickest) ⚡

Best for: Quick experiments, prototyping, first-time users

```python
from hybridts import HybridForecaster

# Everything configured with sensible defaults
forecaster = HybridForecaster()
forecaster.fit(df, model="xgboost")
forecast = forecaster.predict(horizon=30)
```

**What's included by default:**
- Test size: 30 days
- Prophet with multiplicative seasonality
- XGBoost/LightGBM with optimized hyperparameters
- Expanding window cross-validation

### Mode 2: Programmatic Configuration (Flexible) 🔧

Best for: Custom experiments, parameter tuning, notebooks

```python
from hybridts import HybridForecaster

config = {
    'test_size': 30,
    'cv_params': {
        'initial_window': 300,
        'step_length': 30
    },
    'models': {
        'prophet': {
            'param_grid': {
                'changepoint_prior_scale': [0.05, 0.1],
                'seasonality_prior_scale': [5.0, 10.0],
                'seasonality_mode': ['multiplicative']
            }
        },
        'xgboost': {
            'param_grid': {
                'window_length': [21, 28],
                'estimator__max_depth': [5, 7],
                'estimator__learning_rate': [0.05, 0.1]
            }
        }
    }
}

forecaster = HybridForecaster(config=config)
forecaster.fit(df, model="xgboost")
```

### Mode 3: YAML Configuration (Production) 📋

Best for: Production pipelines, reproducible experiments, team collaboration

**settings.yaml:**
```yaml
test_size: 30

cv_params:
  initial_window: 300
  step_length: 30

models:
  prophet:
    param_grid:
      changepoint_prior_scale: [0.05, 0.1, 0.5]
      seasonality_prior_scale: [5.0, 10.0]
      seasonality_mode: ['multiplicative']
    cv_params:
      initial: '350 days'
      period: '30 days'
      horizon: '30 days'
      parallel: 'threads'
  
  xgboost:
    param_grid:
      window_length: [21, 28]
      estimator__max_depth: [5, 7]
      estimator__learning_rate: [0.05, 0.1]
    static:
      n_estimators: 200
      max_depth: 5
      learning_rate: 0.05
```

**Python:**
```python
from hybridts import HybridForecaster
from hybridts.config import load_config

config = load_config("settings.yaml")
forecaster = HybridForecaster(config=config)
forecaster.fit(df, model="xgboost")
```

**Benefits:**
- ✅ Version control your configurations
- ✅ Share configurations across team
- ✅ Easy A/B testing (swap config files)
- ✅ Separate code from parameters

---

## Key Features

### 1. Hybrid Models

Combine Prophet with gradient boosting for superior accuracy:

```python
from hybridts import HybridForecaster, TimeSeriesProcessor

processor = TimeSeriesProcessor()
forecaster = HybridForecaster(config=config, processor=processor)

# Choose your ML model
forecaster.fit(df, escolha_modelo="xgboost")    # Prophet + XGBoost
forecaster.fit(df, escolha_modelo="lightgbm")   # Prophet + LightGBM
forecaster.fit(df, escolha_modelo="sxgboost")   # Prophet + XGBoost (fast mode)
```

### 2. Automated Feature Engineering

Built-in features for common time series patterns:

```python
from hybridts import create_features, get_brazilian_paydays

# Generate payday dates
paydays = get_brazilian_paydays(2023, 2025, country="BR", state="SP")

# Create features automatically
X = create_features(
    df_dates=df[['ds']],
    paydays_set=paydays,
    min_year=2023,
    max_year=2025,
    holidays_country="BR",
    holidays_state="SP"
)

# Features created:
# - is_weekend, is_month_start, is_month_end
# - day_of_week, day_of_month
# - is_payday, is_adiantamento (salary advance)
# - is_holiday, is_holiday_eve, is_post_holiday
# - dias_desde_pagamento (days since last payday)
```

### 3. Custom Holidays

Support for any country and custom events:

```python
from hybridts import create_holidays_prophet

# Brazilian holidays (default)
holidays_br = create_holidays_prophet(
    years=[2023, 2024, 2025],
    country="BR",
    state="SP"
)

# US holidays with custom events
holidays_us = create_holidays_prophet(
    years=[2023, 2024, 2025],
    country="US",
    state=None,
    custom_events=[
        {'holiday': 'Black_Friday', 'ds': '2023-11-24', 'lower_window': -2, 'upper_window': 2},
        {'holiday': 'Cyber_Monday', 'ds': '2023-11-27', 'lower_window': 0, 'upper_window': 2}
    ]
)
```

### 4. Model Validation

Proper time series validation with temporal cross-validation:

```python
# Split data
df_train = df.iloc[:-30]
df_test = df.iloc[-30:]

# Train
forecaster.fit(df_train, escolha_modelo="xgboost")

# Validate
metrics = forecaster.validate(df_train, df_test)

print(f"MAPE: {metrics['mape']:.2%}")
print(f"RMSE: {metrics['rmse']:.2f}")
print(f"MAE: {metrics['mae']:.2f}")
```

### 5. MLflow Integration

Track experiments and version models:

```python
# Train and save
forecaster.fit(df, escolha_modelo="xgboost")
metrics = forecaster.validate(df, df_test)

run_id = forecaster.save_to_mlflow(
    target="Revenue",
    metrics=metrics,
    experiment_path="my_forecasting_project",
    registry_path="./models/latest_model.json"
)

# Load later
from hybridts import HybridForecaster, TimeSeriesProcessor

forecaster, metadata = HybridForecaster.load_from_mlflow(
    config=config,
    processor=processor,
    registry_path="./models/latest_model.json"
)

print(f"Model trained on: {metadata['trained_at']}")
print(f"MAPE: {metadata['mape']:.2%}")
```

---

## Configuration

HybridTS uses YAML for configuration. Create a `config.yaml` file:

```yaml
# Temporal cross-validation settings
cv_params:
  initial_window: 365
  step_length: 30

# Model hyperparameters
models:
  prophet:
    param_grid:
      changepoint_prior_scale: [0.01, 0.1]
      seasonality_prior_scale: [1.0, 10.0]
      seasonality_mode: ['multiplicative']
    cv_params:
      initial: '365 days'
      period: '30 days'
      horizon: '30 days'
      parallel: 'threads'

  xgboost:
    param_grid:
      window_length: [14, 28]
      estimator__max_depth: [5, 6]
      estimator__learning_rate: [0.05, 0.1]
      estimator__n_estimators: [200]

# Test set size
test_size: 30
```

**Or configure programmatically:**

```python
config = {
    'test_size': 30,
    'models': {
        'prophet': {
            'param_grid': {
                'changepoint_prior_scale': [0.05],
                'seasonality_mode': ['multiplicative']
            }
        },
        'xgboost': {
            'param_grid': {
                'window_length': [14],
                'estimator__max_depth': [5]
            }
        }
    }
}

forecaster = HybridForecaster(config=config, processor=processor)
```

---

## Examples

Check out the `examples/` directory for complete examples:

- **quickstart.py** - Basic usage and configuration
- **notebooks/** - Jupyter notebooks with detailed workflows

---

## Use Cases

HybridTS works well for:

- 📈 **Revenue forecasting** - Predict future revenue with trend and seasonality
- 🛒 **Demand forecasting** - Forecast product demand for inventory planning
- 💰 **Financial metrics** - TPV, transaction volume, user growth
- 📊 **Business KPIs** - Any metric with temporal patterns

**Not recommended for:**
- High-frequency data (sub-hourly) - Prophet is designed for daily+ granularity
- Very short time series (<6 months) - Not enough data for reliable patterns
- Non-stationary processes - Requires preprocessing/differencing

---

## Performance Tips

### Speed Up Training

Grid search can be slow. For faster iterations:

```python
# Use static parameters (no grid search)
forecaster.fit(df, escolha_modelo="sxgboost")  # 's' prefix = static/fast mode
```

### Reduce Memory Usage

For large datasets:

```python
# Use smaller cross-validation windows
config['cv_params']['initial_window'] = 180  # Instead of 365
config['cv_params']['step_length'] = 60      # Instead of 30
```

---

## Contributing

Contributions are welcome! Please:

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/amazing-feature`)
3. Commit your changes (`git commit -m 'Add amazing feature'`)
4. Push to the branch (`git push origin feature/amazing-feature`)
5. Open a Pull Request

---

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

---

## Citation

If you use HybridTS in your research or project, please cite:

```bibtex
@software{hybridts2026,
  author = {Franco, Davi},
  title = {HybridTS: Hybrid Time Series Forecasting},
  year = {2026},
  url = {https://github.com/davifrancamaciel/hybridts}
}
```

---

## Acknowledgments

Built on top of excellent open-source projects:

- [Prophet](https://facebook.github.io/prophet/) by Meta
- [XGBoost](https://xgboost.readthedocs.io/) by DMLC
- [LightGBM](https://lightgbm.readthedocs.io/) by Microsoft
- [sktime](https://www.sktime.net/) for time series utilities

---

## Contact

**Davi Franco**  
📧 alcanfordavi@gmail.com  
🐙 [@davifrancamaciel](https://github.com/davifrancamaciel)

---

**Made with ❤️ for the time series forecasting community**
