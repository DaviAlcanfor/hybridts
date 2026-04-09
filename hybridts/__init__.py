"""
HybridTS - Hybrid Time Series Forecasting Library
==================================================

A flexible library for combining Prophet's trend detection with gradient boosting
(XGBoost/LightGBM) for residual correction.

Quick Start:
    >>> from hybridts import HybridForecaster, ProphetModel, XGBoostModel
    >>>
    >>> forecaster = HybridForecaster(
    ...     primary_model=ProphetModel(param_grid=..., cv_params=...),
    ...     secondary_model=XGBoostModel(...),
    ...     test_size=30,
    ... )
    >>> forecaster.fit(df)
    >>> forecast = forecaster.predict(horizon=30)

Main Classes:
    - HybridForecaster: Main forecasting pipeline
    - TimeSeriesProcessor: Data preprocessing and validation
    - ProphetModel: Prophet wrapper with CV hyperparameter tuning
    - XGBoostModel: XGBoost residual forecaster with CV hyperparameter tuning
    - LightGBMModel: LightGBM residual forecaster with CV hyperparameter tuning
    - ForecastMetrics: Forecast accuracy metrics

Utilities:
    - create_holidays_prophet: Generate holiday features for Prophet
    - get_brazilian_paydays: Generate payday dates
    - create_features: Feature engineering pipeline
"""

from loguru import logger

logger.disable("hybridts")

__version__ = "0.4.0"
__author__ = "Davi Franco"
__email__ = "alcanfordavi@gmail.com"

from .pipeline import HybridForecaster
from .preprocessing import TimeSeriesProcessor
from .models.primary.prophet import ProphetModel
from .models.secondary.xgboost_model import XGBoostModel
from .models.secondary.lightgbm_model import LightGBMModel
from .features.holidays import create_holidays_prophet, get_brazilian_paydays
from .features.engineering import create_features
from .metrics.forecast import ForecastMetrics

__all__ = [
    "HybridForecaster",
    "TimeSeriesProcessor",
    "ProphetModel",
    "XGBoostModel",
    "LightGBMModel",
    "ForecastMetrics",
    "create_holidays_prophet",
    "get_brazilian_paydays",
    "create_features",
    "__version__",
]
