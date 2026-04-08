"""
HybridTS - Hybrid Time Series Forecasting Library
==================================================

A flexible library for combining Prophet's trend detection with gradient boosting
(XGBoost/LightGBM) for residual correction.

Quick Start:
    >>> from hybridts import HybridForecaster, ProphetModel, XGBoostTuner
    >>>
    >>> forecaster = HybridForecaster(
    ...     primary_model=ProphetModel(param_grid=..., cv_params=...),
    ...     secondary_model=XGBoostTuner(...),
    ...     test_size=30,
    ... )
    >>> forecaster.fit(df)
    >>> forecast = forecaster.predict(horizon=30)

Main Classes:
    - HybridForecaster: Main forecasting pipeline
    - TimeSeriesProcessor: Data preprocessing and validation
    - ProphetModel: Prophet wrapper with CV hyperparameter tuning
    - XGBoostTuner: XGBoost wrapper with CV hyperparameter tuning
    - LightGBMTuner: LightGBM wrapper with CV hyperparameter tuning

Utilities:
    - create_holidays_prophet: Generate holiday features for Prophet
    - get_brazilian_paydays: Generate payday dates
    - create_features: Feature engineering pipeline
"""

from loguru import logger

logger.disable("hybridts")

__version__ = "0.2.0"
__author__ = "Davi Franco"
__email__ = "alcanfordavi@gmail.com"

from hybridts.src.pipeline.pipeline import HybridForecaster
from hybridts.src.features.data_processor import TimeSeriesProcessor
from hybridts.src.models.primary.prophet import ProphetModel
from hybridts.src.models.secondary.xgboost import XGBoostTuner
from hybridts.src.models.secondary.lightgbm import LightGBMTuner
from hybridts.src.features.holidays import create_holidays_prophet, get_brazilian_paydays
from hybridts.src.features.engineering import create_features

__all__ = [
    "HybridForecaster",
    "TimeSeriesProcessor",
    "ProphetModel",
    "XGBoostTuner",
    "LightGBMTuner",
    "create_holidays_prophet",
    "get_brazilian_paydays",
    "create_features",
    "__version__",
]
