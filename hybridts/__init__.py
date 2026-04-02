"""
HybridTS - Hybrid Time Series Forecasting Library
==================================================

A flexible library for time series forecasting using hybrid models that combine
Prophet's trend detection with gradient boosting (XGBoost/LightGBM) for residuals.

Quick Start (3 lines):
    >>> from hybridts import HybridForecaster
    >>> forecaster = HybridForecaster()
    >>> forecaster.fit(df, model="xgboost")
    >>> forecast = forecaster.predict(horizon=30)

Three Configuration Modes:
    
    1. Default Configuration (quickest):
        >>> forecaster = HybridForecaster()
        >>> forecaster.fit(df, model="xgboost")
    
    2. Programmatic Configuration:
        >>> config = {'test_size': 30, 'models': {...}}
        >>> forecaster = HybridForecaster(config=config)
    
    3. YAML Configuration (production):
        >>> from hybridts import load_config
        >>> config = load_config("settings.yaml")
        >>> forecaster = HybridForecaster(config=config)

Main Classes:
    - HybridForecaster: Main forecasting pipeline (Prophet + ML)
    - TimeSeriesProcessor: Data preprocessing and validation
    - ProphetModel: Prophet model wrapper (advanced usage)
    - XGBoostModel: XGBoost model wrapper (advanced usage)
    - LightGBMModel: LightGBM model wrapper (advanced usage)

Utilities:
    - load_config: Load configuration from YAML
    - create_holidays_prophet: Generate holiday features
    - get_brazilian_paydays: Generate payday dates
    - create_features: Feature engineering pipeline
"""

__version__ = "0.1.0"
__author__ = "Davi Franco"
__email__ = "alcanfordavi@gmail.com"

# Core classes
from hybridts.src.pipeline.pipeline import HybridForecaster
from hybridts.src.features.data_processor import TimeSeriesProcessor

# Model classes (advanced usage)
from hybridts.src.training.prophet import ProphetModel
from hybridts.src.training.xgboost import XGBoostModel
from hybridts.src.training.lightgbm import LightGBMTuner as LightGBMModel

# Utilities
from hybridts.config.loader import load_config
from hybridts.src.features.holidays import (
    create_holidays_prophet,
    get_brazilian_paydays,
)
from hybridts.src.features.engineering import create_features

# Public API
__all__ = [
    # Core
    "HybridForecaster",
    "TimeSeriesProcessor",
    # Models
    "ProphetModel",
    "XGBoostModel",
    "LightGBMModel",
    # Utilities
    "load_config",
    "create_holidays_prophet",
    "get_brazilian_paydays",
    "create_features",
    # Metadata
    "__version__",
]
