"""
Configuração compartilhada para todos os testes (pytest fixtures globais).
"""

import pytest
import pandas as pd
import numpy as np


@pytest.fixture
def sample_timeseries_data():
    """
    Fixture global: Gera série temporal sintética com 365 dias.
    
    Características:
    - Tendência crescente
    - Sazonalidade semanal
    - Ruído aleatório
    """
    dates = pd.date_range('2023-01-01', periods=365, freq='D')
    
    # Trend + seasonality + noise
    trend = np.linspace(10000, 15000, 365)
    seasonality = 2000 * np.sin(2 * np.pi * np.arange(365) / 7)  # Weekly
    noise = np.random.normal(0, 500, 365)
    
    values = trend + seasonality + noise
    values = np.maximum(values, 0)  # Non-negative
    
    return pd.DataFrame({'ds': dates, 'y': values})


@pytest.fixture
def minimal_config():
    """
    Fixture global: Config mínima válida para HybridForecaster.
    """
    return {
        'test_size': 30,
        'cv_params': {
            'initial_window': 300,
            'step_length': 30
        },
        'models': {
            'prophet': {
                'param_grid': {
                    'changepoint_prior_scale': [0.05],
                    'seasonality_prior_scale': [5.0],
                    'seasonality_mode': ['multiplicative']
                },
                'cv_params': {
                    'initial': '300 days',
                    'period': '30 days',
                    'horizon': '30 days',
                    'parallel': 'threads'
                }
            },
            'xgboost': {
                'param_grid': {
                    'window_length': [14],
                    'estimator__max_depth': [5],
                    'estimator__learning_rate': [0.05],
                    'estimator__n_estimators': [100]
                },
                'static': {
                    'n_estimators': 100,
                    'max_depth': 5,
                    'learning_rate': 0.05,
                    'subsample': 0.8,
                    'tree_method': 'hist',
                    'random_state': 42
                }
            }
        }
    }
