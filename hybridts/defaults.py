"""
Default configuration for HybridTS.

These sensible defaults allow users to get started without writing config files.
Based on empirical best practices across multiple forecasting tasks.
"""

DEFAULT_CONFIG = {
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
            },
            'cv_params': {
                'initial': '350 days',
                'period': '30 days',
                'horizon': '30 days',
                'parallel': 'threads'
            }
        },
        
        'xgboost': {
            'param_grid': {
                'window_length': [21, 28],
                'estimator__max_depth': [5, 7],
                'estimator__learning_rate': [0.05, 0.1],
                'estimator__n_estimators': [200, 300]
            },
            'static': {
                'n_estimators': 200,
                'max_depth': 5,
                'learning_rate': 0.05,
                'subsample': 0.8,
                'colsample_bytree': 0.8,
                'tree_method': 'hist',
                'random_state': 42
            }
        },
        
        'lightgbm': {
            'param_grid': {
                'window_length': [21, 28],
                'estimator__max_depth': [5, 7],
                'estimator__learning_rate': [0.05, 0.1],
                'estimator__n_estimators': [200, 300],
                'estimator__num_leaves': [31, 63]
            },
            'static': {
                'n_estimators': 200,
                'max_depth': 5,
                'learning_rate': 0.05,
                'num_leaves': 31,
                'subsample': 0.8,
                'colsample_bytree': 0.8,
                'random_state': 42,
                'verbose': -1
            }
        }
    }
}


def get_default_config() -> dict:
    """
    Get a copy of the default configuration.
    
    Returns:
        dict: Default configuration dictionary
    """
    import copy
    return copy.deepcopy(DEFAULT_CONFIG)
