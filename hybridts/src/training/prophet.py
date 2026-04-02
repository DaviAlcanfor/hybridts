# %pip install prophet

import itertools
from prophet.diagnostics import cross_validation, performance_metrics
from prophet import Prophet
import numpy as np
import pandas as pd

class ProphetModel:
    """
    Prophet-based baseline forecaster for hybrid time series models.
    
    Handles trend, seasonality, and holiday effects. Supports hyperparameter
    tuning via cross-validation to find optimal parameters.
    
    Args:
        config: Configuration dictionary containing:
            - models.prophet.param_grid: Parameters to search (changepoint_prior_scale, etc.)
            - models.prophet.cv_params: Cross-validation parameters (initial, period, horizon)
    """
    def __init__(self, config: dict):
        self.param_grid = config['models']['prophet']['param_grid']
        self.cv_params = config['models']['prophet']['cv_params']

    def _get_all_params(self, param_grid: dict) -> list:
        """Generate all possible combinations of parameters from a parameter grid."""

        return [dict(zip(param_grid.keys(), v)) for v in itertools.product(*param_grid.values())]
    
    def find_best_params(self, df_train: pd.DataFrame, prophet_holidays: pd.DataFrame) -> dict:
        """
        Train Prophet model with cross-validation and return best params

        Args:
            df_train: pd.Dataframe
            years_in_data: np.ndarray
            prophet_holidays: pd.Dataframe
        
        Returns:
            best_prophet_params: dict
        """
        all_params = self._get_all_params(self.param_grid)
        mapes = []

        for params in all_params:
            m = Prophet(
                holidays=prophet_holidays,
                yearly_seasonality=True,
                weekly_seasonality=True,
                daily_seasonality=False,
                **params
            )
          
            m.fit(df_train)
            
            df_cv = cross_validation(m, **self.cv_params)
            
            df_p = performance_metrics(df_cv, rolling_window=1)
            mapes.append(df_p['mape'].values[0])

        best_index = np.argmin(mapes)
        best_prophet_params = all_params[best_index]

        return best_prophet_params
