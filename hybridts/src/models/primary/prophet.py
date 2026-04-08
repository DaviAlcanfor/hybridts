import itertools
from typing import Optional
from prophet.diagnostics import cross_validation, performance_metrics
from prophet import Prophet
import numpy as np
import pandas as pd

from hybridts.src.exception.model_exception import ModelTrainingException, ModelPredictionException, model_error_handler


class ProphetModel:
    def __init__(
        self,
        param_grid: dict = None,
        cv_params: dict = None,
        yearly_seasonality: bool = True,
        weekly_seasonality: bool = True,
        daily_seasonality: bool = False,
        static_params: dict = None
    ):
        self.param_grid = param_grid
        self.cv_params = cv_params
        self.yearly_seasonality = yearly_seasonality
        self.weekly_seasonality = weekly_seasonality
        self.daily_seasonality = daily_seasonality
        self.static_params = static_params


    def _get_all_params(self, param_grid: dict) -> list:
        """
        Generate all possible combinations of parameters from a parameter grid.
        """

        return [dict(zip(param_grid.keys(), v)) for v in itertools.product(*param_grid.values())]
    
    
    def find_best_params(
        self, 
        df_train: pd.DataFrame, 
        holidays: pd.DataFrame
    ) -> dict:
        """
        Train Prophet model with cross-validation and return best params

        Args:
            df_train: pd.Dataframe
            years_in_data: np.ndarray
            holidays: pd.Dataframe
        
        Returns:
            best_prophet_params: dict
        """
        all_params = self._get_all_params(self.param_grid)
        mapes = []

        for params in all_params:
            m = Prophet(holidays=holidays, **params)
            m.fit(df_train)
            
            # cross validation
            df_cv = cross_validation(m, **self.cv_params)
            df_p = performance_metrics(df_cv, rolling_window=1)
            mapes.append(df_p['mape'].values[0])

        best_index = np.argmin(mapes)
        best_prophet_params = all_params[best_index]

        return best_prophet_params
    
    @model_error_handler(ModelTrainingException)
    def fit(
        self,
        df_train: pd.DataFrame,
        holidays: Optional[pd.DataFrame] = None,
    ) -> 'ProphetModel':
        best_params = self.find_best_params(df_train, holidays)
        self.model_ = Prophet(
            holidays=holidays,
            yearly_seasonality=self.yearly_seasonality,
            weekly_seasonality=self.weekly_seasonality,
            daily_seasonality=self.daily_seasonality,
            **best_params
        )
        self.model_.fit(df_train)
        self.best_params_ = best_params
        return self

    @model_error_handler(ModelTrainingException)
    def fit_static(
        self,
        df_train: pd.DataFrame,
        holidays: Optional[pd.DataFrame] = None,
    ) -> 'ProphetModel':
        """
        Fit Prophet model with fixed parameters (no CV). Faster, ideal for daily retraining when hyperparameters are already known.
        
        Args:
            df_train: pd.Dataframe
            holidays: pd.Dataframe
            
        Returns:
            self: ProphetModel with fitted model
        """
        
        
        self.model_ = Prophet(
            holidays=holidays,
            yearly_seasonality=self.yearly_seasonality,
            weekly_seasonality=self.weekly_seasonality,
            daily_seasonality=self.daily_seasonality,
            **self.static_params
        )
        self.model_.fit(df_train)
        return self
        
    @model_error_handler(ModelPredictionException)
    def predict(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Predicts future values using the fitted Prophet model.
        
        Args:
            df: pd.DataFrame with a 'ds' column containing the dates for prediction
        
        Returns:
            pd.DataFrame with predictions, including 'ds' and 'yhat' columns
        """
        
        return self.model_.predict(df)

