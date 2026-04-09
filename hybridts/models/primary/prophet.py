import itertools
from typing import Optional

import numpy as np
import pandas as pd
from prophet import Prophet
from prophet.diagnostics import cross_validation, performance_metrics

from ...exceptions import ModelPredictionException, ModelTrainingException, model_error_handler


class ProphetModel:
    """
    Prophet wrapper with cross-validation hyperparameter tuning.

    Args:
        param_grid: Grid of parameters to search over.
        cv_params: Parameters passed to Prophet's cross_validation().
        yearly_seasonality: Enable yearly seasonality (default: True).
        weekly_seasonality: Enable weekly seasonality (default: True).
        daily_seasonality: Enable daily seasonality (default: False).
        static_params: Fixed parameters for fit_static() (no CV).
    """

    def __init__(
        self,
        param_grid: dict = None,
        cv_params: dict = None,
        yearly_seasonality: bool = True,
        weekly_seasonality: bool = True,
        daily_seasonality: bool = False,
        static_params: dict = None,
    ):
        self.param_grid = param_grid
        self.cv_params = cv_params
        self.yearly_seasonality = yearly_seasonality
        self.weekly_seasonality = weekly_seasonality
        self.daily_seasonality = daily_seasonality
        self.static_params = static_params

    def _get_all_params(self, param_grid: dict) -> list:
        return [dict(zip(param_grid.keys(), v)) for v in itertools.product(*param_grid.values())]

    def find_best_params(self, df_train: pd.DataFrame, holidays: pd.DataFrame) -> dict:
        """
        Train Prophet with cross-validation over param_grid and return the best parameters.

        Args:
            df_train: Training DataFrame with 'ds' and 'y' columns.
            holidays: Holidays DataFrame for Prophet.

        Returns:
            Dictionary of best parameters.
        """
        all_params = self._get_all_params(self.param_grid)
        mapes = []

        for params in all_params:
            m = Prophet(holidays=holidays, **params)
            m.fit(df_train)
            df_cv = cross_validation(m, **self.cv_params)
            df_p = performance_metrics(df_cv, rolling_window=1)
            mapes.append(df_p["mape"].values[0])

        best_index = np.argmin(mapes)
        self.best_params_ = all_params[best_index]
        return self.best_params_

    @model_error_handler(ModelTrainingException)
    def fit(
        self,
        df_train: pd.DataFrame,
        holidays: Optional[pd.DataFrame] = None,
    ) -> "ProphetModel":
        """
        Fit Prophet using cross-validation to find the best hyperparameters.

        Args:
            df_train: Training DataFrame with 'ds' and 'y' columns.
            holidays: Holidays DataFrame for Prophet.

        Returns:
            self
        """
        best_params = self.find_best_params(df_train, holidays)
        self.model_ = Prophet(
            holidays=holidays,
            yearly_seasonality=self.yearly_seasonality,
            weekly_seasonality=self.weekly_seasonality,
            daily_seasonality=self.daily_seasonality,
            **best_params,
        )
        self.model_.fit(df_train)
        return self

    @model_error_handler(ModelTrainingException)
    def fit_static(
        self,
        df_train: pd.DataFrame,
        holidays: Optional[pd.DataFrame] = None,
    ) -> "ProphetModel":
        """
        Fit Prophet with fixed parameters (no CV). Faster, ideal for daily retraining
        when hyperparameters are already known.

        Args:
            df_train: Training DataFrame with 'ds' and 'y' columns.
            holidays: Holidays DataFrame for Prophet.

        Returns:
            self
        """
        self.model_ = Prophet(
            holidays=holidays,
            yearly_seasonality=self.yearly_seasonality,
            weekly_seasonality=self.weekly_seasonality,
            daily_seasonality=self.daily_seasonality,
            **self.static_params,
        )
        self.model_.fit(df_train)
        return self

    @model_error_handler(ModelPredictionException)
    def predict(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Generate predictions using the fitted Prophet model.

        Args:
            df: DataFrame with a 'ds' column containing forecast dates.

        Returns:
            DataFrame with 'ds' and 'yhat' columns (among others).
        """
        return self.model_.predict(df)
