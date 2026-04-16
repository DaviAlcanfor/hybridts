from typing import Optional

import numpy as np
import pandas as pd
from lightgbm import LGBMRegressor
from sktime.forecasting.compose import make_reduction
from sktime.forecasting.model_selection import ExpandingWindowSplitter, ForecastingGridSearchCV
from sktime.performance_metrics.forecasting import MeanAbsolutePercentageError

from ..base import ResidualModel
from ...exceptions import ModelPredictionException, ModelTrainingException, model_error_handler


class LightGBMModel(ResidualModel):
    """
    LightGBM residual forecaster with optional cross-validation hyperparameter tuning.

    Args:
        lgbm_regressor_params: LGBMRegressor parameters. Must include a 'strategy' key
                               (e.g. "recursive", "direct") which is popped before passing
                               to the regressor.
        param_grid: Parameter grid for grid search CV (optional).
        fh: Forecast horizon used during CV and prediction.
        initial_window: Initial window size for ExpandingWindowSplitter.
        step_length: Step length for ExpandingWindowSplitter.
        window_length: Window length for fit_static().
    """

    def __init__(
        self,
        lgbm_regressor_params: dict = None,
        fh: int = None,
        initial_window: int = None,
        step_length: int = None,
        window_length: int = None,
        param_grid: Optional[dict] = None,
    ):
        params = dict(lgbm_regressor_params)
        self.strategy = params.pop("strategy")
        self.lgbm_regressor_params = params
        self.fh = fh
        self.param_grid = param_grid
        self.initial_window = initial_window
        self.step_length = step_length
        self.window_length = window_length

    @model_error_handler(ModelTrainingException)
    def fit(
        self,
        residuals_train: pd.Series,
        X_train: Optional[pd.DataFrame] = None,
    ) -> "LightGBMModel":
        """
        Fit LightGBM via sktime using GridSearchCV with an ExpandingWindowSplitter.

        Args:
            residuals_train: Residual series from the primary model.
            X_train: Exogenous feature DataFrame (optional).

        Returns:
            self
        """
        fh = np.arange(1, self.fh + 1)
        mape = MeanAbsolutePercentageError(symmetric=False)

        cv = ExpandingWindowSplitter(
            initial_window=self.initial_window,
            step_length=self.step_length,
            fh=fh,
        )

        lgbm_forecaster = make_reduction(
            estimator=LGBMRegressor(**self.lgbm_regressor_params),
            strategy=self.strategy,
        )

        gscv = ForecastingGridSearchCV(
            forecaster=lgbm_forecaster,
            cv=cv,
            param_grid=self.param_grid,
            scoring=mape,
        )

        gscv.fit(y=residuals_train, X=X_train)
        self.model_ = gscv.best_forecaster_
        return self

    @model_error_handler(ModelTrainingException)
    def fit_static(
        self,
        residuals_train: pd.Series,
        X_train: Optional[pd.DataFrame] = None,
    ) -> "LightGBMModel":
        """
        Fit LightGBM with fixed parameters (no CV). Faster, ideal for daily retraining.

        Args:
            residuals_train: Residual series from the primary model.
            X_train: Exogenous feature DataFrame (optional).

        Returns:
            self
        """
        self.model_ = make_reduction(
            estimator=LGBMRegressor(**self.lgbm_regressor_params),
            strategy=self.strategy,
            window_length=self.window_length,
        )
        self.model_.fit(y=residuals_train, X=X_train)
        return self

    @model_error_handler(ModelPredictionException)
    def predict(
        self,
        fh,
        X: Optional[pd.DataFrame] = None,
    ) -> pd.Series:
        """
        Generate predictions using the fitted LightGBM model.

        Args:
            fh: Forecast horizon (array or int).
            X: Exogenous feature DataFrame for the forecast period (optional).

        Returns:
            Series of predicted residuals.
        """
        return self.model_.predict(fh=fh, X=X)
