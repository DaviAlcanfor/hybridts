from typing import Optional

import pandas as pd
from sktime.forecasting.compose import make_reduction
from sktime.forecasting.model_selection import ExpandingWindowSplitter, ForecastingGridSearchCV
from sktime.performance_metrics.forecasting import MeanAbsolutePercentageError
from xgboost import XGBRegressor

from ..base import ResidualModel
from ...exceptions import ModelPredictionException, ModelTrainingException, model_error_handler


class XGBoostModel(ResidualModel):
    """
    XGBoost residual forecaster with optional cross-validation hyperparameter tuning.

    Args:
        param_grid: Parameter grid for grid search CV.
        static_params: Fixed parameters used by fit_static() (no CV).
        regressor_params: Base XGBRegressor parameters (applied in both modes).
        cv_initial_window: Initial window size for ExpandingWindowSplitter.
        cv_step_length: Step length for ExpandingWindowSplitter.
        window_length: Window length for fit_static().
        fh: Forecast horizon used during CV.
        strategy: sktime reduction strategy (e.g. "recursive", "direct").
        fh: Forecast horizon used during CV and prediction.
    """

    def __init__(
        self,
        param_grid: dict,
        static_params: dict,
        regressor_params: dict,
        cv_initial_window: int,
        cv_step_length: int,
        window_length: int,
        fh: int,
        strategy: str,
    ):
        self.param_grid = param_grid
        self.static_params = static_params
        self.regressor_params = regressor_params
        self.cv_initial_window = cv_initial_window
        self.cv_step_length = cv_step_length
        self.static_window_length = window_length
        self.strategy = strategy
        self.fh = fh

    @model_error_handler(ModelTrainingException)
    def fit(
        self,
        residuals_train: pd.Series,
        X_train: Optional[pd.DataFrame] = None,
    ) -> "XGBoostModel":
        """
        Fit XGBoost using GridSearchCV with an ExpandingWindowSplitter.

        Args:
            residuals_train: Residual series from the primary model.
            X_train: Exogenous feature DataFrame (optional).

        Returns:
            self
        """
        mape = MeanAbsolutePercentageError(symmetric=False)

        xgb_forecaster = make_reduction(
            estimator=XGBRegressor(**self.regressor_params),
            strategy=self.strategy,
        )

        cv = ExpandingWindowSplitter(
            initial_window=self.cv_initial_window,
            step_length=self.cv_step_length,
            fh=self.fh,
        )

        gscv = ForecastingGridSearchCV(
            forecaster=xgb_forecaster,
            cv=cv,
            param_grid=self.param_grid,
            scoring=mape,
            backend_params={"n_jobs": -1},
        )

        gscv.fit(y=residuals_train, X=X_train)
        self.model_ = gscv.best_forecaster_
        return self

    @model_error_handler(ModelTrainingException)
    def fit_static(
        self,
        residuals_train: pd.Series,
        X_train: Optional[pd.DataFrame] = None,
    ) -> "XGBoostModel":
        """
        Fit XGBoost with fixed parameters (no CV). Faster, ideal for daily retraining.

        Args:
            residuals_train: Residual series from the primary model.
            X_train: Exogenous feature DataFrame (optional).

        Returns:
            self
        """
        xgb_forecaster = make_reduction(
            estimator=XGBRegressor(**self.static_params),
            window_length=self.static_window_length,
            strategy=self.strategy,
        )
        xgb_forecaster.fit(y=residuals_train, X=X_train)
        self.model_ = xgb_forecaster
        return self

    @model_error_handler(ModelPredictionException)
    def predict(
        self,
        fh,
        X: Optional[pd.DataFrame] = None,
    ) -> pd.Series:
        """
        Generate predictions using the fitted XGBoost model.

        Args:
            fh: Forecast horizon (array or int).
            X: Exogenous feature DataFrame for the forecast period (optional).

        Returns:
            Series of predicted residuals.
        """
        return self.model_.predict(fh=fh, X=X)
