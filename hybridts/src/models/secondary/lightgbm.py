from typing import Optional

from lightgbm import LGBMRegressor
from sktime.forecasting.model_selection import ForecastingGridSearchCV, ExpandingWindowSplitter
from sktime.performance_metrics.forecasting import MeanAbsolutePercentageError
from sktime.forecasting.compose import make_reduction
import numpy as np
import pandas as pd

from hybridts.src.exception.model_exception import ModelTrainingException, ModelPredictionException, model_error_handler


class LightGBMTuner():
    def __init__(
        self, 
        lgbm_regressor_params: dict = None,
        test_size: int = None,
        initial_window: int = None,
        step_length: int = None,
        window_length: int = None,
        param_grid: Optional[dict] = None,
    ):
        self.strategy = lgbm_regressor_params.pop('strategy')
        self.lgbm_regressor_params = lgbm_regressor_params
        self.test_size = test_size
        self.param_grid = param_grid
        self.initial_window = initial_window
        self.step_length = step_length 
        self.window_length = window_length  


    @model_error_handler(ModelTrainingException)
    def fit(
        self,
        X_train: pd.DataFrame,
        residuals_train: pd.Series
    ) -> 'LightGBMTuner':
        """
        Treina um regressor LightGBM (via sktime) nos resíduos da série temporal 
        usando validação cruzada (Expanding Window).

        Args:
            X_train: pd.DataFrame com as features exógenas (feriados, paydays, etc).
            residuals_train: pd.Series com os resíduos do modelo base (Prophet).

        Returns:
            best_lgbm_model: O melhor forecaster LightGBM já treinado.
        """
        fh = np.arange(1, self.test_size + 1)
        mape = MeanAbsolutePercentageError(symmetric=False)

        cv = ExpandingWindowSplitter(
            initial_window=self.initial_window,
            step_length=self.step_length,
            fh=fh
        )

        lgbm_forecaster = make_reduction(
            estimator=LGBMRegressor(
                **self.lgbm_regressor_params
            ),
            strategy=self.strategy
        )

        gscv_lgbm = ForecastingGridSearchCV(
            forecaster=lgbm_forecaster,
            cv=cv,
            param_grid=self.param_grid,
            scoring=mape
        )

        gscv_lgbm.fit(y=residuals_train, X=X_train)
        self.model_ = gscv_lgbm.best_forecaster_
        return self
    
    
    @model_error_handler(ModelTrainingException)
    def fit_static(
        self,
        X_train: pd.DataFrame,
        residuals_train: pd.Series
    ) -> 'LightGBMTuner':
        """
        Treina o LightGBM com parâmetros fixos, sem fazer busca/CV.
        Útil para treinamento rápido ou quando os recursos são limitados.

        Args:
            X_train: pd.DataFrame com as features exógenas (feriados, paydays, etc).
            residuals_train: pd.Series com os resíduos do modelo base (Prophet).
        Returns:
            lgbm_model: Forecaster LightGBM treinado com parâmetros fixos.
        """
        
        self.model_ = make_reduction(
            estimator=LGBMRegressor(
                **self.lgbm_regressor_params
            ),
            strategy=self.strategy,
            window_length=self.window_length
        )
        self.model_.fit(y=residuals_train, X=X_train)
        return self
    
    
    @model_error_handler(ModelPredictionException)
    def predict(self, X: pd.DataFrame = None) -> pd.Series:
        """
        Faz previsões usando o modelo LightGBM treinado.

        Args:
            X: pd.DataFrame com as features exógenas para o período de previsão.

        Returns:
            predictions: np.ndarray com as previsões do modelo LightGBM.
        """
        return self.model_.predict(fh=self.test_size, X=X)