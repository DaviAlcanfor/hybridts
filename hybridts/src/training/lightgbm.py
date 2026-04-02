from lightgbm import LGBMRegressor
from sktime.forecasting.model_selection import ForecastingGridSearchCV, ExpandingWindowSplitter
from sktime.performance_metrics.forecasting import MeanAbsolutePercentageError
from sktime.forecasting.compose import make_reduction
import numpy as np
import pandas as pd

class LightGBMTuner:
    """
    LightGBM-based residual forecaster for hybrid time series models.
    
    Trains LightGBM to model residuals from a baseline forecaster (e.g., Prophet).
    Supports cross-validated hyperparameter tuning via expanding window.
    
    Args:
        config: Configuration dictionary containing:
            - test_size: Number of periods for test set
            - models.lightgbm.param_grid: Grid search parameters
            - cv_params: Cross-validation window configuration
    """
    def __init__(self, config: dict):
        self.test_size = config['test_size']
        self.param_grid = config['models']['lightgbm']['param_grid']
        self.initial_window = config['cv_params']['initial_window']
        self.step_length = config['cv_params']['step_length'] 


    def train_cv(self, X_train: pd.DataFrame, residuals_train: pd.Series):
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

        cv_otimizado = ExpandingWindowSplitter(
            initial_window=self.initial_window,
            step_length=self.step_length,
            fh=fh
        )

        lgbm_forecaster = make_reduction(
            estimator=LGBMRegressor(
                random_state=42, 
                verbose=-1, 
                n_jobs=1,
            ),
            strategy='recursive'  
        )

        gscv_lgbm = ForecastingGridSearchCV(
            forecaster=lgbm_forecaster,
            cv=cv_otimizado,
            param_grid=self.param_grid,
            scoring=mape
        )

        gscv_lgbm.fit(y=residuals_train, X=X_train)

        return gscv_lgbm.best_forecaster_