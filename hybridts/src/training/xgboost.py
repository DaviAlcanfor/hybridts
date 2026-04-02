
from sktime.forecasting.model_selection import ForecastingGridSearchCV, ExpandingWindowSplitter
from sktime.performance_metrics.forecasting import MeanAbsolutePercentageError
from sktime.forecasting.compose import make_reduction
from xgboost import XGBRegressor
import numpy as np

class XGBoostModel:
    """
    XGBoost-based residual forecaster for hybrid time series models.
    
    Trains XGBoost to model residuals from a baseline forecaster (e.g., Prophet).
    Supports both cross-validated hyperparameter tuning and static parameter training.
    
    Args:
        config: Configuration dictionary containing:
            - test_size: Number of periods for test set
            - models.xgboost.param_grid: Grid search parameters
            - models.xgboost.static: Static parameters for quick training
            - cv_params: Cross-validation window configuration
    """
    def __init__(self, config: dict):
        self.test_size = config['test_size']
        self.param_grid = config['models']['xgboost']['param_grid']
        self.static_params = config['models']['xgboost']['static']
        self.cv_initial_window = config['cv_params']['initial_window'] 
        self.cv_step_length = config['cv_params']['step_length']   
        self.static_window_length = 28

    def train_cv(self, X_train, residuals_train):
        """
        Treina o XGBoost usando GridSearch e Validação Cruzada (Expanding Window).
        Ideal para encontrar os melhores hiperparâmetros.
        """
        xgb_forecaster = make_reduction(
            estimator=XGBRegressor(tree_method="hist", random_state=42),
            strategy='recursive'  
        )

        cv = ExpandingWindowSplitter(
            initial_window=self.cv_initial_window, 
            step_length=self.cv_step_length,    
            fh=np.arange(1, self.test_size + 1)
        )

        gscv = ForecastingGridSearchCV(
            forecaster=xgb_forecaster,
            cv=cv,
            param_grid=self.param_grid,
            scoring=MeanAbsolutePercentageError(symmetric=False),
            backend_params={"n_jobs": -1}
        )

        gscv.fit(y=residuals_train, X=X_train)

        return gscv.best_forecaster_

    def train_static(self, X_train, residuals_train):
        """
        Treina o XGBoost com parâmetros fixos, sem fazer busca/CV.
        Mais rápido, ideal para retreinos diários quando os hiperparâmetros já são conhecidos.
        """
        
        xgb_forecaster = make_reduction(
            estimator=XGBRegressor(**self.static_params),
            window_length=self.static_window_length,
            strategy='recursive'
        )

        xgb_forecaster.fit(y=residuals_train, X=X_train)
        
        return xgb_forecaster