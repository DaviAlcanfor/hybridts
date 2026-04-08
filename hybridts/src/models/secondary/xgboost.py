import pandas as pd
from sktime.forecasting.model_selection import ForecastingGridSearchCV, ExpandingWindowSplitter
from sktime.performance_metrics.forecasting import MeanAbsolutePercentageError
from sktime.forecasting.compose import make_reduction
from xgboost import XGBRegressor

from hybridts.src.exception.model_exception import ModelTrainingException, ModelPredictionException, model_error_handler


class XGBoostTuner():
    def __init__(
        self, 
        test_size: int, 
        param_grid: dict, 
        static_params: dict, 
        cv_initial_window: int, 
        cv_step_length: int,
        window_length: int,
        fh: int,
        strategy: str,
        regressor_params: dict
    ):
        self.test_size = test_size
        self.param_grid = param_grid
        self.static_params = static_params
        self.cv_initial_window = cv_initial_window
        self.cv_step_length = cv_step_length
        self.static_window_length = window_length
        self.regressor_params = regressor_params
        self.strategy = strategy
        self.fh = fh

    @model_error_handler(ModelTrainingException)
    def fit(self, X_train, residuals_train) -> 'XGBoostTuner':
        """
        Treina o XGBoost usando GridSearch e Validação Cruzada (Expanding Window).
        Ideal para encontrar os melhores hiperparâmetros.
        
        Args:
            X_train: pd.DataFrame com as features exógenas (feriados, paydays, etc).
            residuals_train: pd.Series com os resíduos do modelo base (Prophet).
            
        Returns:
            xgb_model: O melhor forecaster XGBoost já treinado.
        """
        mape = MeanAbsolutePercentageError(symmetric=False)
        
        xgb_forecaster = make_reduction(
            estimator=XGBRegressor(**self.regressor_params),
            strategy=self.strategy  
        )

        cv = ExpandingWindowSplitter(
            initial_window=self.cv_initial_window, 
            step_length=self.cv_step_length,    
            fh=self.fh
        )

        gscv = ForecastingGridSearchCV(
            forecaster=xgb_forecaster,
            cv=cv,
            param_grid=self.param_grid,
            scoring=mape,
            backend_params={"n_jobs": -1} # garante que seja mais rápido usando todos os núcleos
        )

        gscv.fit(y=residuals_train, X=X_train)
        
        self.model_ = gscv.best_forecaster_
        return self

    @model_error_handler(ModelTrainingException)
    def fit_static(self, X_train, residuals_train) -> 'XGBoostTuner':
        """
        Treina o XGBoost com parâmetros fixos, sem fazer busca/CV.
        Mais rápido, ideal para retreinos diários quando os hiperparâmetros já são conhecidos.
        
        Args:
            X_train: pd.DataFrame com as features exógenas (feriados, paydays, etc).
            residuals_train: pd.Series com os resíduos do modelo base (Prophet).
        Returns:
            xgb_model: Forecaster XGBoost treinado com parâmetros fixos.
        """
        
        xgb_forecaster = make_reduction(
            estimator=XGBRegressor(**self.static_params),
            window_length=self.static_window_length,
            strategy=self.strategy
        )

        xgb_forecaster.fit(y=residuals_train, X=X_train)
        
        self.model_ = xgb_forecaster
        return self
    
    @model_error_handler(ModelPredictionException)
    def predict(self, X_test: pd.DataFrame = None) -> pd.Series:
        """
        Faz previsões usando o modelo XGBoost treinado.
        
        Args:
            X_test: pd.DataFrame com as features exógenas para o período de teste.
            
        Returns:
            predictions: np.ndarray com as previsões do modelo XGBoost.
        """
        return self.model_.predict(fh=self.fh, X=X_test)
    