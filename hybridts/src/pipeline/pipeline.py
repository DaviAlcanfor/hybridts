import json
import joblib
import mlflow
import mlflow.prophet
import mlflow.sklearn
import pandas as pd
import numpy as np
from datetime import datetime
from prophet import Prophet
from typing import Any, Dict, Tuple, Optional
from sklearn.metrics import mean_absolute_percentage_error, mean_squared_error

from hybridts.src.features.engineering import create_features
from hybridts.src.features.holidays import create_holidays_prophet, get_brazilian_paydays
from hybridts.src.features.data_processor import TimeSeriesProcessor
from hybridts.src.training.prophet import ProphetModel
from hybridts.src.training.xgboost import XGBoostModel
from hybridts.src.training.lightgbm import LightGBMTuner
from hybridts.defaults import get_default_config

class HybridForecaster:
    """
    Hybrid time series forecaster combining Prophet with gradient boosting.
    
    Supports three usage modes:
    1. Default configuration (no arguments)
    2. Custom dict configuration
    3. YAML configuration (via load_config)
    
    Args:
        config: Optional configuration dictionary. If None, uses sensible defaults.
        processor: Optional TimeSeriesProcessor. If None, creates one automatically.
        
    Example:
        >>> # Mode 1: Defaults
        >>> forecaster = HybridForecaster()
        >>> forecaster.fit(df, model="xgboost")
        
        >>> # Mode 2: Custom config
        >>> config = {'test_size': 30, 'models': {...}}
        >>> forecaster = HybridForecaster(config=config)
        
        >>> # Mode 3: YAML
        >>> from hybridts.config import load_config
        >>> config = load_config("settings.yaml")
        >>> forecaster = HybridForecaster(config=config)
    """
    def __init__(
        self, 
        config: Optional[dict] = None, 
        processor: Optional[TimeSeriesProcessor] = None
    ):
        # Use defaults if no config provided
        self.config = config if config is not None else get_default_config()
        
        # Create processor if not provided
        self.processor = processor if processor is not None else TimeSeriesProcessor()
        
        self.prophet_model = None
        self.ml_model = None
        self.best_prophet_params = None
        self.escolha_modelo = None
        self._is_fitted = False
    
    def _calculate_residuals(
            self, 
            df: pd.DataFrame, 
            prophet_predictions: pd.DataFrame
        ) -> pd.Series:
        """
        Calculates residuals between actual values and Prophet predictions.

        Args:
            df (pd.DataFrame): DataFrame with columns 'ds' and 'y' (actual values).
            prophet_predictions (pd.DataFrame): DataFrame with Prophet predictions ('yhat').

        Returns:
            pd.Series: Series of residuals indexed by date.
        """
        
        y = df.set_index('ds')['y']
        y.index = pd.PeriodIndex(y.index, freq='D')
        residuals = y.values - prophet_predictions['yhat'].values

        return pd.Series(residuals, index=y.index)
    
    def _train_ml_model(
            self, 
            X: pd.DataFrame, 
            residuals: pd.Series
        ):
        """
        Trains the ML model on Prophet residuals.

        Args:
            X (pd.DataFrame): Feature matrix.
            residuals (pd.Series): Target residuals.

        Returns:
            Any: Trained ML model.

        Raises:
            ValueError: If an invalid model is selected.
        """

        match self.escolha_modelo:
            case "XGBoost":
                tuner = XGBoostModel(self.config)
                return tuner.train_cv(X, residuals)
            case "sXGBoost":
                tuner = XGBoostModel(self.config)
                return tuner.train_static(X, residuals)
            case "LightGBM":
                tuner = LightGBMTuner(self.config)
                return tuner.train_cv(X, residuals)
            case _:
                raise ValueError(f"Modelo '{self.escolha_modelo}' inválido.")

    def fit(
            self,
            df: pd.DataFrame, 
            model: str = "xgboost",
            paydays_set: Optional[set] = None,
            holidays_country: str = "BR",
            holidays_state: Optional[str] = None
        ) -> 'HybridForecaster':
        """
        Fits the hybrid model (Prophet + ML) on the provided data.

        Args:
            df: Training data with columns 'ds' (datetime) and 'y' (values)
            model: ML model choice - "xgboost", "sxgboost" (fast), or "lightgbm"
            paydays_set: Optional set of payday dates for feature engineering.
                        If None, generates Brazilian paydays automatically.
            holidays_country: Country code for holidays (default: "BR" for Brazil)
            holidays_state: State/subdivision code (default: None = country-level only)

        Returns:
            Self (fitted forecaster)

        Raises:
            ValueError: If an invalid model is selected
            
        Example:
            >>> forecaster = HybridForecaster()
            >>> forecaster.fit(df, model="xgboost")
            >>> # Or with custom holidays
            >>> forecaster.fit(df, model="lightgbm", holidays_country="US")
        """
        # Normalize model name (case-insensitive, handle variants)
        model_lower = model.lower()
        if model_lower in ["xgboost", "xgb"]:
            self.escolha_modelo = "XGBoost"
        elif model_lower in ["sxgboost", "sxgb", "static_xgboost"]:
            self.escolha_modelo = "sXGBoost"
        elif model_lower in ["lightgbm", "lgbm", "lgb"]:
            self.escolha_modelo = "LightGBM"
        else:
            raise ValueError(
                f"Invalid model '{model}'. Choose from: 'xgboost', 'sxgboost', 'lightgbm'"
            )
        
        # Auto-generate paydays if not provided
        if paydays_set is None:
            min_year, max_year = self.processor.get_min_max_years(df)
            paydays_set = get_brazilian_paydays(min_year, max_year + 1)
        
        # Store for later use in predict
        self._paydays_set = paydays_set
        self._holidays_country = holidays_country
        self._holidays_state = holidays_state
        
        # Get years and create holidays
        min_year, max_year = self.processor.get_min_max_years(df)
        years_in_data = df['ds'].dt.year.unique()
        prophet_holidays = create_holidays_prophet(
            years=years_in_data,
            country=holidays_country,
            state=holidays_state
        )
        
        # Train Prophet
        prophet_tuner = ProphetModel(self.config)
        self.best_prophet_params = prophet_tuner.find_best_params(
            df_train=df, 
            prophet_holidays=prophet_holidays
        )
        
        self.prophet_model = Prophet(
            holidays=prophet_holidays,
            yearly_seasonality=True,
            weekly_seasonality=True,
            daily_seasonality=False,
            **self.best_prophet_params
        )
        self.prophet_model.fit(df)
        
        # Train ML on residuals
        prophet_predictions = self.prophet_model.predict(df[['ds']])
        residuals = self._calculate_residuals(df, prophet_predictions)
        
        X = create_features(
            df[['ds']], 
            paydays_set, 
            min_year, 
            max_year + 1,
            holidays_country=holidays_country,
            holidays_state=holidays_state
        )
        self.ml_model = self._train_ml_model(X, residuals)
        
        self._is_fitted = True
        print(f"✓ Model trained: Prophet + {self.escolha_modelo}")
        
        return self

    def predict(
            self,
            horizon: int,
            paydays_set: Optional[set] = None
        ) -> pd.DataFrame:
        """
        Generates forecasts for the next N days.

        Args:
            horizon: Number of days to forecast
            paydays_set: Optional set of paydays. If None, uses paydays from fit()

        Returns:
            DataFrame with columns:
            - data: Forecast dates
            - forecast_prophet_base: Prophet baseline forecast
            - ajuste_residual_ml: ML residual corrections
            - forecast_hibrido_final: Final hybrid forecast (baseline + corrections)

        Raises:
            RuntimeError: If the model is not fitted

        Example:
            >>> forecaster.fit(df, model="xgboost")
            >>> forecast = forecaster.predict(horizon=30)
            >>> print(forecast[['data', 'forecast_hibrido_final']])
        """
        if not self._is_fitted:
            raise RuntimeError("Model not fitted. Call fit() first.")
        
        # Use paydays from fit if not provided
        if paydays_set is None:
            paydays_set = self._paydays_set

        from datetime import timedelta
        hoje = pd.Timestamp.now().normalize()
        datas_futuras = pd.date_range(start=hoje + timedelta(days=1), periods=horizon, freq='D')
        df_futuro = pd.DataFrame({'ds': datas_futuras})
        
        previsao_prophet = self.prophet_model.predict(df_futuro)
        
        min_year, max_year = self.processor.get_min_max_years(df_futuro)
        X_futuro = create_features(
            df_futuro[['ds']], 
            paydays_set, 
            min_year, 
            max_year + 1,
            holidays_country=self._holidays_country,
            holidays_state=self._holidays_state
        )
        
        fh = np.arange(1, horizon + 1)
        previsao_ml = self.ml_model.predict(fh=fh, X=X_futuro)
        
        return pd.DataFrame({
            'data': df_futuro['ds'],
            'forecast_prophet_base': previsao_prophet['yhat'].values,
            'ajuste_residual_ml': previsao_ml.values,
            'forecast_hibrido_final': (previsao_prophet['yhat'].values + previsao_ml.values).astype(int)
        })


    def validate(
            self,
            df: pd.DataFrame,
            escolha_modelo: str,
            paydays_set: set,
            test_size: Optional[int] = None
        ) -> Tuple[Dict[str, float], np.ndarray, np.ndarray]:
        """
        Evaluates the model using a holdout set (no data leakage).

        Args:
            df (pd.DataFrame): Full dataset with columns 'ds' and 'y'.
            escolha_modelo (str): ML model choice.
            paydays_set (set): Set of paydays for feature engineering.
            test_size (Optional[int]): Number of days for the test set.

        Returns:
            Dict[str, float]: Dictionary with validation metrics (MAPE, RMSE, MAE).
        """
        test_size = test_size or self.config['test_size']
        
        df_train, df_test = self.processor.df_train_test_split(df, test_size)
        
        temp_forecaster = HybridForecaster(self.config, self.processor)
        temp_forecaster.fit(df_train, escolha_modelo, paydays_set)
        
        df_pred = temp_forecaster.predict(horizon=test_size, paydays_set=paydays_set)
        
        y_true = df_test['y'].values
        y_pred = df_pred['forecast_hibrido_final'].values
        
        metrics = {
            'mape': mean_absolute_percentage_error(y_true, y_pred) * 100,
            'rmse': np.sqrt(mean_squared_error(y_true, y_pred)),
            'mae': np.mean(np.abs(y_true - y_pred)),
            'mdape': np.median(np.abs((y_true - y_pred) / y_true)) * 100
        }
        
        print(f" Validation Metrics (holdout {test_size} dias):")
        print(f"   MAPE: {metrics['mape']:.2f}%")
        print(f"   RMSE: {metrics['rmse']:,.0f}")
        print(f"   MAE:  {metrics['mae']:,.0f}")
        print(f"   MdAPE: {metrics['mdape']:.2f}%")
        
        return metrics, y_true, y_pred
    

  

    def validate_and_fit(
        self,
        df: pd.DataFrame,
        escolha_modelo: str,
        paydays_set: set,
        test_size: Optional[int] = None
    ) -> Tuple['HybridForecaster', Dict[str, float]]:
        """
        Validates the model, then retrains on the full dataset.

        Args:
            df (pd.DataFrame): Full dataset with columns 'ds' and 'y'.
            escolha_modelo (str): ML model choice.
            paydays_set (set): Set of paydays for feature engineering.
            test_size (Optional[int]): Number of days for the test set.

        Returns:
            Tuple[HybridForecaster, Dict[str, float]]: Fitted forecaster and validation metrics.
        """

        test_size = test_size or self.config['test_size']
        df_train, df_test = self.processor.df_train_test_split(df, test_size)

        metrics, y_true, y_pred = self.validate(df, escolha_modelo, paydays_set, test_size)  # ← recebe os 3
        
        self._validation_context = {
            'test_start': str(df_test['ds'].min()),
            'test_end': str(df_test['ds'].max()),
            'test_size': test_size,
            'y_true': y_true.tolist(),
            'y_pred': y_pred.tolist()
        }
        
        print(" Retreinando no dataset completo...")
        self.fit(df, escolha_modelo, paydays_set)
        
        return self, metrics

    

    def save_to_mlflow(
        self,
        target: str,
        metrics: dict = None,
        experiment_path: Optional[str] = None,
        registry_path: Optional[str] = None,
        tracking_uri: Optional[str] = None
    ) -> str:
        """
        Saves the trained model to MLflow and writes the run_id to a JSON file.

        Args:
            target (str): Target metric (e.g., 'TPV', 'MAT', 'revenue').
            metrics (dict): Validation metrics from validate().
            experiment_path (str): MLflow experiment path. 
                                  Default: f"tpv_forecast_{target}"
            registry_path (str): Path to save the JSON registry file.
                                Default: "./models/latest_model.json"
            tracking_uri (str): MLflow tracking URI (e.g., "http://localhost:5000").
                               Default: None (uses local ./mlruns directory)

        Returns:
            str: MLflow run ID.
        
        Example:
            # Local MLflow (default)
            run_id = forecaster.save_to_mlflow(target="TPV", metrics=metrics)
            
            # Remote MLflow server
            run_id = forecaster.save_to_mlflow(
                target="TPV",
                tracking_uri="http://mlflow.company.com",
                experiment_path="/shared/experiments/tpv_forecast"
            )
        """
        if not self._is_fitted:
            raise RuntimeError("Modelo não treinado. Chame fit() primeiro.")

        # Configurar tracking URI se fornecido
        if tracking_uri:
            mlflow.set_tracking_uri(tracking_uri)

        # Defaults sensatos
        experiment_path = experiment_path or f"tpv_forecast_{target}"
        registry_path = registry_path or "./models/latest_model.json"

        mlflow.set_experiment(experiment_path)

        run_name = f"{target}_{self.escolha_modelo}_{datetime.now().strftime('%Y%m')}"

        with mlflow.start_run(run_name=run_name):
            mlflow.log_params({
                "target": target,
                "ml_model": self.escolha_modelo,
                "prophet_params": str(self.best_prophet_params),
                "test_size": self.config['test_size'],
            })

            if metrics:
                mlflow.log_metrics(metrics)

            mlflow.prophet.log_model(self.prophet_model, "prophet_model")
            mlflow.sklearn.log_model(self.ml_model, "ml_model")

            if hasattr(self, '_validation_context'):
                mlflow.log_dict(self._validation_context, "validation_context.json")


            run_id = mlflow.active_run().info.run_id

            registry = {
                "run_id": run_id,
                "target": target,
                "modelo": self.escolha_modelo,
                "mape": metrics.get('mape') if metrics else None,
                "mdape": metrics.get('mdape') if metrics else None,  
                "trained_at": str(datetime.now().date())
            }

        # Criar diretório se não existir
        import os
        os.makedirs(os.path.dirname(registry_path) or ".", exist_ok=True)

        with open(registry_path, "w") as f:
            json.dump(registry, f)

        print(f"✓ Modelo salvo no MLflow — Run ID: {run_id}")
        print(f"✓ Registry salvo em: {registry_path}")
        return run_id

    @classmethod
    def load_from_mlflow(
            cls,
            config: dict,
            processor: 'TimeSeriesProcessor',
            registry_path: Optional[str] = None,
            tracking_uri: Optional[str] = None
        ) -> Tuple['HybridForecaster', dict]:
        """
        Loads a trained model from MLflow using the JSON registry.

        Args:
            config (dict): Pipeline config from settings.yaml or Config object.
            processor (TimeSeriesProcessor): Processor instance.
            registry_path (str): Path to the JSON registry file.
                                Default: "./models/latest_model.json"
            tracking_uri (str): MLflow tracking URI.
                               Default: None (uses local ./mlruns)

        Returns:
            Tuple[HybridForecaster, dict]: Loaded forecaster and registry metadata.
        
        Example:
            config = load_config("config.yaml")
            processor = TimeSeriesProcessor()
            forecaster, metadata = HybridForecaster.load_from_mlflow(
                config=config,
                processor=processor,
                registry_path="./models/latest_model.json"
            )
        """
        registry_path = registry_path or "./models/latest_model.json"
        
        if tracking_uri:
            mlflow.set_tracking_uri(tracking_uri)

        with open(registry_path, "r") as f:
            registry = json.load(f)

        run_id = registry['run_id']

        instance = cls(config=config, processor=processor)
        instance.prophet_model = mlflow.prophet.load_model(f"runs:/{run_id}/prophet_model")
        instance.ml_model = mlflow.sklearn.load_model(f"runs:/{run_id}/ml_model")
        instance.escolha_modelo = registry['modelo']
        instance._is_fitted = True

        print(f"✓ Modelo carregado — Run ID: {run_id}")
        print(f"  Treinado em: {registry['trained_at']} | MAPE: {registry.get('mape', 'N/A')}")

        return instance, registry