import pandas as pd
import numpy as np
from datetime import timedelta
from typing import Dict, Tuple, Optional
from loguru import logger

from hybridts.src.features.engineering import create_features
from hybridts.src.features.holidays import create_holidays_prophet, get_brazilian_paydays
from hybridts.src.features.data_processor import TimeSeriesProcessor
from hybridts.src.metrics.forecast import ForecastMetrics

# Sentinel for "user did not pass this argument" — distinct from None 
_UNSET = object()


class HybridForecaster:
    """
    Hybrid time series forecaster combining a primary baseline model (e.g. Prophet)
    with a secondary residual model (e.g. XGBoost, LightGBM).

    Args:
        primary_model: Primary model instance (must implement fit/predict).
        secondary_model: Secondary model instance (must implement fit/predict).
        test_size: Holdout size in days used in validate(). Default: 30.
        paydays_set: Set of payday Timestamps for feature engineering.
                     If None and features are auto-generated, Brazilian paydays are used.
        holidays_country: Country code for auto-generated holidays. Default: "BR".
        holidays_state: State/subdivision for auto-generated holidays. Default: None.

    Example:
        >>> forecaster = HybridForecaster(
        ...     primary_model=ProphetModel(param_grid=..., cv_params=...),
        ...     secondary_model=XGBoostModel(...),
        ...     test_size=30,
        ... )
        >>> forecaster.fit(df)
        >>> forecast = forecaster.predict(horizon=30)
    """

    def __init__(
        self,
        primary_model,
        secondary_model,
        test_size: int = 30,
        paydays_set: Optional[set] = None,
        holidays_country: str = "BR",
        holidays_state: Optional[str] = None,
    ):
        self.primary_model = primary_model
        self.secondary_model = secondary_model
        self.test_size = test_size
        self.paydays_set = paydays_set
        self.holidays_country = holidays_country
        self.holidays_state = holidays_state
        self.processor = TimeSeriesProcessor()
        self._is_fitted = False

    def _calculate_residuals(
        self,
        df: pd.DataFrame,
        primary_predictions: pd.DataFrame
    ) -> pd.Series:
        """Residuals between actual values and primary model predictions."""
        y = df.set_index('ds')['y']
        y.index = pd.PeriodIndex(y.index, freq='D')
        residuals = y.values - primary_predictions['yhat'].values
        return pd.Series(residuals, index=y.index)

    def fit(
        self,
        df: pd.DataFrame,
        holidays=_UNSET,
        features=_UNSET,
    ) -> 'HybridForecaster':
        """
        Fits the hybrid model on the provided data.

        Args:
            df: Training data with columns 'ds' (datetime) and 'y' (float).
            holidays: Holiday DataFrame for the primary model.
                      - Omit: auto-generated from holidays_country/state.
                      - None: no holidays used.
                      - DataFrame: used as-is.
            features: Exogenous feature DataFrame for the secondary model.
                      - Omit: auto-generated from paydays_set and holidays config.
                      - None: secondary model trained without exogenous features.
                      - DataFrame: used as-is.

        Returns:
            self
        """
        min_year, max_year = self.processor.get_min_max_years(df)

        if holidays is _UNSET:
            holidays = create_holidays_prophet(
                years=df['ds'].dt.year.unique(),
                country=self.holidays_country,
                state=self.holidays_state
            )

        if features is _UNSET:
            paydays = self.paydays_set or get_brazilian_paydays(min_year, max_year + 1)
            features = create_features(
                df[['ds']],
                paydays_set=paydays,
                min_year=min_year,
                max_year=max_year + 1,
                holidays_country=self.holidays_country,
                holidays_state=self.holidays_state
            )

        self._fit_min_year = min_year
        self._fit_max_year = max_year

        # Train primary model
        self.primary_model.fit(df, holidays)

        # Train secondary model on residuals
        primary_predictions = self.primary_model.predict(df[['ds']])
        residuals = self._calculate_residuals(df, primary_predictions)

        if features is not None:
            self.secondary_model.fit(features, residuals)
        else:
            self.secondary_model.fit(residuals)

        self._is_fitted = True
        logger.success(f"Model trained: {type(self.primary_model).__name__} + {type(self.secondary_model).__name__}")

        return self

    def predict(
        self,
        horizon: int,
        features=_UNSET,
    ) -> pd.DataFrame:
        """
        Generates forecasts for the next N days.

        Args:
            horizon: Number of days to forecast.
            features: Exogenous features for the forecast horizon.
                      - Omit: auto-generated using same config as fit().
                      - None: predict without exogenous features.
                      - DataFrame: used as-is.

        Returns:
            DataFrame with columns:
                - data: forecast dates
                - forecast_primary_base: primary model baseline
                - residual_correction: secondary model residual adjustment
                - forecast_final: final hybrid forecast
        """
        if not self._is_fitted:
            raise RuntimeError("Model not fitted. Call fit() first.")

        future_dates = pd.date_range(
            start=pd.Timestamp.now().normalize() + timedelta(days=1),
            periods=horizon,
            freq='D'
        )
        df_future = pd.DataFrame({'ds': future_dates})

        primary_forecast = self.primary_model.predict(df_future)

        if features is _UNSET:
            paydays = self.paydays_set or get_brazilian_paydays(
                self._fit_min_year, self._fit_max_year + 1
            )
            features = create_features(
                df_future[['ds']],
                paydays_set=paydays,
                min_year=self._fit_min_year,
                max_year=self._fit_max_year + 1,
                holidays_country=self.holidays_country,
                holidays_state=self.holidays_state
            )

        fh = np.arange(1, horizon + 1)
        if features is not None:
            residual_forecast = self.secondary_model.predict(fh=fh, X=features)
        else:
            residual_forecast = self.secondary_model.predict(fh=fh)

        return pd.DataFrame({
            'data': df_future['ds'],
            'forecast_primary_base': primary_forecast['yhat'].values,
            'residual_correction': residual_forecast.values,
            'forecast_final': (primary_forecast['yhat'].values + residual_forecast.values).astype(int)
        })

    def evaluate(
        self,
        df: pd.DataFrame,
        test_size: Optional[int] = None,
        holidays=_UNSET,
        features=_UNSET,
    ) -> Tuple[Dict[str, float], np.ndarray, np.ndarray]:
        """
        Evaluates the model using a holdout set (no data leakage).

        Args:
            df: Full dataset with columns 'ds' and 'y'.
            test_size: Holdout size in days. Defaults to self.test_size.
            holidays: Passed through to fit(). See fit() docstring.
            features: Passed through to fit(). See fit() docstring.

        Returns:
            Tuple of (metrics dict, y_true array, y_pred array).
            Metrics are also stored as self.metrics_, self.y_true_, self.y_pred_.
        """
        test_size = test_size if test_size is not None else self.test_size

        df_train, df_test = self.processor.df_train_test_split(df, test_size)

        temp_forecaster = HybridForecaster(
            primary_model=self.primary_model,
            secondary_model=self.secondary_model,
            test_size=test_size,
            paydays_set=self.paydays_set,
            holidays_country=self.holidays_country,
            holidays_state=self.holidays_state,
        )
        temp_forecaster.fit(df_train, holidays=holidays, features=features)
        df_pred = temp_forecaster.predict(horizon=test_size)

        y_true = df_test['y'].values
        y_pred_primary = df_pred['forecast_primary_base'].values
        y_pred_final   = df_pred['forecast_final'].values

        self.primary_metrics_report_ = ForecastMetrics(y_true, y_pred_primary)
        self.metrics_report_ = ForecastMetrics(y_true, y_pred_final)

        self.metrics_ = self.metrics_report_.all_metrics()
        self.y_true_  = y_true
        self.y_pred_  = y_pred_final

        return self.metrics_, y_true, y_pred_final

    def evaluate_and_fit(
        self,
        df: pd.DataFrame,
        test_size: Optional[int] = None,
        holidays=_UNSET,
        features=_UNSET,
    ) -> Tuple['HybridForecaster', Dict[str, float]]:
        """
        Evaluates on a holdout set, then retrains on the full dataset.

        Returns:
            Tuple of (self, metrics dict).
        """
        test_size = test_size if test_size is not None else self.test_size
        _, df_test = self.processor.df_train_test_split(df, test_size)

        metrics, y_true, y_pred = self.evaluate(
            df, test_size=test_size, holidays=holidays, features=features
        )

        self._validation_context = {
            'test_start': str(df_test['ds'].min()),
            'test_end': str(df_test['ds'].max()),
            'test_size': test_size,
            'y_true': y_true.tolist(),
            'y_pred': y_pred.tolist()
        }

        logger.info("Retraining on full dataset...")
        self.fit(df, holidays=holidays, features=features)

        return self, metrics
