import copy
from datetime import timedelta
from typing import Dict, Optional, Tuple

import numpy as np
import pandas as pd
from loguru import logger

from hybridts.models.base import PrimaryModel, ResidualModel
from .features.engineering import create_features
from .features.holidays import create_holidays_prophet, get_brazilian_paydays
from .metrics.forecast import ForecastMetrics
from .preprocessing.processor import TimeSeriesProcessor

# Sentinel for "user did not pass this argument" — distinct from None
_UNSET = object()


class HybridForecaster:
    """
    Hybrid time series forecaster combining a primary baseline model (e.g. Prophet)
    with a secondary residual model (e.g. XGBoostModel, LightGBMModel).

    Args:
        primary_model: Primary model instance (must implement fit/predict).
        secondary_model: Secondary model instance (must implement fit/predict).
        test_size: Holdout size in days used in evaluate(). Default: 30.
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
        primary_model: PrimaryModel,
        secondary_model: ResidualModel,
        test_size: int = 30,
        paydays_set: Optional[set] = None, # TODO remove this asap and find an alternative
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
        primary_predictions: pd.DataFrame,
    ) -> pd.Series:
        """Residuals between actual values and primary model predictions."""
        y = df.set_index("ds")["y"]
        y.index = pd.PeriodIndex(y.index, freq="D")
        residuals = y.to_numpy() - primary_predictions["yhat"].to_numpy()
        return pd.Series(residuals, index=y.index)

    def fit(
        self,
        df: pd.DataFrame,
        holidays=_UNSET,
        features=_UNSET,
    ) -> "HybridForecaster":
        """
        Fit the hybrid model on the provided data.

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
                years=df["ds"].dt.year.unique(),
                country=self.holidays_country,
                state=self.holidays_state,
            )

        if features is _UNSET:
            paydays = self.paydays_set or get_brazilian_paydays(min_year, max_year + 1)
            features = create_features(
                df[["ds"]],
                paydays_set=paydays,
                min_year=min_year,
                max_year=max_year + 1,
                holidays_country=self.holidays_country,
                holidays_state=self.holidays_state,
            )

        self._fit_min_year = min_year
        self._fit_max_year = max_year

        self.primary_model.fit(df, holidays)

        primary_predictions = self.primary_model.predict(df[["ds"]])
        residuals = self._calculate_residuals(df, primary_predictions)

        self.secondary_model.fit(residuals, X_train=features)

        self._is_fitted = True
        logger.success(f"Model trained: {type(self.primary_model).__name__} + {type(self.secondary_model).__name__}")
        return self

    def predict(
        self,
        horizon: int,
        features=_UNSET,
        start_date: Optional[pd.Timestamp] = None,
    ) -> pd.DataFrame:
        """
        Generate forecasts for the next N days.

        Args:
            horizon: Number of days to forecast.
            features: Exogenous features for the forecast horizon.
                      - Omit: auto-generated using the same config as fit().
                      - None: predict without exogenous features.
                      - DataFrame: used as-is.
            start_date: Origin date for the forecast window (exclusive).
                        Defaults to today. Pass the last training date when
                        forecasting over a known holdout period (e.g. in evaluate()).

        Returns:
            DataFrame with columns:
                - data: forecast dates
                - forecast_primary_base: primary model baseline
                - residual_correction: secondary model residual adjustment
                - forecast_final: final hybrid forecast
        """
        if not self._is_fitted:
            raise RuntimeError("Model not fitted. Call fit() first.")

        origin = start_date if start_date is not None else pd.Timestamp.now().normalize()
        future_dates = pd.date_range(
            start=origin + timedelta(days=1),
            periods=horizon,
            freq="D",
        )
        df_future = pd.DataFrame({"ds": future_dates})

        primary_forecast = self.primary_model.predict(df_future)

        if features is _UNSET:
            paydays = self.paydays_set or get_brazilian_paydays(
                self._fit_min_year, self._fit_max_year + 1
            )
            features = create_features(
                df_future[["ds"]],
                paydays_set=paydays,
                min_year=self._fit_min_year,
                max_year=self._fit_max_year + 1,
                holidays_country=self.holidays_country,
                holidays_state=self.holidays_state,
            )

        fh = np.arange(1, horizon + 1)
        residual_forecast = self.secondary_model.predict(fh=fh, X=features)

        self.forecast_ = pd.DataFrame({
            "data": df_future["ds"],
            "forecast_primary_base": primary_forecast["yhat"].values,
            "residual_correction": residual_forecast.values,
            "forecast_final": (primary_forecast["yhat"].values + residual_forecast.values).round().astype(int),
        })
        self.forecast_plot_df_ = self.forecast_[["data", "forecast_final"]].rename(
            columns={"data": "ds", "forecast_final": "yhat"}
        )
        self.primary_plot_df_ = self.forecast_[["data", "forecast_primary_base"]].rename(
            columns={"data": "ds", "forecast_primary_base": "yhat"}
        )
        return self.forecast_

    def evaluate(
        self,
        df: pd.DataFrame,
        test_size: Optional[int] = None,
        holidays=_UNSET,
        features=_UNSET,
    ) -> Tuple[Dict[str, float], np.ndarray, np.ndarray]:
        """
        Evaluate the model using a holdout set (no data leakage).

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
            primary_model=copy.deepcopy(self.primary_model),
            secondary_model=copy.deepcopy(self.secondary_model),
            test_size=test_size,
            paydays_set=self.paydays_set,
            holidays_country=self.holidays_country,
            holidays_state=self.holidays_state,
        )
        temp_forecaster.fit(df_train, holidays=holidays, features=features)
        df_pred = temp_forecaster.predict(horizon=test_size, start_date=df_train["ds"].max())

        y_true = df_test["y"].to_numpy()
        y_pred_primary = df_pred["forecast_primary_base"].to_numpy()
        y_pred_final = df_pred["forecast_final"].to_numpy()

        self.primary_metrics_report_ = ForecastMetrics(y_true, y_pred_primary)
        self.metrics_report_ = ForecastMetrics(y_true, y_pred_final)

        self.metrics_ = self.metrics_report_.all_metrics()
        self.y_true_ = y_true
        self.y_pred_ = y_pred_final
        self.df_test_ = df_test
        self.eval_forecast_df_ = df_pred[["data", "forecast_final"]].rename(
            columns={"data": "ds", "forecast_final": "yhat"}
        )

        return self.metrics_, y_true, y_pred_final

    def evaluate_and_fit(
        self,
        df: pd.DataFrame,
        test_size: Optional[int] = None,
        holidays=_UNSET,
        features=_UNSET,
    ) -> Tuple["HybridForecaster", Dict[str, float]]:
        """
        Evaluate on a holdout set, then retrain on the full dataset.

        Returns:
            Tuple of (self, metrics dict).
        """
        test_size = test_size if test_size is not None else self.test_size

        metrics, y_true, y_pred = self.evaluate(
            df, test_size=test_size, holidays=holidays, features=features
        )

        self._validation_context = {
            "test_start": str(self.df_test_["ds"].min()),
            "test_end": str(self.df_test_["ds"].max()),
            "test_size": test_size,
            "y_true": y_true.tolist(),
            "y_pred": y_pred.tolist(),
        }

        logger.info("Retraining on full dataset...")
        self.fit(df, holidays=holidays, features=features)

        return self, metrics

    def plot_forecast(
        self,
        df: pd.DataFrame,
        forecast_df: Optional[pd.DataFrame] = None,
        show_primary: bool = True,
        ax=None,
        figsize: tuple = (12, 6),
        title: Optional[str] = None,
    ):
        """
        Plot actual values vs forecast.

        Args:
            df: Historical data with 'ds' and 'y' columns (used as actual values).
            forecast_df: Forecast DataFrame in plot format (ds, yhat).
                         If None, uses self.forecast_plot_df_ from the last predict() call.
            show_primary: Whether to overlay the primary model baseline. Default: True.
            ax: Matplotlib axes. If None, a new figure is created.
            figsize: Figure size.
            title: Plot title.

        Returns:
            fig, ax: Matplotlib figure and axes.
        """
        from .plotting import plot_forecast as _plot_forecast

        if forecast_df is None and not hasattr(self, "forecast_plot_df_"):
            raise RuntimeError("No forecast found. Call predict() first or pass forecast_df.")

        fc = forecast_df if forecast_df is not None else self.forecast_plot_df_
        primary = self.primary_plot_df_ if show_primary and hasattr(self, "primary_plot_df_") else None

        return _plot_forecast(
            actual_df=df,
            forecast_df=fc,
            primary_pred=primary,
            ax=ax,
            figsize=figsize,
            title=title,
        )

    def plot_evaluation(
        self,
        ax=None,
        figsize: tuple = (12, 6),
        title: Optional[str] = None,
    ):
        """
        Plot evaluation results from the last evaluate() or evaluate_and_fit() call.

        Args:
            ax: Matplotlib axes. If None, a new figure is created.
            figsize: Figure size.
            title: Plot title.

        Returns:
            fig, ax: Matplotlib figure and axes.

        Raises:
            RuntimeError: If evaluate() has not been called yet.
        """
        from .plotting import plot_forecast as _plot_forecast

        if not hasattr(self, "eval_forecast_df_"):
            raise RuntimeError("No evaluation results found. Call evaluate() or evaluate_and_fit() first.")

        return _plot_forecast(
            actual_df=self.df_test_,
            forecast_df=self.eval_forecast_df_,
            ax=ax,
            figsize=figsize,
            title=title or "Evaluation — Holdout Period",
        )
