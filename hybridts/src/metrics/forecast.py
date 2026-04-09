import numpy as np


class ForecastMetrics:
    """
    Computes and exposes forecast accuracy metrics for a y_true / y_pred pair.
    All metrics are calculated on instantiation and accessible as attributes.

    Args:
        y_true: Array of actual observed values.
        y_pred: Array of predicted values.

    Attributes:
        mae, mse, rmse, mape, smape, r_squared, bias

    Example:
        >>> report = ForecastMetrics(y_true, y_pred)
        >>> report.mape
        >>> report.summary()
        >>> report.all_metrics()
    """

    def __init__(
        self,
        y_true: np.ndarray,
        y_pred: np.ndarray,
    ):
        self.y_true = np.asarray(y_true, dtype=float)
        self.y_pred = np.asarray(y_pred, dtype=float)

        residuals     = self.y_true - self.y_pred
        nonzero       = self.y_true != 0
        abs_residuals = np.abs(residuals)

        self.mae = float(np.mean(abs_residuals))
        self.mse = float(np.mean(residuals ** 2))
        self.rmse = float(np.sqrt(self.mse))
        self.mape = float(np.mean(abs_residuals[nonzero] / np.abs(self.y_true[nonzero])) * 100)
        self.smape = self._smape(abs_residuals)
        self.r_squared = float(1 - np.sum(residuals ** 2) / np.sum((self.y_true - self.y_true.mean()) ** 2))
        self.bias = float(residuals.mean())

    def _smape(self, abs_residuals: np.ndarray) -> float:
        add     = np.abs(self.y_true) + np.abs(self.y_pred)
        nonzero = add != 0
        return float(np.mean(2 * abs_residuals[nonzero] / add[nonzero]) * 100)

    def mean_absolute_error(self) -> float:
        return self.mae

    def mean_squared_error(self) -> float:
        return self.mse

    def root_mean_squared_error(self) -> float:
        return self.rmse

    def mean_absolute_percentage_error(self) -> float:
        return self.mape

    def symmetric_mean_absolute_percentage_error(self) -> float:
        return self.smape

    def all_metrics(self) -> dict:
        """Returns all metrics as a dictionary."""
        return {
            "MAE":       self.mae,
            "MSE":       self.mse,
            "RMSE":      self.rmse,
            "MAPE":      self.mape,
            "sMAPE":     self.smape,
            "R-squared": self.r_squared,
            "Bias":      self.bias,
        }

    def summary(self) -> str:
        """Returns a formatted string of all metrics."""
        lines = ["Forecast Metrics Summary:"] + [f"{k}: {v:.4f}" for k, v in self.all_metrics().items()]
        return "\n".join(lines)
