import pytest
import pandas as pd
import numpy as np


@pytest.fixture
def sample_timeseries():
    dates = pd.date_range("2022-01-01", periods=400, freq="D")
    trend = np.linspace(10000, 15000, 400)
    seasonality = 2000 * np.sin(2 * np.pi * np.arange(400) / 7)
    noise = np.random.default_rng(42).normal(0, 500, 400)
    values = np.maximum(trend + seasonality + noise, 0)
    return pd.DataFrame({"ds": dates, "y": values})


class FakePrimaryModel:
    def fit(self, df, holidays=None):
        self._n = len(df)
        return self

    def predict(self, df):
        return pd.DataFrame({"yhat": np.ones(len(df)) * 1000})


class FakeSecondaryModel:
    def fit(self, X_or_residuals, residuals=None, X=None):
        return self

    def predict(self, fh, X=None):
        return pd.Series(np.zeros(len(fh)), index=fh)


@pytest.fixture
def fake_primary():
    return FakePrimaryModel()


@pytest.fixture
def fake_secondary():
    return FakeSecondaryModel()
