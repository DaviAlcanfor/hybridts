# hybridts/models/base.py
from abc import ABC, abstractmethod
import numpy as np
import pandas as pd


class Model(ABC):
    """Marker base class for all models."""
    pass


class PrimaryModel(Model):
    @abstractmethod
    def fit(self, df: pd.DataFrame, **kwargs) -> "PrimaryModel":
        ...

    @abstractmethod
    def fit_static(self, df: pd.DataFrame, **kwargs) -> "PrimaryModel":
        ...

    @abstractmethod
    def predict(self, df: pd.DataFrame) -> pd.DataFrame:
        ...


class ResidualModel(Model):
    @abstractmethod
    def fit(self, residuals: pd.Series, **kwargs) -> "ResidualModel":
        ...

    @abstractmethod
    def fit_static(self, residuals: pd.Series, **kwargs) -> "ResidualModel":
        ...

    @abstractmethod
    def predict(self, fh: np.ndarray, **kwargs) -> pd.Series:
        ...