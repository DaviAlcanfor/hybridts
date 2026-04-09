from typing import Callable, Optional, Tuple

import pandas as pd


class TimeSeriesProcessor:
    """
    Time series data processor for preparing and splitting forecast data.

    Handles data loading, train/test splitting, and date utilities.
    """

    def df_train_test_split(
        self,
        df: pd.DataFrame,
        split_size: int,
    ) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """
        Splits a DataFrame into train and test sets by the last N rows.

        Args:
            df: DataFrame to split.
            split_size: Number of rows for the test set.

        Returns:
            Tuple of (df_train, df_test).
        """
        df = df.copy()

        if split_size > len(df):
            raise ValueError("Split size must be smaller than the DataFrame length.")
        if df.dropna().shape[0] != df.shape[0]:
            raise ValueError("DataFrame contains null values.")

        return df.iloc[:-split_size], df.iloc[-split_size:]

    def get_min_max_years(self, df: pd.DataFrame) -> Tuple[int, int]:
        """
        Returns the minimum and maximum year present in the 'ds' column.

        Args:
            df: DataFrame with a datetime 'ds' column.

        Returns:
            Tuple of (min_year, max_year).
        """
        if df["ds"].isnull().any():
            raise ValueError("Null values found in the 'ds' column.")

        return int(df["ds"].dt.year.min()), int(df["ds"].dt.year.max())

    def prepare_data(
        self,
        df: Optional[pd.DataFrame] = None,
        data_loader: Optional[Callable[[], pd.DataFrame]] = None,
    ) -> pd.DataFrame:
        """
        Loads, validates, and fills calendar gaps in time series data.

        Usage:
            processor.prepare_data(df=df)
            processor.prepare_data(data_loader=lambda: pd.read_parquet("data.parquet"))

        Args:
            df: DataFrame with columns ['ds', 'y'].
            data_loader: Callable that returns a DataFrame with columns ['ds', 'y'].

        Returns:
            Validated DataFrame with columns ['ds', 'y'], sorted by date, no gaps.
        """
        if df is not None:
            data = df.copy()
        elif data_loader is not None:
            data = data_loader()
        else:
            raise ValueError(
                "Provide data via 'df' (DataFrame) or 'data_loader' (callable)."
            )

        if "ds" not in data.columns or "y" not in data.columns:
            raise ValueError("DataFrame must contain columns 'ds' and 'y'.")

        data["y"] = data["y"].astype(float)

        if data["y"].isnull().any():
            raise ValueError("Null values detected in column 'y'.")
        if not (data["y"] >= 0).all():
            raise ValueError("Negative values are not allowed in column 'y'.")

        data["ds"] = pd.to_datetime(data["ds"])
        data = data.sort_values("ds").reset_index(drop=True)

        date_range = pd.date_range(data["ds"].min(), data["ds"].max(), freq="D")
        missing = set(date_range) - set(data["ds"])

        if missing:
            print(f"{len(missing)} missing dates found. Filling with zero...")
            data = data.set_index("ds")
            data = data.reindex(pd.DatetimeIndex(date_range), fill_value=0)
            data = data.rename_axis("ds").reset_index()

        return data[["ds", "y"]]
