import numpy as np
import pandas as pd
import pytest

from hybridts import TimeSeriesProcessor


@pytest.fixture
def df_with_gaps():
    dates = pd.date_range("2023-01-01", periods=100, freq="D").tolist()
    dates = [d for i, d in enumerate(dates) if i % 10 != 5]
    return pd.DataFrame({"ds": dates, "y": np.random.uniform(1000, 2000, len(dates))})


def test_train_test_split(sample_timeseries):
    df_train, df_test = TimeSeriesProcessor().df_train_test_split(sample_timeseries, split_size=30)

    assert len(df_train) == len(sample_timeseries) - 30
    assert len(df_test) == 30


def test_train_test_split_invalid_size(sample_timeseries):
    with pytest.raises(ValueError, match="Split size must be smaller"):
        TimeSeriesProcessor().df_train_test_split(sample_timeseries, split_size=9999)


def test_train_test_split_rejects_nulls():
    df = pd.DataFrame({"ds": pd.date_range("2023-01-01", periods=10), "y": [1, None] + [1] * 8})

    with pytest.raises(ValueError, match="null values"):
        TimeSeriesProcessor().df_train_test_split(df, split_size=3)


def test_get_min_max_years(sample_timeseries):
    min_year, max_year = TimeSeriesProcessor().get_min_max_years(sample_timeseries)

    assert min_year == sample_timeseries["ds"].dt.year.min()
    assert max_year == sample_timeseries["ds"].dt.year.max()


def test_prepare_data_basic(sample_timeseries):
    result = TimeSeriesProcessor().prepare_data(df=sample_timeseries)

    assert list(result.columns) == ["ds", "y"]
    assert len(result) == len(sample_timeseries)


def test_prepare_data_fills_gaps(df_with_gaps):
    result = TimeSeriesProcessor().prepare_data(df=df_with_gaps)
    expected_days = (result["ds"].max() - result["ds"].min()).days + 1

    assert len(result) == expected_days
    assert (result["y"] == 0).any()


def test_prepare_data_rejects_nulls():
    df = pd.DataFrame({"ds": pd.date_range("2023-01-01", periods=5), "y": [1, None, 3, 4, 5]})

    with pytest.raises(ValueError, match="Null values"):
        TimeSeriesProcessor().prepare_data(df=df)


def test_prepare_data_rejects_negatives():
    df = pd.DataFrame({"ds": pd.date_range("2023-01-01", periods=5), "y": [1, -2, 3, 4, 5]})

    with pytest.raises(ValueError, match="Negative values"):
        TimeSeriesProcessor().prepare_data(df=df)


def test_prepare_data_rejects_wrong_columns():
    df = pd.DataFrame({"date": pd.date_range("2023-01-01", periods=5), "value": range(5)})

    with pytest.raises(ValueError, match="'ds'"):
        TimeSeriesProcessor().prepare_data(df=df)


def test_prepare_data_with_data_loader(sample_timeseries, tmp_path):
    csv = tmp_path / "data.csv"
    sample_timeseries.to_csv(csv, index=False)
    result = TimeSeriesProcessor().prepare_data(
        data_loader=lambda: pd.read_csv(csv, parse_dates=["ds"])
    )

    assert len(result) == len(sample_timeseries)


def test_prepare_data_requires_input():
    with pytest.raises(ValueError, match="Provide data"):
        TimeSeriesProcessor().prepare_data()
