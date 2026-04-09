import pytest
import pandas as pd
import numpy as np
from hybridts.features.engineering import create_features


@pytest.fixture
def dates_df():
    dates = pd.date_range("2023-01-01", periods=60, freq="D")
    return pd.DataFrame({"ds": dates})


def test_returns_expected_columns(dates_df):
    result = create_features(dates_df, min_year=2023, max_year=2023)
    expected = {
        "is_weekend", "is_month_start", "is_month_end",
        "day_of_week", "day_of_month",
        "is_payday", "is_salary_advance", "is_payday_friday", "days_since_payday",
        "is_holiday", "is_holiday_eve", "is_post_holiday",
    }
    assert expected.issubset(set(result.columns))


def test_index_is_period_index(dates_df):
    result = create_features(dates_df, min_year=2023, max_year=2023)
    assert isinstance(result.index, pd.PeriodIndex)


def test_is_weekend_correct(dates_df):
    result = create_features(dates_df, min_year=2023, max_year=2023)
    # 2023-01-01 is Sunday (weekday=6) → weekend
    assert result.loc[pd.Period("2023-01-01", "D"), "is_weekend"] == 1
    # 2023-01-02 is Monday → not weekend
    assert result.loc[pd.Period("2023-01-02", "D"), "is_weekend"] == 0


def test_without_paydays_payday_columns_are_zero(dates_df):
    result = create_features(dates_df, min_year=2023, max_year=2023, paydays_set=None)
    assert (result["is_payday"] == 0).all()
    assert (result["is_salary_advance"] == 0).all()
    assert (result["is_payday_friday"] == 0).all()
    assert (result["days_since_payday"] == 0).all()


def test_with_paydays_marks_correctly():
    dates = pd.date_range("2023-01-01", periods=30, freq="D")
    df = pd.DataFrame({"ds": dates})
    payday = pd.Timestamp("2023-01-10")
    result = create_features(df, paydays_set={payday}, min_year=2023, max_year=2023)
    assert result.loc[pd.Period("2023-01-10", "D"), "is_payday"] == 1
    assert result.loc[pd.Period("2023-01-11", "D"), "is_payday"] == 0


def test_is_holiday_natal():
    dates = pd.date_range("2023-12-20", periods=10, freq="D")
    df = pd.DataFrame({"ds": dates})
    result = create_features(df, min_year=2023, max_year=2023)
    # 2023-12-25 is Christmas — national holiday in BR
    assert result.loc[pd.Period("2023-12-25", "D"), "is_holiday"] == 1


def test_is_holiday_eve_and_post():
    dates = pd.date_range("2023-12-23", periods=5, freq="D")
    df = pd.DataFrame({"ds": dates})
    result = create_features(df, min_year=2023, max_year=2023)
    assert result.loc[pd.Period("2023-12-24", "D"), "is_holiday_eve"] == 1
    assert result.loc[pd.Period("2023-12-26", "D"), "is_post_holiday"] == 1
