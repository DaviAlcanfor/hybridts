from datetime import timedelta
from typing import Optional, Set

import holidays
import pandas as pd


def create_features(
    df_dates: pd.DataFrame,
    paydays_set: Optional[Set] = None,
    min_year: int = None,
    max_year: int = None,
    holidays_country: str = "BR",
    holidays_state: Optional[str] = "SP",
    payday_advance_day: int = 20,
    default_days_since_payday: int = 15,
) -> pd.DataFrame:
    """
    Create time series features from dates.

    Args:
        df_dates: DataFrame with a 'ds' column (dates).
        paydays_set: Set of payday Timestamps.
        min_year: Minimum year for holiday generation.
        max_year: Maximum year for holiday generation.
        holidays_country: Country code for holidays (default: "BR").
        holidays_state: State/subdivision for holidays (default: "SP").
                        Set to None for country-level holidays only.
        payday_advance_day: Day of month for salary advance (default: 20).
        default_days_since_payday: Default value when no previous payday is found.

    Returns:
        DataFrame with engineered features indexed by date (PeriodIndex).
    """
    if holidays_state:
        country_holidays = holidays.country_holidays(
            holidays_country,
            subdiv=holidays_state,
            years=range(min_year, max_year + 1),
        )
    else:
        country_holidays = holidays.country_holidays(
            holidays_country,
            years=range(min_year, max_year + 1),
        )

    df = df_dates.copy()
    df["ds"] = pd.to_datetime(df["ds"])

    df["is_weekend"] = df["ds"].dt.weekday.isin([5, 6]).astype(int)
    df["is_month_start"] = df["ds"].dt.day.lt(10).astype(int)
    df["is_month_end"] = df["ds"].dt.is_month_end.astype(int)
    df["day_of_week"] = df["ds"].dt.dayofweek
    df["day_of_month"] = df["ds"].dt.day

    if paydays_set is not None:
        df["is_payday"] = df["ds"].isin(paydays_set).astype(int)
        df["is_salary_advance"] = df["ds"].apply(
            lambda x: 1
            if (x.day == payday_advance_day and x.weekday() < 5)
            or (x.day == payday_advance_day - 1 and x.weekday() == 4)
            or (x.day == payday_advance_day - 2 and x.weekday() == 4)
            else 0
        )
        df["is_payday_friday"] = (
            (df["day_of_week"] == 4) & (df["is_payday"] == 1)
        ).astype(int)
        df["_last_payday"] = df["ds"].where(df["is_payday"] == 1)
        df["_last_payday"] = df["_last_payday"].ffill()
        df["days_since_payday"] = (df["ds"] - df["_last_payday"]).dt.days
        df["days_since_payday"] = df["days_since_payday"].fillna(default_days_since_payday)
        df.drop(columns=["_last_payday"], inplace=True)
    else:
        df["is_payday"] = 0
        df["is_salary_advance"] = 0
        df["is_payday_friday"] = 0
        df["days_since_payday"] = 0

    df["is_holiday"] = df["ds"].apply(lambda x: 1 if x in country_holidays else 0)
    df["is_holiday_eve"] = df["ds"].apply(
        lambda x: 1 if (x + timedelta(days=1)) in country_holidays else 0
    )
    df["is_post_holiday"] = df["ds"].apply(
        lambda x: 1 if (x - timedelta(days=1)) in country_holidays else 0
    )

    df = df.set_index("ds")
    df.index = pd.PeriodIndex(df.index, freq="D")

    return df
