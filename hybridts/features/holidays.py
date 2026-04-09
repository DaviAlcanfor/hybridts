from datetime import date, datetime, timedelta
from typing import List, Optional

import holidays
import pandas as pd
from dateutil.easter import easter


def second_sunday_may(year: int) -> int:
    """Returns the day of month of the second Sunday of May (Mother's Day in Brazil)."""
    d = datetime(year, 5, 1)
    days_until_sunday = (6 - d.weekday()) % 7
    d += timedelta(days=days_until_sunday + 7)
    return d.day


def second_sunday_august(year: int) -> int:
    """Returns the day of month of the second Sunday of August (Father's Day in Brazil)."""
    d = datetime(year, 8, 1)
    days_until_sunday = (6 - d.weekday()) % 7
    d += timedelta(days=days_until_sunday + 7)
    return d.day


def last_friday_november(year: int) -> int:
    """Returns the day of month of the last Friday of November (Black Friday)."""
    d = datetime(year, 11, 30)
    while d.weekday() != 4:
        d -= timedelta(days=1)
    return d.day


def create_holidays_prophet(
    years: List[int],
    country: str = "BR",
    state: Optional[str] = "SP",
    custom_events: Optional[List[dict]] = None,
) -> pd.DataFrame:
    """
    Create a holidays DataFrame for Prophet.

    Args:
        years: List of years to generate holidays for.
        country: Country code (default: "BR").
        state: State/subdivision code (default: "SP"). Set to None for country-level only.
        custom_events: Optional list of custom events to add.
                       Format: [{'holiday': 'name', 'ds': 'YYYY-MM-DD',
                                 'lower_window': -N, 'upper_window': N}]
                       If None and country is "BR", uses Brazilian commercial events.

    Returns:
        DataFrame with columns: holiday, ds, lower_window, upper_window.
    """
    if state:
        holidays_obj = holidays.country_holidays(country, subdiv=state, years=years)
    else:
        holidays_obj = holidays.country_holidays(country, years=years)

    rows = []

    if custom_events is None and country == "BR":
        for year in years:
            pascoa = easter(year)
            carnaval = pascoa - timedelta(days=47)
            rows.extend([
                {"holiday": "Volta_as_aulas",    "ds": f"{year}-02-03",                              "lower_window": -7, "upper_window": 2},
                {"holiday": "Carnaval",          "ds": carnaval.strftime("%Y-%m-%d"),                "lower_window": 0,  "upper_window": 5},
                {"holiday": "Pascoa",            "ds": pascoa.strftime("%Y-%m-%d"),                  "lower_window": -4, "upper_window": 4},
                {"holiday": "Dia_das_maes",      "ds": f"{year}-05-{second_sunday_may(year)}",       "lower_window": -2, "upper_window": 1},
                {"holiday": "Dia_dos_namorados", "ds": f"{year}-06-12",                              "lower_window": -3, "upper_window": 2},
                {"holiday": "Dia_dos_pais",      "ds": f"{year}-08-{second_sunday_august(year)}",    "lower_window": -2, "upper_window": 1},
                {"holiday": "Dia_das_criancas",  "ds": f"{year}-10-12",                              "lower_window": 0,  "upper_window": 0},
                {"holiday": "Black_Friday",      "ds": f"{year}-11-{last_friday_november(year)}",    "lower_window": -2, "upper_window": 2},
                {"holiday": "Natal",             "ds": f"{year}-12-25",                              "lower_window": -3, "upper_window": 1},
                {"holiday": "Ano_novo",          "ds": f"{year}-01-01",                              "lower_window": -2, "upper_window": 2},
            ])
    elif custom_events:
        rows.extend(custom_events)

    for dt, name in holidays_obj.items():
        rows.append({
            "holiday": name.replace(" ", "_"),
            "ds": dt.strftime("%Y-%m-%d"),
            "lower_window": -1,
            "upper_window": 1,
        })

    return pd.DataFrame(rows).drop_duplicates().sort_values("ds")


def get_brazilian_paydays(
    start_year: int,
    end_year: int,
    country: str = "BR",
    state: Optional[str] = "SP",
    business_day: int = 5,
) -> set:
    """
    Generate payday dates (Nth business day of each month).

    Args:
        start_year: Start year.
        end_year: End year (inclusive).
        country: Country code for holidays (default: "BR").
        state: State/subdivision for holidays (default: "SP").
        business_day: Which business day is payday (default: 5th business day).

    Returns:
        Set of payday Timestamps.
    """
    if state:
        holidays_obj = holidays.country_holidays(
            country, subdiv=state, years=range(start_year, end_year + 1)
        )
    else:
        holidays_obj = holidays.country_holidays(
            country, years=range(start_year, end_year + 1)
        )

    paydays = set()

    for year in range(start_year, end_year + 1):
        for month in range(1, 13):
            current = date(year, month, 1)
            business_days_counted = 0

            while business_days_counted < business_day:
                if current.weekday() < 5 and current not in holidays_obj:
                    business_days_counted += 1

                if business_days_counted == business_day:
                    paydays.add(pd.Timestamp(current))
                    break
                current += timedelta(days=1)

    return paydays
