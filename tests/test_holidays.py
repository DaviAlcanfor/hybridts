import pytest
import pandas as pd
from hybridts.features.holidays import create_holidays_prophet, get_brazilian_paydays


def test_create_holidays_prophet_columns():
    result = create_holidays_prophet(years=[2023])
    assert {"holiday", "ds", "lower_window", "upper_window"}.issubset(result.columns)


def test_create_holidays_prophet_includes_natal():
    result = create_holidays_prophet(years=[2023])
    assert result["holiday"].str.contains("Natal").any()


def test_create_holidays_prophet_includes_ano_novo():
    result = create_holidays_prophet(years=[2023])
    assert result["holiday"].str.contains("Ano_novo").any()


def test_create_holidays_prophet_no_duplicates():
    result = create_holidays_prophet(years=[2023])
    dupes = result.duplicated(subset=["holiday", "ds"])
    assert not dupes.any()


def test_create_holidays_prophet_custom_events():
    custom = [{"holiday": "Evento_teste", "ds": "2023-07-04", "lower_window": 0, "upper_window": 0}]
    result = create_holidays_prophet(years=[2023], custom_events=custom)
    assert result["holiday"].str.contains("Evento_teste").any()


def test_get_brazilian_paydays_returns_set():
    result = get_brazilian_paydays(2023, 2023)
    assert isinstance(result, set)


def test_get_brazilian_paydays_has_12_months():
    result = get_brazilian_paydays(2023, 2023)
    months = {ts.month for ts in result}
    assert months == set(range(1, 13))


def test_get_brazilian_paydays_no_sundays():
    result = get_brazilian_paydays(2023, 2023)
    assert all(ts.weekday() != 6 for ts in result)
