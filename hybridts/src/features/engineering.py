import pandas as pd
from datetime import timedelta
import holidays
from typing import Optional, Set


def create_features(
    df_dates: pd.DataFrame,
    paydays_set: Optional[Set] = None,
    min_year: int = None,
    max_year: int = None,
    holidays_country: str = "BR",
    holidays_state: Optional[str] = "SP",
    payday_advance_day: int = 20,
    default_days_since_payday: int = 15
):
    """
    Create time series features from dates.

    Args:
        df_dates: DataFrame with 'ds' column (dates)
        paydays_set: Set of payday dates
        min_year: Minimum year for holidays
        max_year: Maximum year for holidays
        holidays_country: Country code for holidays (default: "BR" for Brazil)
        holidays_state: State/subdivision for holidays (default: "SP" for São Paulo)
                       Set to None for country-level holidays only
        payday_advance_day: Day of month for salary advance (default: 20)
        default_days_since_payday: Default value when no previous payday found (default: 15)

    Returns:
        DataFrame with engineered features indexed by date (PeriodIndex)
    """
    # Obter feriados do país/estado especificado
    if holidays_state:
        br_holidays = holidays.country_holidays(
            holidays_country, 
            subdiv=holidays_state,
            years=range(min_year, max_year + 1)
        )
    else:
        br_holidays = holidays.country_holidays(
            holidays_country,
            years=range(min_year, max_year + 1)
        )
    
    df_exo = df_dates.copy()
    df_exo['ds'] = pd.to_datetime(df_exo['ds'])
    
    df_exo['is_weekend'] = df_exo['ds'].dt.weekday.isin([5, 6]).astype(int)
    df_exo['is_month_start'] = df_exo['ds'].dt.day.lt(10).astype(int)
    df_exo['is_month_end'] = df_exo['ds'].dt.is_month_end.astype(int)
    df_exo['day_of_week'] = df_exo['ds'].dt.dayofweek
    df_exo['day_of_month'] = df_exo['ds'].dt.day
    if paydays_set is not None:
        df_exo['is_payday'] = df_exo['ds'].isin(paydays_set).astype(int)
        df_exo['is_adiantamento'] = df_exo['ds'].apply(
            lambda x: 1 if (x.day == payday_advance_day and x.weekday() < 5) or
                           (x.day == payday_advance_day - 1 and x.weekday() == 4) or
                           (x.day == payday_advance_day - 2 and x.weekday() == 4) else 0
        )
        df_exo['sextou_com_dinheiro'] = ((df_exo['day_of_week'] == 4) & (df_exo['is_payday'] == 1)).astype(int)
        df_exo['data_temp'] = df_exo['ds']
        df_exo.loc[df_exo['is_payday'] == 0, 'data_temp'] = pd.NaT
        df_exo['ultimo_pagamento'] = df_exo['data_temp'].ffill()
        df_exo['dias_desde_pagamento'] = (df_exo['ds'] - df_exo['ultimo_pagamento']).dt.days
        df_exo['dias_desde_pagamento'] = df_exo['dias_desde_pagamento'].fillna(default_days_since_payday)
        df_exo.drop(columns=['data_temp', 'ultimo_pagamento'], inplace=True)
    else:
        df_exo['is_payday'] = 0
        df_exo['is_adiantamento'] = 0
        df_exo['sextou_com_dinheiro'] = 0
        df_exo['dias_desde_pagamento'] = 0

    df_exo['is_holiday'] = df_exo['ds'].apply(lambda x: 1 if x in br_holidays else 0)
    df_exo['is_holiday_eve'] = df_exo['ds'].apply(lambda x: 1 if (x + timedelta(days=1)) in br_holidays else 0)
    df_exo['is_post_holiday'] = df_exo['ds'].apply(lambda x: 1 if (x - timedelta(days=1)) in br_holidays else 0)
    df_exo = df_exo.set_index('ds')
    df_exo.index = pd.PeriodIndex(df_exo.index, freq='D')
    
    return df_exo

