
from datetime import datetime
from datetime import timedelta
from datetime import date
from typing import Optional, List
import pandas as pd
import holidays   
from dateutil.easter import easter

def second_sunday_may(year):
    """
    Calculate the second Sunday of May for a given year (Mother's Day in Brazil/US).
    
    Args:
        year: Year to calculate for
        
    Returns:
        Day of month (1-31) of the second Sunday
    """
    first_may_day = datetime(year, 5, 1)
    sunday_counter = 0 
    while sunday_counter < 2: 
        if first_may_day.weekday() == 6: sunday_counter += 1
        else: first_may_day += timedelta(days=1)
    return first_may_day.day

def second_sunday_august(year):
    """
    Calculate the second Sunday of August for a given year (Father's Day in Brazil).
    
    Args:
        year: Year to calculate for
        
    Returns:
        Day of month (1-31) of the second Sunday
    """
    first_august_day = datetime(year, 8, 1)
    sunday_counter = 0
    while sunday_counter < 2:
        if first_august_day.weekday() == 6: sunday_counter += 1
        else: first_august_day += timedelta(days=1)
    return first_august_day.day 

def last_friday_november(year):
    """
    Calculate the last Friday of November for a given year (Black Friday).
    
    Args:
        year: Year to calculate for
        
    Returns:
        Day of month (1-30) of the last Friday
    """
    last_day_november = datetime(year, 11, 30)
    while last_day_november.weekday() != 4:
        last_day_november -= timedelta(days=1)
    return last_day_november.day

def create_holidays_prophet(
    years: List[int],
    country: str = "BR",
    state: Optional[str] = "SP",
    custom_events: Optional[List[dict]] = None
):
    """
    Create holidays DataFrame for Prophet.

    Args:
        years: List of years to generate holidays for
        country: Country code (default: "BR" for Brazil)
        state: State/subdivision code (default: "SP"). Set to None for country-level only
        custom_events: Optional list of custom events to add.
                      Format: [{'holiday': 'name', 'ds': 'YYYY-MM-DD', 'lower_window': -N, 'upper_window': N}]
                      If None, uses Brazilian commercial holidays (Mother's Day, Black Friday, etc.)

    Returns:
        DataFrame with columns: holiday, ds, lower_window, upper_window
    """
    custom_holidays = []
    
    # Carregar feriados oficiais do país
    if state:
        holidays_obj = holidays.country_holidays(country, subdiv=state, years=years)
    else:
        holidays_obj = holidays.country_holidays(country, years=years)

    # Se custom_events não foi fornecido, usar eventos comerciais brasileiros (padrão legacy)
    if custom_events is None and country == "BR":
        for year in years:
            pascoa = easter(year)
            carnaval = pascoa - timedelta(days=47)  

            custom_holidays.extend([
                {'holiday': 'Volta_as_aulas',   'ds': f'{year}-02-03', 'lower_window': -7, 'upper_window': 2},
                {'holiday': 'Carnaval',         'ds': carnaval.strftime('%Y-%m-%d'), 'lower_window': 0, 'upper_window': 5},  
                {'holiday': 'Pascoa',           'ds': pascoa.strftime('%Y-%m-%d'), 'lower_window': -4, 'upper_window': 4},
                {'holiday': 'Dia_das_maes',     'ds': f'{year}-05-{second_sunday_may(year)}', 'lower_window': -2, 'upper_window': 1},  
                {'holiday': 'Dia_dos_namorados','ds': f'{year}-06-12', 'lower_window': -3, 'upper_window': 2},
                {'holiday': 'Dia_dos_pais',     'ds': f'{year}-08-{second_sunday_august(year)}', 'lower_window': -2, 'upper_window': 1},  
                {'holiday': 'Dia_das_criancas', 'ds': f'{year}-10-12', 'lower_window': 0, 'upper_window': 0},
                {'holiday': 'Black_Friday',     'ds': f'{year}-11-{last_friday_november(year)}', 'lower_window': -2, 'upper_window': 2}, 
                {'holiday': 'Natal',            'ds': f'{year}-12-25', 'lower_window': -3, 'upper_window': 1},
                {'holiday': 'Ano_novo',         'ds': f'{year}-01-01', 'lower_window': -2, 'upper_window': 2},
            ])
    elif custom_events:
        custom_holidays.extend(custom_events)

    # Adicionar feriados oficiais do país
    for dt, name in holidays_obj.items():
        custom_holidays.append({
            'holiday': name.replace(" ", "_"),
            'ds': dt.strftime('%Y-%m-%d'),
            'lower_window': -1, 'upper_window': 1,
        })

    custom_holidays = pd.DataFrame(custom_holidays).drop_duplicates().sort_values('ds')
    return custom_holidays




def get_brazilian_paydays(
    start_year: int,
    end_year: int,
    country: str = "BR",
    state: Optional[str] = "SP",
    business_day: int = 5
):
    """
    Generate payday dates (Nth business day of each month).

    Args:
        start_year: Start year
        end_year: End year (inclusive)
        country: Country code for holidays (default: "BR")
        state: State/subdivision for holidays (default: "SP")
        business_day: Which business day is payday (default: 5 = 5th business day)

    Returns:
        Set of payday Timestamps
    """
    if state:
        holidays_obj = holidays.country_holidays(country, subdiv=state, years=range(start_year, end_year + 1))
    else:
        holidays_obj = holidays.country_holidays(country, years=range(start_year, end_year + 1))
    
    paydays = set()
    
    for year in range(start_year, end_year + 1):
        for month in range(1, 13):
            current = date(year, month, 1)
            business_days_counted = 0
            
            while business_days_counted < business_day:
                if current.weekday() != 6 and current not in holidays_obj:
                    business_days_counted += 1
                
                if business_days_counted == business_day:
                    paydays.add(pd.Timestamp(current))
                    break
                current += timedelta(days=1)
                
    return paydays