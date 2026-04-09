
import pandas as pd
import numpy as np 
from datetime import date, timedelta, datetime
import matplotlib.pyplot as plt
from prophet import Prophet
from xgboost import XGBRegressor
from sktime.forecasting.compose import make_reduction
from sktime.utils.plotting import plot_series
from sktime.performance_metrics.forecasting import mean_absolute_error, mean_squared_error, mean_absolute_percentage_error
from statsmodels.stats.diagnostic import acorr_ljungbox


def second_sunday_may(year):
    first_may_day = datetime(year, 5, 1)
    sunday_counter = 0 
    while sunday_counter < 2: 
        if first_may_day.weekday() == 6: sunday_counter += 1
        else: first_may_day += timedelta(days=1)
    return first_may_day.day

def second_sunday_august(year):
    first_august_day = datetime(year, 8, 1)
    sunday_counter = 0
    while sunday_counter < 2:
        if first_august_day.weekday() == 6: sunday_counter += 1
        else: first_august_day += timedelta(days=1)
    return first_august_day.day 

def last_friday_november(year):
    last_day_november = datetime(year, 11, 30)
    while last_day_november.weekday() != 4:
        last_day_november -= timedelta(days=1)
    return last_day_november.day

def create_holidays_prophet(years):
    import holidays   
    custom_holidays = []
    holidays_br = holidays.BR(years=years, subdiv="SP")

    for year in years:
        custom_holidays.extend([
            {'holiday': 'Volta_as_aulas',   'ds': f'{year}-02-03', 'lower_window': -7, 'upper_window': 2},
            {'holiday': 'Carnaval',         'ds': f'{year}-02-12', 'lower_window': 0, 'upper_window': 5},  
            {'holiday': 'Pascoa',           'ds': f'{year}-03-31', 'lower_window': -4, 'upper_window': 4},
            {'holiday': 'Dia_das_maes',     'ds': f'{year}-05-{second_sunday_may(year)}', 'lower_window': -2, 'upper_window': 1},  
            {'holiday': 'Dia_dos_namorados','ds': f'{year}-06-12', 'lower_window': -3, 'upper_window': 2},
            {'holiday': 'Dia_dos_pais',     'ds': f'{year}-08-{second_sunday_august(year)}', 'lower_window': -2, 'upper_window': 1},  
            {'holiday': 'Dia_das_criancas', 'ds': f'{year}-10-12', 'lower_window': 0, 'upper_window': 0},
            {'holiday': 'Black_Friday',     'ds': f'{year}-11-{last_friday_november(year)}', 'lower_window': -2, 'upper_window': 2}, 
            {'holiday': 'Natal',            'ds': f'{year}-12-25', 'lower_window': -3, 'upper_window': 1},
            {'holiday': 'Ano_novo',         'ds': f'{year}-01-01', 'lower_window': -2, 'upper_window': 2},
        ])

    for dt, name in holidays_br.items():
        custom_holidays.append({
            'holiday': name.replace(" ", "_"),
            'ds': dt.strftime('%Y-%m-%d'),
            'lower_window': -1, 'upper_window': 1,
        })

    custom_holidays = pd.DataFrame(custom_holidays).drop_duplicates().sort_values('ds')
    return custom_holidays

def create_exogenous_features(df_dates, paydays_set):
    import holidays
    br_holidays = holidays.BR(years=range(2022, 2028), state='SP')
    
    df_exo = df_dates.copy()
    df_exo['ds'] = pd.to_datetime(df_exo['ds'])
    
    df_exo['is_weekend'] = df_exo['ds'].dt.weekday.isin([5, 6]).astype(int)
    df_exo['is_month_start'] = df_exo['ds'].dt.day.lt(10).astype(int)
    df_exo['is_month_end'] = df_exo['ds'].dt.is_month_end.astype(int)
    df_exo['day_of_week'] = df_exo['ds'].dt.dayofweek
    df_exo['day_of_month'] = df_exo['ds'].dt.day
    
    df_exo['is_payday'] = df_exo['ds'].isin(paydays_set).astype(int)
    
    df_exo['is_adiantamento'] = df_exo['ds'].apply(
        lambda x: 1 if (x.day == 20 and x.weekday() < 5) or 
                       (x.day == 19 and x.weekday() == 4) or 
                       (x.day == 18 and x.weekday() == 4) else 0
    )
    
    df_exo['is_holiday'] = df_exo['ds'].apply(lambda x: 1 if x in br_holidays else 0)
    df_exo['is_holiday_eve'] = df_exo['ds'].apply(lambda x: 1 if (x + timedelta(days=1)) in br_holidays else 0)
    
    df_exo = df_exo.set_index('ds')
    df_exo.index = pd.PeriodIndex(df_exo.index, freq='D')
    
    return df_exo

def create_exogenous_features_v2(df_dates, paydays_set):
    import holidays
    br_holidays = holidays.BR(years=range(2022, 2028), state='SP')
    
    df_exo = df_dates.copy()
    df_exo['ds'] = pd.to_datetime(df_exo['ds'])
    
    df_exo['is_weekend'] = df_exo['ds'].dt.weekday.isin([5, 6]).astype(int)
    df_exo['is_month_start'] = df_exo['ds'].dt.day.lt(10).astype(int)
    df_exo['is_month_end'] = df_exo['ds'].dt.is_month_end.astype(int)
    df_exo['day_of_week'] = df_exo['ds'].dt.dayofweek
    df_exo['day_of_month'] = df_exo['ds'].dt.day
    df_exo['is_payday'] = df_exo['ds'].isin(paydays_set).astype(int)
    df_exo['is_adiantamento'] = df_exo['ds'].apply(
        lambda x: 1 if (x.day == 20 and x.weekday() < 5) or 
                       (x.day == 19 and x.weekday() == 4) or 
                       (x.day == 18 and x.weekday() == 4) else 0
    )
    df_exo['is_holiday'] = df_exo['ds'].apply(lambda x: 1 if x in br_holidays else 0)
    df_exo['is_holiday_eve'] = df_exo['ds'].apply(lambda x: 1 if (x + timedelta(days=1)) in br_holidays else 0)
    df_exo['sextou_com_dinheiro'] = ((df_exo['day_of_week'] == 4) & (df_exo['is_payday'] == 1)).astype(int)
    df_exo['data_temp'] = df_exo['ds']
    df_exo.loc[df_exo['is_payday'] == 0, 'data_temp'] = pd.NaT 
    df_exo['ultimo_pagamento'] = df_exo['data_temp'].ffill()  
    df_exo['dias_desde_pagamento'] = (df_exo['ds'] - df_exo['ultimo_pagamento']).dt.days
    df_exo['dias_desde_pagamento'] = df_exo['dias_desde_pagamento'].fillna(15) 
    df_exo.drop(columns=['data_temp', 'ultimo_pagamento'], inplace=True)    
    df_exo['is_post_holiday'] = df_exo['ds'].apply(lambda x: 1 if (x - timedelta(days=1)) in br_holidays else 0)
    df_exo = df_exo.set_index('ds')
    df_exo.index = pd.PeriodIndex(df_exo.index, freq='D')
    
    return df_exo


def metrics(test, pred):
    mae = mean_absolute_error(test, pred)
    rmse = np.sqrt(mean_squared_error(test, pred))
    mape = mean_absolute_percentage_error(test, pred) * 100
    print(f"MAE:  {mae:.2f}")
    print(f"RMSE: {rmse:.2f}")
    print(f"MAPE: {mape:.2f}%")

def get_brazilian_paydays(start_year, end_year):
    import holidays
    br_holidays = holidays.BR(years=range(start_year, end_year + 1), state='SP')
    paydays = set()
    
    for year in range(start_year, end_year + 1):
        for month in range(1, 13):
            current = date(year, month, 1)
            business_days_counted = 0
            
            while business_days_counted < 5:
                if current.weekday() != 6 and current not in br_holidays:
                    business_days_counted += 1
                
                if business_days_counted == 5:
                    paydays.add(pd.Timestamp(current))
                    break
                current += timedelta(days=1)
                
    return paydays


### Query - coletando o TPV Crédito da `<your_schema.tpv_table>`


# Forneça seus dados como CSV/Parquet com colunas: ['ds', 'y']
# ds: data (YYYY-MM-DD), y: valor TPV

df = pd.read_csv('seu_arquivo_tpv.csv', parse_dates=['ds'])

# Alternativa Parquet:
# df = pd.read_parquet('seu_arquivo_tpv.parquet')

# LEGACY Databricks (descomente se necessário):
# spark = SparkSession.builder.appName("Sparkify").getOrCreate() 
# query = """
# SELECT 
#     date(reference_date) ds, 
#     sum(value) y
# FROM 
#     <your_schema.tpv_table>
# WHERE 
#     product = 'Crédito'
#     AND value <> 0 
#     AND consumer_id IS NOT NULL
# group by all
# ORDER BY 1 
# """
# df = spark.sql(query).toPandas()


df['ds'] = pd.to_datetime(df['ds'])
df = df[['ds', 'y']].sort_values('ds').reset_index(drop=True)

TEST_SIZE = 30

df_train = df.iloc[:-TEST_SIZE].copy()
df_test = df.iloc[-TEST_SIZE:].copy()

min_year = df['ds'].dt.year.min()
max_year = df['ds'].dt.year.max()
paydays_set = get_brazilian_paydays(min_year, max_year + 1)

X_train = create_exogenous_features(df_train[['ds']], paydays_set)
X_test = create_exogenous_features(df_test[['ds']], paydays_set)

y_test = df_test.set_index('ds')['y']
y_test.index = pd.PeriodIndex(y_test.index, freq='D')


import itertools
from prophet.diagnostics import cross_validation, performance_metrics

print("1. Iniciando Grid Search do Prophet...")

years_in_data = df['ds'].dt.year.unique()
prophet_holidays = create_holidays_prophet(years_in_data)

param_grid = {  
    'changepoint_prior_scale': [0.01, 0.05, 0.1],
    'seasonality_prior_scale': [1.0, 10.0, 20.0],
    'seasonality_mode': ['additive', 'multiplicative']
}

all_params = [dict(zip(param_grid.keys(), v)) for v in itertools.product(*param_grid.values())]
mapes = []

for params in all_params:
    m = Prophet(
        holidays=prophet_holidays,
        yearly_seasonality=True,
        weekly_seasonality=True,
        daily_seasonality=False,
        **params
    )
    m.fit(df_train)
    df_cv = cross_validation(m, 
                             initial='400 days', 
                             period='30 days', 
                             horizon='30 days', 
                             parallel="processes")
    
    df_p = performance_metrics(df_cv, rolling_window=1)
    mapes.append(df_p['mape'].values[0])

best_prophet_params = all_params[np.argmin(mapes)]
print(f"Melhores parâmetros do Prophet: {best_prophet_params}")


print("2. Treinando o Prophet Final e extraindo resíduos...")

best_prophet_model = Prophet(
    holidays=prophet_holidays,
    yearly_seasonality=True,
    weekly_seasonality=True,
    daily_seasonality=False,
    **best_prophet_params
)
best_prophet_model.fit(df_train)

prophet_fitted = best_prophet_model.predict(df_train[['ds']])

df_train_indexed = df_train.set_index('ds')
df_train_indexed.index = pd.PeriodIndex(df_train_indexed.index, freq='D')

prophet_fitted_indexed = prophet_fitted.set_index('ds')
prophet_fitted_indexed.index = pd.PeriodIndex(prophet_fitted_indexed.index, freq='D')

if best_prophet_params['seasonality_mode'] == 'multiplicative':
    residuals_train = df_train_indexed['y'].astype(float) - prophet_fitted_indexed['yhat']
else:
    residuals_train = df_train_indexed['y'].astype(float) - prophet_fitted_indexed['yhat']

lb_prophet = acorr_ljungbox(residuals_train.dropna(), lags=[7, 14], return_df=True)
print("\n", lb_prophet)


from sktime.forecasting.model_selection import ForecastingGridSearchCV, ExpandingWindowSplitter
from sktime.performance_metrics.forecasting import MeanAbsolutePercentageError

print("3. Iniciando Grid Search do XGBoost nos resíduos...")

xgb_forecaster = make_reduction(
    estimator=XGBRegressor(tree_method="hist", random_state=42),
    strategy='recursive'  
)

cv = ExpandingWindowSplitter(
    initial_window=90, 
    step_length=15,    
    fh=np.arange(1, TEST_SIZE + 1)
)

param_grid_xgb = {
    "window_length": [7, 14, 28], 
    "estimator__max_depth": [4, 5, 6],
    "estimator__learning_rate": [0.01, 0.05, 0.1],
    "estimator__n_estimators": [100, 200]
}

gscv = ForecastingGridSearchCV(
    forecaster=xgb_forecaster,
    cv=cv,
    param_grid=param_grid_xgb,
    scoring=MeanAbsolutePercentageError(symmetric=False),
    #backend="joblib",
    backend_params={"n_jobs": -1}
)

gscv.fit(y=residuals_train, X=X_train)

# O gscv já salva automaticamente o melhor modelo treinado!
best_xgb_model = gscv.best_forecaster_
print(f"Melhores parâmetros do XGBoost: {gscv.best_params_}")


future_dates = best_prophet_model.make_future_dataframe(periods=TEST_SIZE, freq='D')
prophet_forecast_full = best_prophet_model.predict(future_dates)

prophet_forecast_test = prophet_forecast_full.tail(TEST_SIZE).set_index('ds')['yhat']
prophet_forecast_test.index = pd.PeriodIndex(prophet_forecast_test.index, freq='D')

fh = np.arange(1, TEST_SIZE + 1)
xgb_forecast_test = best_xgb_model.predict(fh=fh, X=X_test)

hybrid_forecast = prophet_forecast_test + xgb_forecast_test

print("--- Métricas do Modelo Híbrido Otimizado ---")
metrics(test=y_test, pred=hybrid_forecast)

plot_series(y_test, hybrid_forecast, prophet_forecast_test, 
            labels=["Real", "Previsão Híbrida (Prophet+XGB)", "Apenas Prophet Baseline"])
plt.title("Forecast Híbrido Otimizado: Previsão Diária de TPV")
plt.show()


residuos_finais = hybrid_forecast - y_test.astype(float)
lb_final = acorr_ljungbox(residuos_finais.dropna(), lags=[7, 14], return_df=True)
print("\n", lb_final)


# df_prev_fev = hybrid_forecast.to_frame(name="yhat")
# df_prev_fev['yhat'].astype(float)
# df_prev_filtrado = df_prev_fev[df_prev_fev.index.month == 2]
# round(sum(df_prev_filtrado['yhat']))