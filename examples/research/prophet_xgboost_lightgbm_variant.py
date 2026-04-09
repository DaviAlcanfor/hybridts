import pandas as pd
import numpy as np 
from datetime import date, timedelta, datetime
import matplotlib.pyplot as plt
from prophet import Prophet
from xgboost import XGBRegressor
from sktime.forecasting.compose import make_reduction
from sktime.utils.plotting import plot_series
from sktime.performance_metrics.forecasting import mean_absolute_error, mean_squared_error, mean_absolute_percentage_error

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

df = pd.read_csv('seu_arquivo_mat.csv', parse_dates=['ds'])

# Alternativa: df = pd.read_parquet('seu_arquivo_mat.parquet')


# Modelo V1 s/ CV
# Mape: 9.35% - com ambos os modelos com desempenho equilibrado


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


years_in_data = df['ds'].dt.year.unique()
prophet_holidays = create_holidays_prophet(years_in_data)

prophet_model = Prophet(
    holidays=prophet_holidays,
    seasonality_mode='multiplicative',
    yearly_seasonality=True,
    weekly_seasonality=True,
    daily_seasonality=False
)

prophet_model.fit(df_train)

prophet_fitted = prophet_model.predict(df_train[['ds']])


df_train_indexed = df_train.set_index('ds')
df_train_indexed.index = pd.PeriodIndex(df_train_indexed.index, freq='D')

prophet_fitted_indexed = prophet_fitted.set_index('ds')
prophet_fitted_indexed.index = pd.PeriodIndex(prophet_fitted_indexed.index, freq='D')

residuals_train = df_train_indexed['y'].astype(float) - prophet_fitted_indexed['yhat']


xgb_forecaster = make_reduction(
    estimator=XGBRegressor(
        n_estimators=200,
        max_depth=6,
        learning_rate=0.05,
        subsample=0.8,
        tree_method="hist",
        random_state=42
    ),
    window_length=28,
    strategy='recursive'  
)

xgb_forecaster.fit(y=residuals_train, X=X_train)


future_dates = prophet_model.make_future_dataframe(periods=TEST_SIZE, freq='D')
prophet_forecast_full = prophet_model.predict(future_dates)

prophet_forecast_test = prophet_forecast_full.tail(TEST_SIZE).set_index('ds')['yhat']
prophet_forecast_test.index = pd.PeriodIndex(prophet_forecast_test.index, freq='D')

fh = np.arange(1, TEST_SIZE + 1)
xgb_forecast_test = xgb_forecaster.predict(fh=fh, X=X_test)

hybrid_forecast = prophet_forecast_test + xgb_forecast_test

print("--- Métricas do Modelo Híbrido ---")
metrics(test=y_test, pred=hybrid_forecast)

plot_series(y_test, hybrid_forecast, prophet_forecast_test, 
            labels=["Real", "Previsão Híbrida (Prophet+XGB)", "Apenas Prophet Baseline"])
plt.title("Forecast Híbrido: Previsão Diária de TPV")
plt.show()


# Modelo V2 c/ CV 
# Mape: 9.38% - porém o XGB não teve bom desempenho, ao contrário do Prophet que performou muito bem
# *Esse modelo é mais seguro do que "hardocodar" os parametros no V1*


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


# Modelo V3 (usando LightGBM) c/ CV 


X_train_v2 = create_exogenous_features_v2(df_train[['ds']], paydays_set)
X_test_v2 = create_exogenous_features_v2(df_test[['ds']], paydays_set)


from lightgbm import LGBMRegressor
from sktime.forecasting.model_selection import ForecastingGridSearchCV, ExpandingWindowSplitter
from sktime.performance_metrics.forecasting import MeanAbsolutePercentageError

print("1. Iniciando Grid Search do LightGBM OTIMIZADO (CV Inteligente)...")

cv_otimizado = ExpandingWindowSplitter(
    initial_window=730,
    step_length=30,
    fh=np.arange(1, TEST_SIZE + 1)
)

lgbm_forecaster = make_reduction(
    estimator=LGBMRegressor(random_state=42, verbose=-1, n_jobs=1),
    strategy='recursive'  
)

param_grid_lgbm = {
    "window_length": [7, 14, 28], 
    "estimator__max_depth": [4, 5, 6],
    "estimator__learning_rate": [0.01, 0.05, 0.1],
    "estimator__n_estimators": [100, 200]
}

gscv_lgbm = ForecastingGridSearchCV(
    forecaster=lgbm_forecaster,
    cv=cv_otimizado,
    param_grid=param_grid_lgbm,
    scoring=MeanAbsolutePercentageError(symmetric=False)
)

gscv_lgbm.fit(y=residuals_train, X=X_train_v2)

best_lgbm_model = gscv_lgbm.best_forecaster_
print(f"Melhores parâmetros do LightGBM validados: {gscv_lgbm.best_params_}")

fh = np.arange(1, TEST_SIZE + 1)
lgbm_forecast_test_cv = best_lgbm_model.predict(fh=fh, X=X_test_v2)

hybrid_forecast_lgbm_cv = prophet_forecast_test + lgbm_forecast_test_cv

print("--- Métricas Híbrido: Prophet + LightGBM (CV Seguro + Novas Features) ---")
metrics(test=y_test, pred=hybrid_forecast_lgbm_cv)


plot_series(y_test, hybrid_forecast_lgbm_cv, prophet_forecast_test, 
            labels=["Real", "Previsão Híbrida Definitiva", "Apenas Prophet Baseline"])
plt.title("Forecast Final: Prophet + LGBM (Com CV Otimizado e Novas Features)")
plt.show()


# Prophet + lgbm refit 

df_full = df.copy().sort_values(by='ds').reset_index(drop=True)

FUTURE_DAYS = 30
last_date = df_full['ds'].max()
future_dates_index = pd.date_range(start=last_date + timedelta(days=1), periods=FUTURE_DAYS, freq='D')
df_future_dates = pd.DataFrame({'ds': future_dates_index})


all_dates = pd.concat([df_full[['ds']], df_future_dates]).reset_index(drop=True)
X_all = create_exogenous_features_v2(all_dates, paydays_set)

new_prophet_model = Prophet(
    holidays=prophet_holidays,
    yearly_seasonality=True,
    weekly_seasonality=True,
    daily_seasonality=False,
    **best_prophet_params
)

new_prophet_model.fit(df_full)

future_df_prophet = new_prophet_model.make_future_dataframe(periods=FUTURE_DAYS, freq='D')
forecast_prophet_full = new_prophet_model.predict(future_df_prophet)

prophet_future_pred = forecast_prophet_full.tail(FUTURE_DAYS).set_index('ds')['yhat']
prophet_future_pred.index = pd.PeriodIndex(prophet_future_pred.index, freq='D')

gscv_lgbm.fit(y=residuals_train, X=X_train_v2)
lgbm_pred = best_lgbm_model.predict(fh=np.arange(1, FUTURE_DAYS + 1), X=X_all.tail(FUTURE_DAYS))
hybrid_forecast_lgbm = prophet_future_pred + lgbm_pred

plot_series(prophet_future_pred, hybrid_forecast_lgbm, 
            labels=["Prophet", "Híbrido Final"])
plt.title("Forecast Final: Prophet + LGBM (Com CV Otimizado e Novas Features)")
plt.show()


df_full = df.copy().sort_values(by='ds').reset_index(drop=True)

FUTURE_DAYS = 38
last_date = df_full['ds'].max()
future_dates_index = pd.date_range(start=last_date + timedelta(days=1), periods=FUTURE_DAYS, freq='D')
df_future_dates = pd.DataFrame({'ds': future_dates_index})

all_dates = pd.concat([df_full[['ds']], df_future_dates]).reset_index(drop=True)
X_all = create_exogenous_features_v2(all_dates, paydays_set)
X_train_full = X_all.iloc[:-FUTURE_DAYS].copy()
X_future     = X_all.iloc[-FUTURE_DAYS:].copy()

new_prophet_model = Prophet(
    holidays=prophet_holidays,
    yearly_seasonality=True,
    weekly_seasonality=True,
    daily_seasonality=False,
    **best_prophet_params
)

new_prophet_model.fit(df_full)

future_df_prophet = new_prophet_model.make_future_dataframe(periods=FUTURE_DAYS, freq='D')
forecast_prophet_full = new_prophet_model.predict(future_df_prophet)

prophet_future_pred = forecast_prophet_full.tail(FUTURE_DAYS).set_index('ds')['yhat']
prophet_future_pred.index = pd.PeriodIndex(prophet_future_pred.index, freq='D')


prophet_past_pred = forecast_prophet_full.iloc[:-FUTURE_DAYS].set_index('ds')['yhat']

df_full_indexed = df_full.set_index('ds')
df_full_indexed.index = pd.PeriodIndex(df_full_indexed.index, freq='D')
prophet_past_pred.index = pd.PeriodIndex(prophet_past_pred.index, freq='D')

residuals_full = df_full_indexed['y'].astype(float) - prophet_past_pred

new_lgbm_model = make_reduction(
    estimator=LGBMRegressor(
        **gscv_lgbm.best_params_
    ),
    window_length=28,
    strategy='recursive'
)

new_lgbm_model.fit(y=residuals_full, X=X_train_full)

fh_future = np.arange(1, FUTURE_DAYS + 1)
lgbm_future_pred = new_lgbm_model.predict(fh=fh_future, X=X_future)

final_forecast = prophet_future_pred + lgbm_future_pred

plt.figure(figsize=(15, 6))
plt.plot(df_full['ds'].tail(90), df_full['y'].tail(90), label='Histórico Recente (90 dias)', marker='o')
plt.plot(df_future_dates['ds'], final_forecast.values, label='Previsão Futura (30 Dias)', color='orange', linestyle='--', marker='x')
plt.title("O Futuro do MAT: Previsão Híbrida Final")
plt.legend()
plt.grid(True, alpha=0.3)
plt.show()

df_resultado = pd.DataFrame({'Data': df_future_dates['ds'], 'Previsao_TPV': final_forecast.values})
df_resultado['Data'] = pd.to_datetime(df_resultado['Data']).dt.strftime('%Y-%m-%d')
df_resultado['Previsao_TPV'] = df_resultado['Previsao_TPV'].astype(int)
display(df_resultado.tail(FUTURE_DAYS))


df_march_2023 = df[(df['ds'].dt.month == 3) & (df['ds'].dt.year == 2023)].set_index('ds')
df_march_2024 = df[(df['ds'].dt.month == 3) & (df['ds'].dt.year == 2024)].set_index('ds')
df_march_2025 = df[(df['ds'].dt.month == 3) & (df['ds'].dt.year == 2025)].set_index('ds')

plot_series(df_march_2025, title="MAT em Março de 2025")
plot_series(df_march_2024, title="MAT em Março de 2024")
plot_series(df_march_2023, title="MAT em Março de 2023")