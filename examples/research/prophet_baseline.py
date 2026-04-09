# NOTA: dbutils removido - Databricks: descomente dbutils.library.restartPython()


import numpy as np 
import pandas as pd                                
from sktime.utils.plotting import plot_series
from datetime import date, timedelta, datetime
import matplotlib.pyplot as plt
from sktime.utils.plotting import plot_series
from sktime.performance_metrics.forecasting import mean_absolute_error, mean_squared_error, mean_absolute_percentage_error


# Query


# NOTA: SparkSession removido
# Forneça CSV/Parquet com colunas ['ds', 'y']
df = pd.read_csv('seu_arquivo_tpv.csv', parse_dates=['ds'])

# Alternativa: df = pd.read_parquet('seu_arquivo_tpv.parquet')


# Functions
# * get_season
# * create_features
# * second_sunday_may
# * second_sunday_august
# * last_friday_november
# * create_holidays
# * metrics 
# * make_graph


### `get_season(date):`


def get_season(date):
    """Retorna a estacao em que a data recebida está"""
    m = date.month
    d = date.day

    if (m == 12 and d >= 21) or (m in [1, 2]) or (m == 3 and d < 21): return "Summer"
    if (m == 3 and d >= 21) or (m in [4, 5]) or (m == 6 and d < 21): return "Autumn"
    if (m == 6 and d >= 21) or (m in [7, 8]) or (m == 9 and d < 23): return "Winter"
    if (m == 9 and d >= 23) or (m in [10, 11]) or (m == 12 and d < 21): return "Spring"


### `create_features(df):`


def create_features(df):
    """Cria features sem vazamento de dados"""
    df_exo = df.copy()
    df_exo['ds'] = pd.to_datetime(df_exo['ds'])
    
    # Features temporais (seguras)
    df_exo['is_weekend'] = (df_exo['ds'].dt.dayofweek >= 5).astype(int)
    df_exo['is_month_start'] = (df_exo['ds'].dt.day < 10).astype(int)
    df_exo['is_vacation_time'] = (df_exo['ds'].dt.month.isin([12, 1, 7])).astype(int)
    
    return df_exo


### `second_sunday_may(year):`
### `second_sunday_august(year):`
### `last_friday_november(year):`


# Dia das maes - segundo domingo de maio
def second_sunday_may(year):
    """Retorna o segundo domingo de maio"""

    first_may_day = datetime(year, 5, 1)
    sunday_counter = 0 

    while sunday_counter < 2: 
        if first_may_day.weekday() == 6: 
            sunday_counter += 1
        first_may_day = first_may_day + timedelta(days=1)

    return first_may_day.day


# Dia dos pais - segundo domingo de agosto
def second_sunday_august(year):
    """Retorna o segundo domingo de agosto"""

    first_may_day = datetime(year, 8, 1)

    sunday_counter = 0

    while sunday_counter < 2:
        if first_may_day.weekday() == 6:
            sunday_counter += 1
        first_may_day = first_may_day + timedelta(days=1)

    return first_may_day.day 


# Black Friday - ultima sexta de novembro
def last_friday_november(year):
    """Retorna a ultima sexta-feira de novembro"""

    last_day_november = datetime(year, 11, 30)

    while last_day_november.weekday() != 4:
        last_day_november = last_day_november - timedelta(days=1)
    return last_day_november.day


### `create_holidays(years):`


###Datas contidas:
# * Volta as aulas - 3 a 5 de fevereiro
# * Carnaval - 47 dias antes da pascoa
# * Páscoa - 22/25 de abril
# * Dia das mães - segundo domingo de maio
# * Dia dos namorados - 12 de junho 
# * Dia dos pais - segundo domindo de agosto
# * Dia das crianças - 12/10
# * Halloween - 31/10
# * Black Friday - ultima sexta de novembro
# * Natal - 25/12
# * Ano novo - 01/01


def create_holidays(years):
    """Adiciona os feriados em que as pessoas mais gastam dinheiro"""

    import holidays    

    # melhorar a flexibilidade das datas

    custom_holidays = []
    holidays_br = holidays.BR(years=years, subdiv="SP")

    for year in years:
        custom_holidays.extend([
            {'holiday': 'Volta_as_aulas',   'ds': f'{year}-02-03', 'lower_window': -7, 'upper_window': 2},
            {'holiday': 'Carnaval',         'ds': f'{year}-02-12', 'lower_window': -5, 'upper_window': 5},  
            {'holiday': 'Pascoa',           'ds': f'{year}-03-31', 'lower_window': -4, 'upper_window': 4},
            {'holiday': 'Dia_das_maes',     'ds': f'{year}-05-{second_sunday_may(year)}', 'lower_window': -2, 'upper_window': 1},  
            {'holiday': 'Dia_dos_namorados','ds': f'{year}-06-12', 'lower_window': -3, 'upper_window': 2},
            {'holiday': 'Dia_dos_pais',     'ds': f'{year}-08-{second_sunday_august(year)}', 'lower_window': -2, 'upper_window': 1},  
            {'holiday': 'Dia_das_crianças', 'ds': f'{year}-10-12', 'lower_window': 0, 'upper_window': 0},
            {'holiday': 'Halloween',        'ds': f'{year}-10-31', 'lower_window': -1, 'upper_window': 0},
            {'holiday': 'Black_Friday',     'ds': f'{year}-11-{last_friday_november(year)}', 'lower_window': -2, 'upper_window': 2}, 
            {'holiday': 'Ano_novo',         'ds': f'{year}-01-01', 'lower_window': -2, 'upper_window': 2},
        ])

    # Adicionando os feriados do pacote holidays junto aos customizados
    for date, name in holidays_br.items():
        custom_holidays.append({
            'holiday': name.replace(" ", "_"),
            'ds': date.strftime('%Y-%m-%d'),
            'lower_window': -1,
            'upper_window': 1,
        })

    # transformando em Dataframe pois o modelo so aceita lower/upper quando em df
    custom_holidays = pd.DataFrame(custom_holidays)
    custom_holidays = custom_holidays.drop_duplicates() 
    custom_holidays = custom_holidays.sort_values('ds')


    print("Numero total de feriados: ",len(custom_holidays))
    print(f"Periodo: de {years[0]} até {years[-1]}")

    return custom_holidays


### `metrics(forecast, real):`


def metrics(forecast, real):
    """Calcula as metricas de avaliacao"""
    
    mae = mean_absolute_error(real['y'], forecast['yhat'])
    rmse = np.sqrt(mean_squared_error(real['y'], forecast['yhat']))
    mape = mean_absolute_percentage_error(real['y'], forecast['yhat']) * 100

    print(f"Métricas de Avaliação:")
    print(f"MAE:  {mae}")
    print(f"RMSE: {rmse}")
    print(f"MAPE: {mape}%")


### `train_test_split(df, sep_period)`


def train_test_split(df, test_size):
    """
    Separa um dataframe entre treino e teste baseado no periodo de separacao recebido

    Args:
    - df: Dataframe
    - sep_period: Datetime

    Returns:
    - df_train: Dataframe de treino
    - df_test: Dataframe de teste 
    """

    df_train = df.head(len(df) - test_size)
    df_test = df.tail(test_size)
    
    return df_train, df_test


### `make_graph(forecast, real)`


def make_graph(forecast, real):
    """Gera um grafico com os dados de treino, teste e previsao"""

    forecast_plot = model.plot(forecast)

    axes = forecast_plot.gca()
    last_training_date = forecast['ds'].max()

    plt.plot(real['ds'], real['y'], 'ro',markersize=3, label='True Test Data')
    plt.legend()


# Separacao treino e teste


# trazendo treino e teste com features
# SPLIT_DATE = pd.to_datetime(datetime.today() - timedelta(days=90)) # periodo de 3 meses
test_size = 30

df['ds'] = pd.to_datetime(df['ds'])
df_exogenous = create_features(df)

df_train, df_test = train_test_split(df=df_exogenous, test_size=test_size)


# Adicao de feriados no modelo


YEARS = list(range(2024, 2026))

holidays_br = create_holidays(YEARS)
# holidays_br.tail()


# Configuracao do modelo (prophet)


from prophet import Prophet

# modelo capta automaticamente sazonalidade 
# holidays sao adicionados automaticamente ao modelo

model = Prophet(
    holidays=holidays_br,
    seasonality_mode='multiplicative',
    weekly_seasonality=True,
    daily_seasonality=True,
    yearly_seasonality=True,
)

model.add_seasonality(name='monthly', period=30.5, fourier_order=10)  # ciclo mensal
model.add_seasonality(name='quarterly', period=91.25, fourier_order=8)  # ciclo trimestral


# Adicionando os regressoes (identificacao das variaveis)


# todas as variaveis da funcao `create_features()`
regressors = [col for col in df_exogenous.columns if col not in ['ds', 'y']]

for reg in regressors:
    model.add_regressor(reg)
    print(f"Added: {reg}")


# Fitting


# adicionando features ao treino para que o modelo consiga entender o que esta acontecendo em cada parte
model.fit(df_train)


# Forecasting


future = model.make_future_dataframe(periods=len(df_test))

# Adicionar apenas features temporais (sem rolling)
future = create_features(future)

# Para rolling features, usar apenas histórico conhecido
last_known_values = df_train[['rolling_mean_7', 'rolling_mean_14', 'rolling_mean_28']].iloc[-1]
for col in ['rolling_mean_7', 'rolling_mean_14', 'rolling_mean_28']:
    future[col] = last_known_values[col]  # Propagar último valor conhecido

forecast = model.predict(future)


future


# Metricas de avaliacao | Plot forecast


forecast_30 = forecast.tail(30)

metrics(forecast=forecast_30, real=df_test)
make_graph(forecast=forecast, real=df_test)


plot_series(df_test['y'], forecast_30['yhat'])


from prophet.diagnostics import cross_validation, performance_metrics

df_cv = cross_validation(model, initial='365 days', period='180 days', horizon='90 days')
df_p = performance_metrics(df_cv)
df_p


# df_cv = cross_validation(model, initial='730 days', period='180 days', horizon='365 days')
# performance_metrics(df_cv)


# Define the hyperparameter grid
param_grid = {
    'seasonality_mode': ['additive', 'multiplicative'],
    'changepoint_prior_scale': [0.01, 0.1, 0.5],
    'seasonality_prior_scale': [1, 10, 30],
}


# Helper function to evaluate the model
def evaluate_model(model, metric_func):
    df_cv = cross_validation(model, initial='730 days', period='180 days', horizon='365 days')
    return metric_func(df_cv['y'], df_cv['yhat'])


# Grid search
best_params = {}
best_score = float('inf')

for mode in param_grid['seasonality_mode']:
    for cps in param_grid['changepoint_prior_scale']:
        for sps in param_grid['seasonality_prior_scale']:
            # Create a model with the current hyperparameters
            model = Prophet(seasonality_mode=mode, changepoint_prior_scale=cps, seasonality_prior_scale=sps)
            model.fit(df_train)

            # Evaluate the model using Mean Absolute Error (MAE)
            score = evaluate_model(model, mean_absolute_error)

            # Update best parameters if necessary
            if score < best_score:
                best_score = score
                best_params = {
                    'seasonality_mode': mode,
                    'changepoint_prior_scale': cps,
                    'seasonality_prior_scale': sps
                }


print(best_params)
print(best_score)

# Create the best model with the optimal hyperparameters
best_model = Prophet(**best_params)
best_model.fit(df_train)
future = best_model.make_future_dataframe(periods=30, freq='D')
predict = best_model.predict(future)
forecast_30 = predict.tail(30)
metrics(forecast=forecast_30, real=df_test)
plot_series(df_test['y'], forecast_30['yhat'], labels=['test', 'pred'])


f = forecast_30[['ds', 'yhat']]
display(f)