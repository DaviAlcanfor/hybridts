# Modelo híbrido de série temporal SARIMAX + XGBoost
# ARIMA capta inicialmente sazonalidade e tendencia, para robustes e consistencia. Já o XGB vai refinar mais precisamente as "pontas" do forecast estatico do arima


import pandas as pd
import numpy as np 
import pmdarima
from datetime import date, timedelta, datetime
import matplotlib.pyplot as plt
from sktime.utils.plotting import plot_series
from sktime.forecasting.base import ForecastingHorizon
from sktime.forecasting.model_selection import temporal_train_test_split
from sklearn.compose import make_column_transformer
from sktime.forecasting.model_selection import SlidingWindowSplitter
from sktime.forecasting.model_selection import ForecastingGridSearchCV
from sktime.forecasting.compose import make_reduction
from sktime.forecasting.compose import RecursiveTimeSeriesRegressionForecaster
from sktime.forecasting.compose import DirectReductionForecaster
from xgboost import XGBRegressor
from sktime.forecasting.arima import AutoARIMA
from pmdarima import auto_arima


# essa query pega:
# - data
# - tpv
# em crédito e transações aprovadas


# Forneça CSV/Parquet com colunas: ['ds' ou 'dt', 'y']
df = pd.read_csv('seu_arquivo_tpv.csv', parse_dates=['dt'])
# Renomear 'dt' para 'ds' se necessário: df = df.rename(columns={'dt': 'ds'})

# Alternativa: df = pd.read_parquet('seu_arquivo_tpv.parquet')


# Funções 


def normalize_series(df):
    """
    Normaliza os valores target de um dataframe e adiciona a data de inicio para um indice

    Args:
    - df: Dataframe
    - start_date: Timestamp
    - periods: int

    Returns:
    - df_normal: Dataframe
    """
    
    return pd.DataFrame(data={'y': df.values}, index=df.index)


def normalize(df, target, start_date, periods):
    """
    Normaliza os valores target de um dataframe e adiciona a data de inicio para um indice

    Args:
    - df: Dataframe
    - start_date: Timestamp
    - periods: int

    Returns:
    - df_normal: Dataframe
    """

    df_normal = df.copy()

    index = pd.date_range(start=start_date, periods=periods, freq='D')

    df_normal = pd.DataFrame(data={'y': target.values}, index=index)

    return df_normal


def normalize_X(start_date, periods):
    """
    Normaliza apenas as colunas de datas, criando indices

    Args:
    - start_date: Timestamp
    - periods: int
    
    Returns:
    - df_exo: Dataframe
    """

    index = pd.date_range(start=start_date, periods=periods, freq='D')
    return pd.DataFrame(index=index)


def minimize(string):
    """ Minimiza uma string, tirando espacos e substituindo por underline, colocando tudo em minusculo e removendo aspas"""
    return string.lower().replace(" ", "_").replace("'","")


# def get_season(date):
#     """Retorna a estacao em que a dt recebida está"""
#     m = date.month
#     d = date.day
#     if (m == 12 and d >= 21) or (m in [1, 2]) or (m == 3 and d < 21): return "Summer"
#     if (m == 3 and d >= 21) or (m in [4, 5]) or (m == 6 and d < 21): return "Autumn"
#     if (m == 6 and d >= 21) or (m in [7, 8]) or (m == 9 and d < 23): return "Winter"
#     if (m == 9 and d >= 23) or (m in [10, 11]) or (m == 12 and d < 21): return "Spring"


from datetime import datetime, timedelta

def second_sunday_may():
    """Retorna o dia do segundo domingo de maio do ano corrente.
    Inicia em 1º de maio e itera até encontrar o segundo domingo (weekday == 6)."""
    first_may_day = datetime(datetime.today().year, 5, 1)
    sunday_counter = 0
    while sunday_counter < 2:
        if first_may_day.weekday() == 6:  # Domingo
            sunday_counter += 1
        first_may_day += timedelta(days=1)
    return first_may_day.day

def second_sunday_august():
    """Retorna o dia do segundo domingo de agosto do ano corrente.
    Inicia em 1º de agosto e itera até encontrar o segundo domingo (weekday == 6)."""
    first_august_day = datetime(datetime.today().year, 8, 1)
    sunday_counter = 0
    while sunday_counter < 2:
        if first_august_day.weekday() == 6:  # Domingo
            sunday_counter += 1
        first_august_day += timedelta(days=1)
    return first_august_day.day

def last_friday_november():
    """Retorna o dia da última sexta-feira de novembro do ano corrente.
    Inicia em 30 de novembro e retrocede até encontrar uma sexta-feira (weekday == 4)."""
    last_day_november = datetime(datetime.today().year, 11, 30)
    while last_day_november.weekday() != 4:  # Sexta-feira
        last_day_november -= timedelta(days=1)
    return last_day_november.day


### Datas que as pessoas mais gastam dinheiro
# * Volta as aulas - 3 a 5 de fevereiro
# * Carnaval - 47 dias antes da pascoa
# * Páscoa - 22/25 de abril
# * Dia das mães - segundo domingo de maio
# * Dia dos namorados - 12 de junho ???
# * Dia dos pais - segundo domindo de agosto
# * Dia das crianças - 12/10
# * Halloween - 31/10
# * Black Friday - ultima sexta de novembro
# * Natal - 25/12
# * Ano novo - 01/01


def make_exo(df: pd.DataFrame, lag=False, holiday=False, compressed=False):
    """
    Cria variáveis exógenas relacionadas ao tempo em que ocorrem mais transações financeiras.
    
    Args:
    - df: Dataframe

    Returns:
    - df_exo: Dataframe
    """

    import holidays
    br_holidays = holidays.BR(years=range(2022, 2025), state='SP')

    df_exo = df.copy()

    # Cria colunas binárias para feriados, finais de semana, início do mês, férias e características temporais
    df_exo['ds'] = pd.to_datetime(df_exo.index)
    df_exo['is_holiday'] = df_exo['ds'].isin(br_holidays).astype(int)
    df_exo['is_weekend'] = df_exo['ds'].dt.weekday.isin([5, 6]).astype(int)
    df_exo['is_month_start'] = df_exo['ds'].dt.day.lt(10).astype(int)

    if not compressed:
        df_exo['is_vacation_time'] = df_exo['ds'].dt.month.isin([12, 1, 7]).astype(int)
        df_exo['year'] = df_exo['ds'].dt.year
        df_exo['month'] = df_exo['ds'].dt.month
        df_exo['quarter'] = df_exo['ds'].dt.quarter
        df_exo['day_of_week'] = df_exo['ds'].dt.dayofweek
        df_exo['day_of_year'] = df_exo['ds'].dt.dayofyear
        df_exo['day_of_month'] = df_exo['ds'].dt.day
        df_exo['week_of_year'] = df_exo['ds'].dt.isocalendar().week
        df_exo['season_Autumn'] = df_exo['ds'].apply(lambda date: 1 if get_season(date) == "Autumn" else 0)
        df_exo['season_Winter'] = df_exo['ds'].apply(lambda date: 1 if get_season(date) == "Winter" else 0)
        df_exo['season_Spring'] = df_exo['ds'].apply(lambda date: 1 if get_season(date) == "Spring" else 0)

    if holiday:
        df_exo = df_exo.drop(columns=['is_holiday'])
        for dt, holiday in br_holidays.items():
            df_exo[minimize(holiday)] = df_exo['ds'].apply(lambda date: 1 if date.month == dt.month and date.day == dt.day else 0)
        df_exo['back_to_school'] = df_exo['ds'].apply(lambda date: 1 if date.month == 2 and date.day in [3, 4, 5] else 0)
        df_exo['is_black_friday'] = df_exo['ds'].apply(lambda date: 1 if date.month == 11 and date.day == last_friday_november() else 0)
        df_exo['halloween'] = df_exo['ds'].apply(lambda date: 1 if date.month == 10 and date.day == 31 else 0)
        df_exo['mothers_day'] = df_exo['ds'].apply(lambda date: 1 if date.month == 5 and date.day == 8 else 0)
        df_exo['childs_day'] = df_exo['ds'].apply(lambda date: 1 if date.day == 12 and date.month == 10 else 0)
        df_exo['lovers_day'] = df_exo['ds'].apply(lambda date: 1 if date.day == 12 and date.month == 6 else 0)
        df_exo['fathers_day'] = df_exo['ds'].apply(lambda date: 1 if date.month == 8 and date.day == second_sunday_august() else 0)
        df_exo['easter'] = df_exo['ds'].apply(lambda date: 1 if date.day == 20 and date.month == 4 else 0)

    # Adiciona lags e estatísticas de janela móvel se solicitado
    # if lag:
    #     df_exo['lag_1'] = df_exo['y'].shift(1)
    #     df_exo['lag_2'] = df_exo['y'].shift(7)
    #     df_exo['lag_3'] = df_exo['y'].shift(14)
    #     df_exo['lag_4'] = df_exo['y'].shift(28)
    #     # df_exo['lag_5'] = df_exo['y'].shift(5)
    #     # df_exo['lag_6'] = df_exo['y'].shift(6)
    #     # df_exo['lag_7'] = df_exo['y'].shift(7)

    #     df_exo['rolling_mean_7'] = df_exo['y'].rolling(window=7).mean()
    #     df_exo['rolling_mean_14'] = df_exo['y'].rolling(window=14).mean()
    #     df_exo['rolling_mean_28'] = df_exo['y'].rolling(window=28).mean()   
    #     # df_exo['rolling_std_7'] = df_exo['y'].rolling(window=7).std()
    #     # df_exo['rolling_min_7'] = df_exo['y'].rolling(window=7).min()
    #     # df_exo['rolling_max_7'] = df_exo['y'].rolling(window=7).max()
    #     # df_exo['rolling_sum_7'] = df_exo['y'].rolling(window=7).sum()
    #     # df_exo['rolling_median_7'] = df_exo['y'].rolling(window=7).median() 

    # Preenche valores nulos gerados pelas operações de lag e rolling
    df_exo = df_exo.fillna(0)
    df_exo.drop(columns=['ds'], inplace=True)

    return df_exo


def metrics(test, pred):
    # Importa métricas de avaliação de séries temporais do sktime
    from sktime.performance_metrics.forecasting import (
        mean_absolute_error,
        mean_squared_error,
        mean_absolute_percentage_error,
    )

    # Calcula o erro absoluto médio (MAE)
    mae = int(mean_absolute_error(test, pred))
    # Calcula o erro quadrático médio (MSE)
    mse = int(mean_squared_error(test, pred))
    # Calcula o erro percentual absoluto médio (MAPE)
    mape = mean_absolute_percentage_error(test, pred)

    # Exibe as métricas calculadas
    print("MAE: ", mae)
    print("MSE: ", mse)
    print(f"MAPE: {mape:.2f}")


# def summarize(df):
#     """
#     Faz o sumario dos dados dos periodos de um dataframe
#     Args:
#     - df: Dataframe
#     Returns:
#     - Valor correspondente a cada periodo
#     - Valor total
#     - Periodo do maior valor registrado
#     """
#     for data, valor in df.items():
#         print(data.strftime("%d") + " - " + str(round(valor))) 
#     total = str(round(sum(df)))
#     print("\nTOTAL:\n" + total ) # total[:1] + "," + total[1:3] + )
#     print("\nMaior valor foi no dia " + str(df.idxmax().day) + ":\n" + str(round(max(df))))
#     print("\nMenor valor foi no dia " + str(df.idxmin().day) + ":\n" + str(round(min(df))))


# Constantes


DATA_CORTE = pd.Timestamp(pd.to_datetime('2024-01-01')) # caso haja necessidade de corte de data
DATA_INICIO = pd.Timestamp(pd.to_datetime('2022-04-01')) # data do primeiro dado coletado de tpv/dia
COMECO_NOVEMBRO = pd.Timestamp(pd.to_datetime('2025-11-01')) # comeco de novembro para fazer predicoes
MONTH_PERIOD = 30 # periodo mensal
PERIODOS = len(df) # numero de periodos do dataframe
FREQ = 'D' # frequencia de indice/data do dataframe


# Preparando X e y


# normaliza os dados coletados de tpv/dia e transforma em uma série
df = normalize(df=df, target=df['y'], start_date=DATA_INICIO, periods=PERIODOS)

# Criando dataframe com:

# lag features
df_exo_lag = make_exo(df, lag=True, holiday=True, compressed=True)
y_lag = df_exo_lag['y'].astype(float)
X_lag = df_exo_lag.drop(columns=['y'])

y_train_lag, y_test_lag, X_train_lag, X_test_lag = temporal_train_test_split(X=X_lag, y=y_lag, test_size=0.1)

# non lag features
df_exo = make_exo(df, compressed=True)
y = df['y'].astype(float)
X = df_exo.drop(columns=['y'])

y_train, y_test, X_train, X_test = temporal_train_test_split(y=y, X=X, test_size=0.1)

# compressed


# Configuração do ARIMA


fh = ForecastingHorizon(np.arange(1, len(y_test_lag) + 1), is_relative=True) 

arima_model = AutoARIMA(
    # deixa de reserva caso use o auto_arima do pmdarima
    # y=y_train, 
    # X=X_train,

    seasonal=True,                 # (S)       
    stepwise=True,                 # ~= GridSearchCV (testa os melhores parametros para o modelo) 
    trace=False,                   # mostra o progresso do stepwise no terminal + AIC 
    start_p=1, start_q=1,          # (AR) Define em quantos passos atrás o modelo irá começar a analisar os dados
    start_P=1, start_Q=1,          # S(AR) Define em quantos passos atrás o modelo irá começar a analisar os dados
    # m=7,                           # (pmdarima auto_arima) Sazonalidade da série
    d=None,                        # Diferença não sazonal para se tornar estacionária
    D=1,                           # Ordem de diferenciação sazonal
    max_p=5, max_q=5, max_d=2,     # O máximo de n parametros que serão testados
    max_P=2, max_D=1, max_Q=2,     #
    max_order=10,                  # Limite máximo da soma dos parâmetros não sazonais
    error_action='ignore',         # Ignora o erro e parte pro próximo teste
    sp=30,                          # (sktime AutoARIMA) Sazonalidade da série 
    suppress_warnings=True,        # Não aparece o erro 
    n_jobs=-1,                     # N de núcleos usados para o processamento (-1 = todos)
    random_state=42
)


# Config do XGB


# o XGBoost funciona como várias árvores de decisão, então pode ser usado para fazer predições recursivamente
xgb = make_reduction(
    estimator=XGBRegressor(
        n_estimators=300,
        max_depth=6,
        learning_rate=0.05,
        subsample=0.8,
        colsample_bytree=0.8,
        tree_method="hist",
        n_jobs=1
    ),
    window_length=28,
    strategy='recursive' # se baseia em valores passados
)


# Fit e residuo do ARIMA 


# fitting
arima_fitted = arima_model.fit(y=y_train)

# coletando os fitted values para poder treinar o xgboost com o residuo do treino
fh_train = ForecastingHorizon(y_train.index, is_relative=False)
arima_fitted_values = arima_fitted.predict(fh=fh_train)

# residual do treino ARIMA
arima_train_residuals = y_train - arima_fitted_values


# Fitting do XGB (treino com o residuo do arima)


# O fit do xgb está sendo feito com o residual do treino do ARIMA,
# isso faz com que o xgb entenda onde o ARIMA falhou e corrige os possíveis 
# futuros erros do ARIMA
xgb_fitted = xgb.fit(y=arima_train_residuals, X=X_train_lag)


# Forecast ambos


arima_forecast = arima_model.predict(fh=fh) 
xgb_forecast = xgb_fitted.predict(fh=fh, X=X_test_lag)

# Juncao dos modelos (hibrido)
hybrid_forecast = arima_forecast + xgb_forecast


# Resultado do forecast híbrido


metrics(test=y_test, pred=hybrid_forecast)
plot_series(y_test, hybrid_forecast, labels=["real", "predicted"])