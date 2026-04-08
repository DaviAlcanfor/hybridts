import pandas as pd
from typing import Tuple, Any, Optional, Union, Callable
import warnings

class TimeSeriesProcessor:
    """
    Time series data processor for preparing and splitting forecast data.
    
    Handles data loading, train/test splitting, and date utilities.
    Originally designed for Databricks but now works with local data sources.
    
    Args:
        mapa_queries: (Optional) Dictionary of SQL queries - kept for backward compatibility.
                     No longer required for usage outside Databricks.
    """
    def __init__(self, mapa_queries: Optional[dict] = None):
        self.mapa_queries = mapa_queries or {}

    def df_train_test_split(
            self,
            df: pd.DataFrame, 
            split_size: int
        ) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """
        Separa um dataframe entre treino e teste de acordo com o tamanho do split.

        Args:
            df: dataframe a ser separado
            split_size: tamanho do split

        Returns:
            df_train: dataframe de treino
            df_test: dataframe de teste
        """
        
        df = df.copy()

        if split_size > len(df):
            raise ValueError("Split size must be smaller than the dataframe length")
        if df.dropna().shape[0] != df.shape[0]:
            raise ValueError("Existem valores nulos no dataframe após o processamento.")

        df_train = df.iloc[:-split_size]
        df_test = df.iloc[-split_size:]

        return df_train, df_test

    def get_min_max_years(self, df: pd.DataFrame) -> Tuple[int, int]:
        """
        Retorna o ano mínimo e máximo do dataframe.

        Args:
            df: dataframe a ser analisado

        Returns:
            min_year: ano mínimo
            max_year: ano máximo
        """

        if df['ds'].isnull().any():
            raise ValueError("Existem valores nulos na coluna de datas (ds).")

        min_year = df['ds'].dt.year.min()
        max_year = df['ds'].dt.year.max()

        return min_year, max_year

    def preparar_dados_completos(
            self,
            escolha_target: Optional[str] = None,
            spark: Any = None,
            df: Optional[pd.DataFrame] = None,
            data_loader: Optional[Callable[[], pd.DataFrame]] = None
        ) -> pd.DataFrame:
        """
        Busca os dados, valida valores e preenche buracos no calendário.

        Três formas de uso:
        
        1. **Fornecendo DataFrame diretamente (recomendado):**
           ```python
           df = pd.read_csv('dados.csv', parse_dates=['ds'])
           processor.preparar_dados_completos(df=df)
           ```
        
        2. **Fornecendo função de carregamento:**
           ```python
           def carregar_dados():
               return pd.read_parquet('dados.parquet')
           processor.preparar_dados_completos(data_loader=carregar_dados)
           ```
        
        3. **Usando Spark (DEPRECATED - para retrocompatibilidade Databricks):**
           ```python
           processor.preparar_dados_completos(escolha_target='TPV', spark=spark)
           ```

        Args:
            escolha_target: (DEPRECATED) Chave para buscar query no mapa_queries
            spark: (DEPRECATED) Sessão do Spark - mantido para retrocompatibilidade
            df: DataFrame pandas com colunas ['ds', 'y']
            data_loader: Função que retorna DataFrame pandas
            
        Returns:
            df: dataframe preparado e validado com colunas ['ds', 'y']
        """
        
        # Prioridade: df > data_loader > spark (legacy)
        if df is not None:
            dados = df.copy()
        elif data_loader is not None:
            dados = data_loader()
        elif spark is not None and escolha_target is not None:
            # Modo legacy (Databricks)
            warnings.warn(
                "Uso de 'spark' está deprecated. Prefira passar DataFrame diretamente via 'df' "
                "ou usar 'data_loader' com função de carregamento.",
                DeprecationWarning,
                stacklevel=2
            )
            if not self.mapa_queries:
                raise ValueError(
                    "mapa_queries não foi fornecido no construtor. "
                    "Para usar spark, inicialize TimeSeriesProcessor com dicionário de queries."
                )
            query_escolhida = self.mapa_queries.get(escolha_target)
            if not query_escolhida:
                raise ValueError(f"Target '{escolha_target}' não encontrado em mapa_queries")
            dados = spark.sql(query_escolhida).toPandas()
        else:
            raise ValueError(
                "Forneça os dados via 'df' (DataFrame), 'data_loader' (função), "
                "ou use o modo legacy com 'spark' + 'escolha_target'"
            )
        
        # Validação e processamento
        if 'ds' not in dados.columns or 'y' not in dados.columns:
            raise ValueError("DataFrame deve conter colunas 'ds' (data) e 'y' (valor target)")
        
        dados['y'] = dados['y'].astype(float)

        if dados['y'].isnull().any():
            raise ValueError("Valores NULL detectados na coluna 'y'")
        if not (dados['y'] >= 0).all():
            raise ValueError("Valores negativos não permitidos na coluna 'y'")
        
        dados['ds'] = pd.to_datetime(dados['ds'])
        dados = dados.sort_values('ds').reset_index(drop=True)
        
        date_range = pd.date_range(dados['ds'].min(), dados['ds'].max(), freq='D')
        faltando = set(date_range) - set(dados['ds'])

        if faltando:
            print(f"{len(faltando)} datas faltando. Preenchendo com zero...")
            dados = dados.set_index('ds')
            dados = dados.reindex(pd.DatetimeIndex(date_range), fill_value=0)
            dados = dados.rename_axis('ds').reset_index()
        
        return dados[['ds', 'y']]