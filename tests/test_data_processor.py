"""
Testes para TimeSeriesProcessor

Exemplos básicos de testes automatizados usando pytest.
Para rodar: pytest tests/test_data_processor.py
"""

import pytest
import pandas as pd
import numpy as np
from datetime import datetime, timedelta

from hybridts import TimeSeriesProcessor


@pytest.fixture
def sample_df():
    """Fixture: DataFrame de exemplo com 100 dias de dados."""
    dates = pd.date_range('2023-01-01', periods=100, freq='D')
    values = np.random.uniform(1000, 2000, 100)
    return pd.DataFrame({'ds': dates, 'y': values})


@pytest.fixture
def sample_df_with_gaps():
    """Fixture: DataFrame com datas faltando."""
    dates = pd.date_range('2023-01-01', periods=100, freq='D')
    # Remover algumas datas
    dates_with_gaps = [d for i, d in enumerate(dates) if i % 10 != 5]
    values = np.random.uniform(1000, 2000, len(dates_with_gaps))
    return pd.DataFrame({'ds': dates_with_gaps, 'y': values})


class TestTimeSeriesProcessor:
    """Testes da classe TimeSeriesProcessor."""
    
    def test_init_without_queries(self):
        """Testa inicialização sem mapa de queries."""
        processor = TimeSeriesProcessor()
        assert processor.mapa_queries == {}
    
    def test_init_with_queries(self):
        """Testa inicialização com mapa de queries."""
        queries = {'TPV': 'SELECT * FROM ...', 'MAT': 'SELECT * FROM ...'}
        processor = TimeSeriesProcessor(mapa_queries=queries)
        assert processor.mapa_queries == queries
    
    def test_train_test_split_valid(self, sample_df):
        """Testa split com tamanho válido."""
        processor = TimeSeriesProcessor()
        df_train, df_test = processor.df_train_test_split(sample_df, split_size=30)
        
        assert len(df_train) == 70
        assert len(df_test) == 30
        assert df_train.iloc[-1]['y'] != df_test.iloc[0]['y']  # Não há overlap
    
    def test_train_test_split_invalid_size(self, sample_df):
        """Testa que erro é lançado quando split_size > len(df)."""
        processor = TimeSeriesProcessor()
        
        with pytest.raises(ValueError, match="Split size must be smaller"):
            processor.df_train_test_split(sample_df, split_size=200)
    
    def test_train_test_split_with_nulls_raises_error(self):
        """Testa que erro é lançado se houver valores nulos."""
        processor = TimeSeriesProcessor()
        df_with_nulls = pd.DataFrame({
            'ds': pd.date_range('2023-01-01', periods=10),
            'y': [1, 2, None, 4, 5, 6, 7, 8, 9, 10]
        })
        
        with pytest.raises(ValueError, match="valores nulos"):
            processor.df_train_test_split(df_with_nulls, split_size=3)
    
    def test_get_min_max_years(self, sample_df):
        """Testa extração de anos mínimo e máximo."""
        processor = TimeSeriesProcessor()
        min_year, max_year = processor.get_min_max_years(sample_df)
        
        assert min_year == 2023
        assert max_year == 2023
    
    def test_preparar_dados_completos_with_df(self, sample_df):
        """Testa preparação de dados passando DataFrame diretamente."""
        processor = TimeSeriesProcessor()
        df_processed = processor.preparar_dados_completos(df=sample_df)
        
        assert 'ds' in df_processed.columns
        assert 'y' in df_processed.columns
        assert len(df_processed) == len(sample_df)
        assert df_processed['y'].dtype == float
    
    def test_preparar_dados_completos_fills_gaps(self, sample_df_with_gaps):
        """Testa que datas faltando são preenchidas com zero."""
        processor = TimeSeriesProcessor()
        df_processed = processor.preparar_dados_completos(df=sample_df_with_gaps)
        
        # DataFrame processado deve ter 100 dias (sem gaps)
        date_range = pd.date_range(
            df_processed['ds'].min(),
            df_processed['ds'].max(),
            freq='D'
        )
        assert len(df_processed) == len(date_range)
        
        # Valores preenchidos devem ser 0
        assert (df_processed[df_processed['y'] == 0].shape[0] > 0)
    
    def test_preparar_dados_completos_validates_nulls(self):
        """Testa validação de valores nulos."""
        processor = TimeSeriesProcessor()
        df_with_nulls = pd.DataFrame({
            'ds': pd.date_range('2023-01-01', periods=10),
            'y': [1, 2, None, 4, 5, 6, 7, 8, 9, 10]
        })
        
        with pytest.raises(ValueError, match="NULL detectados"):
            processor.preparar_dados_completos(df=df_with_nulls)
    
    def test_preparar_dados_completos_validates_negative_values(self):
        """Testa validação de valores negativos."""
        processor = TimeSeriesProcessor()
        df_with_negatives = pd.DataFrame({
            'ds': pd.date_range('2023-01-01', periods=10),
            'y': [1, 2, -5, 4, 5, 6, 7, 8, 9, 10]
        })
        
        with pytest.raises(ValueError, match="negativos não permitidos"):
            processor.preparar_dados_completos(df=df_with_negatives)
    
    def test_preparar_dados_completos_requires_ds_and_y_columns(self):
        """Testa que erro é lançado se colunas obrigatórias estiverem faltando."""
        processor = TimeSeriesProcessor()
        df_missing_cols = pd.DataFrame({
            'date': pd.date_range('2023-01-01', periods=10),
            'value': range(10)
        })
        
        with pytest.raises(ValueError, match="deve conter colunas 'ds'"):
            processor.preparar_dados_completos(df=df_missing_cols)
    
    def test_preparar_dados_completos_with_data_loader(self, sample_df, tmp_path):
        """Testa preparação usando função de carregamento."""
        # Salvar CSV temporário
        csv_path = tmp_path / "test_data.csv"
        sample_df.to_csv(csv_path, index=False)
        
        processor = TimeSeriesProcessor()
        
        # Função de carregamento
        def load_data():
            return pd.read_csv(csv_path, parse_dates=['ds'])
        
        df_processed = processor.preparar_dados_completos(data_loader=load_data)
        
        assert len(df_processed) == len(sample_df)
        assert df_processed['ds'].dtype == 'datetime64[ns]'
    
    def test_preparar_dados_completos_raises_when_no_input(self):
        """Testa que erro é lançado se nenhuma fonte de dados for fornecida."""
        processor = TimeSeriesProcessor()
        
        with pytest.raises(ValueError, match="Forneça os dados via"):
            processor.preparar_dados_completos()


if __name__ == "__main__":
    # Permite rodar os testes diretamente: python tests/test_data_processor.py
    pytest.main([__file__, "-v"])
