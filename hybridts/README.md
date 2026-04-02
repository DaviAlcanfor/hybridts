## Arquitetura

### Configuração

O arquivo `config/settings.yaml` centraliza:
- Queries de TPV/MAT
- Parâmetros de validação
- Configurações de cada modelo

### Modelos

| Modelo | Papel |
|--------|-------|
| Prophet | Principal — captura tendência e sazonalidade |
| XGBoost | Auxiliar — modela resíduos |
| LightGBM | Auxiliar — modela resíduos |

### Fluxo de Treinamento

1. **Validação cruzada (Prophet)** → seleciona melhores hiperparâmetros
2. **Geração de resíduos** → calcula erro do treino do Prophet
3. **Validação cruzada (modelo auxiliar)**
   - Modelo selecionado via widget do notebook
   - Treina sobre os resíduos
   - Seleciona melhores hiperparâmetros
4. **HybridForecaster** (`src/pipeline/pipeline.py`)
   - Combina Prophet + modelo auxiliar
   - `fit()` e `predict()`
   - `save_to_mlflow()` → salva no MLflow Experiments

### Features

- **Exógenas:** `src/features/engineering.py`
- **Feriados:** `src/features/holidays.py` (com windows)

### Notebooks

| Notebook | Quando usar | Tempo |
|----------|-------------|-------|
| `Run Pipeline` | Treina e salva modelo no MLflow | ~1h |
| `Run Inference` | Carrega modelo salvo e gera forecast | ~2min |

> **Frequência sugerida:** `Run Pipeline` 1x/mês, `Run Inference` no dia-a-dia.
