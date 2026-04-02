# Examples

This directory contains example scripts and research notebooks demonstrating how to use HybridTS.

## 📁 Directory Structure

```
examples/
├── quickstart.py           # Quick start guide - simplest usage
├── README.md               # This file
└── research/               # Research notebooks (exported from Databricks)
    ├── Prophet.py
    ├── Prophet + XGBoost.py
    ├── Prophet + LightGBM.py
    ├── Prophet + XGBoost (static params).py
    ├── Hybrid Time Series - SARIMAX + XGBoost.py
    ├── [MAT] Hybrid TS  - Prophet | Xgboost | LightGBM.py
    └── [TPV] Hybrid TS  - Prophet | Xgboost | LightGBM.py
```

## 🚀 Quick Start

Start with `quickstart.py` for the simplest introduction:

```bash
cd examples/
python quickstart.py
```

## 📊 Research Notebooks

The `research/` directory contains exploratory scripts (originally Jupyter notebooks exported from Databricks) used during the library's development. These demonstrate:

- Different model combinations (Prophet + XGBoost/LightGBM/SARIMAX)
- Hyperparameter tuning approaches
- Cross-validation strategies
- Real-world forecasting workflows

**Note:** These are Python scripts (`.py`), not interactive notebooks. They contain the complete code with comments and markdown cells preserved as comments.

### Running Research Scripts

```bash
cd examples/research/
python "Prophet + XGBoost.py"
```

**Requirements:**
- You'll need to provide your own dataset as CSV/Parquet
- Scripts expect data with columns: `ds` (date) and `y` (value)
- See inline comments for data format examples



## 📚 Usage Modes

HybridTS supports three configuration modes:

### 1. Default Configuration (Quickest)
```python
from hybridts import HybridForecaster

forecaster = HybridForecaster()
forecaster.fit(df)
forecast = forecaster.predict(horizon=30)
```

### 2. Programmatic Configuration
```python
config = {
    'test_size': 30,
    'models': {
        'prophet': {...},
        'xgboost': {...}
    }
}
forecaster = HybridForecaster(config=config)
```

### 3. YAML Configuration (Production)
```python
from hybridts.config import load_config

config = load_config("settings.yaml")
forecaster = HybridForecaster(config=config)
```

See `quickstart.py` for complete examples of all three modes.

## 🐛 Troubleshooting

**Issue:** Script fails with "module not found"
- **Solution:** Install hybridts: `pip install -e ..` (from project root)

**Issue:** Missing data file
- **Solution:** Scripts expect you to provide your own dataset. See inline comments for format.

**Issue:** MLflow errors
- **Solution:** Ensure MLflow is installed: `pip install mlflow`

## 📖 More Resources

- [Main README](../README.md) - Full library documentation
- [Contributing Guide](../CONTRIBUTING.md) - How to contribute
- [API Reference](../docs/) - Detailed API documentation (coming soon)

---

**Questions?** Open an issue on GitHub or check the main README.
