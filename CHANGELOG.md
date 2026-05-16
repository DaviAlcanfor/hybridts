# Changelog

## [0.5.0] - 2026-05-15

### Added
- Hybrid forecasting combining Prophet + XGBoost/LightGBM
- sklearn-style API (`fit`, `predict`, `evaluate`, `evaluate_and_fit`)
- Built-in evaluation metrics (MAE, RMSE, MAPE, sMAPE, R², Bias)
- Auto feature engineering (holidays, payday indicators, calendar features)
- Integrated plotting (`plot_forecast`, `plot_evaluation`)
- YAML and programmatic configuration support
- MLflow integration for experiment tracking
- Input validation in `fit`
- CI/CD with GitHub Actions
