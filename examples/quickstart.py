"""
Quickstart Example - HybridTS Forecasting Library
==================================================

Demonstrates the three configuration modes for HybridTS.
"""

import pandas as pd
import numpy as np
from datetime import datetime, timedelta

# Generate synthetic time series data for demonstration
def generate_sample_data(start_date='2023-01-01', periods=400):
    """Generate synthetic time series data for testing."""
    dates = pd.date_range(start=start_date, periods=periods, freq='D')
    
    # Base trend + weekly seasonality + noise
    trend = np.linspace(10000, 15000, periods)
    seasonality = 2000 * np.sin(2 * np.pi * np.arange(periods) / 7)  # Weekly pattern
    noise = np.random.normal(0, 500, periods)
    
    values = trend + seasonality + noise
    values = np.maximum(values, 0)  # Ensure non-negative
    
    df = pd.DataFrame({
        'ds': dates,
        'y': values
    })
    
    return df


# Example 1: Simplest usage (Mode 1 - Defaults)
def example_defaults():
    """Mode 1: Default configuration - quickest to get started."""
    print("=" * 70)
    print("Example 1: Default Configuration (Quickest)")
    print("=" * 70)
    
    from hybridts import HybridForecaster
    
    # Generate sample data
    df = generate_sample_data(periods=400)
    print(f"✓ Generated {len(df)} days of sample data")
    
    # Initialize with defaults - no config needed!
    forecaster = HybridForecaster()
    
    # Train
    print("\nTraining Prophet + XGBoost with default parameters...")
    forecaster.fit(df, model="xgboost")
    
    # Predict next 30 days
    forecast = forecaster.predict(horizon=30)
    print(f"\n✓ Generated forecast for next 30 days")
    print("\nFirst 5 forecast rows:")
    print(forecast[['data', 'forecast_hibrido_final']].head())
    
    print(f"\nTotal forecast: ${forecast['forecast_hibrido_final'].sum():,.0f}")


# Example 2: Programmatic configuration (Mode 2 - Dict)
def example_custom_config():
    """Mode 2: Custom configuration via dict - flexible for experiments."""
    print("\n" + "=" * 70)
    print("Example 2: Programmatic Configuration (Dict)")
    print("=" * 70)
    
    from hybridts import HybridForecaster
    
    # Define config programmatically
    config = {
        'test_size': 30,
        'cv_params': {
            'initial_window': 300,
            'step_length': 30
        },
        'models': {
            'prophet': {
                'param_grid': {
                    'changepoint_prior_scale': [0.05],
                    'seasonality_prior_scale': [5.0],
                    'seasonality_mode': ['multiplicative']
                },
                'cv_params': {
                    'initial': '350 days',
                    'period': '30 days',
                    'horizon': '30 days',
                    'parallel': 'threads'
                }
            },
            'xgboost': {
                'param_grid': {
                    'window_length': [21],
                    'estimator__max_depth': [5],
                    'estimator__learning_rate': [0.05],
                    'estimator__n_estimators': [200]
                }
            }
        }
    }
    
    df = generate_sample_data(periods=400)
    forecaster = HybridForecaster(config=config)
    
    print("✓ Using custom configuration (no YAML file)")
    forecaster.fit(df, model="xgboost")
    
    forecast = forecaster.predict(horizon=30)
    print(f"\n✓ Forecast generated: ${forecast['forecast_hibrido_final'].sum():,.0f}")


# Example 3: YAML configuration (Mode 3 - Production)
def example_yaml_config():
    """Mode 3: YAML configuration - for production and reproducibility."""
    print("\n" + "=" * 70)
    print("Example 3: YAML Configuration (Production)")
    print("=" * 70)
    
    from hybridts import HybridForecaster, load_config
    
    try:
        # Load config from YAML
        config = load_config("../hybridts/config/settings.yaml")
        df = generate_sample_data(periods=400)
        
        forecaster = HybridForecaster(config=config)
        print("✓ Configuration loaded from settings.yaml")
        
        forecaster.fit(df, model="xgboost")
        forecast = forecaster.predict(horizon=30)
        
        print(f"\n✓ Forecast generated: ${forecast['forecast_hibrido_final'].sum():,.0f}")
        
    except FileNotFoundError:
        print("⚠️  settings.yaml not found (this is fine for demo)")
        print("   In production, you'd have a config file checked into version control")


# Example 4: Custom holidays (US market)
def example_custom_holidays():
    """Using custom holidays for different markets."""
    print("\n" + "=" * 70)
    print("Example 4: Custom Holidays (US Market)")
    print("=" * 70)
    
    from hybridts import HybridForecaster
    
    df = generate_sample_data(periods=400)
    forecaster = HybridForecaster()
    
    # Fit with US holidays instead of Brazilian
    print("Training with US holidays...")
    forecaster.fit(
        df, 
        model="xgboost",
        holidays_country="US",
        holidays_state=None  # Country-level only
    )
    
    forecast = forecaster.predict(horizon=30)
    print(f"✓ Forecast with US holidays: ${forecast['forecast_hibrido_final'].sum():,.0f}")


# Example 5: Different models
def example_model_comparison():
    """Compare different residual models."""
    print("\n" + "=" * 70)
    print("Example 5: Model Comparison")
    print("=" * 70)
    
    from hybridts import HybridForecaster
    
    df = generate_sample_data(periods=400)
    
    models = ["xgboost", "sxgboost", "lightgbm"]
    results = {}
    
    for model_name in models:
        forecaster = HybridForecaster()
        print(f"\nTraining Prophet + {model_name.upper()}...")
        forecaster.fit(df, model=model_name)
        
        forecast = forecaster.predict(horizon=30)
        total = forecast['forecast_hibrido_final'].sum()
        results[model_name] = total
        
        print(f"  Total forecast: ${total:,.0f}")
    
    print("\n" + "-" * 70)
    print("Summary:")
    for model, total in results.items():
        print(f"  {model:12s}: ${total:,.0f}")


if __name__ == "__main__":
    # Run all examples
    example_defaults()           # Start here - simplest usage
    example_custom_config()      # When you need customization
    example_yaml_config()        # For production deployments
    example_custom_holidays()    # For different markets
    example_model_comparison()   # Compare models
    
    print("\n" + "=" * 70)
    print("All examples completed successfully! 🎉")
    print("=" * 70)
    print("\nNext steps:")
    print("  1. Try with your own data (CSV with 'ds' and 'y' columns)")
    print("  2. Experiment with different models and parameters")
    print("  3. Check out examples/research/ for advanced techniques")
    print("  4. Read CONTRIBUTING.md to contribute improvements")

