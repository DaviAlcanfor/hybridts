"""
HybridTS - Hybrid Time Series Forecasting Library
==================================================

A Python library for time series forecasting using hybrid models:
- Prophet + XGBoost
- Prophet + LightGBM

Features:
- Automated feature engineering for time series
- Holiday effects (customizable by country)
- Temporal cross-validation
- MLflow integration for experiment tracking
- Easy-to-use API for quick forecasting
"""

from setuptools import setup, find_packages
from pathlib import Path

# Read README for long_description
this_directory = Path(__file__).parent
long_description = (this_directory / "README.md").read_text(encoding="utf-8")

setup(
    name="hybridts",
    version="0.1.0",
    author="Davi Franco",
    author_email="alcanfordavi@gmail.com",
    description="Hybrid time series forecasting library combining Prophet with gradient boosting",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/davifrancamaciel/hybridts",
    project_urls={
        "Bug Tracker": "https://github.com/davifrancamaciel/hybridts/issues",
        "Documentation": "https://github.com/davifrancamaciel/hybridts#readme",
        "Source Code": "https://github.com/davifrancamaciel/hybridts",
    },
    packages=find_packages(exclude=["tests", "notebooks", "examples"]),
    classifiers=[
        "Development Status :: 3 - Alpha",
        "Intended Audience :: Developers",
        "Intended Audience :: Science/Research",
        "License :: OSI Approved :: MIT License",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
        "Programming Language :: Python :: 3.12",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
        "Operating System :: OS Independent",
    ],
    python_requires=">=3.8",
    install_requires=[
        "pandas>=1.3.0,<2.3.0",
        "numpy>=1.21.0,<2.0.0",
        "scikit-learn>=1.0.0,<1.6.0",
        "prophet>=1.1.0,<1.2.0",
        "xgboost>=1.6.0,<2.2.0",
        "lightgbm>=3.3.0,<4.6.0",
        "sktime>=0.13.0,<0.33.0",
        "holidays>=0.14.0,<0.64.0",
        "mlflow>=2.0.0,<2.19.0",
        "pyyaml>=6.0,<6.1.0",
        "joblib>=1.1.0,<1.5.0",
    ],
    extras_require={
        "dev": [
            "pytest>=7.0.0",
            "pytest-cov>=3.0.0",
            "black>=22.0.0",
            "isort>=5.10.0",
            "flake8>=4.0.0",
            "mypy>=0.950",
        ],
        "docs": [
            "sphinx>=4.5.0",
            "sphinx-rtd-theme>=1.0.0",
        ],
    },
    include_package_data=True,
    zip_safe=False,
)
