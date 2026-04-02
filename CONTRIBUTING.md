# Contributing to HybridTS

Thank you for your interest in contributing to HybridTS! This document provides guidelines and instructions for contributing.

## 🚀 Quick Start

```bash
# Clone the repository
git clone https://github.com/YOUR_USERNAME/hybridts.git
cd hybridts

# Create a virtual environment
python3 -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install in development mode
pip install -e ".[dev]"

# Run tests
pytest

# Run code formatting
black hybridts/ tests/ examples/
```

## 📋 Development Setup

### Prerequisites

- Python 3.8 or higher
- pip and virtualenv
- Git

### Installation for Development

```bash
# Install all dependencies including dev tools
pip install -e ".[dev]"

# Or manually install dev dependencies
pip install pytest pytest-cov black flake8 mypy
```

## 🧪 Testing

We use `pytest` for testing. Tests are located in the `tests/` directory.

### Running Tests

```bash
# Run all tests
pytest

# Run with coverage report
pytest --cov=hybridts --cov-report=html

# Run specific test file
pytest tests/test_data_processor.py

# Run tests matching a pattern
pytest -k "test_train_test_split"
```

### Writing Tests

- Place tests in `tests/` directory
- Name test files as `test_*.py`
- Name test functions as `test_*`
- Use fixtures for common setup (see `tests/conftest.py`)

Example test:

```python
import pytest
from hybridts import TimeSeriesProcessor

def test_train_test_split():
    processor = TimeSeriesProcessor()
    df = generate_sample_data(periods=100)
    
    df_train, df_test = processor.df_train_test_split(df, split_size=30)
    
    assert len(df_train) == 70
    assert len(df_test) == 30
    assert df_train['ds'].max() < df_test['ds'].min()
```

## 🎨 Code Style

We follow PEP 8 with some modifications. Use `black` for automatic formatting.

### Formatting

```bash
# Format all Python files
black hybridts/ tests/ examples/

# Check formatting without making changes
black --check hybridts/
```

### Linting

```bash
# Run flake8
flake8 hybridts/ tests/

# Run type checking
mypy hybridts/
```

### Style Guidelines

- **Line length:** 100 characters (black default)
- **Imports:** Use absolute imports, group by standard library → third-party → local
- **Docstrings:** Google style (see below)
- **Type hints:** Required for public APIs
- **Naming:**
  - Classes: `PascalCase`
  - Functions/variables: `snake_case`
  - Constants: `UPPER_SNAKE_CASE`

### Docstring Format

We use Google-style docstrings:

```python
def function_name(param1: type, param2: type) -> return_type:
    """
    Brief description of what the function does.
    
    Longer description if needed. Can span multiple lines
    and include details about the implementation.
    
    Args:
        param1: Description of param1
        param2: Description of param2
        
    Returns:
        Description of return value
        
    Raises:
        ValueError: When this exception is raised
        TypeError: When this other exception is raised
        
    Example:
        >>> result = function_name(10, "test")
        >>> print(result)
        Expected output
    """
    pass
```

## 📝 Commit Messages

We follow [Conventional Commits](https://www.conventionalcommits.org/):

```
<type>: <description>

[optional body]

[optional footer]
```

### Types

- **feat:** New feature
- **fix:** Bug fix
- **docs:** Documentation changes
- **style:** Code style changes (formatting, no logic change)
- **refactor:** Code refactoring (no feature change or bug fix)
- **test:** Adding or updating tests
- **chore:** Maintenance tasks (dependencies, build, etc.)

### Examples

```bash
feat: add SARIMAX residual model

Implements SARIMAX as an alternative to Prophet for baseline forecasting.
Includes tests and example usage.

Closes #42

---

fix: handle empty dataframes in train_test_split

Previously crashed with IndexError when df had fewer rows than split_size.
Now raises ValueError with clear message.

---

docs: update README with configuration modes

Added section explaining the 3 configuration approaches:
defaults, programmatic dict, and YAML file.
```

## 🔄 Pull Request Process

1. **Fork the repository** and create a feature branch
   ```bash
   git checkout -b feat/your-feature-name
   ```

2. **Make your changes** following the style guidelines

3. **Add tests** for new functionality

4. **Update documentation** (README, docstrings, examples)

5. **Run tests and linting**
   ```bash
   pytest
   black hybridts/ tests/
   flake8 hybridts/
   ```

6. **Commit with conventional commit messages**

7. **Push and create a Pull Request**
   ```bash
   git push origin feat/your-feature-name
   ```

8. **Fill in the PR template** with:
   - Description of changes
   - Motivation and context
   - How to test
   - Screenshots (if UI/output changes)
   - Checklist of completed items

### PR Checklist

Before submitting, ensure:

- [ ] Code follows style guidelines (black, flake8)
- [ ] All tests pass (`pytest`)
- [ ] New code has tests (aim for >80% coverage)
- [ ] Docstrings are complete and accurate
- [ ] CHANGELOG.md is updated (under "Unreleased")
- [ ] Examples still work (if API changed)
- [ ] README is updated (if user-facing change)

## 🐛 Bug Reports

Use GitHub Issues with the bug template. Include:

- **Environment:** Python version, OS, package versions
- **Description:** What happened vs. what you expected
- **Reproduction:** Minimal code example that reproduces the issue
- **Stack trace:** Full error message and traceback

## 💡 Feature Requests

Use GitHub Issues with the feature template. Include:

- **Use case:** Why this feature is needed
- **Proposed solution:** How you envision it working
- **Alternatives:** Other approaches you considered
- **Examples:** Code snippets showing desired API

## 🏗️ Project Structure

```
hybridts/
├── hybridts/              # Main package
│   ├── __init__.py        # Public API exports
│   ├── config/            # Configuration loading
│   ├── models/            # Model implementations (Prophet, XGBoost, LightGBM)
│   ├── features/          # Feature engineering and data processing
│   └── pipeline/          # Pipeline orchestration
├── tests/                 # Test suite
├── examples/              # Usage examples and research notebooks
├── docs/                  # Documentation (Sphinx, coming soon)
├── pyproject.toml         # Project metadata and dependencies
├── setup.py               # Setup script
├── README.md              # Main documentation
├── CHANGELOG.md           # Version history
└── CONTRIBUTING.md        # This file
```

## 📚 Adding New Models

To add a new residual model (e.g., CatBoost):

1. Create `hybridts/models/catboost.py`:
   ```python
   class CatBoostModel:
       """CatBoost-based residual forecaster."""
       
       def __init__(self, config: dict):
           # Load config
           pass
       
       def train_cv(self, X_train, residuals_train):
           # Training with cross-validation
           pass
       
       def train_static(self, X_train, residuals_train):
           # Training with fixed params
           pass
   ```

2. Add tests in `tests/test_catboost.py`

3. Update `pipeline.py` to support the new model

4. Add example in `examples/research/`

5. Update README and CHANGELOG

## 🎓 Resources

- [Project README](README.md)
- [Prophet Documentation](https://facebook.github.io/prophet/)
- [XGBoost Documentation](https://xgboost.readthedocs.io/)
- [sktime Documentation](https://www.sktime.net/)
- [Conventional Commits](https://www.conventionalcommits.org/)

## 📬 Contact

- **Issues:** [GitHub Issues](https://github.com/YOUR_USERNAME/hybridts/issues)
- **Email:** alcanfordavi@gmail.com

## 📄 License

By contributing, you agree that your contributions will be licensed under the MIT License.

---

Thank you for contributing to HybridTS! 🎉
