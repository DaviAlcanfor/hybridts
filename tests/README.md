# Tests directory for tpv-forecast

This directory contains automated tests for the library.

## Running Tests

### Install test dependencies:
```bash
pip install pytest pytest-cov
```

### Run all tests:
```bash
pytest
```

### Run with coverage report:
```bash
pytest --cov=tpv_forecast --cov-report=html
```

### Run specific test file:
```bash
pytest tests/test_data_processor.py
```

### Run specific test:
```bash
pytest tests/test_data_processor.py::TestTimeSeriesProcessor::test_train_test_split_valid
```

## Test Structure

- `conftest.py` - Shared fixtures for all tests
- `test_data_processor.py` - Tests for TimeSeriesProcessor class
- (More test files to be added: test_pipeline.py, test_features.py, etc.)

## Coverage Goals

- **v0.1.0**: No tests (current state - example provided)
- **v0.2.0**: >50% coverage (core classes)
- **v1.0.0**: >70% coverage (production-ready)
