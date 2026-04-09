import pytest
from hybridts.exceptions import (
    ModelTrainingException,
    ModelPredictionException,
    model_error_handler,
)


def test_training_exception_message():
    exc = ModelTrainingException("ProphetModel", ValueError("bad data"))
    assert "ProphetModel" in str(exc)
    assert "bad data" in str(exc)


def test_prediction_exception_message():
    exc = ModelPredictionException("XGBoostTuner", RuntimeError("shape mismatch"))
    assert "XGBoostTuner" in str(exc)
    assert "shape mismatch" in str(exc)


def test_exception_stores_original():
    original = ValueError("root cause")
    exc = ModelTrainingException("ProphetModel", original)
    assert exc.original_exception is original


def test_handler_wraps_as_training_exception():
    class FakeModel:
        @model_error_handler(ModelTrainingException)
        def fit(self):
            raise ValueError("inner error")

    with pytest.raises(ModelTrainingException) as exc_info:
        FakeModel().fit()

    assert "FakeModel" in str(exc_info.value)
    assert "inner error" in str(exc_info.value)


def test_handler_wraps_as_prediction_exception():
    class FakeModel:
        @model_error_handler(ModelPredictionException)
        def predict(self):
            raise KeyError("missing column")

    with pytest.raises(ModelPredictionException):
        FakeModel().predict()


def test_handler_preserves_cause():
    class FakeModel:
        @model_error_handler(ModelTrainingException)
        def fit(self):
            raise RuntimeError("original")

    with pytest.raises(ModelTrainingException) as exc_info:
        FakeModel().fit()

    assert isinstance(exc_info.value.__cause__, RuntimeError)


def test_handler_preserves_function_name():
    class FakeModel:
        @model_error_handler(ModelTrainingException)
        def fit(self):
            pass

    assert FakeModel.fit.__name__ == "fit"


def test_handler_does_not_wrap_on_success():
    class FakeModel:
        @model_error_handler(ModelTrainingException)
        def fit(self):
            return 42

    assert FakeModel().fit() == 42
