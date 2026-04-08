import functools


class ModelTrainingException(Exception):
    """Custom exception for errors during model training."""
    def __init__(self, model_name: str, original_exception: Exception):
        super().__init__(f"An error occurred while training model '{model_name}': {str(original_exception)}")
        self.original_exception = original_exception


class ModelPredictionException(Exception):
    """Custom exception for errors during model prediction."""
    def __init__(self, model_name: str, original_exception: Exception):
        super().__init__(f"An error occurred while making predictions with model '{model_name}': {str(original_exception)}")
        self.original_exception = original_exception


def model_error_handler(exception_class):
    """Decorator that wraps model methods with standardized exception handling."""
    def decorator(func):
        @functools.wraps(func)
        def wrapper(self, *args, **kwargs):
            try:
                return func(self, *args, **kwargs)
            except Exception as e:
                raise exception_class(type(self).__name__, e) from e
        return wrapper
    return decorator
