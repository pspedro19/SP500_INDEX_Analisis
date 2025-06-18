from sklearn.base import BaseEstimator, RegressorMixin

class CatBoostWrapper(BaseEstimator, RegressorMixin):
    def __init__(self, **params):
        try:
            from catboost import CatBoostRegressor
        except Exception as exc:  # pragma: no cover
            raise ImportError("catboost not installed") from exc
        
        # Set default parameters
        self.params = params
        self.model = None
        self._initialize_model()

    def _initialize_model(self):
        """Initialize the model with current parameters."""
        from catboost import CatBoostRegressor
        # Set default verbose=False for CatBoost to avoid output spam
        default_params = {'verbose': False}
        default_params.update(self.params)
        self.model = CatBoostRegressor(**default_params)

    def fit(self, X, y):
        if self.model is None:
            self._initialize_model()
        self.model.fit(X, y)
        return self

    def predict(self, X):
        if self.model is None:
            raise ValueError("Model must be fitted before making predictions")
        return self.model.predict(X)

    def get_params(self, deep=True):
        """Get parameters for this estimator."""
        return self.params.copy()

    def set_params(self, **params):
        """Set the parameters of this estimator."""
        self.params.update(params)
        self._initialize_model()  # Reinitialize with new params
        return self
