from sklearn.neural_network import MLPRegressor
from sklearn.base import BaseEstimator, RegressorMixin


class MLPWrapper(BaseEstimator, RegressorMixin):
    def __init__(self, **params):
        # Set default parameters
        self.params = params
        self.model = None
        self._initialize_model()

    def _initialize_model(self):
        """Initialize the model with current parameters."""
        # Set default max_iter=500 for MLP to ensure convergence
        default_params = {'max_iter': 500, 'random_state': 42}
        default_params.update(self.params)
        self.model = MLPRegressor(**default_params)

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
