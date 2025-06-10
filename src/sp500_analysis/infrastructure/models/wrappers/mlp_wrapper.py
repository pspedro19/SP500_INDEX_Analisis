from sklearn.neural_network import MLPRegressor

class MLPWrapper:
    def __init__(self, **params):
        self.model = MLPRegressor(**params)

    def fit(self, X, y):
        self.model.fit(X, y)

    def predict(self, X):
        return self.model.predict(X)
