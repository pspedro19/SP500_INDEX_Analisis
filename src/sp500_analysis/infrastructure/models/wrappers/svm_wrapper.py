from sklearn.svm import SVR


class SVMWrapper:
    def __init__(self, **params):
        self.model = SVR(**params)

    def fit(self, X, y):
        self.model.fit(X, y)

    def predict(self, X):
        return self.model.predict(X)
