class CatBoostWrapper:
    def __init__(self, **params):
        try:
            from catboost import CatBoostRegressor
        except Exception as exc:  # pragma: no cover
            raise ImportError("catboost not installed") from exc
        self.model = CatBoostRegressor(**params)

    def fit(self, X, y):
        self.model.fit(X, y)

    def predict(self, X):
        return self.model.predict(X)
