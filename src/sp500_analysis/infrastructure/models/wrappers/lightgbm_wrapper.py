class LightGBMWrapper:
    def __init__(self, **params):
        try:
            import lightgbm as lgb
        except Exception as exc:  # pragma: no cover
            raise ImportError("lightgbm not installed") from exc
        self.model = lgb.LGBMRegressor(**params)

    def fit(self, X, y):
        self.model.fit(X, y)

    def predict(self, X):
        return self.model.predict(X)
