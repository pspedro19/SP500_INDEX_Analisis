class XGBoostWrapper:
    def __init__(self, **params):
        try:
            import xgboost as xgb
        except Exception as exc:  # pragma: no cover
            raise ImportError("xgboost not installed") from exc
        self.model = xgb.XGBRegressor(**params)

    def fit(self, X, y):
        self.model.fit(X, y)

    def predict(self, X):
        return self.model.predict(X)
