try:
    from tensorflow.keras.models import Sequential
    from tensorflow.keras.layers import LSTM, Dense, Dropout
    from tensorflow.keras.optimizers import Adam
except Exception:  # pragma: no cover
    Sequential = None


class LSTMWrapper:
    def __init__(
        self, units: int = 64, dropout_rate: float = 0.2, learning_rate: float = 0.001, sequence_length: int = 10
    ):
        if Sequential is None:
            raise ImportError("TensorFlow not installed")
        self.sequence_length = sequence_length
        self.model = Sequential(
            [
                LSTM(units=units, input_shape=(sequence_length, None), return_sequences=False),
                Dropout(dropout_rate),
                Dense(1),
            ]
        )
        self.model.compile(optimizer=Adam(learning_rate=learning_rate), loss="mse")

    def fit(self, X, y, **kwargs):
        self.model.fit(X, y, **kwargs)

    def predict(self, X):
        return self.model.predict(X)
