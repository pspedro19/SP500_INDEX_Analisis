from __future__ import annotations

import numpy as np
from sklearn.model_selection import TimeSeriesSplit
from sklearn.metrics import mean_squared_error
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping


def _create_sequences(X, y, seq_len):
    X_vals = X.values
    y_vals = y.values
    seq_x, seq_y = [], []
    for i in range(len(X_vals) - seq_len):
        seq_x.append(X_vals[i:i+seq_len])
        seq_y.append(y_vals[i+seq_len])
    return np.array(seq_x), np.array(seq_y)


def objective(trial, X, y, cv_splits: int, gap: int, random_seed: int, base_params=None) -> float:
    units = trial.suggest_int("units", 32, 128)
    learning_rate = trial.suggest_float("learning_rate", 1e-4, 1e-2, log=True)
    sequence_length = trial.suggest_int("sequence_length", 5, 20)

    if base_params:
        units = trial.suggest_int("units", max(base_params.get("units", units) - 16, 16), min(base_params.get("units", units) + 16, 256))
        learning_rate = trial.suggest_float("learning_rate", max(base_params.get("learning_rate", learning_rate) * 0.5, 1e-4), min(base_params.get("learning_rate", learning_rate) * 2, 1e-2), log=True)
        sequence_length = trial.suggest_int("sequence_length", max(base_params.get("sequence_length", sequence_length) - 5, 5), min(base_params.get("sequence_length", sequence_length) + 5, 25))

    tscv = TimeSeriesSplit(n_splits=cv_splits, gap=gap)
    rmses = []
    for train_idx, val_idx in tscv.split(X):
        X_train, y_train = X.iloc[train_idx], y.iloc[train_idx]
        X_val, y_val = X.iloc[val_idx], y.iloc[val_idx]
        X_train_seq, y_train_seq = _create_sequences(X_train, y_train, sequence_length)
        X_val_seq, y_val_seq = _create_sequences(X_val, y_val, sequence_length)
        if len(X_train_seq) == 0 or len(X_val_seq) == 0:
            continue
        model = Sequential([
            LSTM(units, input_shape=(sequence_length, X.shape[1])),
            Dense(1)
        ])
        model.compile(optimizer=Adam(learning_rate), loss='mse')
        es = EarlyStopping(patience=5, restore_best_weights=True)
        model.fit(X_train_seq, y_train_seq, epochs=20, batch_size=32, verbose=0, validation_split=0.1, callbacks=[es])
        preds = model.predict(X_val_seq, verbose=0).flatten()
        rmses.append(mean_squared_error(y_val_seq, preds, squared=False))
    return float(np.mean(rmses)) if rmses else float('inf')
