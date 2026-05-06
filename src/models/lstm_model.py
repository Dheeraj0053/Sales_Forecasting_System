from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout


@dataclass
class LSTMForecaster:
    """Univariate LSTM with recursive forecasting."""

    look_back: int = 8
    model: Sequential | None = None
    scaler: MinMaxScaler | None = None
    history_scaled: np.ndarray | None = None

    def _create_sequences(self, values: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        X, y = [], []
        for idx in range(self.look_back, len(values)):
            X.append(values[idx - self.look_back : idx])
            y.append(values[idx])
        return np.array(X), np.array(y)

    def fit(self, train_series: pd.Series) -> "LSTMForecaster":
        values = train_series.to_numpy(dtype=float).reshape(-1, 1)
        if len(values) <= self.look_back:
            raise ValueError("Not enough records to train LSTM.")

        scaler = MinMaxScaler()
        scaled = scaler.fit_transform(values)
        X, y = self._create_sequences(scaled)
        X = X.reshape(X.shape[0], X.shape[1], 1)

        model = Sequential(
            [
                LSTM(32, return_sequences=True, input_shape=(self.look_back, 1)),
                Dropout(0.1),
                LSTM(16),
                Dense(1),
            ]
        )
        model.compile(optimizer="adam", loss="mse")
        model.fit(X, y, epochs=20, batch_size=16, verbose=0)

        self.model = model
        self.scaler = scaler
        self.history_scaled = scaled.flatten()
        return self

    def forecast(self, steps: int) -> np.ndarray:
        if self.model is None or self.scaler is None or self.history_scaled is None:
            raise ValueError("LSTM model is not trained.")

        window = self.history_scaled[-self.look_back :].copy()
        preds_scaled = []
        for _ in range(steps):
            x_input = window.reshape(1, self.look_back, 1)
            pred = float(self.model.predict(x_input, verbose=0)[0][0])
            preds_scaled.append(pred)
            window = np.append(window[1:], pred)

        preds = self.scaler.inverse_transform(np.array(preds_scaled).reshape(-1, 1)).flatten()
        return preds.astype(float)
