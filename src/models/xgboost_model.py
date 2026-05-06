from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
from xgboost import XGBRegressor

from src.feature_engineering import FEATURE_COLUMNS, add_time_features


@dataclass
class XGBoostForecaster:
    """Gradient boosting forecaster with recursive multi-step prediction."""

    model: XGBRegressor | None = None
    history: pd.DataFrame | None = None

    def fit(self, train_df: pd.DataFrame) -> "XGBoostForecaster":
        featured = add_time_features(train_df).dropna()
        if featured.empty:
            raise ValueError("Not enough history to train XGBoost features.")
        X = featured[FEATURE_COLUMNS]
        y = featured["sales"]
        self.model = XGBRegressor(
            n_estimators=300,
            learning_rate=0.05,
            max_depth=6,
            subsample=0.9,
            colsample_bytree=0.9,
            objective="reg:squarederror",
            random_state=42,
        )
        self.model.fit(X, y)
        self.history = train_df[["date", "sales"]].copy().sort_values("date")
        return self

    def forecast(self, steps: int, freq: str = "W") -> np.ndarray:
        if self.model is None or self.history is None:
            raise ValueError("XGBoost model is not trained.")

        history = self.history.copy()
        preds: list[float] = []
        for _ in range(steps):
            next_date = history["date"].max() + pd.tseries.frequencies.to_offset(freq)
            candidate = pd.concat(
                [history, pd.DataFrame([{"date": next_date, "sales": np.nan}])],
                ignore_index=True,
            )
            feat_row = add_time_features(candidate).tail(1)[FEATURE_COLUMNS].fillna(0.0)
            pred = float(self.model.predict(feat_row)[0])
            preds.append(pred)
            history = pd.concat(
                [history, pd.DataFrame([{"date": next_date, "sales": pred}])],
                ignore_index=True,
            )
        return np.array(preds, dtype=float)
