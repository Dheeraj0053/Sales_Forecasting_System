from __future__ import annotations

import numpy as np
import pandas as pd

try:
    from prophet import Prophet
except Exception:  # pragma: no cover
    Prophet = None


class ProphetForecaster:
    """Prophet wrapper. Disabled gracefully if package unavailable."""

    def __init__(self) -> None:
        self.model = None

    def fit(self, df: pd.DataFrame) -> "ProphetForecaster":
        if Prophet is None:
            raise ImportError("Prophet is not installed in this environment.")
        train_df = df.rename(columns={"date": "ds", "sales": "y"})[["ds", "y"]]
        model = Prophet(
            yearly_seasonality=True,
            weekly_seasonality=True,
            daily_seasonality=False,
            seasonality_mode="additive",
        )
        model.fit(train_df)
        self.model = model
        return self

    def forecast(self, steps: int, freq: str = "W") -> np.ndarray:
        if self.model is None:
            raise ValueError("Prophet model is not trained.")
        future = self.model.make_future_dataframe(periods=steps, freq=freq)
        fcst = self.model.predict(future)
        preds = fcst["yhat"].tail(steps).to_numpy(dtype=float)
        return preds
