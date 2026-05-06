from __future__ import annotations

import itertools
from dataclasses import dataclass

import numpy as np
import pandas as pd
from statsmodels.tsa.arima.model import ARIMA


@dataclass
class ArimaModel:
    """ARIMA model with light grid-search for (p,d,q)."""

    order: tuple[int, int, int] | None = None
    fitted: object | None = None

    def fit(self, series: pd.Series) -> "ArimaModel":
        best_aic = float("inf")
        best_fit = None
        best_order = None
        clean_series = series.astype(float)

        for order in itertools.product([0, 1, 2], [0, 1], [0, 1, 2]):
            try:
                candidate = ARIMA(clean_series, order=order).fit()
                if candidate.aic < best_aic:
                    best_aic = candidate.aic
                    best_fit = candidate
                    best_order = order
            except Exception:
                continue

        if best_fit is None or best_order is None:
            raise RuntimeError("ARIMA tuning failed for this state.")

        self.order = best_order
        self.fitted = best_fit
        return self

    def forecast(self, steps: int) -> np.ndarray:
        if self.fitted is None:
            raise ValueError("ARIMA model is not trained.")
        forecast_values = self.fitted.forecast(steps=steps)
        return np.asarray(forecast_values, dtype=float)
