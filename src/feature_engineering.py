from __future__ import annotations

import numpy as np
import pandas as pd


def add_time_features(df: pd.DataFrame) -> pd.DataFrame:
    """Create lag, rolling, and calendar features using only historical values."""
    engineered = df.copy().sort_values("date")

    engineered["lag_1"] = engineered["sales"].shift(1)
    engineered["lag_7"] = engineered["sales"].shift(7)
    engineered["lag_30"] = engineered["sales"].shift(30)

    engineered["roll_mean_7"] = engineered["sales"].shift(1).rolling(window=7).mean()
    engineered["roll_mean_14"] = engineered["sales"].shift(1).rolling(window=14).mean()
    engineered["roll_mean_30"] = engineered["sales"].shift(1).rolling(window=30).mean()
    engineered["roll_std_7"] = engineered["sales"].shift(1).rolling(window=7).std()
    engineered["roll_std_14"] = engineered["sales"].shift(1).rolling(window=14).std()
    engineered["roll_std_30"] = engineered["sales"].shift(1).rolling(window=30).std()

    engineered["day_of_week"] = engineered["date"].dt.dayofweek
    engineered["month"] = engineered["date"].dt.month
    engineered["is_weekend"] = engineered["day_of_week"].isin([5, 6]).astype(int)

    return engineered.replace([np.inf, -np.inf], np.nan)


FEATURE_COLUMNS = [
    "lag_1",
    "lag_7",
    "lag_30",
    "roll_mean_7",
    "roll_mean_14",
    "roll_mean_30",
    "roll_std_7",
    "roll_std_14",
    "roll_std_30",
    "day_of_week",
    "month",
    "is_weekend",
]
