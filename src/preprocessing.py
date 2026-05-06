from __future__ import annotations

import pandas as pd

from src.config import FREQ


def preprocess_sales_data(df: pd.DataFrame) -> pd.DataFrame:
    """
    Prepare clean weekly state-level time series:
    - date conversion and sorting
    - weekly resampling
    - missing value filling
    """
    working = df.copy()
    working["date"] = pd.to_datetime(working["date"], errors="coerce")
    working = working.dropna(subset=["date", "state"]).sort_values(["state", "date"])

    grouped_frames: list[pd.DataFrame] = []
    for state, state_df in working.groupby("state"):
        state_df = state_df[["date", "sales"]].set_index("date").sort_index()
        weekly = state_df.resample(FREQ).sum(min_count=1)
        weekly["state"] = state

        # Forward fill first to carry last known values, then interpolate smooth gaps.
        weekly["sales"] = weekly["sales"].ffill().interpolate(method="linear")
        weekly["sales"] = weekly["sales"].bfill().fillna(0.0)
        grouped_frames.append(weekly.reset_index())

    if not grouped_frames:
        raise ValueError("No valid state-level time series found after preprocessing.")

    return pd.concat(grouped_frames, ignore_index=True)
