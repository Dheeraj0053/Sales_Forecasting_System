from __future__ import annotations

import logging
from typing import Iterable

import pandas as pd

from src.config import (
    DATE_COLUMN_CANDIDATES,
    STATE_COLUMN_CANDIDATES,
    TARGET_COLUMN_CANDIDATES,
)

LOGGER = logging.getLogger(__name__)


def _find_column(columns: Iterable[str], candidates: list[str]) -> str:
    normalized = {c.strip().lower(): c for c in columns}
    for candidate in candidates:
        if candidate in normalized:
            return normalized[candidate]
    raise ValueError(f"Could not find any of columns {candidates} in dataset.")


def load_raw_excel(file_path: str) -> pd.DataFrame:
    """Load raw Excel data and map key columns to standard names."""
    df = pd.read_excel(file_path)
    if df.empty:
        raise ValueError("Input Excel file is empty.")

    state_col = _find_column(df.columns, STATE_COLUMN_CANDIDATES)
    date_col = _find_column(df.columns, DATE_COLUMN_CANDIDATES)
    target_col = _find_column(df.columns, TARGET_COLUMN_CANDIDATES)

    renamed = df.rename(
        columns={
            state_col: "state",
            date_col: "date",
            target_col: "sales",
        }
    )

    data = renamed[["state", "date", "sales"]].copy()
    data["state"] = data["state"].astype(str).str.strip()
    data["date"] = pd.to_datetime(data["date"], errors="coerce")
    data["sales"] = (
        data["sales"]
        .astype(str)
        .str.replace(",", "", regex=False)
        .str.replace("(", "-", regex=False)
        .str.replace(")", "", regex=False)
    )
    data["sales"] = pd.to_numeric(data["sales"], errors="coerce")

    data = data.dropna(subset=["state", "date"])
    missing_sales = data["sales"].isna().sum()
    if missing_sales > 0:
        LOGGER.warning("Found %s rows with missing sales values.", missing_sales)

    return data
