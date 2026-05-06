from __future__ import annotations

import logging
import os
import pickle
from datetime import datetime, timezone
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

from src.config import (
    FORECAST_HORIZON,
    METRICS_PATH,
    MODEL_REGISTRY_VERSION,
    MODELS_DIR,
    RAW_DATA_PATH,
    REGISTRY_PATH,
    TRAIN_RATIO,
)
from src.data_loader import load_raw_excel
from src.evaluate import calculate_metrics
from src.models.arima_model import ArimaModel
from src.models.lstm_model import LSTMForecaster
from src.models.prophet_model import ProphetForecaster
from src.models.xgboost_model import XGBoostForecaster
from src.preprocessing import preprocess_sales_data
from src.utils import ensure_dir, setup_logging, write_json

LOGGER = logging.getLogger(__name__)


def _state_slug(state: str) -> str:
    return "".join(ch.lower() if ch.isalnum() else "_" for ch in state).strip("_")


def _time_split(state_df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    split_idx = int(len(state_df) * TRAIN_RATIO)
    split_idx = max(split_idx, 12)
    train_df = state_df.iloc[:split_idx].copy()
    valid_df = state_df.iloc[split_idx:].copy()
    return train_df, valid_df


def _evaluate_models(train_df: pd.DataFrame, valid_df: pd.DataFrame) -> dict[str, dict[str, float]]:
    y_valid = valid_df["sales"].to_numpy(dtype=float)
    horizon = len(y_valid)
    results: dict[str, dict[str, float]] = {}

    candidates = {
        "arima": lambda: ArimaModel().fit(train_df["sales"]),
        "prophet": lambda: ProphetForecaster().fit(train_df[["date", "sales"]]),
        "xgboost": lambda: XGBoostForecaster().fit(train_df[["date", "sales"]]),
        "lstm": lambda: LSTMForecaster(look_back=8).fit(train_df["sales"]),
    }
    enabled_models = {
        item.strip().lower()
        for item in os.getenv("ENABLED_MODELS", "arima,prophet,xgboost,lstm").split(",")
        if item.strip()
    }

    for model_name, trainer in candidates.items():
        if model_name not in enabled_models:
            continue
        try:
            model = trainer()
            if model_name == "prophet":
                y_pred = model.forecast(horizon, freq="W")
            else:
                y_pred = model.forecast(horizon)
            metrics = calculate_metrics(y_valid, np.asarray(y_pred, dtype=float))
            results[model_name] = metrics
        except Exception as exc:
            LOGGER.warning("Model %s failed during validation: %s", model_name, exc)
            results[model_name] = {"rmse": float("inf"), "mae": float("inf")}
    return results


def _fit_best_model(model_name: str, full_df: pd.DataFrame):
    if model_name == "arima":
        return ArimaModel().fit(full_df["sales"])
    if model_name == "prophet":
        return ProphetForecaster().fit(full_df[["date", "sales"]])
    if model_name == "xgboost":
        return XGBoostForecaster().fit(full_df[["date", "sales"]])
    if model_name == "lstm":
        return LSTMForecaster(look_back=8).fit(full_df["sales"])
    raise ValueError(f"Unsupported model name: {model_name}")


def _save_model_bundle(model, model_name: str, state: str, models_dir: Path) -> dict[str, str]:
    state_dir = models_dir / _state_slug(state)
    ensure_dir(state_dir)

    if model_name == "lstm":
        keras_path = state_dir / "lstm.keras"
        meta_path = state_dir / "lstm_meta.pkl"
        model.model.save(keras_path)
        payload = {
            "look_back": model.look_back,
            "scaler": model.scaler,
            "history_scaled": model.history_scaled,
        }
        with meta_path.open("wb") as fp:
            pickle.dump(payload, fp)
        return {"keras_path": str(keras_path), "meta_path": str(meta_path)}

    artifact_path = state_dir / f"{model_name}.joblib"
    joblib.dump(model, artifact_path)
    return {"artifact_path": str(artifact_path)}


def _build_model_version(state_df: pd.DataFrame, model_name: str) -> str:
    last_date = state_df["date"].max().strftime("%Y%m%d")
    return f"{MODEL_REGISTRY_VERSION}-{model_name}-{last_date}-{len(state_df)}"


def main() -> None:
    setup_logging()
    ensure_dir(MODELS_DIR)

    raw = load_raw_excel(str(RAW_DATA_PATH))
    preprocessed = preprocess_sales_data(raw)

    all_metrics: dict[str, dict[str, dict[str, float]]] = {}
    registry: dict[str, dict] = {}

    for state, state_df in preprocessed.groupby("state"):
        state_df = state_df.sort_values("date").reset_index(drop=True)
        if len(state_df) < 30:
            LOGGER.warning("Skipping %s: too few weekly observations (%s).", state, len(state_df))
            continue

        train_df, valid_df = _time_split(state_df)
        if valid_df.empty:
            LOGGER.warning("Skipping %s: empty validation split.", state)
            continue

        model_scores = _evaluate_models(train_df, valid_df)
        if not model_scores:
            LOGGER.warning("Skipping %s: no models enabled.", state)
            continue
        best_model_name = min(model_scores, key=lambda m: (model_scores[m]["rmse"], model_scores[m]["mae"]))

        best_model = _fit_best_model(best_model_name, state_df)
        artifact_info = _save_model_bundle(best_model, best_model_name, state, MODELS_DIR)
        model_version = _build_model_version(state_df, best_model_name)

        # Save next 8 week forecast at train time for auditability and quick retrieval.
        if best_model_name == "prophet":
            forecast = best_model.forecast(FORECAST_HORIZON, freq="W").tolist()
        else:
            forecast = best_model.forecast(FORECAST_HORIZON).tolist()

        all_metrics[state] = model_scores
        registry[state] = {
            "best_model": best_model_name,
            "model_version": model_version,
            "forecast_horizon": FORECAST_HORIZON,
            "latest_forecast": [float(v) for v in forecast],
            "trained_at_utc": datetime.now(timezone.utc).isoformat(),
            "train_start_date": state_df["date"].min().isoformat(),
            "train_end_date": state_df["date"].max().isoformat(),
            "num_observations": int(len(state_df)),
            **artifact_info,
        }
        LOGGER.info(
            "state model selected",
            extra={
                "event": "model_selection",
                "state": state,
                "model_name": best_model_name,
                "model_version": model_version,
            },
        )

    write_json(METRICS_PATH, all_metrics)
    write_json(REGISTRY_PATH, registry)
    LOGGER.info("training completed", extra={"event": "training_completed"})


if __name__ == "__main__":
    main()
