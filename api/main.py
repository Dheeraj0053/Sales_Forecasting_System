from __future__ import annotations

import pickle
from pathlib import Path

import joblib
from fastapi import FastAPI, HTTPException, Query
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from tensorflow.keras.models import load_model

from src.config import APP_VERSION, FORECAST_HORIZON, REGISTRY_PATH
from src.models.lstm_model import LSTMForecaster
from src.utils import read_json, setup_logging

setup_logging()
app = FastAPI(title="Time Series Forecasting API", version=APP_VERSION)


def _load_registry() -> dict:
    path = Path(REGISTRY_PATH)
    if not path.exists():
        raise RuntimeError("Registry file is missing. Run training first.")
    return read_json(path)


def _load_state_model(state_record: dict):
    model_name = state_record["best_model"]
    if model_name == "lstm":
        keras_path = Path(state_record["keras_path"])
        meta_path = Path(state_record["meta_path"])
        with meta_path.open("rb") as fp:
            meta = pickle.load(fp)
        model = LSTMForecaster(look_back=meta["look_back"])
        model.model = load_model(keras_path)
        model.scaler = meta["scaler"]
        model.history_scaled = meta["history_scaled"]
        return model_name, model

    model = joblib.load(Path(state_record["artifact_path"]))
    return model_name, model


@app.get("/health")
def health() -> dict:
    return {"status": "ok"}


@app.get("/states")
def get_states() -> dict:
    registry = _load_registry()
    return {"states": sorted(list(registry.keys()))}


@app.get("/predict")
def predict(state: str = Query(..., description="State name")) -> dict:
    registry = _load_registry()
    normalized = state.strip()
    if normalized not in registry:
        raise HTTPException(status_code=404, detail=f"State '{state}' not found in registry.")

    state_record = registry[normalized]
    model_name, model = _load_state_model(state_record)
    if model_name == "prophet":
        preds = model.forecast(FORECAST_HORIZON, freq="W")
    else:
        preds = model.forecast(FORECAST_HORIZON)

    return {
        "state": normalized,
        "best_model": model_name,
        "model_version": state_record.get("model_version", "unknown"),
        "forecast": [float(v) for v in preds],
    }

# Mount static directory for the frontend
app.mount("/static", StaticFiles(directory="static"), name="static")

@app.get("/")
def read_index():
    return FileResponse("static/index.html")
