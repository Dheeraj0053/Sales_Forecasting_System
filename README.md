# Production-Ready Sales Forecasting System

## Problem Statement
Build a robust backend pipeline to forecast the next **8 weeks of sales for each state** from an Excel dataset.  
The system must train multiple model families, compare them consistently, auto-select the best model per state, store model artifacts, and expose forecasts through a REST API.

## Solution Overview
This project implements an end-to-end architecture with:
- **Data ingestion** from Excel with flexible column mapping (`State`, `Date`, `Sales/Total`)
- **Preprocessing** for weekly state-level series, missing dates, and missing values
- **Feature engineering** with lag, rolling, and date features (leakage-safe)
- **Multi-model training**: ARIMA, Prophet, XGBoost, LSTM
- **Per-state model selection** using RMSE + MAE
- **Artifact persistence** (model files + registry + metrics)
- **FastAPI inference** endpoint to retrieve 8-week forecasts by state
- **Structured JSON logging** for training and API observability
- **Model version metadata** per state in the model registry
- **Automated CI checks** for tests, training smoke, and API import

## Project Structure
```text
project/
│
├── data/
│   ├── raw.xlsx
│   └── README.md
│
├── src/
│   ├── __init__.py
│   ├── config.py
│   ├── data_loader.py
│   ├── preprocessing.py
│   ├── feature_engineering.py
│   ├── train.py
│   ├── evaluate.py
│   ├── utils.py
│   │
│   └── models/
│       ├── __init__.py
│       ├── arima_model.py
│       ├── prophet_model.py
│       ├── xgboost_model.py
│       └── lstm_model.py
│
├── api/
│   └── main.py
│
├── notebooks/
│   └── exploration.ipynb
│
├── tests/
│   ├── test_data_loader.py
│   ├── test_preprocessing.py
│   └── test_feature_engineering.py
│
├── scripts/
│   └── generate_sample_data.py
│
├── .github/
│   └── workflows/
│       └── ci.yml
│
├── Dockerfile
├── docker-compose.yml
├── .dockerignore
├── pytest.ini
│
├── artifacts/              # auto-generated after training
│   ├── models/
│   ├── metrics.json
│   └── registry.json
│
├── requirements.txt
└── README.md
```

## Data Requirements
Input file: `data/raw.xlsx`

Required columns:
- State identifier (e.g., `State`)
- Date column (e.g., `Date`)
- Target sales column (e.g., `Sales` or `Total`)

### Preprocessing details
- Convert date column to datetime
- Sort chronologically
- Group by state
- Resample to weekly frequency
- Fill missing values using forward fill + interpolation (+ backfill safety)

## Feature Engineering
Implemented in `src/feature_engineering.py`:
- **Lag features**: `t-1`, `t-7`, `t-30`
- **Rolling features**: mean and std for windows `7`, `14`, `30`
- **Date features**: `day_of_week`, `month`
- **Extra feature**: `is_weekend` flag

Leakage prevention:
- Rolling statistics use `shift(1)` so only past values are used.

## Modeling Approach
Each state is trained/evaluated independently.

Models:
1. **ARIMA** (`statsmodels`)  
   - Grid search across `(p,d,q)` combinations
2. **Prophet** (`prophet`)  
   - Weekly + yearly seasonality
3. **XGBoost** (`xgboost`)  
   - Supervised regression on engineered features
4. **LSTM** (`tensorflow.keras`)  
   - Sequence learning with normalization and recursive forecasting

## Train / Validation Strategy
- Time-series split only (no random shuffle)
- Train: first 80%
- Validation: last 20%

## Evaluation and Model Selection
Metrics:
- RMSE
- MAE

For each state:
- Evaluate all model families on validation horizon
- Select best model by `(RMSE, MAE)` ascending
- Refit best model on full state history
- Save trained artifact(s)
- Save `latest_forecast` for next 8 weeks

Outputs:
- `artifacts/metrics.json` - model metrics per state
- `artifacts/registry.json` - best model and artifact paths per state

Registry also stores:
- `model_version`
- `trained_at_utc`
- `train_start_date`, `train_end_date`
- `num_observations`

## API Endpoints
Run API:
```bash
uvicorn api.main:app --reload
```

### Health Check
`GET /health`

Response:
```json
{"status":"ok"}
```

### Prediction
`GET /predict?state=STATE_NAME`

Behavior:
- Loads best trained model for that state
- Returns 8-week forecast

Sample response:
```json
{
  "state": "California",
  "best_model": "xgboost",
  "model_version": "v1-xgboost-20241229-156",
  "forecast": [123.4, 127.8, 126.0, 130.7, 131.2, 133.0, 134.5, 136.1]
}
```

## How To Run
1. Create virtual environment and install dependencies:
   ```bash
   pip install -r requirements.txt
   ```
2. Place dataset file at:
   - `data/raw.xlsx`
3. Run training:
   ```bash
   python -m src.train
   ```
4. Start API:
   ```bash
   uvicorn api.main:app --reload
   ```
5. Test prediction:
   - `http://127.0.0.1:8000/predict?state=California`

### Run tests
```bash
pytest -q
```

## Docker Deployment
Build and run API with Docker:
```bash
docker compose up --build
```

The container mounts:
- `./data` to `/app/data`
- `./artifacts` to `/app/artifacts`

## CI Workflow
GitHub Actions workflow at `.github/workflows/ci.yml` runs:
- Unit tests (`pytest`)
- Syntax compile checks
- Training smoke test on generated sample data (`ENABLED_MODELS=arima,xgboost`)
- API import smoke test

## Structured Logging and Versioning
- Logging is JSON-formatted for easier production ingestion.
- Training emits model selection events with state/model/version fields.
- API response includes `model_version`.
- You can limit training models in constrained environments:
  ```bash
  ENABLED_MODELS=arima,xgboost python -m src.train
  ```

## Production Notes
- Errors are surfaced with clear messages if registry/model artifacts are missing.
- Model artifacts are separated per state for clean deployment and rollback.
- Architecture is modular for maintainability and testability.
- You can add CI/CD, monitoring, and model drift checks on top of this foundation.

## Screenshots
<img width="1920" height="1080" alt="Screenshot (209)" src="https://github.com/user-attachments/assets/2532de98-96d9-4988-972e-302752ad8b93" />
<img width="1920" height="1080" alt="Screenshot (210)" src="https://github.com/user-attachments/assets/9d8236fd-ff0b-4af8-a4e3-2efc75cae2ad" />
<img width="1920" height="1080" alt="Screenshot (212)" src="https://github.com/user-attachments/assets/c50b7335-b271-472f-b356-f877dc67ac1d" />





