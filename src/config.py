from pathlib import Path


BASE_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = BASE_DIR / "data"
RAW_DATA_PATH = DATA_DIR / "raw.xlsx"

ARTIFACTS_DIR = BASE_DIR / "artifacts"
MODELS_DIR = ARTIFACTS_DIR / "models"
METRICS_PATH = ARTIFACTS_DIR / "metrics.json"
REGISTRY_PATH = ARTIFACTS_DIR / "registry.json"

DATE_COLUMN_CANDIDATES = ["date", "ds", "timestamp", "order_date"]
STATE_COLUMN_CANDIDATES = ["state", "region", "province", "location"]
TARGET_COLUMN_CANDIDATES = ["sales", "total", "revenue", "target", "y"]

FREQ = "W"
FORECAST_HORIZON = 8
TRAIN_RATIO = 0.8
RANDOM_SEED = 42
APP_VERSION = "1.1.0"
MODEL_REGISTRY_VERSION = "v1"
