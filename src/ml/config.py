from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]

MODEL_DIR = PROJECT_ROOT / "models"
MODEL_PATH = MODEL_DIR / "diabetes_rf_pipeline.joblib"

TEST_SIZE = 0.2
RANDOM_STATE = 42

RF_N_ESTIMATORS = 100
RF_MAX_DEPTH = 5
RF_MIN_SAMPLES_LEAF = 2
