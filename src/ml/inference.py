import joblib
import pandas as pd

from src.ml.config import MODEL_PATH


def load_model():
    """Load trained model pipeline."""
    if not MODEL_PATH.exists():
        raise FileNotFoundError(f"Model file not found: {MODEL_PATH}. Run train script first.")

    return joblib.load(MODEL_PATH)


def predict_one(model, row: pd.DataFrame) -> float:
    """Predict target value for one row."""
    prediction = model.predict(row)
    return float(prediction[0])
