from pathlib import Path

import joblib
import pandas as pd
from sklearn.datasets import load_diabetes


PROJECT_ROOT = Path(__file__).resolve().parents[2]
MODEL_PATH = PROJECT_ROOT / "models" / "diabetes_rf_pipeline.joblib"


def load_model():
    """Load trained model pipeline."""
    if not MODEL_PATH.exists():
        raise FileNotFoundError(
            f"Model file not found: {MODEL_PATH}. Run train script first."
        )

    return joblib.load(MODEL_PATH)


def build_manual_sample() -> pd.DataFrame:
    """Build one-row sample using mean feature values."""
    data = load_diabetes(as_frame=True)
    X = data.data

    return pd.DataFrame([X.mean()])


def predict_one(model, row: pd.DataFrame) -> float:
    """Predict target value for one row."""
    prediction = model.predict(row)
    return float(prediction[0])


def main() -> None:
    model = load_model()
    sample = build_manual_sample()

    prediction = predict_one(model, sample)

    print("Input sample:")
    print(sample)
    print(f"Prediction: {prediction:.2f}")


if __name__ == "__main__":
    main()