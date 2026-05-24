import pandas as pd
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel

from src.ml.config import FEATURE_COLUMNS
from src.ml.inference import load_model, predict_one


class DiabetesFeatures(BaseModel):
    age: float
    sex: float
    bmi: float
    bp: float
    s1: float
    s2: float
    s3: float
    s4: float
    s5: float
    s6: float


class PredictionResponse(BaseModel):
    prediction: float


app = FastAPI(title="Diabetes ML API")


try:
    model = load_model()
    model_error = None
except FileNotFoundError as exc:
    model = None
    model_error = str(exc)


@app.get("/health")
def health() -> dict:
    return {
        "status": "ok" if model is not None else "degraded",
        "model_loaded": model is not None,
        "model_error": model_error,
    }


def build_input_dataframe(payload: DiabetesFeatures) -> pd.DataFrame:
    row = pd.DataFrame([payload.model_dump()])
    return row[FEATURE_COLUMNS]


def get_model():
    if model is None:
        raise HTTPException(
            status_code=503,
            detail=model_error or "Model is not loaded.",
        )

    return model


@app.post("/predict", response_model=PredictionResponse)
def predict(payload: DiabetesFeatures) -> PredictionResponse:
    row = build_input_dataframe(payload)
    prediction = predict_one(get_model(), row)

    return PredictionResponse(prediction=prediction)
