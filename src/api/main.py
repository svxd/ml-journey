import pandas as pd
from fastapi import FastAPI
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

model = load_model()


@app.get("/health")
def health() -> dict:
    return {
        "status": "ok",
        "model_loaded": model is not None,
    }


def build_input_dataframe(payload: DiabetesFeatures) -> pd.DataFrame:
    row = pd.DataFrame([payload.model_dump()])
    return row[FEATURE_COLUMNS]


@app.post("/predict", response_model=PredictionResponse)
def predict(payload: DiabetesFeatures) -> PredictionResponse:
    row = build_input_dataframe(payload)
    prediction = predict_one(model, row)

    return PredictionResponse(prediction=prediction)
