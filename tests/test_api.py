from fastapi.testclient import TestClient

from src.ml.config import MODEL_PATH
from src.ml.train_diabetes_rf import main as train_model

if not MODEL_PATH.exists():
    train_model()


from src.api.main import app  # noqa: E402

client = TestClient(app)


def test_health_returns_ok() -> None:
    response = client.get("/health")

    assert response.status_code == 200

    data = response.json()

    assert data["status"] == "ok"
    assert data["model_loaded"] is True


def test_predict_returns_prediction() -> None:
    payload = {
        "age": 0.0,
        "sex": 0.0,
        "bmi": 0.0,
        "bp": 0.0,
        "s1": 0.0,
        "s2": 0.0,
        "s3": 0.0,
        "s4": 0.0,
        "s5": 0.0,
        "s6": 0.0,
    }

    response = client.post("/predict", json=payload)

    assert response.status_code == 200

    data = response.json()

    assert "prediction" in data
    assert isinstance(data["prediction"], float)


def test_predict_missing_field_returns_422() -> None:
    payload = {
        "age": 0.0,
        "sex": 0.0,
        "bmi": 0.0,
        "bp": 0.0,
        "s1": 0.0,
        "s2": 0.0,
        "s3": 0.0,
        "s4": 0.0,
        "s5": 0.0,
    }

    response = client.post("/predict", json=payload)

    assert response.status_code == 422
