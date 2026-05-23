import joblib

from src.ml.config import MODEL_DIR, MODEL_PATH
from src.ml.data import load_data, split_data
from src.ml.evaluation import evaluate_model
from src.ml.modeling import build_model


def main() -> None:
    X, y = load_data()
    X_train, X_test, y_train, y_test = split_data(X, y)

    model = build_model()
    model.fit(X_train, y_train)

    metrics = evaluate_model(model, X_test, y_test)

    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    joblib.dump(model, MODEL_PATH)

    print(f"Model saved to: {MODEL_PATH}")
    print(f"MAE : {metrics['MAE']:.4f}")
    print(f"RMSE: {metrics['RMSE']:.4f}")
    print(f"R2  : {metrics['R2']:.4f}")


if __name__ == "__main__":
    main()
