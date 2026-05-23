from src.ml.data import build_manual_sample
from src.ml.inference import load_model, predict_one


def main() -> None:
    model = load_model()
    sample = build_manual_sample()

    prediction = predict_one(model, sample)

    print("Input sample:")
    print(sample)
    print(f"Prediction: {prediction:.2f}")


if __name__ == "__main__":
    main()
