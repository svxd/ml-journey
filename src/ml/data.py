import pandas as pd
from sklearn.datasets import load_diabetes
from sklearn.model_selection import train_test_split

from src.ml.config import RANDOM_STATE, TEST_SIZE


def load_data():
    """Load diabetes dataset."""
    data = load_diabetes(as_frame=True)
    return data.data, data.target


def split_data(X, y):
    """Split data into train and test parts."""
    return train_test_split(
        X,
        y,
        test_size=TEST_SIZE,
        random_state=RANDOM_STATE,
    )


def build_manual_sample() -> pd.DataFrame:
    """Build one-row sample using mean feature values."""
    X, _ = load_data()
    return pd.DataFrame([X.mean()])
