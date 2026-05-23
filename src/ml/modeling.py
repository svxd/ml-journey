from sklearn.ensemble import RandomForestRegressor
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from src.ml.config import (
    RANDOM_STATE,
    RF_MAX_DEPTH,
    RF_MIN_SAMPLES_LEAF,
    RF_N_ESTIMATORS,
)


def build_model() -> Pipeline:
    """Build RandomForest regression pipeline."""
    return Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
            (
                "model",
                RandomForestRegressor(
                    n_estimators=RF_N_ESTIMATORS,
                    max_depth=RF_MAX_DEPTH,
                    min_samples_leaf=RF_MIN_SAMPLES_LEAF,
                    random_state=RANDOM_STATE,
                ),
            ),
        ]
    )
