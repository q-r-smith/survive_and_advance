# models/game_predictor_model.py
import os
import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier
import joblib

MODEL_PATH = os.path.join(os.path.dirname(__file__), "game_predictor.pkl")

NON_FEATURE_COLS = ["season", "season_type", "game_type", "conf_game"]

DEFAULT_PARAMS = {
    "loss": "log_loss",
    "learning_rate": 0.05,
    "max_iter": 500,
    "max_depth": 4,
    "min_samples_leaf": 20,
    "l2_regularization": 0.1,
    "max_bins": 255,
    "early_stopping": True,
    "validation_fraction": 0.1,
    "n_iter_no_change": 20,
    "random_state": 42,
}


class GamePredictor:
    def __init__(self, params=None):
        p = {**DEFAULT_PARAMS, **(params or {})}
        self.model = HistGradientBoostingClassifier(**p)
        self.feature_cols = None

    def fit(self, X: pd.DataFrame, y: pd.Series, sample_weight=None):
        self.feature_cols = [c for c in X.columns if c not in NON_FEATURE_COLS]
        self.model.fit(X[self.feature_cols], y, sample_weight=sample_weight)
        return self

    def predict_proba(self, X: pd.DataFrame) -> np.ndarray:
        """Returns P(home team wins) for each row."""
        return self.model.predict_proba(X[self.feature_cols])[:, 1]

    def predict(self, X: pd.DataFrame, threshold: float = 0.5) -> np.ndarray:
        return (self.predict_proba(X) >= threshold).astype(int)

    def feature_importances(self) -> pd.Series:
        """
        Returns native gain-based importances if available (sklearn >= 1.2),
        otherwise raises AttributeError — callers should fall back to
        permutation_importance from sklearn.inspection.
        """
        return pd.Series(
            self.model.feature_importances_,
            index=self.feature_cols,
        ).sort_values(ascending=False)

    def save(self, path: str = MODEL_PATH):
        joblib.dump(self, path)
        print(f"  saved → {path}")

    @classmethod
    def load(cls, path: str = MODEL_PATH) -> "GamePredictor":
        return joblib.load(path)
