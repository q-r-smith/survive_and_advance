# models/calibrated_predictor.py
"""
Wraps GamePredictor with isotonic calibration (sklearn CalibratedClassifierCV).

Calibration is fit post-training on a held-out chronological slice (last 15% of
training rows, sorted by season), so recent-season games inform the calibration curve.
"""
import os
import numpy as np
import pandas as pd
from sklearn.calibration import CalibratedClassifierCV
import joblib

from models.game_predictor_model import GamePredictor

MODEL_PATH = os.path.join(os.path.dirname(__file__), "calibrated_predictor.pkl")


class CalibratedPredictor:
    def __init__(self, base_predictor: GamePredictor):
        self.base_predictor = base_predictor
        self._calibrated    = None
        self.feature_cols   = None

    def fit(self, X: pd.DataFrame, y: pd.Series):
        # Sort chronologically so the calibration holdout contains the most recent games
        X_sorted = X.sort_values("season")
        y_sorted = y.loc[X_sorted.index].reset_index(drop=True)
        X_sorted = X_sorted.reset_index(drop=True)

        cutoff = int(len(X_sorted) * 0.85)
        X_fit, y_fit = X_sorted.iloc[:cutoff], y_sorted.iloc[:cutoff]
        X_cal, y_cal = X_sorted.iloc[cutoff:], y_sorted.iloc[cutoff:]

        # Train base predictor on the first 85%
        self.base_predictor.fit(X_fit, y_fit)
        self.feature_cols = self.base_predictor.feature_cols

        # Fit isotonic calibration on the last 15% using the already-fitted model
        self._calibrated = CalibratedClassifierCV(
            self.base_predictor.model,
            method="isotonic",
            cv="prefit",
        )
        self._calibrated.fit(X_cal[self.feature_cols], y_cal)
        return self

    def predict_proba(self, X: pd.DataFrame) -> np.ndarray:
        return self._calibrated.predict_proba(X[self.feature_cols])[:, 1]

    def predict(self, X: pd.DataFrame, threshold: float = 0.5) -> np.ndarray:
        return (self.predict_proba(X) >= threshold).astype(int)

    def save(self, path: str = MODEL_PATH):
        joblib.dump(self, path)
        print(f"  saved → {path}")

    @classmethod
    def load(cls, path: str = MODEL_PATH) -> "CalibratedPredictor":
        return joblib.load(path)
