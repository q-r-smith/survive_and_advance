# models/logistic_predictor.py
"""
Logistic regression predictor with StandardScaler + median imputation.

Motivation (Klemm benchmark): logistic regression trained on tournament-only
data with ~6 engineered features hit top 99.5% on ESPN Tournament Challenge.
A linear model avoids overfitting to regular-season noise and generalises
better to tournament dynamics (pace, pressure, single-elimination).

StandardScaler is mandatory — efficiency_ratio, ELO, and SRS live on very
different scales. NaN imputation with training-set medians allows this model
to handle features that are only defined for seeded / conference-tracked teams
(e.g. upset_propensity, non_conf_sos).
"""
import os
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
import joblib

from models.game_predictor_model import NON_FEATURE_COLS

MODEL_PATH = os.path.join(os.path.dirname(__file__), "logistic_predictor.pkl")

DEFAULT_PARAMS = {
    "C":        1.0,
    "penalty":  "l2",
    "solver":   "lbfgs",
    "max_iter": 1000,
}

# Solver required per penalty type
_PENALTY_SOLVER = {
    "l1":         "liblinear",
    "l2":         "lbfgs",
    "elasticnet": "saga",
    "none":       "lbfgs",
}


class LogisticPredictor:
    def __init__(self, C=1.0, penalty="l2", solver=None, max_iter=1000, l1_ratio=None):
        # Auto-select solver if not specified
        if solver is None:
            solver = _PENALTY_SOLVER.get(penalty, "lbfgs")

        clf_kwargs = dict(C=C, penalty=penalty, solver=solver, max_iter=max_iter, random_state=42)
        if penalty == "elasticnet":
            clf_kwargs["l1_ratio"] = l1_ratio if l1_ratio is not None else 0.5

        self.pipeline = Pipeline([
            ("scaler", StandardScaler()),
            ("clf", LogisticRegression(**clf_kwargs)),
        ])
        self.feature_cols = None
        self.medians_: pd.Series = None

    def fit(self, X: pd.DataFrame, y: pd.Series, sample_weight=None):
        self.feature_cols = [
            c for c in X.columns
            if c not in NON_FEATURE_COLS and X[c].dtype != object
        ]
        Xf = X[self.feature_cols]
        # Store training-set medians for predict-time imputation
        self.medians_ = Xf.median()
        X_clean = Xf.fillna(self.medians_)
        if sample_weight is not None:
            self.pipeline.fit(X_clean, y, clf__sample_weight=sample_weight)
        else:
            self.pipeline.fit(X_clean, y)
        return self

    def predict_proba(self, X: pd.DataFrame) -> np.ndarray:
        X_clean = X[self.feature_cols].fillna(self.medians_)
        return self.pipeline.predict_proba(X_clean)[:, 1]

    def predict(self, X: pd.DataFrame, threshold: float = 0.5) -> np.ndarray:
        return (self.predict_proba(X) >= threshold).astype(int)

    def feature_importances(self) -> pd.Series:
        """Absolute scaled coefficients as a proxy for feature importance."""
        coefs = np.abs(self.pipeline.named_steps["clf"].coef_[0])
        return pd.Series(coefs, index=self.feature_cols).sort_values(ascending=False)

    def save(self, path: str = MODEL_PATH):
        joblib.dump(self, path)
        print(f"  saved → {path}")

    @classmethod
    def load(cls, path: str = MODEL_PATH) -> "LogisticPredictor":
        obj = joblib.load(path)
        # Patch models saved with older sklearn that stored multi_class attribute
        clf = obj.pipeline.named_steps.get("clf")
        if clf is not None and not hasattr(clf, "multi_class"):
            clf.multi_class = "auto"
        return obj
