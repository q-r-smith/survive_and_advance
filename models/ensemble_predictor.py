# models/ensemble_predictor.py
"""
HistGradientBoosting + LightGBM stacked under a logistic regression meta-learner.

Out-of-fold (OOF) strategy ensures the meta-learner never trains on predictions
the base models made on their own training data, preventing leakage into the stack.
"""
import os
import json
import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold
import joblib

try:
    from lightgbm import LGBMClassifier
    _HAS_LGBM = True
except (ImportError, OSError):
    # OSError is raised on macOS when libomp is missing even if lightgbm is installed.
    # Fix: brew install libomp
    _HAS_LGBM = False

from models.game_predictor_model import NON_FEATURE_COLS, DEFAULT_PARAMS

PARAMS_PATH = os.path.join(os.path.dirname(__file__), "best_params.json")
MODEL_PATH  = os.path.join(os.path.dirname(__file__), "ensemble_predictor.pkl")

_LGBM_DEFAULTS = {
    "n_estimators":  500,
    "learning_rate": 0.05,
    "num_leaves":    31,
    "random_state":  42,
    "verbose":       -1,
}

_CV_FOLDS     = 5
_RANDOM_STATE = 42

# HistGB params that conflict with a fixed-iteration OOF loop
_EARLY_STOP_KEYS = {"early_stopping", "validation_fraction", "n_iter_no_change"}


class EnsemblePredictor:
    def __init__(self, hist_params=None):
        if not _HAS_LGBM:
            raise ImportError(
                "LightGBM failed to load. On macOS this is usually a missing OpenMP dependency.\n"
                "Fix: brew install libomp\n"
                "Then verify: python -c 'import lightgbm'"
            )

        if hist_params is None and os.path.exists(PARAMS_PATH):
            with open(PARAMS_PATH) as f:
                hist_params = json.load(f)

        self._hist_params     = {**DEFAULT_PARAMS, **(hist_params or {})}
        # Disable early stopping for OOF loop so max_iter is the fixed budget
        self._hist_params_oof = {
            k: v for k, v in self._hist_params.items() if k not in _EARLY_STOP_KEYS
        }
        self._hist_params_oof["early_stopping"] = False

        self.feature_cols = None
        self.hist_model   = None
        self.lgbm_model   = None
        self.meta         = LogisticRegression(C=1.0, random_state=_RANDOM_STATE)

    def fit(self, X: pd.DataFrame, y: pd.Series):
        self.feature_cols = [c for c in X.columns if c not in NON_FEATURE_COLS]
        Xf = X[self.feature_cols].values
        yf = y.values

        cv       = StratifiedKFold(n_splits=_CV_FOLDS, shuffle=True, random_state=_RANDOM_STATE)
        oof_hist = np.zeros(len(yf))
        oof_lgbm = np.zeros(len(yf))

        print(f"  Building OOF predictions ({_CV_FOLDS} folds)...")
        for fold, (tr_idx, vl_idx) in enumerate(cv.split(Xf, yf), 1):
            print(f"    fold {fold}/{_CV_FOLDS}")
            X_tr, X_vl = Xf[tr_idx], Xf[vl_idx]
            y_tr = yf[tr_idx]

            hist = HistGradientBoostingClassifier(**self._hist_params_oof)
            hist.fit(X_tr, y_tr)
            oof_hist[vl_idx] = hist.predict_proba(X_vl)[:, 1]

            lgbm = LGBMClassifier(**_LGBM_DEFAULTS)
            lgbm.fit(X_tr, y_tr)
            oof_lgbm[vl_idx] = lgbm.predict_proba(X_vl)[:, 1]

        # Fit meta-learner on OOF probabilities
        meta_X = np.column_stack([oof_hist, oof_lgbm])
        self.meta.fit(meta_X, yf)

        # Retrain base models on the full training set
        print("  Fitting base models on full training set...")
        self.hist_model = HistGradientBoostingClassifier(**self._hist_params)
        self.hist_model.fit(Xf, yf)

        self.lgbm_model = LGBMClassifier(**_LGBM_DEFAULTS)
        self.lgbm_model.fit(Xf, yf)

        return self

    def predict_proba(self, X: pd.DataFrame) -> np.ndarray:
        Xf       = X[self.feature_cols].values
        hist_p   = self.hist_model.predict_proba(Xf)[:, 1]
        lgbm_p   = self.lgbm_model.predict_proba(Xf)[:, 1]
        meta_X   = np.column_stack([hist_p, lgbm_p])
        return self.meta.predict_proba(meta_X)[:, 1]

    def predict(self, X: pd.DataFrame, threshold: float = 0.5) -> np.ndarray:
        return (self.predict_proba(X) >= threshold).astype(int)

    def save(self, path: str = MODEL_PATH):
        joblib.dump(self, path)
        print(f"  saved → {path}")

    @classmethod
    def load(cls, path: str = MODEL_PATH) -> "EnsemblePredictor":
        return joblib.load(path)
