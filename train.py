# train.py
"""
Train GamePredictor on 2015–2024 data, evaluate on 2025 validation set.

Evaluation reported on:
  - Full training set (overfit check)
  - Full val set (all 2025 games)
  - Val postseason only (the tournament games we care about)

Loads tuned hyperparameters from models/best_params.json if present
(produced by validation/calibration.py). Falls back to defaults otherwise.
"""

import os
import sys
import json
import pandas as pd
from sklearn.metrics import brier_score_loss, accuracy_score
from sklearn.inspection import permutation_importance

sys.path.insert(0, os.path.dirname(__file__))
from models.game_predictor_model import GamePredictor

CACHE_DIR   = os.path.join(os.path.dirname(__file__), "data", "cache")
PARAMS_PATH = os.path.join(os.path.dirname(__file__), "models", "best_params.json")


def load_splits():
    X_train = pd.read_csv(os.path.join(CACHE_DIR, "X_train.csv"))
    y_train = pd.read_csv(os.path.join(CACHE_DIR, "y_train.csv"))["label"]
    X_val   = pd.read_csv(os.path.join(CACHE_DIR, "X_val.csv"))
    y_val   = pd.read_csv(os.path.join(CACHE_DIR, "y_val.csv"))["label"]
    return X_train, y_train, X_val, y_val


def evaluate(name: str, predictor: GamePredictor, X: pd.DataFrame, y: pd.Series):
    proba = predictor.predict_proba(X)
    preds = (proba >= 0.5).astype(int)
    brier = brier_score_loss(y, proba)
    acc   = accuracy_score(y, preds)
    print(f"  [{name}]  Brier: {brier:.4f}  |  Accuracy: {acc:.4f}  (n={len(y):,})")
    return brier, acc


def main():
    print("=" * 50)
    print("STEP 1 — Load splits")
    print("=" * 50)
    X_train, y_train, X_val, y_val = load_splits()
    print(f"  X_train : {X_train.shape}   X_val : {X_val.shape}")

    params = None
    if os.path.exists(PARAMS_PATH):
        with open(PARAMS_PATH) as f:
            params = json.load(f)
        print(f"\n  Loaded tuned params from {PARAMS_PATH}")
    else:
        print("\n  No best_params.json found — using defaults")
        print("  Run `python -m validation.calibration` to tune first.")

    print("\n" + "=" * 50)
    print("STEP 2 — Train")
    print("=" * 50)
    predictor = GamePredictor(params=params)
    predictor.fit(X_train, y_train)
    print(f"  Done. Iterations used: {predictor.model.n_iter_}")

    print("\n" + "=" * 50)
    print("STEP 3 — Evaluate")
    print("=" * 50)
    evaluate("Train       ", predictor, X_train, y_train)
    evaluate("Val (all)   ", predictor, X_val,   y_val)

    post_mask = X_val["season_type"] == "postseason"
    if post_mask.sum() > 0:
        evaluate("Val (post)  ", predictor, X_val[post_mask], y_val[post_mask])
    else:
        print("  [Val (post)] no postseason rows found in val set")

    print("\n" + "=" * 50)
    print("STEP 4 — Feature importances")
    print("=" * 50)
    try:
        print(predictor.feature_importances().to_string())
    except AttributeError:
        print("  Native importances unavailable (sklearn < 1.2) — running permutation importance on val set...")
        feat_cols = predictor.feature_cols
        result = permutation_importance(
            predictor.model,
            X_val[feat_cols],
            y_val,
            n_repeats=10,
            scoring="neg_brier_score",
            random_state=42,
            n_jobs=-1,
        )
        imp = pd.Series(result.importances_mean, index=feat_cols).sort_values(ascending=False)
        print(imp.to_string())

    print("\n" + "=" * 50)
    print("STEP 5 — Save model")
    print("=" * 50)
    predictor.save()

    print("\nDone.")
    return predictor


if __name__ == "__main__":
    main()
