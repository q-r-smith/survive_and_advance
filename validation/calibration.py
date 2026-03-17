# validation/calibration.py
"""
Hyperparameter search for GamePredictor using RandomizedSearchCV.

Scores on neg_brier_score (5-fold stratified CV on training data).
Writes the best param set to models/best_params.json, which train.py
picks up automatically.

Usage:
    python -m validation.calibration              # default 40 iterations
    python -m validation.calibration --n-iter 80  # more thorough search
"""

import os
import sys
import json
import pandas as pd
from scipy.stats import uniform, randint
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.metrics import brier_score_loss, accuracy_score
from sklearn.model_selection import RandomizedSearchCV, StratifiedKFold

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

CACHE_DIR   = os.path.join(os.path.dirname(__file__), "..", "data", "cache")
PARAMS_PATH = os.path.join(os.path.dirname(__file__), "..", "models", "best_params.json")

NON_FEATURE_COLS = ["season", "season_type"]

N_ITER       = 40
CV_FOLDS     = 5
RANDOM_STATE = 42

# Search space
PARAM_DIST = {
    "learning_rate":     uniform(0.01, 0.19),   # 0.01 – 0.20
    "max_iter":          randint(200, 1001),     # 200 – 1000
    "max_depth":         randint(3, 8),          # 3 – 7
    "min_samples_leaf":  randint(10, 61),        # 10 – 60
    "l2_regularization": uniform(0.0, 0.5),
    "max_bins":          [127, 255],
}


def load_train():
    X = pd.read_csv(os.path.join(CACHE_DIR, "X_train.csv"))
    y = pd.read_csv(os.path.join(CACHE_DIR, "y_train.csv"))["label"]
    feat_cols = [c for c in X.columns if c not in NON_FEATURE_COLS]
    return X[feat_cols], y


def load_val(feat_cols):
    X = pd.read_csv(os.path.join(CACHE_DIR, "X_val.csv"))
    y = pd.read_csv(os.path.join(CACHE_DIR, "y_val.csv"))["label"]
    return X[feat_cols], y


def eval_on_val(params, X_train, y_train, X_val, y_val):
    """Train a fresh model with given params on full train set, score on val."""
    model = HistGradientBoostingClassifier(**params, early_stopping=False)
    model.fit(X_train, y_train)
    proba = model.predict_proba(X_val)[:, 1]
    return {
        "val_brier":    round(float(brier_score_loss(y_val, proba)), 6),
        "val_accuracy": round(float(accuracy_score(y_val, proba >= 0.5)), 6),
    }


def main():
    n_iter = N_ITER
    if "--n-iter" in sys.argv:
        n_iter = int(sys.argv[sys.argv.index("--n-iter") + 1])

    print("Loading data...")
    X, y = load_train()
    X_val, y_val = load_val(X.columns.tolist())
    print(f"  train: {X.shape[0]:,} rows × {X.shape[1]} features   val: {X_val.shape[0]:,} rows")

    # Base estimator — disable early stopping so max_iter is the actual budget
    base = HistGradientBoostingClassifier(
        loss="log_loss",
        early_stopping=False,
        random_state=RANDOM_STATE,
    )

    cv = StratifiedKFold(n_splits=CV_FOLDS, shuffle=True, random_state=RANDOM_STATE)

    search = RandomizedSearchCV(
        base,
        PARAM_DIST,
        n_iter=n_iter,
        scoring="neg_brier_score",   # sklearn built-in — avoids make_scorer NaN issues
        cv=cv,
        n_jobs=-1,
        verbose=2,
        random_state=RANDOM_STATE,
        refit=False,
    )

    print(f"\nRunning RandomizedSearchCV ({n_iter} iterations, {CV_FOLDS}-fold CV)...")
    search.fit(X, y)

    results = pd.DataFrame(search.cv_results_)
    results["brier"] = -results["mean_test_score"]
    results["brier_std"] = results["std_test_score"]
    results_sorted = results.sort_values("brier").reset_index(drop=True)

    # ── save top-5 param sets ─────────────────────────────────────────────────
    FIXED = {"loss": "log_loss", "random_state": RANDOM_STATE}
    top5_records = []
    print("\nEvaluating top-5 configs on val set...")
    for rank, (_, row) in enumerate(results_sorted.head(5).iterrows(), start=1):
        params = {**{k: row[f"param_{k}"] for k in PARAM_DIST}, **FIXED}
        val_scores = eval_on_val(params, X, y, X_val, y_val)
        entry = {
            "rank": rank,
            "cv_brier": round(row["brier"], 6),
            "cv_brier_std": round(row["brier_std"], 6),
            **val_scores,
            "params": params,
        }
        print(f"  rank {rank}: cv_brier={entry['cv_brier']:.4f}  val_brier={entry['val_brier']:.4f}  val_acc={entry['val_accuracy']:.4f}")
        top5_records.append(entry)

    os.makedirs(os.path.dirname(PARAMS_PATH), exist_ok=True)

    top5_path = PARAMS_PATH.replace("best_params.json", "top5_params.json")
    with open(top5_path, "w") as f:
        json.dump(top5_records, f, indent=2)
    print(f"\nTop-5 param sets saved → {top5_path}")

    # best_params.json — what train.py loads by default
    best = top5_records[0]["params"]
    with open(PARAMS_PATH, "w") as f:
        json.dump(best, f, indent=2)
    print(f"Best params saved    → {PARAMS_PATH}")

    print(f"\nBest CV Brier score : {top5_records[0]['cv_brier']:.4f} ± {top5_records[0]['cv_brier_std']:.4f}")
    print(f"Best params:\n{json.dumps(best, indent=2)}")

    # ── top-10 display ────────────────────────────────────────────────────────
    display_cols = ["brier", "brier_std"] + [f"param_{k}" for k in PARAM_DIST]
    print(f"\nTop 10 configurations:\n{results_sorted.head(10)[display_cols].to_string()}")


if __name__ == "__main__":
    main()
