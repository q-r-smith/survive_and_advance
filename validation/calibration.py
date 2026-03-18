# validation/calibration.py
"""
Hyperparameter search for GamePredictor (HistGB) and optionally LogisticPredictor.

Scores on neg_brier_score (5-fold stratified CV on training data).
Writes the best param set to models/best_params.json, which train.py
picks up automatically. Top-5 param sets (with val-set scores) are written
to models/top5_params.json for downstream comparison.

Usage:
    python -m validation.calibration                       # default 40 iterations (HistGB)
    python -m validation.calibration --n-iter 80           # more thorough search
    python -m validation.calibration --weighted-cv         # weight recent seasons 2×
    python -m validation.calibration --n-iter 60 --weighted-cv
    python -m validation.calibration --logistic            # grid search logistic params
    python -m validation.calibration --logistic --histgb   # run both searches
"""

import os
import sys
import json
import numpy as np
import pandas as pd
from scipy.stats import uniform, randint
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import brier_score_loss, accuracy_score
from sklearn.model_selection import RandomizedSearchCV, GridSearchCV, StratifiedKFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

CACHE_DIR            = os.path.join(os.path.dirname(__file__), "..", "data", "cache")
PARAMS_PATH          = os.path.join(os.path.dirname(__file__), "..", "models", "best_params.json")
LOGISTIC_PARAMS_PATH = os.path.join(os.path.dirname(__file__), "..", "models", "best_logistic_params.json")

NON_FEATURE_COLS = ["season", "season_type", "game_type", "conf_game"]

N_ITER       = 40
CV_FOLDS     = 5
RANDOM_STATE = 42

# HistGB search space
PARAM_DIST = {
    "learning_rate":     uniform(0.01, 0.19),   # 0.01 – 0.20
    "max_iter":          randint(200, 1001),     # 200 – 1000
    "max_depth":         randint(3, 8),          # 3 – 7
    "min_samples_leaf":  randint(10, 61),        # 10 – 60
    "l2_regularization": uniform(0.0, 0.5),
    "max_bins":          [127, 255],
}

# Logistic grid — explicit param dicts handle solver constraints per penalty
def _logistic_param_grid():
    """
    Returns a list of param dicts for GridSearchCV.
    Solver is auto-selected per penalty to avoid sklearn validation errors:
      l2           → lbfgs
      l1           → liblinear
      elasticnet   → saga  (requires l1_ratio)
    """
    grid = []
    for C in [0.001, 0.01, 0.1, 1.0, 10.0, 100.0]:
        grid.append({"clf__C": [C], "clf__penalty": ["l2"],  "clf__solver": ["lbfgs"]})
        grid.append({"clf__C": [C], "clf__penalty": ["l1"],  "clf__solver": ["liblinear"]})
        for l1r in [0.25, 0.5, 0.75]:
            grid.append({
                "clf__C": [C], "clf__penalty": ["elasticnet"],
                "clf__solver": ["saga"], "clf__l1_ratio": [l1r],
            })
    return grid


def load_train():
    X_raw = pd.read_csv(os.path.join(CACHE_DIR, "X_train.csv"))
    y     = pd.read_csv(os.path.join(CACHE_DIR, "y_train.csv"))["label"]
    feat_cols = [
        c for c in X_raw.columns
        if c not in NON_FEATURE_COLS
        and X_raw[c].dtype != object
    ]
    # Return feature matrix, labels, and raw season array (needed for weighted CV)
    return X_raw[feat_cols], y, X_raw["season"].values


def load_val(feat_cols):
    X = pd.read_csv(os.path.join(CACHE_DIR, "X_val.csv"))
    y = pd.read_csv(os.path.join(CACHE_DIR, "y_val.csv"))["label"]
    valid_cols = [c for c in feat_cols if c in X.columns]
    return X[valid_cols], y


def eval_on_val(params, X_train, y_train, X_val, y_val):
    """Train a fresh HistGB model with given params on full train set, score on val."""
    model = HistGradientBoostingClassifier(**params, early_stopping=False)
    model.fit(X_train, y_train)
    proba = model.predict_proba(X_val)[:, 1]
    return {
        "val_brier":    round(float(brier_score_loss(y_val, proba)), 6),
        "val_accuracy": round(float(accuracy_score(y_val, proba >= 0.5)), 6),
    }


def run_histgb_search(X, y, seasons, X_val, y_val, n_iter, weighted_cv):
    """RandomizedSearchCV over HistGB hyperparameters. Saves best_params.json."""
    base = HistGradientBoostingClassifier(
        loss="log_loss",
        early_stopping=False,
        random_state=RANDOM_STATE,
    )
    cv = StratifiedKFold(n_splits=CV_FOLDS, shuffle=True, random_state=RANDOM_STATE)

    fit_kwargs = {}
    if weighted_cv:
        min_s, max_s   = int(seasons.min()), int(seasons.max())
        sample_weights = (1.0 + (seasons - min_s) / max(max_s - min_s, 1)).astype(float)
        fit_kwargs["sample_weight"] = sample_weights
        print(f"  Weighted CV enabled: seasons {min_s}–{max_s} weighted 1.0–2.0 in training")

    search = RandomizedSearchCV(
        base, PARAM_DIST,
        n_iter=n_iter,
        scoring="neg_brier_score",
        cv=cv, n_jobs=-1, verbose=2,
        random_state=RANDOM_STATE, refit=False,
    )
    print(f"\nRunning RandomizedSearchCV ({n_iter} iterations, {CV_FOLDS}-fold CV)...")
    search.fit(X, y, **fit_kwargs)

    results = pd.DataFrame(search.cv_results_)
    results["brier"]     = -results["mean_test_score"]
    results["brier_std"] = results["std_test_score"]
    results_sorted = results.sort_values("brier").reset_index(drop=True)

    FIXED = {"loss": "log_loss", "random_state": RANDOM_STATE}
    top5_records = []
    print("\nEvaluating top-5 HistGB configs on val set...")
    for rank, (_, row) in enumerate(results_sorted.head(5).iterrows(), start=1):
        params     = {**{k: row[f"param_{k}"] for k in PARAM_DIST}, **FIXED}
        val_scores = eval_on_val(params, X, y, X_val, y_val)
        entry = {
            "rank":         rank,
            "cv_brier":     round(row["brier"], 6),
            "cv_brier_std": round(row["brier_std"], 6),
            **val_scores,
            "params":       params,
        }
        print(
            f"  rank {rank}: cv_brier={entry['cv_brier']:.4f}  "
            f"val_brier={entry['val_brier']:.4f}  val_acc={entry['val_accuracy']:.4f}"
        )
        top5_records.append(entry)

    os.makedirs(os.path.dirname(PARAMS_PATH), exist_ok=True)

    top5_path = PARAMS_PATH.replace("best_params.json", "top5_params.json")
    with open(top5_path, "w") as f:
        json.dump(top5_records, f, indent=2)
    print(f"\nTop-5 HistGB param sets saved → {top5_path}")

    best = top5_records[0]["params"]
    with open(PARAMS_PATH, "w") as f:
        json.dump(best, f, indent=2)
    print(f"Best HistGB params saved      → {PARAMS_PATH}")
    print(f"\nBest CV Brier : {top5_records[0]['cv_brier']:.4f} ± {top5_records[0]['cv_brier_std']:.4f}")
    print(f"Best params:\n{json.dumps(best, indent=2)}")

    display_cols = ["brier", "brier_std"] + [f"param_{k}" for k in PARAM_DIST]
    print(f"\nTop 10 configurations:\n{results_sorted.head(10)[display_cols].to_string()}")


def run_logistic_search(X, y, X_val, y_val):
    """
    GridSearchCV over logistic regression hyperparameters.
    Uses SimpleImputer + StandardScaler pipeline so NaN features are handled.
    Saves best params to models/best_logistic_params.json.
    """
    # Belt-and-suspenders: ensure no object-dtype columns leaked through
    numeric_cols = [c for c in X.columns if X[c].dtype != object]
    X     = X[numeric_cols]
    X_val = X_val[[c for c in numeric_cols if c in X_val.columns]]

    pipe = Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler",  StandardScaler()),
        ("clf",     LogisticRegression(max_iter=1000, random_state=RANDOM_STATE)),
    ])

    param_grid = _logistic_param_grid()
    cv = StratifiedKFold(n_splits=CV_FOLDS, shuffle=True, random_state=RANDOM_STATE)

    print(f"\nRunning GridSearchCV for LogisticRegression ({len(param_grid)} param combos, {CV_FOLDS}-fold CV)...")

    search = GridSearchCV(
        pipe, param_grid,
        scoring="neg_brier_score",
        cv=cv, n_jobs=-1, verbose=1,
        refit=False,
    )
    search.fit(X, y)

    results = pd.DataFrame(search.cv_results_)
    results["brier"] = -results["mean_test_score"]
    results_sorted = results.sort_values("brier").reset_index(drop=True)

    print("\nTop-5 logistic configs:")
    for rank, (_, row) in enumerate(results_sorted.head(5).iterrows(), start=1):
        # Extract the clf__ prefixed params and strip prefix
        raw = {k: v for k, v in row.items() if k.startswith("param_clf__")}
        p = {k.replace("param_clf__", ""): v for k, v in raw.items() if not (isinstance(v, float) and np.isnan(v))}
        # Evaluate on val set
        imputer  = SimpleImputer(strategy="median")
        scaler   = StandardScaler()
        clf_kw   = {k: v for k, v in p.items()}
        clf      = LogisticRegression(max_iter=1000, random_state=RANDOM_STATE, **clf_kw)
        X_imp    = imputer.fit_transform(X)
        X_scaled = scaler.fit_transform(X_imp)
        X_val_imp    = imputer.transform(X_val.values if hasattr(X_val, "values") else X_val)
        X_val_scaled = scaler.transform(X_val_imp)
        clf.fit(X_scaled, y)
        proba     = clf.predict_proba(X_val_scaled)[:, 1]
        val_brier = brier_score_loss(y_val, proba)
        val_acc   = accuracy_score(y_val, proba >= 0.5)
        print(
            f"  rank {rank}: cv_brier={row['brier']:.4f}  "
            f"val_brier={val_brier:.4f}  val_acc={val_acc:.4f}  params={p}"
        )
        if rank == 1:
            best_logistic = p

    os.makedirs(os.path.dirname(LOGISTIC_PARAMS_PATH), exist_ok=True)
    with open(LOGISTIC_PARAMS_PATH, "w") as f:
        json.dump(best_logistic, f, indent=2)
    print(f"\nBest logistic params saved → {LOGISTIC_PARAMS_PATH}")
    print(f"Best params:\n{json.dumps(best_logistic, indent=2)}")


def main():
    n_iter      = N_ITER
    weighted_cv = "--weighted-cv" in sys.argv
    run_logistic = "--logistic" in sys.argv
    run_histgb   = "--histgb" in sys.argv or not run_logistic  # default to histgb unless --logistic only
    if "--n-iter" in sys.argv:
        n_iter = int(sys.argv[sys.argv.index("--n-iter") + 1])

    print("Loading data...")
    X, y, seasons = load_train()
    X_val, y_val  = load_val(X.columns.tolist())
    print(f"  train: {X.shape[0]:,} rows × {X.shape[1]} features   val: {X_val.shape[0]:,} rows")

    if run_histgb:
        print("\n" + "=" * 50)
        print("HistGB hyperparameter search")
        print("=" * 50)
        run_histgb_search(X, y, seasons, X_val, y_val, n_iter, weighted_cv)

    if run_logistic:
        print("\n" + "=" * 50)
        print("Logistic regression hyperparameter search")
        print("=" * 50)
        run_logistic_search(X, y, X_val, y_val)


if __name__ == "__main__":
    main()
