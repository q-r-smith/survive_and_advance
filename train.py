# train.py
"""
Train and evaluate game prediction models.

Default: trains GamePredictor (HistGradientBoostingClassifier) on 2015–2024 data,
         evaluates on 2025 validation set.

Optional flags (combinable):
  --calibrate       Also run CalibratedPredictor (isotonic calibration on recent holdout)
  --ensemble        Also run EnsemblePredictor (HistGB + LightGBM + logistic meta-learner)
  --logistic        Also run LogisticPredictor (StandardScaler + LogisticRegression)
  --tournament-only For each model, also train on postseason-only data and compare
  --all             Run all model types × training regimes and print comparison table

Evaluation output format:
  [label]  Brier: X.XXXX  |  Accuracy: X.XXXX  (n=N)
  Primary metric: postseason Brier score (tournament games).
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


def evaluate(name: str, predictor, X: pd.DataFrame, y: pd.Series):
    proba = predictor.predict_proba(X)
    preds = (proba >= 0.5).astype(int)
    brier = brier_score_loss(y, proba)
    acc   = accuracy_score(y, preds)
    print(f"  [{name}]  Brier: {brier:.4f}  |  Accuracy: {acc:.4f}  (n={len(y):,})")
    return brier, acc


def _train_and_eval(model, X_train, y_train, X_val, y_val, post_mask, label_prefix):
    """Fit model, print train + val (all) + val (post) scores, return post brier."""
    model.fit(X_train, y_train)
    evaluate(f"{label_prefix} train   ", model, X_train, y_train)
    evaluate(f"{label_prefix} val-all ", model, X_val, y_val)
    if post_mask.sum() > 0:
        brier, acc = evaluate(f"{label_prefix} val-post", model, X_val[post_mask], y_val[post_mask])
    else:
        print(f"  [{label_prefix} val-post] no postseason rows in val set")
        brier, acc = None, None
    return brier, acc


def main():
    use_calibrate    = "--calibrate"       in sys.argv
    use_ensemble     = "--ensemble"        in sys.argv
    use_logistic     = "--logistic"        in sys.argv
    use_tourney_only = "--tournament-only" in sys.argv
    use_all          = "--all"             in sys.argv

    if use_all:
        use_logistic = use_tourney_only = True

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

    post_mask       = X_val["season_type"] == "postseason"
    post_train_mask = X_train["season_type"] == "postseason"

    X_train_post = X_train[post_train_mask]
    y_train_post = y_train[post_train_mask]
    print(f"\n  Postseason train rows: {len(X_train_post):,}  (of {len(X_train):,} total)")

    # ── Collect results for --all comparison table ────────────────────────────
    table_rows = []

    # ── Baseline: GamePredictor ───────────────────────────────────────────────
    print("\n" + "=" * 50)
    print("STEP 2 — Train (GamePredictor, all games)")
    print("=" * 50)
    predictor = GamePredictor(params=params)
    brier, acc = _train_and_eval(
        predictor, X_train, y_train, X_val, y_val, post_mask, "HistGBT  all    "
    )
    table_rows.append(("HistGBT", "all 2015–24", brier, acc, post_mask.sum()))

    # ── Tournament-only variant (HistGB) ─────────────────────────────────────
    if use_tourney_only and len(X_train_post) > 0:
        print("\n" + "─" * 50)
        print("--- HistGBT trained on postseason-only ---")
        print("─" * 50)
        pred_post = GamePredictor(params=params)
        brier_t, acc_t = _train_and_eval(
            pred_post, X_train_post, y_train_post, X_val, y_val, post_mask, "HistGBT  post   "
        )
        table_rows.append(("HistGBT", "post 2015–24", brier_t, acc_t, post_mask.sum()))

    # ── Calibrated ───────────────────────────────────────────────────────────
    if use_calibrate:
        from models.calibrated_predictor import CalibratedPredictor
        print("\n" + "─" * 50)
        print("--- Calibrated (isotonic, last-15%-holdout) ---")
        print("─" * 50)
        cal = CalibratedPredictor(GamePredictor(params=params))
        cal.fit(X_train, y_train)
        evaluate("Cal val-all  ", cal, X_val, y_val)
        if post_mask.sum() > 0:
            evaluate("Cal val-post ", cal, X_val[post_mask], y_val[post_mask])
        cal.save()

    # ── Ensemble ─────────────────────────────────────────────────────────────
    if use_ensemble:
        from models.ensemble_predictor import EnsemblePredictor
        print("\n" + "─" * 50)
        print("--- Ensemble (HistGB + LightGBM + logistic meta) ---")
        print("─" * 50)
        ens = EnsemblePredictor(hist_params=params)
        ens.fit(X_train, y_train)
        evaluate("Ens val-all  ", ens, X_val, y_val)
        if post_mask.sum() > 0:
            evaluate("Ens val-post ", ens, X_val[post_mask], y_val[post_mask])
        ens.save()

    # ── Logistic Predictor ───────────────────────────────────────────────────
    if use_logistic:
        from models.logistic_predictor import LogisticPredictor

        logistic_params_path = os.path.join(os.path.dirname(__file__), "models", "best_logistic_params.json")
        lr_kwargs = {}
        if os.path.exists(logistic_params_path):
            with open(logistic_params_path) as f:
                lr_kwargs = json.load(f)
            print(f"\n  Loaded logistic params from {logistic_params_path}")

        print("\n" + "─" * 50)
        print("--- LogisticPredictor (all games) ---")
        print("─" * 50)
        lr_all = LogisticPredictor(**lr_kwargs)
        brier_lr, acc_lr = _train_and_eval(
            lr_all, X_train, y_train, X_val, y_val, post_mask, "LogReg   all    "
        )
        table_rows.append(("LogReg", "all 2015–24", brier_lr, acc_lr, post_mask.sum()))
        lr_all.save()

        if use_tourney_only and len(X_train_post) > 0:
            print("\n" + "─" * 50)
            print("--- LogisticPredictor (postseason-only) ---")
            print("─" * 50)
            lr_post = LogisticPredictor(**lr_kwargs)
            brier_lrp, acc_lrp = _train_and_eval(
                lr_post, X_train_post, y_train_post, X_val, y_val, post_mask, "LogReg   post   "
            )
            table_rows.append(("LogReg", "post 2015–24", brier_lrp, acc_lrp, post_mask.sum()))

    # ── Comparison table (--all) ──────────────────────────────────────────────
    if use_all and table_rows:
        print("\n" + "=" * 65)
        print(" Comparison (val postseason Brier — lower is better)")
        print("=" * 65)
        print(f"  {'Model':<28} {'Train set':<14} {'Brier':>7}  {'Acc':>7}  {'n':>5}")
        print("  " + "-" * 61)
        for model_name, train_set, brier, acc, n in table_rows:
            brier_str = f"{brier:.4f}" if brier is not None else "  N/A "
            acc_str   = f"{acc:.4f}"   if acc   is not None else "  N/A "
            print(f"  {model_name:<28} {train_set:<14} {brier_str:>7}  {acc_str:>7}  {n:>5}")
        print("=" * 65)

    # ── Feature importances (baseline model) ─────────────────────────────────
    print("\n" + "=" * 50)
    print("STEP 3 — Feature importances (baseline HistGBT)")
    print("=" * 50)
    try:
        print(predictor.feature_importances().to_string())
    except AttributeError:
        print("  Native importances unavailable — running permutation importance on val set...")
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

    # ── Save baseline model ───────────────────────────────────────────────────
    print("\n" + "=" * 50)
    print("STEP 4 — Save baseline model")
    print("=" * 50)
    predictor.save()

    print("\nDone.")
    return predictor


if __name__ == "__main__":
    main()
