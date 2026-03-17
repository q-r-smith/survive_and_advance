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

After every run, models/eval_results.json is written with full metrics,
feature importances/coefficients, and hyperparameters for every model evaluated.
"""

import os
import sys
import json
import datetime
import numpy as np
import pandas as pd
from sklearn.metrics import brier_score_loss, accuracy_score, log_loss
from sklearn.inspection import permutation_importance

sys.path.insert(0, os.path.dirname(__file__))
from models.game_predictor_model import GamePredictor

CACHE_DIR        = os.path.join(os.path.dirname(__file__), "data", "cache")
PARAMS_PATH      = os.path.join(os.path.dirname(__file__), "models", "best_params.json")
EVAL_RESULTS_PATH = os.path.join(os.path.dirname(__file__), "models", "eval_results.json")


def load_splits():
    X_train = pd.read_csv(os.path.join(CACHE_DIR, "X_train.csv"))
    y_train = pd.read_csv(os.path.join(CACHE_DIR, "y_train.csv"))["label"]
    X_val   = pd.read_csv(os.path.join(CACHE_DIR, "X_val.csv"))
    y_val   = pd.read_csv(os.path.join(CACHE_DIR, "y_val.csv"))["label"]
    return X_train, y_train, X_val, y_val


def _eval_split(predictor, X: pd.DataFrame, y: pd.Series, label: str) -> dict:
    """Evaluate a predictor on one split. Prints and returns a metrics dict."""
    proba  = predictor.predict_proba(X)
    preds  = (proba >= 0.5).astype(int)
    brier  = float(brier_score_loss(y, proba))
    acc    = float(accuracy_score(y, preds))
    logloss = float(log_loss(y, proba))
    print(f"  [{label}]  Brier: {brier:.4f}  |  Acc: {acc:.4f}  |  LogLoss: {logloss:.4f}  (n={len(y):,})")
    return {"brier": round(brier, 6), "accuracy": round(acc, 6), "log_loss": round(logloss, 6), "n": int(len(y))}


def _get_importances(model, X_val: pd.DataFrame, y_val: pd.Series) -> list:
    """
    Extract feature importances or coefficients depending on model type.
    Returns a list of dicts sorted by absolute importance, ready for JSON.
    """
    from models.logistic_predictor import LogisticPredictor

    if isinstance(model, LogisticPredictor):
        coefs   = model.pipeline.named_steps["clf"].coef_[0]
        feat_cols = model.feature_cols
        rows = sorted(
            [{"feature": f, "coef": round(float(c), 6), "abs_coef": round(float(abs(c)), 6)}
             for f, c in zip(feat_cols, coefs)],
            key=lambda r: r["abs_coef"], reverse=True,
        )
        for rank, r in enumerate(rows, 1):
            r["rank"] = rank
        return rows

    # HistGB-based models (GamePredictor or wrapped)
    raw_model = None
    if isinstance(model, GamePredictor):
        raw_model = model.model
        feat_cols = model.feature_cols
    else:
        # CalibratedPredictor, EnsemblePredictor — best effort
        return []

    try:
        importances = raw_model.feature_importances_
        rows = sorted(
            [{"feature": f, "importance": round(float(v), 6)}
             for f, v in zip(feat_cols, importances)],
            key=lambda r: r["importance"], reverse=True,
        )
    except AttributeError:
        # sklearn < 1.2: fall back to permutation importance on val set
        print("    (native importances unavailable — using permutation importance)")
        result = permutation_importance(
            raw_model, X_val[feat_cols], y_val,
            n_repeats=10, scoring="neg_brier_score", random_state=42, n_jobs=-1,
        )
        rows = sorted(
            [{"feature": f, "importance": round(float(m), 6),
              "importance_std": round(float(s), 6)}
             for f, m, s in zip(feat_cols, result.importances_mean, result.importances_std)],
            key=lambda r: r["importance"], reverse=True,
        )

    for rank, r in enumerate(rows, 1):
        r["rank"] = rank
    return rows


def _record(model, model_name: str, train_regime: str, params_used: dict,
            X_train, y_train, X_val, y_val, post_mask, X_val_post, y_val_post) -> dict:
    """
    Fit model, collect all metrics + importances, return a result dict.
    Also appends to the console comparison table.
    """
    model.fit(X_train, y_train)

    entry = {
        "model":         model_name,
        "train_regime":  train_regime,
        "train_rows":    int(len(X_train)),
        "params":        params_used,
        "metrics": {
            "train":    _eval_split(model, X_train, y_train, f"{model_name} | {train_regime} | train"),
            "val_all":  _eval_split(model, X_val,   y_val,   f"{model_name} | {train_regime} | val-all"),
        },
    }

    if post_mask.sum() > 0:
        entry["metrics"]["val_post"] = _eval_split(
            model, X_val_post, y_val_post, f"{model_name} | {train_regime} | val-post"
        )
    else:
        entry["metrics"]["val_post"] = None

    print(f"    extracting feature importances ...")
    entry["feature_importances"] = _get_importances(model, X_val, y_val)

    return entry


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

    post_mask        = X_val["season_type"] == "postseason"
    post_train_mask  = X_train["season_type"] == "postseason"
    X_val_post       = X_val[post_mask]
    y_val_post       = y_val[post_mask]
    X_train_post     = X_train[post_train_mask]
    y_train_post     = y_train[post_train_mask]
    print(f"\n  Postseason train rows : {len(X_train_post):,}  (of {len(X_train):,} total)")
    print(f"  Postseason val rows   : {len(X_val_post):,}  (of {len(X_val):,} total)")

    # ── Build the run record that will be serialised to JSON ─────────────────
    run_record = {
        "timestamp":  datetime.datetime.now().isoformat(timespec="seconds"),
        "flags":      [a for a in sys.argv[1:] if a.startswith("--")],
        "data": {
            "train_rows":      int(len(X_train)),
            "train_post_rows": int(len(X_train_post)),
            "val_rows":        int(len(X_val)),
            "val_post_rows":   int(len(X_val_post)),
            "features":        [c for c in X_train.columns if c not in ("season", "season_type")],
        },
        "histgb_params": params,
        "models": [],
    }

    table_rows = []  # (model_name, train_regime, post_brier, post_acc)

    def _add(model, model_name, train_regime, params_used, Xtr, ytr):
        print(f"\n{'─'*50}")
        print(f"  {model_name}  |  train: {train_regime}")
        print(f"{'─'*50}")
        entry = _record(model, model_name, train_regime, params_used,
                        Xtr, ytr, X_val, y_val, post_mask, X_val_post, y_val_post)
        run_record["models"].append(entry)
        post = entry["metrics"].get("val_post") or {}
        table_rows.append((model_name, train_regime,
                           post.get("brier"), post.get("accuracy"), post.get("n")))
        return entry

    # ── Baseline: GamePredictor (all games) ──────────────────────────────────
    print("\n" + "=" * 50)
    print("STEP 2 — GamePredictor (all games)")
    print("=" * 50)
    predictor = GamePredictor(params=params)
    baseline_entry = _add(predictor, "HistGBT", "all 2015–24", params or {}, X_train, y_train)

    # ── Tournament-only: HistGB ───────────────────────────────────────────────
    if use_tourney_only and len(X_train_post) > 0:
        pred_post = GamePredictor(params=params)
        _add(pred_post, "HistGBT", "post 2015–24", params or {}, X_train_post, y_train_post)

    # ── Calibrated ───────────────────────────────────────────────────────────
    if use_calibrate:
        from models.calibrated_predictor import CalibratedPredictor
        cal = CalibratedPredictor(GamePredictor(params=params))
        _add(cal, "Calibrated", "all 2015–24", params or {}, X_train, y_train)
        cal.save()

    # ── Ensemble ─────────────────────────────────────────────────────────────
    if use_ensemble:
        from models.ensemble_predictor import EnsemblePredictor
        ens = EnsemblePredictor(hist_params=params)
        _add(ens, "Ensemble", "all 2015–24", params or {}, X_train, y_train)
        ens.save()

    # ── Logistic Predictor ───────────────────────────────────────────────────
    if use_logistic:
        from models.logistic_predictor import LogisticPredictor

        lr_params_path = os.path.join(os.path.dirname(__file__), "models", "best_logistic_params.json")
        lr_kwargs = {}
        if os.path.exists(lr_params_path):
            with open(lr_params_path) as f:
                lr_kwargs = json.load(f)
            run_record["logistic_params"] = lr_kwargs
            print(f"\n  Loaded logistic params from {lr_params_path}")

        lr_all = LogisticPredictor(**lr_kwargs)
        _add(lr_all, "LogReg", "all 2015–24", lr_kwargs, X_train, y_train)
        lr_all.save()

        if use_tourney_only and len(X_train_post) > 0:
            lr_post = LogisticPredictor(**lr_kwargs)
            _add(lr_post, "LogReg", "post 2015–24", lr_kwargs, X_train_post, y_train_post)

    # ── Comparison table ──────────────────────────────────────────────────────
    if len(table_rows) > 1:
        print("\n" + "=" * 68)
        print(" Comparison — val postseason  (primary metric: Brier, lower = better)")
        print("=" * 68)
        print(f"  {'Model':<12} {'Train set':<14} {'Brier':>7}  {'Acc':>7}  {'n':>5}")
        print("  " + "-" * 44)
        best_brier = min((r[2] for r in table_rows if r[2] is not None), default=None)
        for model_name, train_set, brier, acc, n in table_rows:
            brier_str = f"{brier:.4f}" if brier is not None else "   N/A"
            acc_str   = f"{acc:.4f}"   if acc   is not None else "   N/A"
            marker    = " ◀" if brier is not None and brier == best_brier else ""
            print(f"  {model_name:<12} {train_set:<14} {brier_str:>7}  {acc_str:>7}  {n or 0:>5}{marker}")
        print("=" * 68)

    # ── Save eval JSON ────────────────────────────────────────────────────────
    os.makedirs(os.path.dirname(EVAL_RESULTS_PATH), exist_ok=True)
    with open(EVAL_RESULTS_PATH, "w") as f:
        json.dump(run_record, f, indent=2)
    print(f"\n  Eval results saved → {EVAL_RESULTS_PATH}")

    # ── Save baseline model ───────────────────────────────────────────────────
    predictor.save()

    print("\nDone.")
    return predictor


if __name__ == "__main__":
    main()
