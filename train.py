# train.py
"""
Train and evaluate game prediction models.

Default: trains baseline HistGBT (full features, all games) on 2015–2024 data,
         evaluates on 2025 validation set.

--all    Run all experiment variants and print full comparison table:
           B0  HistGBT  full features     all games (baseline)
           B1  HistGBT  trimmed features  all games
           B2  HistGBT  trimmed features  all games + tier weights
           B3  HistGBT  trimmed features  high-signal (NCAA + conf tourn)
           B4  HistGBT  trimmed features  high-signal + weights (NCAA=3, conf=1)
           A0  LogReg   full features     postseason only (current best)
           A2  LogReg   full features     high-signal (NCAA + conf tourn)
           A3  LogReg   full features     high-signal + weights

--calibrate   Also run CalibratedPredictor
--ensemble    Also run EnsemblePredictor

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
from models.game_predictor_model import GamePredictor, NON_FEATURE_COLS

CACHE_DIR         = os.path.join(os.path.dirname(__file__), "data", "cache")
PARAMS_PATH       = os.path.join(os.path.dirname(__file__), "models", "best_params.json")
EVAL_RESULTS_PATH = os.path.join(os.path.dirname(__file__), "models", "eval_results.json")

# HistGBT trimmed feature set (B1): drops 8 features with zero/negative permutation importance
# Dropped: diff_adj_off_rating, diff_net_rating, diff_experience (negative)
#          diff_adj_off_rank, diff_adj_def_rank (redundant with srs)
#          diff_close_game_pct, diff_road_win_pct, diff_non_conf_sos (near-zero)
HISTGBT_FEATURES = [
    "diff_adj_def_rating",
    "diff_efficiency_ratio",
    "diff_upset_propensity",
    "diff_pace",
    "diff_elo",
    "diff_srs",
    "diff_star_power",
    "diff_hot_streak",
    "diff_conf_strength",
    "neutral_site",
    "seed_diff",
]


def load_splits():
    X_train = pd.read_csv(os.path.join(CACHE_DIR, "X_train.csv"))
    y_train = pd.read_csv(os.path.join(CACHE_DIR, "y_train.csv"))["label"]
    X_val   = pd.read_csv(os.path.join(CACHE_DIR, "X_val.csv"))
    y_val   = pd.read_csv(os.path.join(CACHE_DIR, "y_val.csv"))["label"]
    return X_train, y_train, X_val, y_val


# ── Training-tier masks ───────────────────────────────────────────────────────

def _postseason_mask(X):
    """NCAA tournament rows (seasonType == "postseason")."""
    return X["season_type"] == "postseason"


def _conf_tourn_mask(X):
    """Conference tournament rows (game_type == TRNMNT, conf_game == True)."""
    if "game_type" not in X.columns or "conf_game" not in X.columns:
        return pd.Series(False, index=X.index)
    return (X["game_type"] == "TRNMNT") & (X["conf_game"].astype(bool))


def _high_signal_mask(X):
    """All high-signal rows: NCAA tournament + conference tournaments."""
    return _postseason_mask(X) | _conf_tourn_mask(X)


def _tier_weights_all(X):
    """Sample weights for all-game training: NCAA=10, conf tourn=5, regular=1."""
    w = np.ones(len(X))
    w[_postseason_mask(X).values] = 10.0
    w[_conf_tourn_mask(X).values] = 5.0
    return w


def _tier_weights_post(X):
    """Sample weights for high-signal training: NCAA=3, conf tourn=1."""
    w = np.ones(len(X))
    w[_postseason_mask(X).values] = 3.0
    return w


def _trim(X):
    """Return X restricted to HISTGBT_FEATURES + all metadata columns."""
    keep_feat = [c for c in HISTGBT_FEATURES if c in X.columns]
    keep_meta = [c for c in X.columns if c in NON_FEATURE_COLS or c in ("game_type", "conf_game")]
    return X[keep_feat + [c for c in keep_meta if c not in keep_feat]]


# ── Evaluation helpers ────────────────────────────────────────────────────────

def _eval_split(predictor, X: pd.DataFrame, y: pd.Series, label: str) -> dict:
    proba   = predictor.predict_proba(X)
    preds   = (proba >= 0.5).astype(int)
    brier   = float(brier_score_loss(y, proba))
    acc     = float(accuracy_score(y, preds))
    logloss = float(log_loss(y, proba))
    print(f"  [{label}]  Brier: {brier:.4f}  |  Acc: {acc:.4f}  |  LogLoss: {logloss:.4f}  (n={len(y):,})")
    return {"brier": round(brier, 6), "accuracy": round(acc, 6), "log_loss": round(logloss, 6), "n": int(len(y))}


def _get_importances(model, X_val: pd.DataFrame, y_val: pd.Series) -> list:
    from models.logistic_predictor import LogisticPredictor

    if isinstance(model, LogisticPredictor):
        coefs     = model.pipeline.named_steps["clf"].coef_[0]
        feat_cols = model.feature_cols
        rows = sorted(
            [{"feature": f, "coef": round(float(c), 6), "abs_coef": round(float(abs(c)), 6)}
             for f, c in zip(feat_cols, coefs)],
            key=lambda r: r["abs_coef"], reverse=True,
        )
        for rank, r in enumerate(rows, 1):
            r["rank"] = rank
        return rows

    raw_model = None
    if isinstance(model, GamePredictor):
        raw_model = model.model
        feat_cols = model.feature_cols
    else:
        return []

    try:
        importances = raw_model.feature_importances_
        rows = sorted(
            [{"feature": f, "importance": round(float(v), 6)}
             for f, v in zip(feat_cols, importances)],
            key=lambda r: r["importance"], reverse=True,
        )
    except AttributeError:
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


def _record(model, model_name: str, regime_label: str, params_used: dict,
            X_train, y_train, X_val, y_val, post_mask, X_val_post, y_val_post,
            sample_weight=None) -> dict:
    model.fit(X_train, y_train, sample_weight=sample_weight)

    entry = {
        "model":        model_name,
        "train_regime": regime_label,
        "train_rows":   int(len(X_train)),
        "params":       params_used,
        "metrics": {
            "train":   _eval_split(model, X_train, y_train, f"{model_name} | {regime_label} | train"),
            "val_all": _eval_split(model, X_val,   y_val,   f"{model_name} | {regime_label} | val-all"),
        },
    }

    if post_mask.sum() > 0:
        entry["metrics"]["val_post"] = _eval_split(
            model, X_val_post, y_val_post, f"{model_name} | {regime_label} | val-post"
        )
    else:
        entry["metrics"]["val_post"] = None

    print(f"    extracting feature importances ...")
    entry["feature_importances"] = _get_importances(model, X_val, y_val)
    return entry


def main():
    use_all      = "--all"       in sys.argv
    use_calibrate = "--calibrate" in sys.argv
    use_ensemble  = "--ensemble"  in sys.argv

    print("=" * 60)
    print("STEP 1 — Load splits")
    print("=" * 60)
    X_train, y_train, X_val, y_val = load_splits()
    print(f"  X_train : {X_train.shape}   X_val : {X_val.shape}")

    has_tier_cols = "game_type" in X_train.columns and "conf_game" in X_train.columns
    if not has_tier_cols:
        print("  WARNING: game_type/conf_game columns missing — rebuild with: python data/pull.py")

    params = None
    if os.path.exists(PARAMS_PATH):
        with open(PARAMS_PATH) as f:
            params = json.load(f)
        print(f"\n  Loaded tuned params from {PARAMS_PATH}")
    else:
        print("\n  No best_params.json found — using defaults")

    # Val masks / subsets
    post_mask  = _postseason_mask(X_val)
    X_val_post = X_val[post_mask]
    y_val_post = y_val[post_mask]

    # Training subsets
    X_train_post = X_train[_postseason_mask(X_train)]
    y_train_post = y_train[_postseason_mask(X_train)]

    if has_tier_cols:
        hs_mask      = _high_signal_mask(X_train)
        X_train_hs   = X_train[hs_mask]
        y_train_hs   = y_train[hs_mask]
        ct_train     = _conf_tourn_mask(X_train).sum()
        ct_val       = _conf_tourn_mask(X_val).sum()
    else:
        X_train_hs, y_train_hs = X_train_post, y_train_post
        ct_train = ct_val = 0

    print(f"\n  NCAA tournament train rows  : {len(X_train_post):,}  (of {len(X_train):,} total)")
    print(f"  Conference tourn train rows : {ct_train:,}")
    print(f"  High-signal train total     : {len(X_train_hs):,}")
    print(f"  Val postseason rows         : {len(X_val_post):,}  (n for primary Brier)")
    print(f"  Val conf tourn rows         : {ct_val:,}")

    # ── Build run record ──────────────────────────────────────────────────────
    run_record = {
        "timestamp":  datetime.datetime.now().isoformat(timespec="seconds"),
        "flags":      [a for a in sys.argv[1:] if a.startswith("--")],
        "data": {
            "train_rows":       int(len(X_train)),
            "train_post_rows":  int(len(X_train_post)),
            "train_hs_rows":    int(len(X_train_hs)),
            "val_rows":         int(len(X_val)),
            "val_post_rows":    int(len(X_val_post)),
            "features":         [c for c in X_train.columns
                                 if c not in NON_FEATURE_COLS and c not in ("game_type", "conf_game")],
            "histgbt_trimmed":  HISTGBT_FEATURES,
        },
        "histgb_params": params,
        "models": [],
    }

    # table_rows: (model_name, features_label, train_regime, brier, acc, n)
    table_rows = []

    def _add(model, model_name, features_label, train_regime, params_used, Xtr, ytr, sw=None):
        regime_label = f"{features_label} | {train_regime}"
        print(f"\n{'─'*60}")
        print(f"  {model_name}  |  features: {features_label}  |  train: {train_regime}")
        print(f"{'─'*60}")
        entry = _record(model, model_name, regime_label, params_used,
                        Xtr, ytr, X_val, y_val, post_mask, X_val_post, y_val_post,
                        sample_weight=sw)
        run_record["models"].append(entry)
        post = entry["metrics"].get("val_post") or {}
        table_rows.append((model_name, features_label, train_regime,
                           post.get("brier"), post.get("accuracy"), post.get("n")))
        return entry

    # ── B0: HistGBT, full features, all games (baseline) ─────────────────────
    print("\n" + "=" * 60)
    print("B0 — HistGBT  full features  all games  (baseline)")
    print("=" * 60)
    predictor = GamePredictor(params=params)
    _add(predictor, "HistGBT", "full", "all games", params or {}, X_train, y_train)

    if use_all:
        # ── B1: HistGBT, trimmed features, all games ──────────────────────────
        print("\n" + "=" * 60)
        print("B1 — HistGBT  trimmed features  all games")
        print("=" * 60)
        pred_b1 = GamePredictor(params=params)
        _add(pred_b1, "HistGBT", "trimmed", "all games", params or {}, _trim(X_train), y_train)

        if has_tier_cols:
            # ── B2: HistGBT, trimmed, all games + tier weights ────────────────
            print("\n" + "=" * 60)
            print("B2 — HistGBT  trimmed features  all games + tier weights")
            print("=" * 60)
            pred_b2 = GamePredictor(params=params)
            _add(pred_b2, "HistGBT", "trimmed", "all+weighted",
                 params or {}, _trim(X_train), y_train, sw=_tier_weights_all(X_train))

            # ── B3: HistGBT, trimmed, high-signal (NCAA + conf tourn) ─────────
            print("\n" + "=" * 60)
            print("B3 — HistGBT  trimmed features  high-signal (NCAA + conf tourn)")
            print("=" * 60)
            pred_b3 = GamePredictor(params=params)
            _add(pred_b3, "HistGBT", "trimmed", "post+conf",
                 params or {}, _trim(X_train_hs), y_train_hs)

            # ── B4: HistGBT, trimmed, high-signal + weights ───────────────────
            print("\n" + "=" * 60)
            print("B4 — HistGBT  trimmed features  high-signal + weights (NCAA=3, conf=1)")
            print("=" * 60)
            pred_b4 = GamePredictor(params=params)
            _add(pred_b4, "HistGBT", "trimmed", "post+conf+wtd",
                 params or {}, _trim(X_train_hs), y_train_hs, sw=_tier_weights_post(X_train_hs))

        # ── Previous HistGBT postseason-only baseline ─────────────────────────
        print("\n" + "=" * 60)
        print("HistGBT  full features  postseason only (NCAA)")
        print("=" * 60)
        pred_post = GamePredictor(params=params)
        _add(pred_post, "HistGBT", "full", "post only (NCAA)", params or {}, X_train_post, y_train_post)

        # ── LogReg setup ──────────────────────────────────────────────────────
        from models.logistic_predictor import LogisticPredictor
        lr_params_path = os.path.join(os.path.dirname(__file__), "models", "best_logistic_params.json")
        lr_kwargs = {}
        if os.path.exists(lr_params_path):
            with open(lr_params_path) as f:
                lr_kwargs = json.load(f)
            run_record["logistic_params"] = lr_kwargs
            print(f"\n  Loaded logistic params from {lr_params_path}")

        # ── A0: LogReg, full features, postseason only ────────────────────────
        print("\n" + "=" * 60)
        print("A0 — LogReg  full features  postseason only (NCAA)  [current best]")
        print("=" * 60)
        lr_a0 = LogisticPredictor(**lr_kwargs)
        _add(lr_a0, "LogReg", "full", "post only (NCAA)", lr_kwargs, X_train_post, y_train_post)
        lr_a0.save()

        if has_tier_cols:
            # ── A2: LogReg, full features, high-signal ────────────────────────
            print("\n" + "=" * 60)
            print("A2 — LogReg  full features  high-signal (NCAA + conf tourn)")
            print("=" * 60)
            lr_a2 = LogisticPredictor(**lr_kwargs)
            _add(lr_a2, "LogReg", "full", "post+conf", lr_kwargs, X_train_hs, y_train_hs)

            # ── A3: LogReg, full features, high-signal + weights ──────────────
            print("\n" + "=" * 60)
            print("A3 — LogReg  full features  high-signal + weights (NCAA=3, conf=1)")
            print("=" * 60)
            lr_a3 = LogisticPredictor(**lr_kwargs)
            _add(lr_a3, "LogReg", "full", "post+conf+wtd",
                 lr_kwargs, X_train_hs, y_train_hs, sw=_tier_weights_post(X_train_hs))

    # ── Optional model types ──────────────────────────────────────────────────
    if use_calibrate:
        from models.calibrated_predictor import CalibratedPredictor
        cal = CalibratedPredictor(GamePredictor(params=params))
        _add(cal, "Calibrated", "full", "all games", params or {}, X_train, y_train)
        cal.save()

    if use_ensemble:
        from models.ensemble_predictor import EnsemblePredictor
        ens = EnsemblePredictor(hist_params=params)
        _add(ens, "Ensemble", "full", "all games", params or {}, X_train, y_train)
        ens.save()

    # ── Comparison table ──────────────────────────────────────────────────────
    if len(table_rows) > 1:
        best_brier = min((r[3] for r in table_rows if r[3] is not None), default=None)
        print("\n" + "=" * 80)
        print(" Comparison — val postseason  (primary metric: Brier, lower = better)")
        print("=" * 80)
        print(f"  {'Model':<10} {'Features':<10} {'Training set':<20} {'Brier':>7}  {'Acc':>7}  {'n':>5}")
        print("  " + "-" * 62)
        for model_name, feat_label, train_set, brier, acc, n in table_rows:
            brier_str = f"{brier:.4f}" if brier is not None else "   N/A"
            acc_str   = f"{acc:.4f}"   if acc   is not None else "   N/A"
            if brier is not None and brier < 0.15:
                marker = " ★ <0.15!"
            elif brier is not None and brier == best_brier:
                marker = " ◀ best"
            else:
                marker = ""
            print(f"  {model_name:<10} {feat_label:<10} {train_set:<20} {brier_str:>7}  {acc_str:>7}  {n or 0:>5}{marker}")
        print("=" * 80)

    # ── Save eval JSON ────────────────────────────────────────────────────────
    os.makedirs(os.path.dirname(EVAL_RESULTS_PATH), exist_ok=True)
    with open(EVAL_RESULTS_PATH, "w") as f:
        json.dump(run_record, f, indent=2)
    print(f"\n  Eval results saved → {EVAL_RESULTS_PATH}")

    # Save baseline model
    predictor.save()
    print("\nDone.")
    return predictor


if __name__ == "__main__":
    main()
