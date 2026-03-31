# data/rebuild_val_with_injuries.py
"""
Apply injury adjustments to features_by_season[SEASON] and rebuild X_val / y_val.

The original features_by_season.pkl is NOT modified.  Only X_val.csv / y_val.csv
are overwritten so that train.py picks up the adjusted validation set.

Usage:
    python data/rebuild_val_with_injuries.py
    python data/rebuild_val_with_injuries.py --season 2025 --injuries data/injuries_2025.json
    python data/rebuild_val_with_injuries.py --restore   # put originals back
"""

import os
import sys
import copy
import shutil
import argparse

import joblib
import pandas as pd

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from data.injury_adjustments import (
    apply_injury_adjustments, load_player_stats, print_injury_report,
)
from features.builder import build_training_set

CACHE_DIR = os.path.join(os.path.dirname(__file__), "cache")
RAW_DIR   = os.path.join(CACHE_DIR, "raw")
FEAT_PKL  = os.path.join(CACHE_DIR, "features_by_season.pkl")
X_VAL_PATH = os.path.join(CACHE_DIR, "X_val.csv")
Y_VAL_PATH = os.path.join(CACHE_DIR, "y_val.csv")
X_VAL_ORIG = os.path.join(CACHE_DIR, "X_val_original.csv")
Y_VAL_ORIG = os.path.join(CACHE_DIR, "y_val_original.csv")


def save_originals():
    """Back up X_val / y_val if not already backed up."""
    if not os.path.exists(X_VAL_ORIG):
        shutil.copy2(X_VAL_PATH, X_VAL_ORIG)
        print(f"  backed up X_val → X_val_original.csv")
    else:
        print(f"  backup already exists: X_val_original.csv")
    if not os.path.exists(Y_VAL_ORIG):
        shutil.copy2(Y_VAL_PATH, Y_VAL_ORIG)
        print(f"  backed up y_val → y_val_original.csv")


def restore_originals():
    """Restore X_val / y_val from backup and remove backup files."""
    if not os.path.exists(X_VAL_ORIG):
        print("No backup found — nothing to restore.")
        return
    shutil.copy2(X_VAL_ORIG, X_VAL_PATH)
    shutil.copy2(Y_VAL_ORIG, Y_VAL_PATH)
    os.remove(X_VAL_ORIG)
    os.remove(Y_VAL_ORIG)
    print("Restored original X_val.csv / y_val.csv and removed backups.")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--season",   type=int, default=2025)
    parser.add_argument("--injuries", type=str, default="data/injuries_2025.json")
    parser.add_argument("--restore",  action="store_true",
                        help="Restore original X_val / y_val from backup")
    args = parser.parse_args()

    if args.restore:
        restore_originals()
        return

    # ── 1. Load features ──────────────────────────────────────────────────────
    print(f"\nLoading {FEAT_PKL} ...")
    features_by_season = joblib.load(FEAT_PKL)
    if args.season not in features_by_season:
        print(f"ERROR: season {args.season} not in features_by_season.")
        sys.exit(1)

    # ── 2. Apply injury adjustments to the target season ──────────────────────
    print(f"\nApplying injury adjustments  (season={args.season}, file={args.injuries})")
    player_stats_df = load_player_stats(args.season)
    adjusted_season_feats, report = apply_injury_adjustments(
        features_by_season[args.season],
        args.injuries,
        player_stats_df,
    )
    print_injury_report(report)

    # Build a shallow copy with the adjusted season swapped in
    adj_features_by_season = dict(features_by_season)
    adj_features_by_season[args.season] = adjusted_season_feats

    # ── 3. Reload the val games from cache ────────────────────────────────────
    games_path = os.path.join(RAW_DIR, f"games_{args.season}.csv")
    if not os.path.exists(games_path):
        print(f"ERROR: val games cache not found: {games_path}")
        sys.exit(1)

    val_games = pd.read_csv(games_path)
    print(f"\nVal games loaded: {len(val_games):,} rows (season {args.season})")

    # ── 4. Rebuild X_val / y_val ──────────────────────────────────────────────
    print("\nRebuilding X_val / y_val with adjusted features ...")
    X_val, y_val = build_training_set(val_games, adj_features_by_season)
    print(f"  X_val shape: {X_val.shape}")

    val_post = X_val[X_val["season_type"] == "postseason"]
    print(f"  Postseason rows: {len(val_post)}  (expect 121 for 2025)")

    # ── 5. Back up originals and save adjusted splits ─────────────────────────
    print("\nBacking up originals ...")
    save_originals()

    X_val.to_csv(X_VAL_PATH, index=False)
    y_val.to_frame("label").to_csv(Y_VAL_PATH, index=False)
    print(f"  saved adjusted X_val → {X_VAL_PATH}")
    print(f"  saved adjusted y_val → {Y_VAL_PATH}")

    print(f"\nDone. Run `python train.py --all` to evaluate on injury-adjusted val set.")
    print(f"To restore originals: python data/rebuild_val_with_injuries.py --restore")


if __name__ == "__main__":
    main()
