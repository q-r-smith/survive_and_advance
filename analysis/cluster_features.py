# analysis/cluster_features.py
"""
Inject team style cluster features into the supervised pipeline.

What this script does:
  1. Loads features_by_season.pkl and cluster_labels.csv
  2. Adds cluster_label and dist_to_centroid to every team's feature dict
     (NaN for teams that were not clustered — model imputes at train time)
  3. Patches features.builder.TEAM_FEATURES at runtime to include the new features
  4. Rebuilds X_train.csv and X_val.csv via build_training_set with the updated features
  5. Saves updated features_by_season.pkl

Pre-requisites:
  - data/cache/features_by_season.pkl  (run python data/pull.py first)
  - analysis/cache/cluster_labels.csv  (run python analysis/team_profiles.py first)

Usage:
    # Dry run — check injection without overwriting CSVs:
    python analysis/cluster_features.py --all-seasons --dry-run

    # Inject and rebuild splits:
    cp data/cache/features_by_season.pkl data/cache/features_by_season_backup.pkl
    python analysis/cluster_features.py --all-seasons

    # Undo — restore backup (splits need to be rebuilt separately via data/pull.py):
    cp data/cache/features_by_season_backup.pkl data/cache/features_by_season.pkl
    python data/pull.py   # rebuild X_train / X_val without cluster features
"""

import argparse
import os
import sys

import joblib
import numpy as np
import pandas as pd

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

CACHE_DIR  = os.path.join(os.path.dirname(__file__), "..", "data", "cache")
RAW_DIR    = os.path.join(CACHE_DIR, "raw")
FEAT_PKL   = os.path.join(CACHE_DIR, "features_by_season.pkl")
X_TRAIN    = os.path.join(CACHE_DIR, "X_train.csv")
Y_TRAIN    = os.path.join(CACHE_DIR, "y_train.csv")
X_VAL      = os.path.join(CACHE_DIR, "X_val.csv")
Y_VAL      = os.path.join(CACHE_DIR, "y_val.csv")
LABELS_CSV = os.path.join(os.path.dirname(__file__), "cache", "cluster_labels.csv")

# Mirror data/pull.py split constants
TRAIN_SEASONS = range(2015, 2026)   # 2015–2025 inclusive
VAL_SEASON    = 2025

CLUSTER_FEATURES = ["cluster_label", "dist_to_centroid"]


# ── Data loading ──────────────────────────────────────────────────────────────

def load_games(seasons) -> pd.DataFrame:
    """Concatenate raw games CSVs for the given seasons."""
    frames = []
    for s in seasons:
        path = os.path.join(RAW_DIR, f"games_{s}.csv")
        if not os.path.exists(path):
            print(f"  WARNING: games_{s}.csv not found — skipping")
            continue
        frames.append(pd.read_csv(path))
    if not frames:
        print("ERROR: no games CSVs found in data/cache/raw/")
        sys.exit(1)
    return pd.concat(frames, ignore_index=True)


def load_cluster_labels() -> pd.DataFrame:
    if not os.path.exists(LABELS_CSV):
        print(f"ERROR: cluster labels not found at {LABELS_CSV}")
        print("Run: python analysis/team_profiles.py --seasons ... first")
        sys.exit(1)
    df = pd.read_csv(LABELS_CSV)
    required = {"team", "season", "cluster_label", "dist_to_centroid"}
    missing = required - set(df.columns)
    if missing:
        print(f"ERROR: cluster_labels.csv is missing columns: {missing}")
        sys.exit(1)
    return df


# ── Injection ─────────────────────────────────────────────────────────────────

def inject_cluster_features(features_by_season: dict, labels_df: pd.DataFrame) -> dict:
    """
    Add cluster_label and dist_to_centroid to each team's feature dict.
    Teams not in labels_df get NaN (imputed at train time).
    Returns a new dict — the original is not modified.
    """
    import copy
    updated = copy.deepcopy(features_by_season)

    # Build lookup: (team, season) → {cluster_label, dist_to_centroid}
    lookup = {}
    for _, row in labels_df.iterrows():
        key = (row["team"], int(row["season"]))
        lookup[key] = {
            "cluster_label":    float(row["cluster_label"]),
            "dist_to_centroid": float(row["dist_to_centroid"]),
        }

    injected_count = 0
    for season, team_feats in updated.items():
        for team, feats in team_feats.items():
            vals = lookup.get((team, season), None)
            if vals is not None:
                feats["cluster_label"]    = vals["cluster_label"]
                feats["dist_to_centroid"] = vals["dist_to_centroid"]
                injected_count += 1
            else:
                feats["cluster_label"]    = np.nan
                feats["dist_to_centroid"] = np.nan

    total = sum(len(v) for v in updated.values())
    print(f"  Injected cluster features for {injected_count:,} / {total:,} team-seasons")
    nan_count = total - injected_count
    nan_pct = nan_count / total * 100
    print(f"  NaN (not clustered): {nan_count:,}  ({nan_pct:.1f}%)")
    return updated


# ── Split rebuild ─────────────────────────────────────────────────────────────

def rebuild_splits(features_by_season: dict, games: pd.DataFrame) -> tuple:
    """
    Rebuild X_train/y_train/X_val/y_val using the updated features.
    Mirrors the split logic in data/pull.py:
      Training: seasons 2015–2024 (all game types) + 2025 regular season only
      Validation: all 2025 games (regular + postseason)
    """
    import features.builder as fb

    # Temporarily add cluster features to TEAM_FEATURES if not already there
    added = []
    for feat in CLUSTER_FEATURES:
        if feat not in fb.TEAM_FEATURES:
            fb.TEAM_FEATURES.append(feat)
            added.append(feat)

    if added:
        print(f"  Patched TEAM_FEATURES at runtime: added {added}")

    # Training split
    train_mask = (
        games["season"].isin(range(2015, 2025))
        | (
            (games["season"] == 2025)
            & (games["seasonType"] == "regular")
        )
    )
    train_games = games[train_mask]

    # Validation split
    val_games = games[games["season"] == VAL_SEASON]

    print(f"  Training games   : {len(train_games):,}")
    print(f"  Val games        : {len(val_games):,}")

    print("  Building training set ...")
    X_train, y_train = fb.build_training_set(train_games, features_by_season)

    print("  Building validation set ...")
    X_val, y_val = fb.build_training_set(val_games, features_by_season)

    # Restore TEAM_FEATURES if we patched it
    for feat in added:
        if feat in fb.TEAM_FEATURES:
            fb.TEAM_FEATURES.remove(feat)

    return X_train, y_train, X_val, y_val


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="Inject cluster features into supervised pipeline.")
    parser.add_argument("--all-seasons", action="store_true", required=True,
                        help="Inject cluster features for all seasons in features_by_season.pkl")
    parser.add_argument("--dry-run", action="store_true",
                        help="Check injection without overwriting any files")
    args = parser.parse_args()

    # 1. Load features_by_season
    if not os.path.exists(FEAT_PKL):
        print(f"ERROR: {FEAT_PKL} not found. Run `python data/pull.py` first.")
        sys.exit(1)
    print(f"Loading {FEAT_PKL} ...")
    features_by_season = joblib.load(FEAT_PKL)
    all_seasons = sorted(features_by_season.keys())
    print(f"  Seasons in pkl: {all_seasons}")

    # 2. Load cluster labels
    print(f"\nLoading cluster labels from {LABELS_CSV} ...")
    labels_df = load_cluster_labels()
    print(f"  {len(labels_df):,} labeled team-seasons  "
          f"(covering seasons: {sorted(labels_df['season'].unique().tolist())})")

    # 3. Inject
    print("\nInjecting cluster features into features_by_season ...")
    updated_features = inject_cluster_features(features_by_season, labels_df)

    if args.dry_run:
        print("\n[dry-run] Skipping file writes. Injection looks good.")
        # Quick sanity: pick a clustered team and verify
        clustered = labels_df.iloc[0]
        s, t = int(clustered["season"]), clustered["team"]
        if s in updated_features and t in updated_features[s]:
            f = updated_features[s][t]
            print(f"  Sample — {t} ({s}): "
                  f"cluster_label={f.get('cluster_label', 'MISSING')}, "
                  f"dist_to_centroid={f.get('dist_to_centroid', 'MISSING'):.4f}")
        return

    # 4. Load games for all training + val seasons
    print("\nLoading games CSVs ...")
    game_seasons = list(set(list(TRAIN_SEASONS) + [VAL_SEASON]))
    games = load_games(game_seasons)
    print(f"  {len(games):,} total game rows across {len(game_seasons)} seasons")

    # 5. Rebuild splits
    print("\nRebuilding train/val splits with cluster features ...")
    X_train, y_train, X_val, y_val = rebuild_splits(updated_features, games)
    print(f"  X_train shape: {X_train.shape}")
    print(f"  X_val   shape: {X_val.shape}")

    # Verify cluster columns were injected
    cluster_cols = [c for c in X_train.columns if "cluster" in c or "centroid" in c]
    print(f"  Cluster columns in X_train: {cluster_cols}")
    if cluster_cols:
        nan_rates = X_train[cluster_cols].isnull().mean()
        for col in cluster_cols:
            print(f"    {col}: {nan_rates[col]:.1%} NaN")

    # 6. Save updated pkl
    print(f"\nSaving updated features_by_season → {FEAT_PKL}")
    joblib.dump(updated_features, FEAT_PKL)

    # 7. Save splits
    X_train.to_csv(X_TRAIN, index=False)
    y_train.to_frame("label").to_csv(Y_TRAIN, index=False)
    X_val.to_csv(X_VAL, index=False)
    y_val.to_frame("label").to_csv(Y_VAL, index=False)
    print(f"  saved X_train → {X_TRAIN}")
    print(f"  saved y_train → {Y_TRAIN}")
    print(f"  saved X_val   → {X_VAL}")
    print(f"  saved y_val   → {Y_VAL}")

    print(f"""
Done. Next step:
  python train.py --all

To undo:
  cp data/cache/features_by_season_backup.pkl data/cache/features_by_season.pkl
  python data/pull.py   # rebuild X_train / X_val without cluster features
""")


if __name__ == "__main__":
    main()
