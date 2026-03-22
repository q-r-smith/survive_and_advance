# data/pull.py
"""
Pulls all raw data, caches to CSV, builds features, and produces train/val splits.

Train:      2015–2025 games (all regular + postseason)
Validation: 2026 games (postseason = tournament; regular included with season_type column)

No leakage: features for each season are built entirely from that season's
regular-season data. 2026 features never touch 2026 postseason outcomes.
"""

import os
import sys
import pandas as pd

# allow running from project root or data/
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from data.loader import (
    get_games,
    get_team_stats,
    get_player_stats,
    get_roster,
    get_srs,
    get_adjusted_ratings,
    get_elo,
)
from features.builder import build_team_season_features, build_training_set

CACHE_DIR = os.path.join(os.path.dirname(__file__), "cache")
RAW_DIR   = os.path.join(CACHE_DIR, "raw")
TRAIN_SEASONS = range(2015, 2026)   # 2015–2025 inclusive (adds 2025 tournament)
VAL_SEASON    = 2025                # permanent benchmark — complete, stable, n=121
# 2014 is needed as the prior-season feature set for 2015 regular season games.
# We build features from it but never train on 2014 games themselves.
PRIOR_SEASON  = 2014


# ── helpers ──────────────────────────────────────────────────────────────────

def _cache_path(name):
    return os.path.join(RAW_DIR, f"{name}.csv")


def _save(df, path):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    df.to_csv(path, index=False)
    print(f"  saved {path}  ({len(df):,} rows)")


def _load_or_fetch(name, fetch_fn, *args, force=False, **kwargs):
    path = _cache_path(name)
    if not force and os.path.exists(path) and os.path.getsize(path) > 0:
        try:
            df = pd.read_csv(path)
            if len(df) > 0:
                print(f"  cache hit  → {path}")
                return df
        except Exception:
            pass
        print(f"  cache corrupt — re-fetching {name} ...")
    print(f"  fetching {name} ...")
    df = fetch_fn(*args, **kwargs)
    _save(df, path)
    return df


# ── raw pull ─────────────────────────────────────────────────────────────────

def pull_raw(seasons, force=False):
    """
    Pull and cache one CSV per data type, covering all requested seasons.
    Pass force=True to re-fetch even if cache exists.
    """
    all_data = {k: [] for k in ["games", "team_stats", "player_stats", "rosters", "srs", "adjusted_ratings", "elo"]}

    for year in seasons:
        print(f"\n── {year} ──────────────────────────")
        all_data["games"].append(
            _load_or_fetch(f"games_{year}", get_games, year, force=force)
        )
        all_data["team_stats"].append(
            _load_or_fetch(f"team_stats_{year}", get_team_stats, year, force=force)
        )
        all_data["player_stats"].append(
            _load_or_fetch(f"player_stats_{year}", get_player_stats, year, force=force)
        )
        all_data["rosters"].append(
            _load_or_fetch(f"rosters_{year}", get_roster, year, force=force)
        )
        all_data["srs"].append(
            _load_or_fetch(f"srs_{year}", get_srs, year, force=force)
        )
        all_data["adjusted_ratings"].append(
            _load_or_fetch(f"adjusted_ratings_{year}", get_adjusted_ratings, year, force=force)
        )
        all_data["elo"].append(
            _load_or_fetch(f"elo_{year}", get_elo, year, force=force)
        )

    return {k: pd.concat(v, ignore_index=True) for k, v in all_data.items()}


# ── feature build ─────────────────────────────────────────────────────────────

def build_features(data, seasons):
    """
    Compute end-of-season team features for each season.
    Each season's features are built entirely from that season's data — no lookahead.
    """
    features_by_season = {}
    for year in seasons:
        print(f"  building features for {year} ...")

        def _season_slice(df, col="season"):
            return df[df[col] == year]

        features_by_season[year] = build_team_season_features(
            team_stats_df      = _season_slice(data["team_stats"]),
            player_stats_df    = _season_slice(data["player_stats"]),
            roster_df          = _season_slice(data["rosters"]),
            srs_df             = _season_slice(data["srs"]),
            adj_df             = _season_slice(data["adjusted_ratings"]),
            elo_df             = _season_slice(data["elo"]),
            games_df           = data["games"],   # all seasons needed for seed regression
            season             = year,
            features_by_season = features_by_season,  # what's built so far (out-of-fold)
        )
    return features_by_season


# ── split & save ──────────────────────────────────────────────────────────────

def build_splits(data, features_by_season):
    """
    Build train / val splits with strict temporal integrity.

    Training set:
      - All game types for seasons 2015–2024
      - REGULAR SEASON ONLY for season 2025
      (2025 postseason is reserved for val — never in training)

    Validation set:
      - ALL 2025 games (regular + postseason)
      - season_type column preserved for filtering to postseason only

    This prevents the 2025 postseason games from appearing in both sets,
    which would inflate val metrics by leaking training targets into eval.
    """
    games = data["games"]

    # Training: all seasons 2015–2024 (any game type)
    # PLUS 2025 regular season only (no postseason)
    train_mask = (
        (games["season"].isin(range(2015, 2025)))
        | (
            (games["season"] == 2025) &
            (games["seasonType"] == "regular")
        )
    )
    train_games = games[train_mask]

    # Validation: all 2025 games (regular + postseason)
    val_games = games[games["season"] == VAL_SEASON]

    print(f"\n  Training games   : {len(train_games):,}")
    print(f"  Val games        : {len(val_games):,}")

    # Integrity check — 2025 postseason must not be in training
    train_post_2025 = train_games[
        (train_games["season"] == 2025) & (train_games["seasonType"] == "postseason")
    ]
    assert len(train_post_2025) == 0, (
        f"LEAKAGE: {len(train_post_2025)} 2025 postseason games in training set!"
    )
    print(f"  Leakage check    : PASSED (0 postseason 2025 games in training)")

    print(f"\n  building training set  ({len(train_games):,} games) ...")
    X_train, y_train = build_training_set(train_games, features_by_season)

    print(f"  building validation set ({len(val_games):,} games) ...")
    X_val, y_val = build_training_set(val_games, features_by_season)

    return X_train, y_train, X_val, y_val


# ── main ──────────────────────────────────────────────────────────────────────

def main(force=False):
    # Pull one extra prior season so regular-season games in the first train year
    # (2015) can use leak-free features. Features are built for all pulled seasons;
    # only TRAIN_SEASONS + VAL_SEASON games enter the train/val matrices.
    # Pull one extra prior season so regular-season games in the first train year
    # (2015) can use leak-free features. build_splits still filters games to
    # TRAIN_SEASONS + VAL_SEASON via the module-level constants.
    # 2026: pulled for features only (simulation inference target — not in train/val splits)
    SIM_SEASON = 2026
    # Deduplicate: TRAIN_SEASONS includes VAL_SEASON (2025), so avoid pulling twice
    all_seasons = list(dict.fromkeys([PRIOR_SEASON] + list(TRAIN_SEASONS) + [VAL_SEASON, SIM_SEASON]))

    print("=" * 50)
    print("STEP 1 — Pull raw data")
    print("=" * 50)
    data = pull_raw(all_seasons, force=force)

    print("\n" + "=" * 50)
    print("STEP 2 — Build features per season")
    print("=" * 50)
    features_by_season = build_features(data, all_seasons)

    import joblib
    feat_pkl_path = os.path.join(CACHE_DIR, "features_by_season.pkl")
    joblib.dump(features_by_season, feat_pkl_path)
    print(f"  saved features_by_season → {feat_pkl_path}")

    print("\n" + "=" * 50)
    print("STEP 3 — Build train / val splits")
    print("=" * 50)
    X_train, y_train, X_val, y_val = build_splits(data, features_by_season)

    print("\n" + "=" * 50)
    print("STEP 4 — Save splits")
    print("=" * 50)
    _save(X_train,                      os.path.join(CACHE_DIR, "X_train.csv"))
    _save(y_train.to_frame("label"),    os.path.join(CACHE_DIR, "y_train.csv"))
    _save(X_val,                        os.path.join(CACHE_DIR, "X_val.csv"))
    _save(y_val.to_frame("label"),      os.path.join(CACHE_DIR, "y_val.csv"))

    # Sanity checks
    print("\n── Sanity checks ───────────────────────────────")
    print(f"  X_train shape : {X_train.shape}")
    print(f"  X_val shape   : {X_val.shape}")
    print(f"  Train label balance  : {y_train.mean():.3f} (home win rate)")
    print(f"  Val label balance    : {y_val.mean():.3f} (home win rate)")

    val_post = X_val[X_val["season_type"] == "postseason"]
    print(f"  Val postseason rows  : {len(val_post)}  (expect 121)")
    if len(val_post) != 121:
        print(f"  WARNING: expected 121 postseason val rows, got {len(val_post)}")
        print(f"  Check for train/val overlap in build_splits()")

    train_post_seasons = set(
        X_train[X_train["season_type"] == "postseason"]["season"].unique()
    )
    val_post_seasons = set(val_post["season"].unique()) if len(val_post) > 0 else set()
    overlap = train_post_seasons & val_post_seasons
    if overlap:
        print(f"  WARNING: postseason overlap between train and val: {overlap}")
    else:
        print(f"  Train/val postseason overlap: NONE (clean split)")

    train_post = X_train[X_train["season_type"] == "postseason"]
    if "round_num" in train_post.columns:
        print(f"\n  Round distribution (train postseason):")
        print(train_post["round_num"].value_counts().sort_index().to_string())

    non_numeric = [c for c in X_train.columns if X_train[c].dtype == object]
    print(f"  Non-numeric columns  : {non_numeric}  (expect [])")

    missing_train = X_train.isnull().mean().sort_values(ascending=False).head(5)
    print(f"\n  Top missing (train):\n{missing_train.to_string()}")

    missing_val = X_val.isnull().mean().sort_values(ascending=False).head(5)
    print(f"\n  Top missing (val):\n{missing_val.to_string()}")

    print("\nDone.")
    return X_train, y_train, X_val, y_val


if __name__ == "__main__":
    force_refetch = "--force" in sys.argv
    main(force=force_refetch)
