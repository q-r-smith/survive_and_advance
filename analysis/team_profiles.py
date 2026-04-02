# analysis/team_profiles.py
"""
Build style-of-play profiles for tournament teams and cluster them.

Style features (from team shooting data):
  rim_attempt_pct      — share of tracked shots at the rim (layups + dunks)
  three_attempt_pct    — share of tracked shots that are 3-point jumpers
  midrange_attempt_pct — share of tracked shots that are 2-point jumpers
  rim_efficiency       — (layup + dunk makes) / (layup + dunk attempts)
  three_pct            — 3-point field goal percentage
  three_assisted_pct   — share of made 3s that were assisted (system vs. created)
  ft_rate              — free throw rate (aggressiveness / style of attack)

Usage:
    python analysis/team_profiles.py --seasons 2016 2017 2018 2019 2021 2022 2023 2024 2025 \
        --k 7 --tournament-only

    # Sweep k values to find the best silhouette score:
    python analysis/team_profiles.py --seasons 2022 2023 2024 2025 --sweep-k

Output:
    analysis/cache/cluster_labels.csv    — team-season cluster assignments
    analysis/cache/cluster_centroids.csv — centroid feature values per cluster
    analysis/cache/style_features.csv    — raw style features (pre-clustering)
"""

import argparse
import os
import sys
import warnings

import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
from sklearn.preprocessing import StandardScaler

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

RAW_DIR    = os.path.join(os.path.dirname(__file__), "..", "data", "cache", "raw")
CACHE_DIR  = os.path.join(os.path.dirname(__file__), "..", "data", "cache")
OUT_DIR    = os.path.join(os.path.dirname(__file__), "cache")

STYLE_COLS = [
    "rim_attempt_pct",      # how they score — interior
    "three_attempt_pct",    # how they score — perimeter
    "midrange_attempt_pct", # how they score — midrange
    "rim_efficiency",       # can they finish at the rim
    "three_pct",            # three-point quality
    "three_assisted_pct",   # system threes vs. created threes
    "ft_rate",              # aggressiveness / style of attack
]


# ── Data loading ──────────────────────────────────────────────────────────────

def load_shooting(seasons: list[int]) -> pd.DataFrame:
    frames = []
    for s in seasons:
        path = os.path.join(RAW_DIR, f"team_shooting_{s}.csv")
        if not os.path.exists(path):
            print(f"  WARNING: team_shooting_{s}.csv not found — run pull_style_data.py first")
            continue
        df = pd.read_csv(path)
        if "season" not in df.columns:
            df["season"] = s
        frames.append(df)
    return pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()



def load_tournament_teams(seasons: list[int]) -> set[tuple]:
    """Return set of (team, season) tuples that appeared in postseason games."""
    tournament_teams = set()
    for s in seasons:
        path = os.path.join(RAW_DIR, f"games_{s}.csv")
        if not os.path.exists(path):
            continue
        games = pd.read_csv(path)
        post = games[games["seasonType"] == "postseason"]
        for team in pd.concat([post["homeTeam"], post["awayTeam"]]).unique():
            tournament_teams.add((team, s))
    return tournament_teams


# ── Style feature computation ─────────────────────────────────────────────────

def compute_shooting_features(shooting_df: pd.DataFrame) -> pd.DataFrame:
    """Compute per-team shooting style features from the team shooting table."""
    df = shooting_df.copy()

    # ── Attempt percentages ───────────────────────────────────────────────────
    # attemptsBreakdown_* columns are already percentages summing to ~100.
    # Divide by 100 to normalize to 0-1 proportions.
    df["rim_attempt_pct"] = (
        df.get("attemptsBreakdown_layups", pd.Series(0, index=df.index)) +
        df.get("attemptsBreakdown_dunks",  pd.Series(0, index=df.index))
    ) / 100.0

    df["three_attempt_pct"] = (
        df.get("attemptsBreakdown_threePointJumpers",
               pd.Series(0, index=df.index))
    ) / 100.0

    df["midrange_attempt_pct"] = (
        df.get("attemptsBreakdown_twoPointJumpers",
               pd.Series(0, index=df.index))
    ) / 100.0

    # ── Rim efficiency ────────────────────────────────────────────────────────
    # Use raw attempt counts as denominator (not the breakdown percentages).
    # Result should be ~0.55-0.70 (proportion of rim attempts that are made).
    layup_made    = pd.to_numeric(df.get("layups_made",     pd.Series(dtype=float)), errors="coerce")
    dunk_made     = pd.to_numeric(df.get("dunks_made",      pd.Series(dtype=float)), errors="coerce")
    layup_att_raw = pd.to_numeric(df.get("layups_attempted",  pd.Series(dtype=float)), errors="coerce")
    dunk_att_raw  = pd.to_numeric(df.get("dunks_attempted",   pd.Series(dtype=float)), errors="coerce")
    rim_att_raw   = (layup_att_raw + dunk_att_raw).replace(0, np.nan)
    df["rim_efficiency"] = (layup_made + dunk_made) / rim_att_raw

    # ── Percentage features — divide by 100 to normalize to 0-1 ─────────────
    df["three_pct"]          = pd.to_numeric(
        df.get("threePointJumpers_pct",         np.nan), errors="coerce") / 100.0
    df["three_assisted_pct"] = pd.to_numeric(
        df.get("threePointJumpers_assistedPct", np.nan), errors="coerce") / 100.0
    df["ft_rate"]            = pd.to_numeric(
        df.get("freeThrowRate",                 np.nan), errors="coerce") / 100.0

    return df[["team", "season"] + [c for c in STYLE_COLS if c in df.columns]]



def build_style_features(seasons: list[int], tournament_only: bool) -> pd.DataFrame:
    """Load raw data and return a style-feature DataFrame indexed by (team, season)."""
    print("Loading team shooting data ...")
    shooting_df = load_shooting(seasons)
    print(f"  {len(shooting_df):,} shooting rows")

    if shooting_df.empty:
        print("ERROR: No shooting data found. Run pull_style_data.py first.")
        sys.exit(1)

    style = compute_shooting_features(shooting_df)

    # Exclude seasons with known data anomalies
    EXCLUDE_SEASONS = {2021}  # COVID bubble — atypical conditions and shot tracking issues
    style = style[~style["season"].isin(EXCLUDE_SEASONS)].reset_index(drop=True)
    if EXCLUDE_SEASONS:
        print(f"  Excluded seasons: {EXCLUDE_SEASONS}")

    if tournament_only:
        print("Filtering to tournament teams only ...")
        tourn_teams = load_tournament_teams(seasons)
        mask = style.apply(lambda r: (r["team"], r["season"]) in tourn_teams, axis=1)
        style = style[mask].reset_index(drop=True)
        print(f"  {len(style):,} team-seasons after filtering")

    return style


# ── Clustering ────────────────────────────────────────────────────────────────

def cluster_teams(style_df: pd.DataFrame, k: int, seed: int = 42) -> pd.DataFrame:
    """
    Fit K-means on available STYLE_COLS (drops rows with too many NaNs).
    Returns style_df with cluster_label and dist_to_centroid columns added.
    """
    available = [c for c in STYLE_COLS if c in style_df.columns]
    df = style_df[["team", "season"] + available].copy()

    # Drop rows where more than half the style cols are NaN
    nan_threshold = len(available) // 2
    df = df.dropna(thresh=len(available) + 2 - nan_threshold)
    df[available] = df[available].fillna(df[available].median())

    if len(df) < k * 2:
        print(f"WARNING: only {len(df)} rows — too few for k={k}. Skipping clustering.")
        return style_df

    scaler = StandardScaler()
    X = scaler.fit_transform(df[available])

    km = KMeans(n_clusters=k, random_state=seed, n_init=20)
    labels = km.fit_predict(X)
    dists  = km.transform(X).min(axis=1)

    df["cluster_label"]    = labels
    df["dist_to_centroid"] = dists

    # Silhouette score
    if len(df) > k:
        sil = silhouette_score(X, labels)
        print(f"  k={k}  silhouette={sil:.4f}  n={len(df)}")
    else:
        print(f"  k={k}  n={len(df)}  (too few for silhouette)")

    # Centroid table (in original feature space)
    centroid_df = pd.DataFrame(
        scaler.inverse_transform(km.cluster_centers_),
        columns=available,
    )
    centroid_df.index.name = "cluster_label"
    centroid_df = centroid_df.reset_index()

    return df, centroid_df, scaler, km


def sweep_k(style_df: pd.DataFrame, k_range=range(3, 12)) -> dict:
    """Try multiple k values and return silhouette scores."""
    available = [c for c in STYLE_COLS if c in style_df.columns]
    df = style_df[["team", "season"] + available].copy()
    nan_threshold = len(available) // 2
    df = df.dropna(thresh=len(available) + 2 - nan_threshold)
    df[available] = df[available].fillna(df[available].median())

    scaler = StandardScaler()
    X = scaler.fit_transform(df[available])

    results = {}
    print(f"\n  Silhouette sweep (n={len(df)} team-seasons):")
    for k in k_range:
        if len(df) < k * 2:
            continue
        km = KMeans(n_clusters=k, random_state=42, n_init=20)
        labels = km.fit_predict(X)
        sil = silhouette_score(X, labels)
        results[k] = sil
        bar = "█" * int(sil * 60)
        print(f"    k={k:2d}  sil={sil:.4f}  {bar}")

    best_k = max(results, key=results.get)
    print(f"\n  Best k = {best_k}  (silhouette = {results[best_k]:.4f})")
    return results


def describe_clusters(labeled_df: pd.DataFrame, centroid_df: pd.DataFrame) -> None:
    """Print interpretable cluster summaries."""
    available = [c for c in STYLE_COLS if c in labeled_df.columns]
    print("\n── Cluster profiles ───────────────────────────────────────────────")
    for _, row in centroid_df.iterrows():
        c = int(row["cluster_label"])
        members = labeled_df[labeled_df["cluster_label"] == c]
        print(f"\n  Cluster {c}  (n={len(members)})")
        for feat in available:
            if feat in row.index:
                print(f"    {feat:<20s}  {row[feat]:.3f}")

    # f4_rate if we have 2025 or any year where we know Final Four outcomes
    # (informational only — caller can compute this separately)


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="Build team style profiles and cluster them.")
    parser.add_argument("--seasons", nargs="+", type=int, required=True,
                        help="Seasons to include (e.g. 2022 2023 2024 2025)")
    parser.add_argument("--k", type=int, default=7,
                        help="Number of clusters (default: 7)")
    parser.add_argument("--tournament-only", action="store_true",
                        help="Restrict to teams that appeared in postseason games")
    parser.add_argument("--sweep-k", action="store_true",
                        help="Try k=3..11 and report silhouette scores, then exit")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    os.makedirs(OUT_DIR, exist_ok=True)

    # Build style features
    style_df = build_style_features(args.seasons, args.tournament_only)

    # Save raw style features
    style_path = os.path.join(OUT_DIR, "style_features.csv")
    style_df.to_csv(style_path, index=False)
    print(f"\n  saved style features → {style_path}  ({len(style_df):,} rows)")

    available = [c for c in STYLE_COLS if c in style_df.columns]
    nan_counts = style_df[available].isnull().sum()
    print(f"\n  NaN counts per style feature:")
    for col in available:
        pct = nan_counts[col] / len(style_df) * 100
        print(f"    {col:<22s}  {nan_counts[col]:4d}  ({pct:.1f}%)")

    if args.sweep_k:
        sweep_k(style_df)
        return

    # Cluster
    print(f"\n── Clustering (k={args.k}) ─────────────────────────────────────────")
    result = cluster_teams(style_df, k=args.k, seed=args.seed)
    if not isinstance(result, tuple):
        return  # clustering was skipped

    labeled_df, centroid_df, scaler, km = result

    # Save outputs
    labels_path    = os.path.join(OUT_DIR, "cluster_labels.csv")
    centroids_path = os.path.join(OUT_DIR, "cluster_centroids.csv")
    labeled_df.to_csv(labels_path, index=False)
    centroid_df.to_csv(centroids_path, index=False)
    print(f"\n  saved cluster_labels    → {labels_path}")
    print(f"  saved cluster_centroids → {centroids_path}")

    # Describe clusters
    describe_clusters(labeled_df, centroid_df)

    print(f"""
── Validation checklist ──────────────────────────────────────────────────
  1. Silhouette > 0.15?         Check above
  2. Any cluster with f4_rate > 2x avg?  (compute manually from labels)
  3. Cluster radar charts look distinct?  (plot centroid_df)
  4. Named archetypes interpretable?      (check three_pt_rate / rim_rate)

If all four pass:
  cp data/cache/features_by_season.pkl data/cache/features_by_season_backup.pkl
  python analysis/cluster_features.py --all-seasons
  python train.py --all
──────────────────────────────────────────────────────────────────────────
""")


if __name__ == "__main__":
    main()
