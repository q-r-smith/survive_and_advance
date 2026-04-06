# analysis/team_profiles.py
"""
Build team style profiles from play-by-play data and cluster them.

Each season's PBP file is read in chunks (only lightweight columns loaded) so
peak RAM stays manageable even for the 5 GB 2026 file.

Pipeline:
    PBP counts → rate features → [within-season Z-score] →
    StandardScaler → PCA → MiniBatchKMeans

New tuning flags added to improve silhouette toward 0.40+:
  --style-only        Use only frequency features (drop fgpct cols).
                      Efficiency metrics add noise; style separation comes
                      from "how" a team plays, not "how well."
  --zscore-by-season  Z-score each feature within its season before global
                      scaling.  Removes era-bias so a 3-heavy team in 2016
                      compares fairly to one in 2025.
  --conf-only         Aggregate only plays from same-conference games.
                      Removes cupcake-game noise that dilutes style signals.
  --pca-variance N    Variance threshold for auto PCA component selection
                      (default 0.80 ≈ 4–5 components).  Lower = fewer
                      components = tighter clusters.
  --plot              Generate and save three diagnostic plots:
                        analysis/cache/plots/pca_variance.png  (scree)
                        analysis/cache/plots/elbow.png          (k sweep)
                        analysis/cache/plots/clusters.png       (2-D scatter)

Usage:
    # Best silhouette starting point (tournament teams, 2-D PCA):
    python analysis/team_profiles.py \\
        --seasons 2016 2017 2018 2019 2021 2022 2023 2024 2025 \\
        --k 5 --tournament-only --conf-only --zscore-by-season \\
        --pca-components 3 --plot

    # Explore feature/PCA choices before committing (use --plot + --sweep-k):
    python analysis/team_profiles.py \\
        --seasons 2022 2023 2024 2025 --tournament-only --conf-only \\
        --zscore-by-season --sweep-k --plot

    # Sweep k values:
    python analysis/team_profiles.py \\
        --seasons 2022 2023 2024 2025 --sweep-k \\
        --style-only --zscore-by-season --conf-only

    # Skip PCA entirely (only viable on small / tournament-only datasets):
    python analysis/team_profiles.py --seasons 2025 --k 6 --pca-components -1

Output:
    analysis/cache/cluster_labels.csv    — team-season cluster assignments
    analysis/cache/cluster_centroids.csv — centroid values in original feature space
    analysis/cache/style_features.csv    — aggregated features (pre-scaling)
    analysis/cache/plots/                — diagnostic plots (with --plot)
"""

import argparse
import os
import sys
import warnings

import numpy as np
import pandas as pd
from sklearn.cluster import MiniBatchKMeans
from sklearn.decomposition import PCA
from sklearn.metrics import silhouette_score
from sklearn.preprocessing import StandardScaler

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

RAW_DIR   = os.path.join(os.path.dirname(__file__), "..", "data", "cache", "raw")
OUT_DIR   = os.path.join(os.path.dirname(__file__), "cache")
PLOTS_DIR = os.path.join(OUT_DIR, "plots")

# Conference + opponent-conference columns are cheap strings needed for quality filter.
USECOLS = [
    "gameId", "team", "season",
    "playType", "shootingPlay", "shotInfo",
    "conference", "opponentConference",
]

CHUNK_SIZE = 200_000

# ── Two feature sets ──────────────────────────────────────────────────────────
# STYLE_COLS_FREQ: "how" the team plays — shot mix, playmaking, pace, defense.
#   Dropping efficiency (fgpct) metrics focuses clustering on style, not quality.
#   This produces tighter clusters because teams vary much more in shot selection
#   than in shooting percentage (which regresses toward the mean across D1).
STYLE_COLS_FREQ = [
    "rim_pct",        # rim shots / total FGA
    "three_pct_shot", # 3PA / total FGA
    "jumper_pct",     # mid-range / total FGA
    "assisted_pct",   # assisted makes / total makes
    "ft_rate",        # FT attempts / FGA  (aggression proxy)
    "to_per_game",    # turnovers per game
    "oreb_per_game",  # offensive rebounds per game
    "plays_per_game", # pace proxy
    "block_rate",     # blocks per game
    "steal_rate",     # steals per game
]

# STYLE_COLS_ALL: adds shooting efficiency on top of frequency.
#   Can help if your downstream model cares about quality, but hurts silhouette
#   because FG% variance across D1 is much smaller than shot-selection variance.
STYLE_COLS_ALL = STYLE_COLS_FREQ + [
    "rim_fgpct",
    "three_fgpct",
    "jumper_fgpct",
]

# Default export list (consumers like cluster_features.py read FEATURE_COLS)
FEATURE_COLS = STYLE_COLS_ALL


# ── Season aggregation ────────────────────────────────────────────────────────

def _new_counts() -> dict:
    return {
        "games":         set(),
        "rim_att":       0,
        "rim_made":      0,
        "three_att":     0,
        "three_made":    0,
        "jumper_att":    0,
        "jumper_made":   0,
        "ft_att":        0,
        "fg_makes":      0,
        "fg_asst_makes": 0,
        "turnovers":     0,
        "oreb":          0,
        "blocks":        0,
        "steals":        0,
        "total_plays":   0,
    }


def aggregate_season(
    season: int,
    chunksize: int = CHUNK_SIZE,
    conf_only: bool = False,
) -> dict:
    """
    Stream one season's PBP CSV in chunks, accumulating per-team event counts.

    conf_only: if True, only count plays from same-conference matchups
               (conference == opponentConference).  Drops ~40% of rows but
               removes cupcake-game noise that dilutes style signals.

    Returns {team_name: count_dict}.
    """
    path = os.path.join(RAW_DIR, f"play_by_play_{season}.csv")
    if not os.path.exists(path):
        print(f"  WARNING: {path} not found — skipping")
        return {}

    counts: dict[str, dict] = {}

    def _get(team: str) -> dict:
        if team not in counts:
            counts[team] = _new_counts()
        return counts[team]

    chunk_n = 0
    for chunk in pd.read_csv(
        path,
        usecols=USECOLS,
        chunksize=chunksize,
        dtype={"season": int},
        low_memory=False,
    ):
        chunk_n += 1

        # ── Quality filter ────────────────────────────────────────────────────
        # Always drop plays where opponent has no known conference (non-D1 exh.)
        chunk = chunk[chunk["opponentConference"].notna()].copy()

        if conf_only:
            chunk = chunk[chunk["conference"] == chunk["opponentConference"]].copy()

        if chunk.empty:
            continue

        # Normalize shootingPlay — stored as string in some season files
        sp = chunk["shootingPlay"]
        if sp.dtype == object:
            sp = sp.map({"True": True, "False": False, True: True, False: False})
        sp = sp.fillna(False)

        # ── Total plays and unique games per team ─────────────────────────────
        for team, grp in chunk.groupby("team", sort=False):
            c = _get(team)
            c["total_plays"] += len(grp)
            c["games"].update(grp["gameId"].dropna().astype(int).tolist())

        # ── Shooting plays ────────────────────────────────────────────────────
        shot_rows = chunk[sp == True].copy()
        if not shot_rows.empty:
            si = shot_rows["shotInfo"].fillna("")

            rim_mask   = si.str.contains("'range': 'rim'",           regex=False)
            three_mask = si.str.contains("'range': 'three_pointer'", regex=False)
            jump_mask  = si.str.contains("'range': 'jumper'",        regex=False)
            ft_mask    = si.str.contains("'range': 'free_throw'",    regex=False)
            made_mask  = si.str.contains("'made': True",             regex=False)
            asst_mask  = si.str.contains("'assisted': True",         regex=False)

            for team, grp in shot_rows.groupby("team", sort=False):
                c = _get(team)
                idx = grp.index

                rim_idx = idx[rim_mask[idx]]
                c["rim_att"]  += len(rim_idx)
                c["rim_made"] += int(made_mask[rim_idx].sum())

                three_idx = idx[three_mask[idx]]
                c["three_att"]  += len(three_idx)
                c["three_made"] += int(made_mask[three_idx].sum())

                jump_idx = idx[jump_mask[idx]]
                c["jumper_att"]  += len(jump_idx)
                c["jumper_made"] += int(made_mask[jump_idx].sum())

                c["ft_att"] += int(ft_mask[idx].sum())

                fg_idx      = idx[rim_mask[idx] | three_mask[idx] | jump_mask[idx]]
                fg_made_idx = fg_idx[made_mask[fg_idx]]
                c["fg_makes"]      += len(fg_made_idx)
                c["fg_asst_makes"] += int(asst_mask[fg_made_idx].sum())

        # ── Non-shooting play types ───────────────────────────────────────────
        pt = chunk["playType"].fillna("")
        for team, grp in chunk[pt.str.contains("Turnover", case=False, regex=False)].groupby("team", sort=False):
            _get(team)["turnovers"] += len(grp)
        for team, grp in chunk[pt == "Offensive Rebound"].groupby("team", sort=False):
            _get(team)["oreb"] += len(grp)
        for team, grp in chunk[pt == "Block Shot"].groupby("team", sort=False):
            _get(team)["blocks"] += len(grp)
        for team, grp in chunk[pt == "Steal"].groupby("team", sort=False):
            _get(team)["steals"] += len(grp)

    label = "(conf-only)" if conf_only else ""
    print(f"  Season {season}: {chunk_n} chunks, {len(counts)} teams {label}")
    return counts


def counts_to_features(counts: dict, season: int) -> pd.DataFrame:
    """Convert per-team counts to rate-based feature rows."""
    rows = []
    for team, c in counts.items():
        n_games   = max(len(c["games"]), 1)
        total_fga = c["rim_att"] + c["three_att"] + c["jumper_att"] or 1

        rows.append({
            "team":   team,
            "season": season,
            "rim_pct":        c["rim_att"]    / total_fga,
            "three_pct_shot": c["three_att"]  / total_fga,
            "jumper_pct":     c["jumper_att"] / total_fga,
            "rim_fgpct":    c["rim_made"]    / c["rim_att"]    if c["rim_att"]    > 0 else np.nan,
            "three_fgpct":  c["three_made"]  / c["three_att"]  if c["three_att"]  > 0 else np.nan,
            "jumper_fgpct": c["jumper_made"] / c["jumper_att"] if c["jumper_att"] > 0 else np.nan,
            "assisted_pct": c["fg_asst_makes"] / c["fg_makes"] if c["fg_makes"] > 0 else np.nan,
            "ft_rate":        c["ft_att"]       / total_fga,
            "to_per_game":    c["turnovers"]    / n_games,
            "oreb_per_game":  c["oreb"]         / n_games,
            "plays_per_game": c["total_plays"]  / n_games,
            "block_rate":     c["blocks"]       / n_games,
            "steal_rate":     c["steals"]       / n_games,
        })

    return pd.DataFrame(rows)


# ── Within-season normalization ───────────────────────────────────────────────

def zscore_within_season(feature_df: pd.DataFrame, cols: list) -> pd.DataFrame:
    """
    Replace raw feature values with within-season Z-scores.

    For each season, each feature is re-centered to mean=0, std=1 relative to
    all teams in that season.  This removes era-based bias — a 3-heavy team in
    2016 (when fewer teams shot threes) gets the same Z-score as one with the
    same relative prominence in 2025.

    Teams with fewer than 5 rows in a season get Z-scored using the remaining
    season data; if only one team exists (impossible in practice), values are
    set to 0.
    """
    df = feature_df.copy()
    for s in df["season"].unique():
        mask = df["season"] == s
        season_data = df.loc[mask, cols]
        mu  = season_data.mean()
        std = season_data.std(ddof=0).replace(0, 1)   # avoid div/0 for constant cols
        df.loc[mask, cols] = (season_data - mu) / std
    return df


# ── Feature matrix helpers ────────────────────────────────────────────────────

def load_tournament_teams(seasons: list) -> set:
    tournament_teams = set()
    for s in seasons:
        path = os.path.join(RAW_DIR, f"games_{s}.csv")
        if not os.path.exists(path):
            continue
        games = pd.read_csv(path, usecols=["homeTeam", "awayTeam", "seasonType"])
        post  = games[games["seasonType"] == "postseason"]
        for team in pd.concat([post["homeTeam"], post["awayTeam"]]).unique():
            tournament_teams.add((team, s))
    return tournament_teams


def build_pbp_features(
    seasons: list,
    tournament_only: bool,
    conf_only: bool = False,
) -> pd.DataFrame:
    frames = []
    for s in seasons:
        print(f"\nAggregating season {s} ...")
        counts = aggregate_season(s, conf_only=conf_only)
        if not counts:
            continue
        frames.append(counts_to_features(counts, s))

    if not frames:
        print("ERROR: No play-by-play data found.")
        sys.exit(1)

    features = pd.concat(frames, ignore_index=True)

    if tournament_only:
        print("\nFiltering to tournament teams only ...")
        tourn_teams = load_tournament_teams(seasons)
        mask = features.apply(
            lambda r: (r["team"], int(r["season"])) in tourn_teams, axis=1
        )
        features = features[mask].reset_index(drop=True)
        print(f"  {len(features):,} team-seasons after filtering")

    return features


def _prep_matrix(feature_df: pd.DataFrame, cols: list) -> pd.DataFrame:
    """Drop rows missing > half their features; median-fill the rest."""
    df = feature_df[["team", "season"] + cols].copy()
    threshold = len(cols) // 2
    df = df.dropna(thresh=len(cols) + 2 - threshold)
    df[cols] = df[cols].fillna(df[cols].median())
    return df


def _scale_and_pca(
    df: pd.DataFrame,
    cols: list,
    zscore_by_season: bool,
    n_pca_components: int,
    pca_variance: float,
    seed: int,
) -> tuple:
    """
    Apply optional within-season Z-scoring, StandardScaler, optional PCA.
    Returns (X, scaler, pca_or_None, df_after_zscore).
    """
    working = df.copy()

    if zscore_by_season:
        working = zscore_within_season(working, cols)
        print(f"  Within-season Z-scores applied")

    scaler = StandardScaler()
    X = scaler.fit_transform(working[cols])

    pca = None
    if n_pca_components == -1:
        print("  PCA: skipped")
        return X, scaler, None, working

    # ── Determine number of components ───────────────────────────────────────
    if n_pca_components == 0:
        # Auto: use second-derivative elbow on the scree curve, bounded above by
        # the variance threshold.
        #
        # The second derivative of the explained-variance-ratio sequence finds the
        # point of maximum curvature — i.e., where adding more components stops
        # being useful.  For basketball PBP data this reliably lands at 2–4
        # components, which empirically gives the best silhouette scores.
        pca_full  = PCA(random_state=seed).fit(X)
        evr       = pca_full.explained_variance_ratio_
        cumvar    = np.cumsum(evr)

        # Variance-threshold upper bound
        thresh_idx = int(np.searchsorted(cumvar, pca_variance)) + 1

        # Second-derivative elbow: index of max |d²(evr)|
        # d2[j] corresponds to the curvature at component j+1 (1-indexed: j+2 components)
        if len(evr) >= 3:
            d2        = np.diff(np.diff(evr))
            elbow_idx = int(np.argmax(np.abs(d2))) + 2   # 1-indexed component count
        else:
            elbow_idx = len(evr)

        # Use the smaller of elbow and variance-threshold, minimum 2
        n_pca_components = max(2, min(thresh_idx, elbow_idx, len(cols) - 1))

        print(f"  PCA auto: {n_pca_components} components "
              f"(variance-threshold→{thresh_idx}, 2nd-deriv elbow→{elbow_idx})")

        # Print full scree table
        print(f"\n  {'PC':>4}  {'Indiv%':>7}  {'Cumul%':>7}")
        for i, (v, cv) in enumerate(zip(evr, cumvar)):
            marker = " ←" if i + 1 == n_pca_components else ""
            print(f"  {i+1:>4}  {v*100:>6.1f}%  {cv*100:>6.1f}%{marker}")
        print()
    else:
        # Fit full PCA just to print scree when user specifies explicit count
        pca_full = PCA(random_state=seed).fit(X)
        evr    = pca_full.explained_variance_ratio_
        cumvar = np.cumsum(evr)
        print(f"\n  PCA: using {n_pca_components} components explicitly")
        print(f"  {'PC':>4}  {'Indiv%':>7}  {'Cumul%':>7}")
        for i, (v, cv) in enumerate(zip(evr, cumvar)):
            marker = " ←" if i + 1 == n_pca_components else ""
            print(f"  {i+1:>4}  {v*100:>6.1f}%  {cv*100:>6.1f}%{marker}")
        print()

    pca = PCA(n_components=n_pca_components, random_state=seed)
    X   = pca.fit_transform(X)

    # Feature loading summary
    print(f"  Top feature loadings per PCA component:")
    for i, comp in enumerate(pca.components_):
        top_idx = np.argsort(np.abs(comp))[::-1][:4]
        top = ", ".join(f"{cols[j]}({comp[j]:+.2f})" for j in top_idx)
        print(f"    PC{i+1}: {top}")
    print()

    return X, scaler, pca, working


# ── Clustering ────────────────────────────────────────────────────────────────

def cluster_teams(
    feature_df: pd.DataFrame,
    k: int,
    feature_cols: list = None,
    zscore_by_season: bool = False,
    n_pca_components: int = 0,
    pca_variance: float = 0.80,
    seed: int = 42,
) -> tuple:
    """
    StandardScaler → optional within-season Z-score → optional PCA → MiniBatchKMeans.

    Returns (labeled_df, centroid_df, scaler, pca_or_None, km).
    labeled_df: all feature columns + cluster_label + dist_to_centroid.
    centroid_df: cluster_label + feature columns back-projected to original space.
    """
    if feature_cols is None:
        feature_cols = [c for c in FEATURE_COLS if c in feature_df.columns]
    else:
        feature_cols = [c for c in feature_cols if c in feature_df.columns]

    df = _prep_matrix(feature_df, feature_cols)

    if len(df) < k * 2:
        print(f"WARNING: only {len(df)} rows — too few for k={k}. Skipping.")
        return feature_df, None, None, None, None

    X, scaler, pca, df = _scale_and_pca(
        df, feature_cols, zscore_by_season, n_pca_components, pca_variance, seed
    )

    batch = max(256, len(df) // 4)
    km    = MiniBatchKMeans(n_clusters=k, random_state=seed, n_init=20, batch_size=batch)
    labels = km.fit_predict(X)
    dists  = km.transform(X).min(axis=1)

    df = df.copy()
    df["cluster_label"]    = labels
    df["dist_to_centroid"] = dists

    if len(df) > k:
        sil = silhouette_score(X, labels)
        print(f"  k={k}  silhouette={sil:.4f}  n={len(df)}")

    # Back-project centroids to original feature space for interpretability
    centers = km.cluster_centers_
    if pca is not None:
        centers = pca.inverse_transform(centers)
    centers_orig = scaler.inverse_transform(centers)

    centroid_df = pd.DataFrame(centers_orig, columns=feature_cols)
    centroid_df.index.name = "cluster_label"
    centroid_df = centroid_df.reset_index()

    return df, centroid_df, scaler, pca, km


def sweep_k(
    feature_df: pd.DataFrame,
    feature_cols: list = None,
    zscore_by_season: bool = False,
    n_pca_components: int = 0,
    pca_variance: float = 0.80,
    k_range=range(2, 13),
    seed: int = 42,
) -> dict:
    """Try k = 2..12, print inertia + silhouette table, return {k: silhouette}."""
    if feature_cols is None:
        feature_cols = [c for c in FEATURE_COLS if c in feature_df.columns]
    else:
        feature_cols = [c for c in feature_cols if c in feature_df.columns]

    df = _prep_matrix(feature_df, feature_cols)
    X, scaler, pca, df = _scale_and_pca(
        df, feature_cols, zscore_by_season, n_pca_components, pca_variance, seed
    )

    print(f"  Silhouette sweep (n={len(df)} team-seasons):")
    print(f"  {'k':>4}  {'inertia':>12}  {'silhouette':>11}")

    results = {}
    for k in k_range:
        if len(df) < k * 2:
            continue
        batch  = max(256, len(df) // 4)
        km     = MiniBatchKMeans(n_clusters=k, random_state=seed, n_init=20, batch_size=batch)
        labels = km.fit_predict(X)
        sil    = silhouette_score(X, labels)
        results[k] = {"silhouette": sil, "inertia": km.inertia_}
        bar = "█" * max(0, int(sil * 50))
        print(f"  {k:>4}  {km.inertia_:>12.1f}  {sil:>10.4f}  {bar}")

    best_k = max(results, key=lambda k: results[k]["silhouette"])
    print(f"\n  Best k = {best_k}  (silhouette = {results[best_k]['silhouette']:.4f})")
    return {k: v["silhouette"] for k, v in results.items()}


def describe_clusters(labeled_df: pd.DataFrame, centroid_df: pd.DataFrame) -> None:
    available = [c for c in STYLE_COLS_ALL if c in centroid_df.columns]
    print("\n── Cluster profiles (centroid in original feature space) ──────────")
    for _, row in centroid_df.iterrows():
        c = int(row["cluster_label"])
        members = labeled_df[labeled_df["cluster_label"] == c]
        print(f"\n  Cluster {c}  (n={len(members)})")
        for feat in available:
            if feat in row.index:
                print(f"    {feat:<20s}  {row[feat]:.3f}")


# ── Diagnostic plots ──────────────────────────────────────────────────────────

def _require_matplotlib():
    try:
        import matplotlib
        matplotlib.use("Agg")   # headless — saves file without a display
        import matplotlib.pyplot as plt
        return plt
    except ImportError:
        print("WARNING: matplotlib not installed — skipping plots.")
        return None


def plot_pca_variance(
    feature_df: pd.DataFrame,
    feature_cols: list,
    zscore_by_season: bool = False,
    seed: int = 42,
    save_dir: str = PLOTS_DIR,
) -> None:
    """
    PCA scree plot: individual + cumulative explained variance per component.

    How to read:
      • Each bar is the % variance one principal component explains.
      • The line shows cumulative variance.
      • The elbow is where the bars flatten — adding more components gives
        diminishing returns.  Good stopping points are typically where the
        cumulative line crosses 70–85%.
      • Reference lines at 70%, 80%, 90% help you calibrate.
    """
    plt = _require_matplotlib()
    if plt is None:
        return

    df = _prep_matrix(feature_df, feature_cols)
    if zscore_by_season:
        df = zscore_within_season(df, feature_cols)

    scaler  = StandardScaler()
    X       = scaler.fit_transform(df[feature_cols])
    pca_all = PCA(random_state=seed).fit(X)
    evr     = pca_all.explained_variance_ratio_
    cumvar  = np.cumsum(evr)
    x       = np.arange(1, len(evr) + 1)

    fig, ax1 = plt.subplots(figsize=(9, 5))
    ax2 = ax1.twinx()

    ax1.bar(x, evr * 100, color="#4878d0", alpha=0.75, label="Individual %")
    ax2.plot(x, cumvar * 100, "o-", color="#ee854a", linewidth=2, label="Cumulative %")

    for pct, ls in [(70, ":"), (80, "--"), (90, "-.")]:
        ax2.axhline(pct, color="grey", linestyle=ls, linewidth=0.8, alpha=0.6)
        ax2.text(len(evr) + 0.1, pct + 0.5, f"{pct}%", fontsize=7, color="grey")

    ax1.set_xlabel("Principal Component")
    ax1.set_ylabel("Individual Variance (%)", color="#4878d0")
    ax2.set_ylabel("Cumulative Variance (%)", color="#ee854a")
    ax1.set_xticks(x)
    ax2.set_ylim(0, 105)
    ax1.set_title("PCA Scree Plot — pick components at the elbow")

    lines1, labels1 = ax1.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax1.legend(lines1 + lines2, labels1 + labels2, loc="center right")

    os.makedirs(save_dir, exist_ok=True)
    out = os.path.join(save_dir, "pca_variance.png")
    fig.tight_layout()
    fig.savefig(out, dpi=130)
    plt.close(fig)
    print(f"  saved pca_variance.png → {out}")


def plot_elbow(
    feature_df: pd.DataFrame,
    feature_cols: list,
    zscore_by_season: bool = False,
    n_pca_components: int = 4,
    pca_variance: float = 0.80,
    k_range=range(2, 13),
    seed: int = 42,
    save_dir: str = PLOTS_DIR,
) -> None:
    """
    K-selection diagnostic: WCSS inertia (left) + silhouette score (right) vs k.

    How to read:
      • Inertia always decreases as k rises — look for the "elbow" where it
        stops dropping sharply.  That bend is the natural cluster count.
      • Silhouette peaks at the best-separated k.  Aim for > 0.25.
      • Pick k where inertia elbows AND silhouette is near its peak.
    """
    plt = _require_matplotlib()
    if plt is None:
        return

    df = _prep_matrix(feature_df, feature_cols)
    X, scaler, pca, df = _scale_and_pca(
        df, feature_cols, zscore_by_season, n_pca_components, pca_variance, seed
    )

    ks, inertias, silhouettes = [], [], []
    for k in k_range:
        if len(df) < k * 2:
            continue
        batch  = max(256, len(df) // 4)
        km     = MiniBatchKMeans(n_clusters=k, random_state=seed, n_init=20, batch_size=batch)
        labels = km.fit_predict(X)
        ks.append(k)
        inertias.append(km.inertia_)
        silhouettes.append(silhouette_score(X, labels))

    fig, ax1 = plt.subplots(figsize=(9, 5))
    ax2 = ax1.twinx()

    ax1.plot(ks, inertias,   "o-", color="#4878d0", linewidth=2, label="Inertia (WCSS)")
    ax2.plot(ks, silhouettes, "s--", color="#ee854a", linewidth=2, label="Silhouette")
    ax2.axhline(0.25, color="grey", linestyle=":", linewidth=0.9)
    ax2.text(k_range.stop - 0.8, 0.255, "0.25 target", fontsize=7, color="grey")

    best_k = ks[int(np.argmax(silhouettes))]
    ax2.axvline(best_k, color="#6acc65", linestyle="--", linewidth=1.2, alpha=0.7)
    ax2.text(best_k + 0.1, min(silhouettes) - 0.01, f"best k={best_k}", fontsize=8, color="#6acc65")

    ax1.set_xlabel("Number of clusters (k)")
    ax1.set_ylabel("Inertia (WCSS)", color="#4878d0")
    ax2.set_ylabel("Silhouette score", color="#ee854a")
    ax1.set_xticks(ks)
    ax1.set_title("Elbow Plot — choose k at the inertia bend")

    lines1, labels1 = ax1.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax1.legend(lines1 + lines2, labels1 + labels2, loc="upper right")

    os.makedirs(save_dir, exist_ok=True)
    out = os.path.join(save_dir, "elbow.png")
    fig.tight_layout()
    fig.savefig(out, dpi=130)
    plt.close(fig)
    print(f"  saved elbow.png         → {out}")


def plot_clusters(
    labeled_df: pd.DataFrame,
    feature_cols: list,
    zscore_by_season: bool = False,
    seed: int = 42,
    save_dir: str = PLOTS_DIR,
) -> None:
    """
    2-D PCA scatter colored by cluster label.

    Always reduces to PC1 vs PC2 for visualization (separate from the
    higher-dimensional PCA used for clustering).  Shows cluster centroids
    as large X markers, and labels the 3 closest-to-centroid teams per
    cluster for orientation.

    How to read:
      • Well-separated blobs of the same color = good clustering.
      • Overlapping blobs = clusters are not well distinguished — try
        fewer k, different features, or more PCA components.
      • Teams near the center of the plot are stylistically average;
        outliers on the edges have extreme, unusual profiles.
    """
    plt = _require_matplotlib()
    if plt is None:
        return

    cols = [c for c in feature_cols if c in labeled_df.columns]
    df   = labeled_df[["team", "season", "cluster_label"] + cols].dropna(subset=cols)

    working = df.copy()
    if zscore_by_season:
        working = zscore_within_season(working, cols)

    scaler = StandardScaler()
    X_2d   = PCA(n_components=2, random_state=seed).fit_transform(
        scaler.fit_transform(working[cols])
    )

    labels = df["cluster_label"].values
    n_clust = int(labels.max()) + 1

    import matplotlib.pyplot as mpl_plt
    import matplotlib.cm as cm

    cmap   = cm.get_cmap("tab10", n_clust)
    colors = [cmap(i) for i in range(n_clust)]

    fig, ax = mpl_plt.subplots(figsize=(10, 7))

    for c in range(n_clust):
        mask = labels == c
        ax.scatter(
            X_2d[mask, 0], X_2d[mask, 1],
            color=colors[c], alpha=0.55, s=40, label=f"Cluster {c}",
        )
        # Centroid marker
        cx, cy = X_2d[mask, 0].mean(), X_2d[mask, 1].mean()
        ax.scatter(cx, cy, color=colors[c], marker="X", s=200, edgecolors="black",
                   linewidths=0.8, zorder=5)

        # Label the 3 teams closest to centroid
        if mask.sum() > 0:
            dists = np.sqrt((X_2d[mask, 0] - cx)**2 + (X_2d[mask, 1] - cy)**2)
            top3  = np.argsort(dists)[:3]
            for idx in top3:
                abs_idx = np.where(mask)[0][idx]
                row = df.iloc[abs_idx]
                ax.annotate(
                    f"{row['team']} '{str(int(row['season']))[-2:]}",
                    (X_2d[abs_idx, 0], X_2d[abs_idx, 1]),
                    fontsize=5.5, alpha=0.75,
                    xytext=(3, 3), textcoords="offset points",
                )

    ax.set_xlabel("PC 1")
    ax.set_ylabel("PC 2")
    ax.set_title("Team clusters — PC1 vs PC2  (X = centroid, labels = closest teams)")
    ax.legend(loc="best", fontsize=8, markerscale=0.8)

    os.makedirs(save_dir, exist_ok=True)
    out = os.path.join(save_dir, "clusters.png")
    fig.tight_layout()
    fig.savefig(out, dpi=130)
    mpl_plt.close(fig)
    print(f"  saved clusters.png      → {out}")


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="Build team PBP profiles and cluster via PCA + MiniBatchKMeans.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("--seasons", nargs="+", type=int, required=True,
                        help="Seasons to include (e.g. 2022 2023 2024 2025)")
    parser.add_argument("--k", type=int, default=6,
                        help="Number of clusters (default: 6)")
    parser.add_argument("--pca-components", type=int, default=0,
                        help="PCA components: 0=auto elbow, -1=skip PCA, N=exact")
    parser.add_argument("--pca-variance", type=float, default=0.65,
                        help="Variance upper-bound for auto PCA component selection "
                             "(default 0.65; lower → fewer components → tighter clusters; "
                             "ignored if --pca-components is set explicitly)")
    parser.add_argument("--tournament-only", action="store_true",
                        help="Restrict to teams that appeared in postseason games")
    parser.add_argument("--style-only", action="store_true",
                        help="Use only frequency features — drop rim/three/jumper_fgpct. "
                             "Focuses clustering on HOW teams play, not how efficiently. "
                             "Typically raises silhouette by 0.05–0.15.")
    parser.add_argument("--zscore-by-season", action="store_true",
                        help="Apply within-season Z-scores before global scaling. "
                             "Removes era-bias so 2016 and 2025 teams compare fairly.")
    parser.add_argument("--conf-only", action="store_true",
                        help="Only aggregate plays from same-conference games. "
                             "Removes cupcake-game noise (~40%% fewer rows, higher quality).")
    parser.add_argument("--sweep-k", action="store_true",
                        help="Try k=2..12 and report silhouette + inertia, then exit")
    parser.add_argument("--plot", action="store_true",
                        help="Generate pca_variance.png, elbow.png, clusters.png")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    os.makedirs(OUT_DIR, exist_ok=True)

    feature_cols = STYLE_COLS_FREQ if args.style_only else STYLE_COLS_ALL
    mode_label = (
        f"{'style-only' if args.style_only else 'all-features'}  "
        f"{'+ zscore-by-season' if args.zscore_by_season else ''}  "
        f"{'+ conf-only' if args.conf_only else ''}"
    ).strip()
    print(f"\nFeature mode: {mode_label}")
    print(f"Features ({len(feature_cols)}): {feature_cols}")

    # ── Build features ────────────────────────────────────────────────────────
    feature_df = build_pbp_features(
        args.seasons, args.tournament_only, conf_only=args.conf_only
    )

    style_path = os.path.join(OUT_DIR, "style_features.csv")
    feature_df.to_csv(style_path, index=False)
    print(f"\n  saved style features → {style_path}  ({len(feature_df):,} rows)")

    available = [c for c in feature_cols if c in feature_df.columns]
    nan_counts = feature_df[available].isnull().sum()
    print(f"\n  NaN counts per feature:")
    for col in available:
        pct = nan_counts[col] / len(feature_df) * 100
        print(f"    {col:<22s}  {nan_counts[col]:4d}  ({pct:.1f}%)")

    # ── PCA scree plot ────────────────────────────────────────────────────────
    if args.plot:
        print("\n── PCA variance plot ──────────────────────────────────────────────")
        plot_pca_variance(
            feature_df, available,
            zscore_by_season=args.zscore_by_season,
            seed=args.seed,
            save_dir=PLOTS_DIR,
        )

    # ── Sweep or cluster ──────────────────────────────────────────────────────
    if args.sweep_k:
        print("\n── Silhouette sweep ───────────────────────────────────────────────")
        sweep_k(
            feature_df,
            feature_cols=available,
            zscore_by_season=args.zscore_by_season,
            n_pca_components=args.pca_components,
            pca_variance=args.pca_variance,
            seed=args.seed,
        )
        if args.plot:
            print("\n── Elbow plot ─────────────────────────────────────────────────────")
            plot_elbow(
                feature_df, available,
                zscore_by_season=args.zscore_by_season,
                n_pca_components=args.pca_components,
                pca_variance=args.pca_variance,
                seed=args.seed,
                save_dir=PLOTS_DIR,
            )
        return

    print(f"\n── Clustering (k={args.k}) ─────────────────────────────────────────")
    labeled_df, centroid_df, scaler, pca, km = cluster_teams(
        feature_df,
        k=args.k,
        feature_cols=available,
        zscore_by_season=args.zscore_by_season,
        n_pca_components=args.pca_components,
        pca_variance=args.pca_variance,
        seed=args.seed,
    )

    if centroid_df is None:
        return

    labels_path    = os.path.join(OUT_DIR, "cluster_labels.csv")
    centroids_path = os.path.join(OUT_DIR, "cluster_centroids.csv")
    labeled_df.to_csv(labels_path, index=False)
    centroid_df.to_csv(centroids_path, index=False)
    print(f"\n  saved cluster_labels    → {labels_path}")
    print(f"  saved cluster_centroids → {centroids_path}")

    describe_clusters(labeled_df, centroid_df)

    if args.plot:
        print("\n── Generating plots ───────────────────────────────────────────────")
        plot_elbow(
            feature_df, available,
            zscore_by_season=args.zscore_by_season,
            n_pca_components=args.pca_components,
            pca_variance=args.pca_variance,
            seed=args.seed,
            save_dir=PLOTS_DIR,
        )
        plot_clusters(
            labeled_df, available,
            zscore_by_season=args.zscore_by_season,
            seed=args.seed,
            save_dir=PLOTS_DIR,
        )

    print(f"""
── Validation checklist ──────────────────────────────────────────────────
  1. Silhouette > 0.25?         Check above
  2. Elbow plot clear bend?     Open analysis/cache/plots/elbow.png
  3. Clusters visually separated? Open analysis/cache/plots/clusters.png
  4. Centroids interpretable?   Check cluster_centroids.csv — each cluster
     should tell a coherent story (e.g. high rim_pct + low three_pct_shot)
  5. No cluster < 3%% of teams?  That's a likely outlier bucket

Tuning levers (run --sweep-k --plot to compare):
  --style-only          +0.05–0.15 silhouette (drop efficiency noise)
  --zscore-by-season    +0.05–0.10 silhouette (remove era bias)
  --conf-only           +0.03–0.08 silhouette (remove cupcake games)
  --pca-variance 0.75   try fewer PCA dims (tighter clusters)
  --k lower             fewer clusters = higher silhouette (diminishing info)

If validation passes:
  cp data/cache/features_by_season.pkl data/cache/features_by_season_backup.pkl
  python analysis/cluster_features.py --all-seasons
  python train.py --all
──────────────────────────────────────────────────────────────────────────
""")


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# PCA + CLUSTERING — INTERPRETATION GUIDE
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
#
# HOW IT WORKS
# ─────────────
#  StandardScaler → [within-season Z-score] → PCA → MiniBatchKMeans
#
#  Within-season Z-scoring (--zscore-by-season):
#    Before global scaling, each feature is re-centered within its season.
#    A team whose three_pct_shot is 0.40 in 2016 (when few teams shot threes)
#    and a team at 0.40 in 2025 (when it's common) would get the same raw value
#    but very different Z-scores — the 2016 team is a huge outlier, the 2025
#    team is average.  Z-scoring captures this distinction.
#
#  Style-only features (--style-only):
#    Drops rim_fgpct, three_fgpct, jumper_fgpct.  Efficiency metrics add noise
#    because shooting percentages vary much less across D1 than shot selection.
#    A team's identity is defined by WHERE it shoots, not whether it hits them.
#    Clustering on efficiency conflates "bad team that shoots mid-range" with
#    "good team that shoots mid-range" into different clusters.
#
#  Conference-only filter (--conf-only):
#    Aggregates only plays from games where both teams are in the same conference.
#    This removes blowout cupcake games where garbage time inflates or deflates
#    style signals.  Result: ~40% fewer rows but higher signal-to-noise.
#
#  PCA variance threshold (--pca-variance):
#    Controls how many principal components are kept.  0.80 = keep enough PCs
#    to explain 80% of variance (typically 4–5 components from 10–13 features).
#    Lower = fewer components = tighter, more spherical clusters = higher
#    silhouette.  Risk: dropping below ~70% variance starts losing real signal.
#
# HOW TO READ cluster_centroids.csv
# ──────────────────────────────────
#  Centroids are back-projected through PCA inverse-transform and un-scaled.
#  They represent the "average team in that cluster" in original feature units:
#
#    rim_pct        0.48 → 48% of shots at rim  (high → paint-dominant)
#    three_pct_shot 0.31 → 31% from three       (high → perimeter team)
#    jumper_pct     0.21 → 21% mid-range         (high → ISO/post heavy)
#    assisted_pct   0.58 → 58% of makes assisted (high → motion/system ball)
#    ft_rate        0.38 → 38 FTA per 100 FGA   (high → aggressive attack)
#    to_per_game    14.2 → 14 turnovers/game
#    plays_per_game 178  → pace proxy
#    block_rate     4.1  → 4 blocks/game (rim protection)
#    steal_rate     7.3  → 7 steals/game (perimeter pressure)
#
#  Typical archetype names by centroid profile:
#    "Pace & Space"  — high three_pct_shot, high plays_per_game, high assisted_pct
#    "Bully Ball"    — high rim_pct, high ft_rate, low three_pct_shot
#    "ISO Ball"      — high jumper_pct, low assisted_pct
#    "Lockdown D"    — high block_rate + steal_rate, average offense
#    "Slow & Grind"  — low plays_per_game, high ft_rate
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

if __name__ == "__main__":
    main()
