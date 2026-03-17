# features/builder.py
import pandas as pd
import numpy as np
from datetime import date

# All per-team features that will be differenced in a matchup vector
TEAM_FEATURES = [
    "adj_off_rating", "adj_def_rating",
    "pace", "elo", "srs",
    # team four factors
    "efg_pct", "tov_ratio", "orb_pct", "ft_rate",
    # opponent four factors (how well defense holds opponent to these)
    "opp_efg_pct", "opp_tov_ratio", "opp_orb_pct", "opp_ft_rate",
    # composite / computed
    "experience", "depth_score", "star_power", "three_pt_volume", "hot_streak",
]


def _season_start_date(season):
    """Season N tips off ~Nov 1 of year N-1 (e.g. 2023 season = Nov 2022)."""
    return date(season - 1, 11, 1)


def compute_experience(player_stats_df, roster_df, season):
    """
    % of total team win shares from players who were 21+ at season start.
    Returns dict: {team: experience_ratio}
    """
    season_start = _season_start_date(season)
    players = player_stats_df[player_stats_df["season"] == season].copy()
    roster = roster_df[roster_df["season"] == season][["team", "playerId", "dateOfBirth"]].copy()

    roster["dateOfBirth"] = pd.to_datetime(roster["dateOfBirth"], errors="coerce").dt.date
    roster["age_at_start"] = roster["dateOfBirth"].apply(
        lambda dob: (season_start - dob).days / 365.25 if pd.notna(dob) else np.nan
    )

    merged = players.merge(
        roster, left_on=["team", "athleteId"], right_on=["team", "playerId"], how="left"
    )

    result = {}
    for team, grp in merged.groupby("team"):
        total_ws = grp["winShares_total"].sum()
        if total_ws <= 0:
            result[team] = 0.0
        else:
            veteran_ws = grp.loc[grp["age_at_start"] >= 21, "winShares_total"].sum()
            # clip to [0,1]: negative win-share players can make total_ws < veteran_ws
            result[team] = float(np.clip(veteran_ws / total_ws, 0.0, 1.0))
    return result


def compute_depth_score(player_stats_df, season):
    """
    % of total team win shares attributed to each team's top 3 players.
    Lower = more distributed roster; higher = star-dependent.
    Returns dict: {team: depth_score}
    """
    players = player_stats_df[player_stats_df["season"] == season].copy()
    result = {}
    for team, grp in players.groupby("team"):
        total_ws = grp["winShares_total"].sum()
        if total_ws <= 0:
            result[team] = 0.0
        else:
            top3_ws = grp.nlargest(3, "winShares_total")["winShares_total"].sum()
            # clip to [0,1]: negative win-share players shrink total_ws below top3_ws
            result[team] = float(np.clip(top3_ws / total_ws, 0.0, 1.0))
    return result


def compute_star_power(player_stats_df, season):
    """Max PORPAG on roster — peak individual impact. Returns dict: {team: max_porpag}"""
    players = player_stats_df[player_stats_df["season"] == season]
    return players.groupby("team")["PORPAG"].max().to_dict()


def compute_elo(elo_df, season):
    """
    End-of-season ELO from the /ratings/elo endpoint.
    Returns dict: {team: elo}
    """
    return elo_df[elo_df["season"] == season].set_index("team")["elo"].to_dict()


def compute_hot_streak(games_df, season, n=10):
    """
    Recency-weighted win rate over each team's last n regular season games.
    Weights decay exponentially — most recent game has highest weight.
    Only uses games where seasonType == 'regular' (no tournament games).
    Returns dict: {team: hot_streak}
    """
    reg = games_df[
        (games_df["season"] == season) & (games_df["seasonType"] == "regular")
    ].sort_values("startDate")

    all_teams = pd.concat([reg["homeTeam"], reg["awayTeam"]]).unique()
    result = {}

    for team in all_teams:
        team_games = reg[
            (reg["homeTeam"] == team) | (reg["awayTeam"] == team)
        ].tail(n)

        if len(team_games) == 0:
            result[team] = 0.5
            continue

        weights = np.exp(np.linspace(-2, 0, len(team_games)))
        wins = [
            (1 if g["homeWinner"] else 0) if g["homeTeam"] == team
            else (0 if g["homeWinner"] else 1)
            for _, g in team_games.iterrows()
        ]
        result[team] = float(np.average(wins, weights=weights))

    return result


def build_team_season_features(team_stats_df, player_stats_df, roster_df, srs_df, adj_df, elo_df, games_df, season):
    """
    Compute end-of-season feature dict per team for a given season.
    IMPORTANT: all inputs must be available before tournament selection date.
    Returns dict: {team_name: {feature: value, ...}}
    """
    ts = team_stats_df[team_stats_df["season"] == season]
    srs_map = srs_df[srs_df["season"] == season].set_index("team")["rating"].to_dict()
    adj = adj_df[adj_df["season"] == season].set_index("team")

    experience = compute_experience(player_stats_df, roster_df, season)
    depth = compute_depth_score(player_stats_df, season)
    star = compute_star_power(player_stats_df, season)
    elo = compute_elo(elo_df, season)
    hot_streak = compute_hot_streak(games_df, season)

    features = {}
    for _, row in ts.iterrows():
        team = row["team"]

        # Three-point volume: share of field goal attempts that are threes
        fga = row.get("teamStats_fieldGoals_attempted", 0)
        three_pa = row.get("teamStats_threePointFieldGoals_attempted", 0)
        three_pt_vol = float(three_pa / fga) if fga > 0 else 0.0

        adj_off = adj.at[team, "offensiveRating"] if team in adj.index else np.nan
        adj_def = adj.at[team, "defensiveRating"] if team in adj.index else np.nan
        # clip extreme values from sparse-data small programs
        adj_off = np.clip(adj_off, -50, 200)
        adj_def = np.clip(adj_def, -50, 200)

        features[team] = {
            "adj_off_rating": adj_off,
            "adj_def_rating": adj_def,
            "pace": row.get("pace", np.nan),
            "elo": elo.get(team, np.nan),
            "srs": srs_map.get(team, np.nan),
            # Team four factors
            "efg_pct": row.get("teamStats_fourFactors_effectiveFieldGoalPct", np.nan),
            "tov_ratio": row.get("teamStats_fourFactors_turnoverRatio", np.nan),
            "orb_pct": row.get("teamStats_fourFactors_offensiveReboundPct", np.nan),
            "ft_rate": row.get("teamStats_fourFactors_freeThrowRate", np.nan),
            # Opponent four factors
            "opp_efg_pct": row.get("opponentStats_fourFactors_effectiveFieldGoalPct", np.nan),
            "opp_tov_ratio": row.get("opponentStats_fourFactors_turnoverRatio", np.nan),
            "opp_orb_pct": row.get("opponentStats_fourFactors_offensiveReboundPct", np.nan),
            "opp_ft_rate": row.get("opponentStats_fourFactors_freeThrowRate", np.nan),
            # Composite / computed
            "experience": experience.get(team, np.nan),
            "depth_score": depth.get(team, np.nan),
            "star_power": star.get(team, np.nan),
            "three_pt_volume": three_pt_vol,
            "hot_streak": hot_streak.get(team, np.nan),
        }
    return features


def build_matchup_features(team_a_feats, team_b_feats, neutral_site=False, seed_a=None, seed_b=None):
    """
    Represent a matchup as feature differentials (A - B).
    Positive values favor team A, negative favor team B.
    neutral_site and seed_diff are added as raw values (not differenced).
    """
    row = {
        f"diff_{f}": team_a_feats.get(f, np.nan) - team_b_feats.get(f, np.nan)
        for f in TEAM_FEATURES
    }
    row["neutral_site"] = int(neutral_site)
    if seed_a is not None and seed_b is not None:
        row["seed_diff"] = seed_b - seed_a  # positive = A is the higher seed (lower number)
    return row


def build_training_set(games_df, team_features_by_season):
    """
    Build (X, y) across all seasons and game types.
    y=1 means home team won. season_type column preserved for filtering at train time.

    Feature vintage is chosen to prevent look-ahead leakage:
      - Regular season games  → prior season's end-of-season features
        (current-season stats are still accumulating during these games)
      - Postseason games      → current season's features
        (regular season is fully complete before the tournament begins)

    This mirrors deployment: tournament predictions use end-of-regular-season stats.
    """
    rows, labels = [], []

    for _, game in games_df.iterrows():
        season = game["season"]
        season_type = game.get("seasonType", "regular")

        feat_season = season if season_type != "regular" else season - 1

        if feat_season not in team_features_by_season:
            continue

        feats = team_features_by_season[feat_season]
        home, away = game["homeTeam"], game["awayTeam"]
        if home not in feats or away not in feats:
            continue

        matchup = build_matchup_features(
            feats[home],
            feats[away],
            neutral_site=bool(game.get("neutralSite", False)),
            seed_a=game.get("homeSeed") or None,
            seed_b=game.get("awaySeed") or None,
        )
        matchup["season_type"] = season_type
        matchup["season"] = season
        rows.append(matchup)
        labels.append(1 if game["homeWinner"] else 0)

    return pd.DataFrame(rows), pd.Series(labels)
