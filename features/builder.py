# features/builder.py
import pandas as pd
import numpy as np
from datetime import date

# All per-team features that will be differenced in a matchup vector
TEAM_FEATURES = [
    # efficiency core
    "adj_off_rating", "adj_def_rating", "net_rating", "efficiency_ratio",
    # normalized season ranks (0 = best, 1 = worst)
    "adj_off_rank", "adj_def_rank",
    # upset signal (efficiency vs seed expectation; NaN for unseeded teams)
    "upset_propensity",
    # other team-level
    "pace", "elo", "srs",
    # composite / computed
    "experience", "star_power", "hot_streak",
    # schedule / conference context
    "conf_strength", "non_conf_sos", "road_win_pct", "close_game_pct",
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


def compute_conf_strength(srs_df, season):
    """
    Leave-one-out average SRS of conference peers.
    Requires a 'conference' column in srs_df; returns {} if absent.
    Returns dict: {team: conf_strength}
    """
    s = srs_df[srs_df["season"] == season].copy()
    if "conference" not in s.columns or s.empty:
        return {}

    s["conf_sum"]   = s.groupby("conference")["rating"].transform("sum")
    s["conf_count"] = s.groupby("conference")["rating"].transform("count")

    # leave-one-out: exclude the team's own rating
    s["loo_mean"] = np.where(
        s["conf_count"] > 1,
        (s["conf_sum"] - s["rating"]) / (s["conf_count"] - 1),
        np.nan,
    )
    return dict(zip(s["team"], s["loo_mean"].astype(float)))


def compute_non_conf_sos(games_df, srs_df, season):
    """
    Average SRS of non-conference opponents in regular-season games.
    Falls back to np.nan if a team played fewer than 3 non-conference games.
    Requires 'conference' column in srs_df; returns {} if absent.
    Returns dict: {team: non_conf_sos}
    """
    s = srs_df[srs_df["season"] == season]
    if "conference" not in s.columns or s.empty:
        return {}

    team_conf = s.set_index("team")["conference"].to_dict()
    srs_map   = s.set_index("team")["rating"].to_dict()

    reg = games_df[
        (games_df["season"] == season) & (games_df["seasonType"] == "regular")
    ]

    all_teams = pd.concat([reg["homeTeam"], reg["awayTeam"]]).unique()
    result = {}

    for team in all_teams:
        team_c = team_conf.get(team)
        team_games = reg[(reg["homeTeam"] == team) | (reg["awayTeam"] == team)]

        opp_srs = []
        for _, g in team_games.iterrows():
            opp = g["awayTeam"] if g["homeTeam"] == team else g["homeTeam"]
            opp_c = team_conf.get(opp)
            if team_c and opp_c and team_c != opp_c and opp in srs_map:
                opp_srs.append(srs_map[opp])

        result[team] = float(np.mean(opp_srs)) if len(opp_srs) >= 3 else np.nan

    return result


def compute_road_win_pct(games_df, season):
    """
    Win rate in true road games (non-neutral site) during the regular season.
    Falls back to np.nan if fewer than 5 road games.
    Returns dict: {team: road_win_pct}
    """
    reg = games_df[
        (games_df["season"] == season) & (games_df["seasonType"] == "regular")
    ].copy()
    # neutralSite may arrive as bool, int, or string from CSV
    reg["_neutral"] = reg["neutralSite"].fillna(False).astype(str).str.lower().isin(["true", "1"])
    road = reg[~reg["_neutral"]]

    result = {}
    for team in road["awayTeam"].unique():
        team_road = road[road["awayTeam"] == team]
        if len(team_road) < 5:
            result[team] = np.nan
        else:
            wins = (~team_road["homeWinner"].astype(bool)).sum()
            result[team] = float(wins / len(team_road))

    return result


def compute_close_game_pct(games_df, season):
    """
    Fraction of regular-season games decided by ≤ 5 points (win or loss).
    Returns dict: {team: close_game_pct}
    """
    reg = games_df[
        (games_df["season"] == season) & (games_df["seasonType"] == "regular")
    ].copy()

    reg["margin"] = (
        pd.to_numeric(reg["homePoints"], errors="coerce") -
        pd.to_numeric(reg["awayPoints"], errors="coerce")
    ).abs()
    reg["close"] = reg["margin"] <= 5

    all_teams = pd.concat([reg["homeTeam"], reg["awayTeam"]]).unique()
    result = {}
    for team in all_teams:
        team_games = reg[(reg["homeTeam"] == team) | (reg["awayTeam"] == team)]
        if len(team_games) == 0:
            result[team] = np.nan
        else:
            result[team] = float(team_games["close"].mean())

    return result


def compute_upset_propensity(games_df, season, efficiency_ratio_map):
    """
    Deviation of a team's efficiency_ratio from the mean for their seed group.
    Positive = overperformer vs seed; negative = underperformer vs seed.
    NaN for unseeded teams.
    Returns dict: {team: upset_propensity}
    """
    from collections import defaultdict
    post = games_df[
        (games_df["season"] == season) & (games_df["seasonType"] == "postseason")
    ]

    team_seed = {}
    for _, g in post.iterrows():
        if pd.notna(g.get("homeSeed")):
            team_seed[g["homeTeam"]] = int(g["homeSeed"])
        if pd.notna(g.get("awaySeed")):
            team_seed[g["awayTeam"]] = int(g["awaySeed"])

    if not team_seed:
        return {}

    seed_ratios = defaultdict(list)
    for team, seed in team_seed.items():
        er = efficiency_ratio_map.get(team)
        if er is not None and np.isfinite(er):
            seed_ratios[seed].append(er)

    seed_mean = {seed: float(np.mean(ratios)) for seed, ratios in seed_ratios.items()}

    result = {}
    for team, seed in team_seed.items():
        er = efficiency_ratio_map.get(team)
        if er is not None and np.isfinite(er) and seed in seed_mean:
            result[team] = float(er - seed_mean[seed])
        else:
            result[team] = np.nan
    return result


def build_team_season_features(team_stats_df, player_stats_df, roster_df, srs_df, adj_df, elo_df, games_df, season):
    """
    Compute end-of-season feature dict per team for a given season.
    IMPORTANT: all inputs must be available before tournament selection date.
    Returns dict: {team_name: {feature: value, ...}}
    """
    ts = team_stats_df[team_stats_df["season"] == season]
    srs_map = srs_df[srs_df["season"] == season].set_index("team")["rating"].to_dict()
    adj = adj_df[adj_df["season"] == season].drop_duplicates(subset="team").set_index("team")
    if not adj.index.is_unique:
        adj = adj.groupby(level=0).first()  # belt-and-suspenders: guarantee scalar .at[] lookups

    # Precompute normalized rank maps for the full season
    adj_season = adj_df[adj_df["season"] == season].drop_duplicates(subset="team").copy()
    if not adj_season.empty and "team" in adj_season.columns:
        n = len(adj_season)
        adj_season["_off_rank"] = adj_season["offensiveRating"].rank(ascending=False)
        adj_season["_def_rank"] = adj_season["defensiveRating"].rank(ascending=True)
        adj_season["_off_rank_norm"] = (adj_season["_off_rank"] - 1) / max(n - 1, 1)
        adj_season["_def_rank_norm"] = (adj_season["_def_rank"] - 1) / max(n - 1, 1)
        adj_off_rank_map = adj_season.set_index("team")["_off_rank_norm"].to_dict()
        adj_def_rank_map = adj_season.set_index("team")["_def_rank_norm"].to_dict()
    else:
        adj_off_rank_map, adj_def_rank_map = {}, {}

    experience    = compute_experience(player_stats_df, roster_df, season)
    star          = compute_star_power(player_stats_df, season)
    elo           = compute_elo(elo_df, season)
    hot_streak    = compute_hot_streak(games_df, season)
    conf_strength = compute_conf_strength(srs_df, season)
    non_conf_sos  = compute_non_conf_sos(games_df, srs_df, season)
    road_win_pct  = compute_road_win_pct(games_df, season)
    close_game    = compute_close_game_pct(games_df, season)

    features = {}
    for _, row in ts.iterrows():
        team = row["team"]

        adj_off = adj.at[team, "offensiveRating"] if team in adj.index else np.nan
        adj_def = adj.at[team, "defensiveRating"] if team in adj.index else np.nan
        # clip extreme values from sparse-data small programs
        adj_off = np.clip(adj_off, -50, 200)
        adj_def = np.clip(adj_def, -50, 200)

        # Efficiency ratio: Klemm's primary feature; captures multiplicative relationship
        efficiency_ratio = float(adj_off / adj_def) if (adj_def != 0 and np.isfinite(adj_def)) else np.nan
        # Net rating from API (already computed server-side)
        net_rating = adj.at[team, "netRating"] if (team in adj.index and "netRating" in adj.columns) else (
            float(adj_off - adj_def) if (np.isfinite(adj_off) and np.isfinite(adj_def)) else np.nan
        )

        features[team] = {
            # Efficiency core
            "adj_off_rating":  adj_off,
            "adj_def_rating":  adj_def,
            "net_rating":      net_rating,
            "efficiency_ratio": efficiency_ratio,
            # Normalized season ranks (0 = best, 1 = worst)
            "adj_off_rank":    adj_off_rank_map.get(team, np.nan),
            "adj_def_rank":    adj_def_rank_map.get(team, np.nan),
            # upset_propensity filled in second pass below
            "upset_propensity": np.nan,
            # Other team-level
            "pace":            row.get("pace", np.nan),
            "elo":             elo.get(team, np.nan),
            "srs":             srs_map.get(team, np.nan),
            # Composite / computed
            "experience":      experience.get(team, np.nan),
            "star_power":      star.get(team, np.nan),
            "hot_streak":      hot_streak.get(team, np.nan),
            # Schedule / conference context
            "conf_strength":   conf_strength.get(team, np.nan),
            "non_conf_sos":    non_conf_sos.get(team, np.nan),
            "road_win_pct":    road_win_pct.get(team, np.nan),
            "close_game_pct":  close_game.get(team, np.nan),
        }

    # Second pass: upset_propensity requires efficiency_ratio for all teams first
    efficiency_ratio_map = {t: f["efficiency_ratio"] for t, f in features.items()}
    upset_prop = compute_upset_propensity(games_df, season, efficiency_ratio_map)
    for team in features:
        features[team]["upset_propensity"] = upset_prop.get(team, np.nan)

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
    y=1 means home team won. season_type, game_type, conf_game columns preserved
    for training-tier filtering at train time (not used as model features).

    Feature vintage is chosen to prevent look-ahead leakage:
      - Regular season games       → prior season's end-of-season features
      - Postseason games           → current season's features
      - Conference tournaments     → current season's features
        (conference tournaments occur after the regular season is complete, just
         like the NCAA tournament, so current-season stats are fully available.
         They carry seasonType=="regular" in the raw data, so we detect them via
         gameType=="TRNMNT" and conferenceGame==True.)

    This mirrors deployment: tournament predictions use end-of-regular-season stats.
    """
    rows, labels = [], []

    for _, game in games_df.iterrows():
        season      = game["season"]
        season_type = game.get("seasonType", "regular")
        game_type   = game.get("gameType", "STD")
        conf_game   = bool(game.get("conferenceGame", False))

        # Conference tournament games are labeled seasonType=="regular" but happen
        # after the regular season ends → use current-season features.
        is_conf_tourn = (game_type == "TRNMNT" and conf_game)
        feat_season   = season if (season_type != "regular" or is_conf_tourn) else season - 1

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
        matchup["season"]      = season
        matchup["game_type"]   = game_type
        matchup["conf_game"]   = conf_game
        rows.append(matchup)
        labels.append(1 if game["homeWinner"] else 0)

    return pd.DataFrame(rows), pd.Series(labels)
