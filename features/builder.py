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
    # committee vs metrics gap (replaces upset_propensity)
    "predicted_seed_delta",
    # other team-level
    "pace", "elo", "srs",
    # composite / computed
    "experience", "star_power", "hot_streak",
    # schedule / conference context
    "conf_strength", "non_conf_sos", "road_win_pct", "close_game_pct",
    # tournament-context feature (NaN for non-postseason rows; imputed at train time)
    "bracket_win_rate",
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


def compute_hot_streak(games_df, season, n_recent=5, n_total=15, elo_scale=150.0):
    """
    Quality-weighted, recency-split win rate over each team's last n_total
    regular season games.

    Each game's contribution is weighted by two factors multiplied together:
      1. Recency: split-window — last n_recent games get weight 2.0,
                  older games get weight 1.0
      2. Quality: opponent pre-game ELO centered at 1500 (D1 average),
                  scaled by elo_scale.
                  1700 ELO → 2.33x, 1500 ELO → 1.0x, 1300 ELO → floored to 0.1x
                  Unknown opponent ELO → 1.0 (treated as average).

    Uses homeTeamEloStart / awayTeamEloStart from the games DataFrame.
    Only uses regular season games (seasonType == 'regular').
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
        ].tail(n_total)

        if len(team_games) == 0:
            result[team] = 0.5
            continue

        actual_n = len(team_games)
        wins, recency_weights, quality_weights = [], [], []

        for i, (_, g) in enumerate(team_games.iterrows()):
            # Win/loss
            if g["homeTeam"] == team:
                wins.append(1.0 if g["homeWinner"] else 0.0)
                opp_elo = float(g.get("awayTeamEloStart", np.nan))
            else:
                wins.append(0.0 if g["homeWinner"] else 1.0)
                opp_elo = float(g.get("homeTeamEloStart", np.nan))

            # Recency: last n_recent games (tail of window) get weight 2.0
            recency_weights.append(2.0 if i >= actual_n - n_recent else 1.0)

            # Quality from opponent pre-game ELO
            if np.isfinite(opp_elo):
                qw = max(1.0 + (opp_elo - 1500.0) / elo_scale, 0.1)
            else:
                qw = 1.0
            quality_weights.append(qw)

        combined = np.array(recency_weights) * np.array(quality_weights)
        result[team] = float(np.average(wins, weights=combined))

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


def compute_predicted_seed_delta(games_df, season, features_by_season, current_season_features=None):
    """
    For each NCAA tournament team in this season, compute the residual between
    their actual seed and their predicted seed from an efficiency-based regression.

    Predicted seed is fit using all OTHER seasons (out-of-fold) to prevent
    leakage. Features used: efficiency_ratio, elo, adj_off_rating, adj_def_rating,
    net_rating.

    current_season_features: the just-built first-pass feature dict for this season.
    Must be passed explicitly because features_by_season does not yet contain the
    current season at call time (it is being built right now).

    Positive delta = seeded worse than metrics suggest (potential overperformer)
    Negative delta = seeded better than metrics suggest (potential underperformer)
    Zero / NaN = no seed data or unseeded team

    Returns dict: {team: predicted_seed_delta}
    """
    from sklearn.linear_model import Ridge
    from sklearn.preprocessing import StandardScaler
    from sklearn.pipeline import Pipeline

    REGRESSION_FEATURES = [
        "efficiency_ratio", "elo", "adj_off_rating", "adj_def_rating", "net_rating"
    ]

    # Get tournament seeds for the current season
    post = games_df[
        (games_df["season"] == season) &
        (games_df["seasonType"] == "postseason")
    ]
    current_seeds: dict = {}
    for _, g in post.iterrows():
        if pd.notna(g.get("homeSeed")):
            current_seeds[g["homeTeam"]] = int(float(g["homeSeed"]))
        if pd.notna(g.get("awaySeed")):
            current_seeds[g["awayTeam"]] = int(float(g["awaySeed"]))

    if not current_seeds:
        return {}

    # Build training data from all other seasons
    X_rows, y_seeds = [], []
    for hist_season, hist_features in features_by_season.items():
        if hist_season == season:
            continue  # out-of-fold: exclude current season
        hist_post = games_df[
            (games_df["season"] == hist_season) &
            (games_df["seasonType"] == "postseason")
        ]
        hist_seeds: dict = {}
        for _, g in hist_post.iterrows():
            if pd.notna(g.get("homeSeed")):
                hist_seeds[g["homeTeam"]] = int(float(g["homeSeed"]))
            if pd.notna(g.get("awaySeed")):
                hist_seeds[g["awayTeam"]] = int(float(g["awaySeed"]))

        for team, seed in hist_seeds.items():
            if team not in hist_features:
                continue
            feats = hist_features[team]
            row = [feats.get(f, np.nan) for f in REGRESSION_FEATURES]
            if any(np.isnan(v) for v in row):
                continue
            X_rows.append(row)
            y_seeds.append(seed)

    if len(X_rows) < 50:
        # Not enough historical data — return zeros
        return {team: 0.0 for team in current_seeds}

    X_hist = np.array(X_rows)
    y_hist = np.array(y_seeds, dtype=float)

    model = Pipeline([
        ("scaler", StandardScaler()),
        ("ridge",  Ridge(alpha=1.0))
    ])
    model.fit(X_hist, y_hist)

    # Predict seeds for current tournament teams and compute delta
    # Use explicitly passed first-pass features; fall back to features_by_season
    # only if not provided (should not normally happen).
    result = {}
    current_feats = current_season_features if current_season_features is not None else features_by_season.get(season, {})
    for team, actual_seed in current_seeds.items():
        if team not in current_feats:
            result[team] = np.nan
            continue
        feats = current_feats[team]
        row = [feats.get(f, np.nan) for f in REGRESSION_FEATURES]
        if any(np.isnan(v) for v in row):
            result[team] = np.nan
            continue
        predicted_seed = float(model.predict([row])[0])
        predicted_seed = np.clip(predicted_seed, 1.0, 16.0)
        result[team] = float(actual_seed - predicted_seed)

    return result


def compute_bracket_win_rate(games_df, season, tournament_teams):
    """
    Win rate in regular season games against teams that made the NCAA tournament
    that same season. Only computed for tournament teams themselves.

    Args:
        games_df: full games DataFrame
        season: int
        tournament_teams: set of team names that made the NCAA tournament this season

    Returns:
        dict: {team: bracket_win_rate} — np.nan if no games played vs bracket teams
    """
    reg = games_df[
        (games_df["season"] == season) &
        (games_df["seasonType"] == "regular")
    ]

    result = {}
    for team in tournament_teams:
        team_games = reg[(reg["homeTeam"] == team) | (reg["awayTeam"] == team)]

        wins, total = 0, 0
        for _, g in team_games.iterrows():
            opp = g["awayTeam"] if g["homeTeam"] == team else g["homeTeam"]
            if opp not in tournament_teams:
                continue
            total += 1
            won = (
                (g["homeTeam"] == team and bool(g["homeWinner"])) or
                (g["awayTeam"] == team and not bool(g["homeWinner"]))
            )
            if won:
                wins += 1

        result[team] = float(wins / total) if total > 0 else np.nan

    return result


def build_team_season_features(team_stats_df, player_stats_df, roster_df, srs_df, adj_df, elo_df, games_df, season,
                               features_by_season=None):
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
            # predicted_seed_delta filled in second pass below
            "predicted_seed_delta": np.nan,
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
            # Filled per-game in build_training_set for postseason rows only
            "bracket_win_rate": np.nan,
        }

    # Second pass: compute predicted_seed_delta (requires features from other seasons)
    if features_by_season is not None:
        seed_delta = compute_predicted_seed_delta(
            games_df=games_df,
            season=season,
            features_by_season=features_by_season,
            current_season_features=features,  # first-pass features for current season
        )
    else:
        seed_delta = {}

    for team in features:
        features[team]["predicted_seed_delta"] = seed_delta.get(team, np.nan)

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
    y=1 means home team won. season_type, game_type, conf_game, round_num columns
    preserved for training-tier filtering / analysis (not used as model features).

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
    from data.bracket_builder import _parse_round_from_notes, build_bracket_from_cache

    rows, labels = [], []
    _bracket_teams_cache    = {}  # season → set of tournament team names
    _bracket_win_rate_cache = {}  # season → {team: bracket_win_rate}

    for _, game in games_df.iterrows():
        season      = game["season"]
        season_type = game.get("seasonType", "regular")
        game_type   = game.get("gameType", "STD")
        conf_game   = bool(game.get("conferenceGame", False))

        # Conference tournament games are labeled seasonType=="regular" but happen
        # after the regular season ends → use current-season features.
        is_conf_tourn = (game_type == "TRNMNT" and conf_game)
        is_postseason = (season_type == "postseason")
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

        # ── round_num metadata (not a model feature) ──────────────────────────
        if is_postseason:
            notes     = str(game.get("gameNotes", ""))
            round_num = _parse_round_from_notes(notes)
            if round_num == -2:   # "3rd Round" sentinel → R32 in both conventions
                round_num = 2
        else:
            round_num = -1   # regular season or conference tournament

        # ── diff_bracket_win_rate (postseason only) ───────────────────────────
        if is_postseason:
            if season not in _bracket_teams_cache:
                try:
                    bd = build_bracket_from_cache(season)
                    _bracket_teams_cache[season] = set(bd["seeds"].keys())
                except Exception:
                    _bracket_teams_cache[season] = set()

            tournament_teams = _bracket_teams_cache[season]
            if tournament_teams:
                if season not in _bracket_win_rate_cache:
                    _bracket_win_rate_cache[season] = compute_bracket_win_rate(
                        games_df, season, tournament_teams
                    )
                bwr      = _bracket_win_rate_cache[season]
                home_bwr = bwr.get(home, np.nan)
                away_bwr = bwr.get(away, np.nan)
                matchup["diff_bracket_win_rate"] = (
                    home_bwr - away_bwr
                    if not (np.isnan(home_bwr) or np.isnan(away_bwr))
                    else np.nan
                )
            else:
                matchup["diff_bracket_win_rate"] = np.nan
        else:
            matchup["diff_bracket_win_rate"] = np.nan

        matchup["round_num"]   = round_num
        matchup["season_type"] = season_type
        matchup["season"]      = season
        matchup["game_type"]   = game_type
        matchup["conf_game"]   = conf_game
        rows.append(matchup)
        labels.append(1 if game["homeWinner"] else 0)

    return pd.DataFrame(rows), pd.Series(labels)
