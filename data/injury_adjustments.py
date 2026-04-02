# data/injury_adjustments.py
"""
Pre-simulation injury adjustment layer.

Applies player-absence adjustments to the in-memory features dict before
BracketSimulator builds its probability matrices. The features_by_season.pkl
cache is never modified — only the working copy is changed.

Usage:
    from data.injury_adjustments import (
        apply_injury_adjustments, load_player_stats, print_injury_report
    )
    player_stats_df = load_player_stats(season)
    features, report = apply_injury_adjustments(features, injury_file, player_stats_df)
    print_injury_report(report)
"""

import os
import copy
import json
import warnings

import numpy as np
import pandas as pd


# ── Player stats loader ───────────────────────────────────────────────────────

def load_player_stats(season: int) -> pd.DataFrame:
    """
    Load player stats from the raw cache for a given season.
    Returns DataFrame with at minimum: name, team, PORPAG, winShares_total columns.
    """
    path = os.path.join("data", "cache", "raw", f"player_stats_{season}.csv")
    if not os.path.exists(path):
        raise FileNotFoundError(
            f"Player stats cache not found: {path}\n"
            f"Run `python data/pull.py` first."
        )
    df = pd.read_csv(path)
    required = {"name", "team", "PORPAG", "winShares_total"}
    missing_cols = required - set(df.columns)
    if missing_cols:
        raise ValueError(
            f"player_stats_{season}.csv is missing columns: {missing_cols}\n"
            f"Found: {list(df.columns)}"
        )
    return df


# ── Team name fuzzy lookup ────────────────────────────────────────────────────

_TEAM_ALIASES = {
    "BYU":            ["Brigham Young", "BYU Cougars"],
    "Brigham Young":  ["BYU"],
    "UConn":          ["Connecticut"],
    "Connecticut":    ["UConn"],
    "NC State":       ["North Carolina State"],
    "North Carolina State": ["NC State"],
    "LSU":            ["Louisiana State"],
    "Louisiana State": ["LSU"],
    "USC":            ["Southern California"],
    "Southern California": ["USC"],
    "VCU":            ["Virginia Commonwealth"],
    "Virginia Commonwealth": ["VCU"],
    "SIUE":           ["SIU Edwardsville"],
    "SIU Edwardsville": ["SIUE"],
}


def _find_team_in_features(team_name: str, features: dict) -> str | None:
    """
    Return the key used in features for team_name, or None if not found.
    Tries exact match, then known aliases.
    """
    if team_name in features:
        return team_name
    for alias in _TEAM_ALIASES.get(team_name, []):
        if alias in features:
            return alias
    return None


# ── Player fuzzy lookup ───────────────────────────────────────────────────────

def _find_players(player_name: str, team_df: pd.DataFrame) -> pd.DataFrame:
    """Case-insensitive contains match on the name column."""
    mask = team_df["name"].str.contains(player_name, case=False, na=False)
    return team_df[mask]


# ── Core adjustment function ──────────────────────────────────────────────────

def apply_injury_adjustments(
    features: dict,
    injury_file: str,
    player_stats_df: pd.DataFrame,
    hot_streak_window: int = 15,
) -> tuple[dict, list[dict]]:
    """
    Apply injury adjustments to a features dict (in-memory copy only).

    Args:
        features:          dict of {team_name: {feature: value}} from features_by_season
        injury_file:       path to injuries JSON file
        player_stats_df:   player stats DataFrame for the current season
        hot_streak_window: number of games in the hot_streak calculation window (default 15)

    Returns:
        adjusted_features: deep copy of features with modifications applied
        adjustment_report: list of dicts describing each adjustment made
    """
    adjusted = copy.deepcopy(features)

    with open(injury_file) as f:
        injury_data = json.load(f)

    report: list[dict] = []

    for team_name, players_list in injury_data.items():
        # Resolve team name in features dict
        api_name = _find_team_in_features(team_name, adjusted)
        if api_name is None:
            warnings.warn(
                f"Injury adjustment: team '{team_name}' not found in features "
                f"(tried aliases too). Skipping."
            )
            continue

        feats = adjusted[api_name]

        # Filter player stats to this team
        team_df = player_stats_df[
            player_stats_df["team"].str.lower() == api_name.lower()
        ].copy()
        if team_df.empty:
            # Try the injury-file name as well
            team_df = player_stats_df[
                player_stats_df["team"].str.lower() == team_name.lower()
            ].copy()

        if team_df.empty:
            warnings.warn(
                f"Injury adjustment: no player stats found for '{api_name}'. "
                f"Cannot compute win-share fractions — skipping team."
            )
            continue

        team_total_ws = float(team_df["winShares_total"].sum())
        if team_total_ws <= 0:
            warnings.warn(
                f"Injury adjustment: '{api_name}' has zero total win shares — skipping."
            )
            continue

        # Determine whether any player is "questionable"
        any_questionable = any(
            p.get("status", "out").lower() == "questionable"
            for p in players_list
        )

        # Collect data for all missing players
        found_players: list[dict] = []
        total_ws_missing = 0.0
        hot_streak_multiplier = 1.0
        all_porpags = team_df["PORPAG"].values
        team_max_porpag = float(np.nanmax(all_porpags)) if len(all_porpags) > 0 else 0.0

        for player in players_list:
            pname = player["name"]
            games_out = int(player.get("games_out", 1))

            matched = _find_players(pname, team_df)
            if matched.empty:
                warnings.warn(
                    f"Injury adjustment: player '{pname}' not found in "
                    f"'{api_name}' stats — skipping this player."
                )
                continue

            # If multiple matches, take the one with highest win shares
            row = matched.loc[matched["winShares_total"].idxmax()]
            player_ws = float(row["winShares_total"])
            player_porpag = float(row["PORPAG"])

            total_ws_missing += player_ws

            # Hot streak multiplier contribution for this player
            games_missed_in_window = min(games_out, hot_streak_window)
            fraction_missed = games_missed_in_window / hot_streak_window
            player_ws_pct = player_ws / team_total_ws
            hot_streak_multiplier *= (1.0 - fraction_missed * player_ws_pct)

            found_players.append({
                "name":         pname,
                "games_out":    games_out,
                "status":       player.get("status", "out"),
                "win_shares":   player_ws,
                "PORPAG":       player_porpag,
            })

        if not found_players:
            continue

        production_lost_pct = float(np.clip(total_ws_missing / team_total_ws, 0.0, 1.0))

        # Is any missing player the team's star (max PORPAG)?
        missing_porpags = [p["PORPAG"] for p in found_players]
        is_star_anchor = any(
            abs(porpag - team_max_porpag) < 1e-6
            for porpag in missing_porpags
        ) and team_max_porpag > 0

        # Next best PORPAG among remaining players
        missing_athlete_ids = set()
        for p in found_players:
            matched = _find_players(p["name"], team_df)
            if not matched.empty:
                missing_athlete_ids.update(matched.index.tolist())
        remaining_df = team_df.drop(index=list(missing_athlete_ids), errors="ignore")
        next_best_porpag = (
            float(remaining_df["PORPAG"].max())
            if (is_star_anchor and not remaining_df.empty)
            else float("nan")
        )

        # Halve adjustments if questionable (average plays/doesn't-play scenarios)
        adj_scale = 0.5 if any_questionable else 1.0

        # Capture "before" values
        # NOTE on model coefficients (as of current trained model):
        #   diff_elo              coef ≈ +1.58  ← dominant feature, must reduce
        #   diff_hot_streak       coef ≈ +0.52  ← positive (quality-weighted); reduce for injuries
        #   diff_srs              coef ≈ -0.42  ← not adjusted (roster-level SRS; not meaningful)
        #   diff_efficiency_ratio coef ≈  0.00  ← collinear with adj ratings; near-zero effect
        #   diff_star_power       coef ≈  0.00  ← near-zero
        #   diff_experience       coef ≈ +0.01  ← very small but correct sign
        before = {
            "elo":              float(feats.get("elo",              float("nan"))),
            "efficiency_ratio": float(feats.get("efficiency_ratio", float("nan"))),
            "hot_streak":       float(feats.get("hot_streak",       float("nan"))),
            "star_power":       float(feats.get("star_power",       float("nan"))),
            "experience":       float(feats.get("experience",       float("nan"))),
        }

        # elo — primary adjustment (dominant positive-coefficient feature).
        # Scale down proportionally to production lost, with 0.15 dampening so
        # a 20%-production loss reduces elo by ~3%.
        _ELO_DAMPENING = 0.15
        if np.isfinite(before["elo"]):
            feats["elo"] = (
                before["elo"]
                * (1.0 - production_lost_pct * _ELO_DAMPENING * adj_scale)
            )

        # efficiency_ratio — secondary adjustment (near-zero coefficient, kept for
        # conceptual completeness; does not meaningfully move win probabilities).
        if np.isfinite(before["efficiency_ratio"]):
            feats["efficiency_ratio"] = (
                before["efficiency_ratio"]
                * (1.0 - production_lost_pct * adj_scale)
            )

        # hot_streak — reduce proportionally to production lost. The quality-weighted
        # hot_streak has a positive coefficient (≈ +0.52), so reducing it correctly
        # decreases win probability for an injured team. Dampened 0.5× because hot_streak
        # reflects recent team outcomes, not just the missing player's contribution.
        _HOT_STREAK_DAMPENING = 0.5
        if np.isfinite(before["hot_streak"]):
            feats["hot_streak"] = (
                before["hot_streak"]
                * (1.0 - production_lost_pct * _HOT_STREAK_DAMPENING * adj_scale)
            )

        # star_power — adjust only if the missing player was the team's best PORPAG scorer.
        if is_star_anchor and np.isfinite(next_best_porpag):
            delta = (before["star_power"] - next_best_porpag) * adj_scale
            feats["star_power"] = before["star_power"] - delta

        # experience — scale down (small positive coefficient; dampened 0.5× because
        # experience is a roster-wide metric).
        if np.isfinite(before["experience"]):
            feats["experience"] = (
                before["experience"]
                * (1.0 - production_lost_pct * 0.5 * adj_scale)
            )

        # Capture "after" values
        after = {
            "elo":              float(feats.get("elo",              float("nan"))),
            "efficiency_ratio": float(feats.get("efficiency_ratio", float("nan"))),
            "hot_streak":       float(feats.get("hot_streak",       float("nan"))),
            "star_power":       float(feats.get("star_power",       float("nan"))),
            "experience":       float(feats.get("experience",       float("nan"))),
        }

        report.append({
            "team":                api_name,
            "players":             [p["name"] for p in found_players],
            "statuses":            [p["status"] for p in found_players],
            "games_out":           [p["games_out"] for p in found_players],
            "production_lost_pct": production_lost_pct,
            "is_star_anchor":      is_star_anchor,
            "any_questionable":    any_questionable,
            "adjustments": {
                feat: {"before": before[feat], "after": after[feat]}
                for feat in ("elo", "efficiency_ratio", "hot_streak", "star_power", "experience")
            },
        })

    return adjusted, report


# ── Report printer ────────────────────────────────────────────────────────────

def print_injury_report(report: list[dict]) -> None:
    """Print a human-readable summary of all injury adjustments before simulation."""
    if not report:
        print("\n  [Injury adjustments] No adjustments applied.\n")
        return

    print("\n" + "=" * 65)
    print(" Injury adjustments (pre-simulation)")
    print("=" * 65)

    for entry in report:
        team = entry["team"]
        players = entry["players"]
        statuses = entry["statuses"]
        games_out = entry["games_out"]
        prod_lost = entry["production_lost_pct"]
        is_star = entry["is_star_anchor"]
        any_q = entry["any_questionable"]

        # Header line
        player_parts = []
        for name, status, g_out in zip(players, statuses, games_out):
            player_parts.append(f"{name} ({status}, {g_out} games)")
        print(f"\n{team} — {', '.join(player_parts)}")
        if any_q:
            print(f"  NOTE: questionable player(s) — adjustments halved")
        print(f"  Production lost: {prod_lost:.1%} of team win shares")
        print(f"  Is star anchor:  {'Yes' if is_star else 'No'}"
              f"{'  (star_power adjusted)' if is_star else '  (star_power unchanged)'}")

        adj = entry["adjustments"]
        col_w = 21

        # Table
        print(f"  ┌{'─'*col_w}┬{'─'*10}┬{'─'*10}┬{'─'*10}┐")
        print(f"  │ {'Feature':<{col_w-2}} │ {'Before':>8} │ {'After':>8} │ {'Change':>8} │")
        print(f"  ├{'─'*col_w}┼{'─'*10}┼{'─'*10}┼{'─'*10}┤")

        for feat, vals in adj.items():
            b = vals["before"]
            a = vals["after"]
            if np.isfinite(b) and np.isfinite(a) and b != 0:
                pct_chg = (a - b) / abs(b) * 100.0
                chg_str = f"{pct_chg:+.1f}%"
            else:
                chg_str = "  —"
            # Use integer format for large values like elo (>= 100)
            fmt = ".1f" if (np.isfinite(b) and abs(b) >= 100) else ".3f"
            b_str = f"{b:{fmt}}" if np.isfinite(b) else "  —"
            a_str = f"{a:{fmt}}" if np.isfinite(a) else "  —"
            print(f"  │ {feat:<{col_w-2}} │ {b_str:>8} │ {a_str:>8} │ {chg_str:>8} │")

        print(f"  └{'─'*col_w}┴{'─'*10}┴{'─'*10}┴{'─'*10}┘")

    print(f"\n{'='*65}\n")
