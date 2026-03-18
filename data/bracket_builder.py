# data/bracket_builder.py
"""
Reconstructs NCAA tournament bracket structure and results from cached game data.

Primary filter: tournament == "NCAA" (reliable across all seasons)
Confirmation:   gameNotes contains "Basketball Championship" (case-insensitive)
Round inference: parsed directly from gameNotes — no date clustering needed.
                 gameNotes encodes round name explicitly:
                   "1st Round" → R64 (1), "2nd Round" → R32 (2),
                   "Sweet 16" / "3rd Round" → S16 (3),
                   "Elite 8" → E8 (4), "Final Four" → F4 (5),
                   "National Championship" → Champ (6),
                   "First Four" → play-in (0)
"""

import os
import pandas as pd
import numpy as np


# ── Round inference from gameNotes ────────────────────────────────────────────

def _parse_round_from_notes(notes: str) -> int:
    """
    Extract round number from gameNotes string.

    Two naming conventions exist in the data:
      New (2016+): First Four→0, 1st Round→1, 2nd Round→2, Sweet 16→3,
                   Elite 8→4, Final Four→5, National Championship→6
      Old (2014-2015): 1st Round→play-in(0), 2nd Round→R64(1),
                       3rd Round→R32(2), Sweet 16→S16(3), ...

    "3rd Round" is ambiguous: it means R32 in old naming and doesn't appear
    in new naming. We return -2 as a sentinel and resolve it after detecting
    which naming convention is in use.

    Returns -1 if unrecognized.
    """
    n = notes.lower()
    if "first four" in n:
        return 0
    if "1st round" in n:
        return 1
    if "2nd round" in n:
        return 2
    if "3rd round" in n:
        return -2   # sentinel: old-style R32, resolved after detection
    if "sweet 16" in n:
        return 3
    if "elite 8" in n:
        return 4
    if "final four" in n:
        return 5
    if "national championship" in n:
        return 6
    return -1


def _parse_region_from_notes(notes: str) -> str | None:
    """Extract region name from gameNotes. Returns None for cross-region games."""
    n = notes.lower()
    if "south" in n:
        return "South"
    if "midwest" in n:
        return "Midwest"
    if "east" in n:
        return "East"
    if "west" in n:
        return "West"
    return None


# ── Standard bracket slot ordering ───────────────────────────────────────────
# In a standard bracket, R1 games are ordered by lower seed:
#   Slot 1: 1v16, Slot 2: 8v9, Slot 3: 5v12, Slot 4: 4v13,
#   Slot 5: 6v11, Slot 6: 3v14, Slot 7: 7v10, Slot 8: 2v15
SEED_TO_SLOT = {1: 1, 8: 2, 5: 3, 4: 4, 6: 5, 3: 6, 7: 7, 2: 8}


def _slot_key(home_seed, away_seed):
    seeds = [s for s in [home_seed, away_seed] if s is not None]
    if not seeds:
        return 99
    lower = min(int(s) for s in seeds)
    return SEED_TO_SLOT.get(lower, 99)


# ── Main builder ──────────────────────────────────────────────────────────────

def build_bracket_from_cache(season: int) -> dict:
    """
    Reconstructs the full NCAA tournament bracket and results for a given season.

    Returns:
        {
          "season":   int,
          "matchups": list of game dicts with round_num assigned,
          "results":  {team: {round_num: 1 or 0}},
          "seeds":    {team: int},
          "champion": str or None,
          "rounds":   {round_num: [list of game dicts]},
          "bracket":  JSON-compatible bracket dict (for BracketSimulator),
        }
    """
    if season == 2020:
        raise ValueError("No NCAA tournament in 2020 (COVID).")

    path = os.path.join("data", "cache", "raw", f"games_{season}.csv")
    if not os.path.exists(path):
        raise FileNotFoundError(
            f"games_{season}.csv not found. Run `python data/pull.py` first."
        )

    games_raw = pd.read_csv(path)

    # ── Primary filter: tournament == "NCAA" ──────────────────────────────────
    ncaa = games_raw[games_raw["tournament"] == "NCAA"].copy()

    if ncaa.empty:
        # Fallback: gameNotes contains "Basketball Championship"
        ncaa = games_raw[
            games_raw["gameNotes"].fillna("").str.contains(
                "Basketball Championship", case=False
            )
        ].copy()

    if ncaa.empty:
        raise ValueError(
            f"No NCAA tournament games found for {season}. "
            f"Check games_{season}.csv tournament and gameNotes columns."
        )

    # Validate count
    if not (60 <= len(ncaa) <= 72):
        print(f"  WARNING: {season} has {len(ncaa)} NCAA rows (expected 63-68) "
              f"— check filter")

    # ── Assign round numbers from gameNotes ───────────────────────────────────
    ncaa["round_num"] = ncaa["gameNotes"].fillna("").apply(_parse_round_from_notes)

    # ── Detect naming convention and resolve sentinels ────────────────────────
    # Old naming (2014-2015): "1st Round"=play-in, "2nd Round"=R64, "3rd Round"=R32
    # New naming (2016+):     "First Four"=play-in, "1st Round"=R64, "2nd Round"=R32
    # Detect by: if round 2 has ≥ 28 games, it's the R64 → old naming
    r2_count = (ncaa["round_num"] == 2).sum()
    r1_count  = (ncaa["round_num"] == 1).sum()
    old_naming = (r2_count >= 28 and r1_count <= 4)

    if old_naming:
        remap = {1: 0, 2: 1, -2: 2, 3: 3, 4: 4, 5: 5, 6: 6, -1: -1}
        ncaa["round_num"] = ncaa["round_num"].map(remap)
    else:
        # New naming: -2 sentinel ("3rd Round") shouldn't appear but handle it
        ncaa.loc[ncaa["round_num"] == -2, "round_num"] = 2

    unrecognized = ncaa[ncaa["round_num"] == -1]
    if len(unrecognized) > 0:
        print(f"  WARNING: {len(unrecognized)} games in {season} have "
              f"unrecognized gameNotes round — falling back to date clustering")
        ncaa["date"] = pd.to_datetime(ncaa["startDate"]).dt.date
        ncaa = ncaa.sort_values("date").reset_index(drop=True)
        unique_dates = sorted(ncaa[ncaa["round_num"] == -1]["date"].unique())
        date_to_round = {d: i + 1 for i, d in enumerate(unique_dates)}
        mask = ncaa["round_num"] == -1
        ncaa.loc[mask, "round_num"] = ncaa.loc[mask, "date"].map(date_to_round)

    # ── Extract seeds ──────────────────────────────────────────────────────────
    seeds: dict[str, int] = {}
    seeded = ncaa[ncaa["homeSeed"].notna() & ncaa["awaySeed"].notna()]
    for _, row in seeded.iterrows():
        try:
            seeds[row["homeTeam"]] = int(float(row["homeSeed"]))
            seeds[row["awayTeam"]] = int(float(row["awaySeed"]))
        except (ValueError, TypeError):
            pass

    # ── Build results, matchups, rounds ───────────────────────────────────────
    results: dict[str, dict[int, int]] = {}
    matchups: list[dict] = []
    rounds: dict[int, list] = {r: [] for r in range(0, 7)}

    for _, game in ncaa.iterrows():
        round_num = int(game["round_num"])
        home      = game["homeTeam"]
        away      = game["awayTeam"]
        home_won  = bool(game["homeWinner"])
        winner    = home if home_won else away
        loser     = away if home_won else home

        # Capture seeds from First Four too
        for team in [home, away]:
            if team not in results:
                results[team] = {}

        if round_num > 0:  # exclude First Four from results/scoring
            results[winner][round_num] = 1
            results[loser][round_num]  = 0

        game_dict = {
            "round_num":   round_num,
            "home_team":   home,
            "away_team":   away,
            "home_seed":   seeds.get(home),
            "away_seed":   seeds.get(away),
            "home_points": game.get("homePoints"),
            "away_points": game.get("awayPoints"),
            "winner":      winner,
            "loser":       loser,
            "region":      _parse_region_from_notes(game.get("gameNotes", "")),
            "game_notes":  game.get("gameNotes", ""),
        }
        matchups.append(game_dict)
        rounds[round_num].append(game_dict)

    # ── Validate Round 1 seed sums ────────────────────────────────────────────
    bad = []
    for g in rounds[1]:
        hs, as_ = g["home_seed"], g["away_seed"]
        if hs is not None and as_ is not None:
            s = int(hs) + int(as_)
            if s != 17:
                bad.append((g["home_team"], g["away_team"], s))
    if bad:
        print(f"  WARNING: {len(bad)} Round 1 games with seed_sum != 17 in {season}: {bad}")

    # ── Champion ───────────────────────────────────────────────────────────────
    r6 = rounds.get(6, [])
    champion = r6[0]["winner"] if r6 else None

    # Print summary
    round_counts = {r: len(v) for r, v in rounds.items() if v}
    print(f"  {season}: {len(ncaa)} games, rounds={round_counts}  "
          f"champion={champion}")

    # ── Reconstruct bracket JSON for BracketSimulator ─────────────────────────
    bracket_json = _reconstruct_bracket_json(rounds, seeds)

    return {
        "season":   season,
        "matchups": matchups,
        "results":  results,
        "seeds":    seeds,
        "champion": champion,
        "rounds":   rounds,
        "bracket":  bracket_json,
    }


def _reconstruct_bracket_json(rounds: dict, seeds: dict) -> dict:
    """
    Build a bracket dict in the JSON format expected by BracketSimulator,
    reconstructed from actual tournament game data.
    """
    r1_games = rounds.get(1, [])

    region_games: dict[str, list] = {
        "South": [], "East": [], "West": [], "Midwest": []
    }

    for game in r1_games:
        region = game.get("region")
        if region in region_games:
            region_games[region].append(game)
        else:
            # Try seeds to guess region (last resort)
            region_games["South"].append(game)

    bracket_json: dict = {
        "regions": {},
        "final_four_slots": {
            "SF1": {"feeds_from": ["South_winner", "East_winner"]},
            "SF2": {"feeds_from": ["West_winner", "Midwest_winner"]},
            "Championship": {"feeds_from": ["SF1_winner", "SF2_winner"]},
        },
    }

    for region, games in region_games.items():
        games_sorted = sorted(
            games,
            key=lambda g: _slot_key(g.get("home_seed"), g.get("away_seed"))
        )
        prefix = region[0]
        matchups = []
        for i, g in enumerate(games_sorted, 1):
            matchups.append({
                "slot":   f"{prefix}_R1_G{i}",
                "team_a": g["home_team"],
                "seed_a": g.get("home_seed"),
                "team_b": g["away_team"],
                "seed_b": g.get("away_seed"),
            })
        bracket_json["regions"][region] = {"matchups": matchups}

    return bracket_json


def build_all_brackets(seasons: list) -> dict:
    """Build brackets for multiple seasons. Returns {season: bracket_dict}."""
    brackets = {}
    for season in seasons:
        try:
            brackets[season] = build_bracket_from_cache(season)
        except (FileNotFoundError, ValueError) as e:
            print(f"  {season}: skipped — {e}")
    return brackets
