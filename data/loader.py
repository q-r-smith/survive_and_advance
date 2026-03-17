# data/loader.py
import requests
import pandas as pd
import os
from dotenv import load_dotenv

load_dotenv(override=True)
API_KEY = os.getenv("COLLEGE_FOOTBALL_API_KEY")
BASE = "https://api.collegebasketballdata.com"
HEADERS = {"Authorization": f"Bearer {API_KEY}"}


def _get(endpoint, params=None):
    resp = requests.get(f"{BASE}{endpoint}", params=params, headers=HEADERS)
    resp.raise_for_status()
    return resp.json()


def get_games(season, season_type=None):
    """
    Pull all games for a given season by fetching month-by-month.
    The API caps results at 3,000 rows per request, so a full season
    (Nov–Apr, ~5,000 games) requires multiple date-range calls.
    Season N runs Nov (N-1) through Apr N.
    """
    # Build monthly windows covering the full season
    months = [
        (f"{season - 1}-11-01", f"{season - 1}-11-30"),
        (f"{season - 1}-12-01", f"{season - 1}-12-31"),
        (f"{season}-01-01",     f"{season}-01-31"),
        (f"{season}-02-01",     f"{season}-02-28"),
        (f"{season}-03-01",     f"{season}-03-31"),
        (f"{season}-04-01",     f"{season}-04-30"),
    ]
    chunks = []
    for start, end in months:
        params = {"season": season, "startDateRange": start, "endDateRange": end}
        if season_type:
            params["seasonType"] = season_type
        data = _get("/games", params)
        if data:
            chunks.append(pd.DataFrame(data))
    if not chunks:
        return pd.DataFrame()
    return pd.concat(chunks, ignore_index=True).drop_duplicates(subset=["id"])


def get_team_stats(season):
    """Season-level team stats including four factors for team and opponent sides."""
    data = _get("/stats/team/season", {"season": season})
    return pd.json_normalize(data, sep="_")


def get_player_stats(season):
    """Season-level player stats including win shares and PORPAG."""
    data = _get("/stats/player/season", {"season": season})
    return pd.json_normalize(data, sep="_")


def get_roster(season):
    """
    Roster data with dateOfBirth for experience calculation.
    Flattens nested players array into one row per player.
    """
    data = _get("/teams/roster", {"season": season})
    rows = []
    for entry in data:
        for player in entry.get("players", []):
            rows.append({
                "teamId": entry["teamId"],
                "team": entry["team"],
                "season": entry["season"],
                "playerId": player["id"],
                "dateOfBirth": player.get("dateOfBirth"),
            })
    return pd.DataFrame(rows)


def get_srs(season):
    """Simple Rating System — used as strength of record proxy."""
    return pd.DataFrame(_get("/ratings/srs", {"season": season}))


def get_adjusted_ratings(season):
    """Adjusted offensive and defensive efficiency ratings."""
    return pd.DataFrame(_get("/ratings/adjusted", {"season": season}))


def get_elo(season):
    """End-of-season ELO rating per team."""
    return pd.DataFrame(_get("/ratings/elo", {"season": season}))


def load_all_seasons(start=2015, end=2023):
    """
    Load all required datasets across seasons.
    Returns a dict of DataFrames keyed by data type.
    """
    buckets = {k: [] for k in ["games", "team_stats", "player_stats", "rosters", "srs", "adjusted_ratings", "elo"]}

    for year in range(start, end + 1):
        print(f"Loading {year}...")
        buckets["games"].append(get_games(year))
        buckets["team_stats"].append(get_team_stats(year))
        buckets["player_stats"].append(get_player_stats(year))
        buckets["rosters"].append(get_roster(year))
        buckets["srs"].append(get_srs(year))
        buckets["adjusted_ratings"].append(get_adjusted_ratings(year))
        buckets["elo"].append(get_elo(year))

    return {k: pd.concat(v, ignore_index=True) for k, v in buckets.items()}
