# data/loader.py
import cbbd
import requests
import pandas as pd
import os
from dotenv import load_dotenv
from pathlib import Path
from datetime import datetime

load_dotenv(Path(__file__).parent.parent / ".env", override=True)
API_KEY = os.getenv("COLLEGE_FOOTBALL_API_KEY")

# Raw HTTP fallback for any endpoint not yet in the cbbd package
BASE    = "https://api.collegebasketballdata.com"
HEADERS = {"Authorization": f"Bearer {API_KEY}"}


def _configuration():
    return cbbd.Configuration(access_token=API_KEY)


def _to_df(result):
    """Convert a list of cbbd Pydantic objects to a flat DataFrame."""
    return pd.json_normalize([r.to_dict() for r in result], sep="_")


def _get(endpoint, params=None):
    """Raw HTTP fallback for endpoints not in the cbbd package."""
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
    months = [
        (datetime(season - 1, 11, 1),  datetime(season - 1, 11, 30)),
        (datetime(season - 1, 12, 1),  datetime(season - 1, 12, 31)),
        (datetime(season,     1,  1),  datetime(season,      1, 31)),
        (datetime(season,     2,  1),  datetime(season,      2, 28)),
        (datetime(season,     3,  1),  datetime(season,      3, 31)),
        (datetime(season,     4,  1),  datetime(season,      4, 30)),
    ]
    chunks = []
    with cbbd.ApiClient(_configuration()) as client:
        api = cbbd.GamesApi(client)
        for start, end in months:
            data = api.get_games(
                season=season,
                start_date_range=start,
                end_date_range=end,
            )
            if data:
                df = _to_df(data)
                if season_type:
                    df = df[df["seasonType"] == season_type]
                chunks.append(df)
    if not chunks:
        return pd.DataFrame()
    return pd.concat(chunks, ignore_index=True).drop_duplicates(subset=["id"])


def get_team_stats(season):
    """Season-level team stats including pace and four factors."""
    with cbbd.ApiClient(_configuration()) as client:
        result = cbbd.StatsApi(client).get_team_season_stats(season=season)
    return _to_df(result)


def get_player_stats(season):
    """Season-level player stats including win shares and PORPAG."""
    with cbbd.ApiClient(_configuration()) as client:
        result = cbbd.StatsApi(client).get_player_season_stats(season=season)
    return _to_df(result)


def get_roster(season):
    """
    Roster data with dateOfBirth for experience calculation.
    Flattens nested players array into one row per player.
    """
    with cbbd.ApiClient(_configuration()) as client:
        result = cbbd.TeamsApi(client).get_team_roster(season=season)
    rows = []
    for entry in result:
        d = entry.to_dict()
        for player in d.get("players", []):
            rows.append({
                "teamId":      d["teamId"],
                "team":        d["team"],
                "season":      d["season"],
                "playerId":    player["id"],
                "dateOfBirth": player.get("dateOfBirth"),
            })
    return pd.DataFrame(rows)


def get_adjusted_ratings(season):
    """Adjusted offensive and defensive efficiency ratings."""
    with cbbd.ApiClient(_configuration()) as client:
        result = cbbd.RatingsApi(client).get_adjusted_efficiency(season=season)
    return _to_df(result)


def get_elo(season):
    """End-of-season ELO rating per team."""
    with cbbd.ApiClient(_configuration()) as client:
        result = cbbd.RatingsApi(client).get_elo(season=season)
    return _to_df(result)


def get_team_shooting(season):
    """Season-level team shooting stats (shot selection and efficiency)."""
    with cbbd.ApiClient(_configuration()) as client:
        result = cbbd.StatsApi(client).get_team_season_shooting_stats(season=season)
    return _to_df(result)


def get_player_shooting(season):
    """Season-level player shooting stats."""
    with cbbd.ApiClient(_configuration()) as client:
        result = cbbd.StatsApi(client).get_player_season_shooting_stats(season=season)
    return _to_df(result)


def get_lineups(season):
    """Season-level lineup data by team."""
    with cbbd.ApiClient(_configuration()) as client:
        result = cbbd.LineupsApi(client).get_lineups_by_team_season(season=season)
    return _to_df(result)


def get_plays_by_tournament(season, tournament="NCAA"):
    """All play-by-play data for a given season's tournament."""
    with cbbd.ApiClient(_configuration()) as client:
        result = cbbd.PlaysApi(client).get_plays_by_tournament(
            season=season, tournament=tournament
        )
    return _to_df(result)


def load_all_seasons(start=2015, end=2023):
    """
    Load all required datasets across seasons.
    Returns a dict of DataFrames keyed by data type.
    """
    buckets = {k: [] for k in [
        "games", "team_stats", "player_stats", "rosters", "adjusted_ratings", "elo"
    ]}
    for year in range(start, end + 1):
        print(f"Loading {year}...")
        buckets["games"].append(get_games(year))
        buckets["team_stats"].append(get_team_stats(year))
        buckets["player_stats"].append(get_player_stats(year))
        buckets["rosters"].append(get_roster(year))
        buckets["adjusted_ratings"].append(get_adjusted_ratings(year))
        buckets["elo"].append(get_elo(year))
    return {k: pd.concat(v, ignore_index=True) for k, v in buckets.items()}
