# analysis/pull_style_data.py
"""
Pull team shooting data for style-of-play analysis.
Saves to data/cache/raw/ alongside the main pipeline's cached CSVs.

Usage:
    python analysis/pull_style_data.py --seasons 2023 2024 2025 2026
    python analysis/pull_style_data.py --seasons 2016 2017 2018 2019 2021 2022 2023 2024 2025 --force
"""

import argparse
import os
from pathlib import Path
import sys
import time

import pandas as pd
import requests

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from dotenv import load_dotenv

load_dotenv(Path(__file__).parent.parent / ".env", override=True)

API_KEY = os.getenv("COLLEGE_FOOTBALL_API_KEY")
BASE = "https://api.collegebasketballdata.com"
HEADERS = {"Authorization": f"Bearer {API_KEY}"}

RAW_DIR = os.path.join(os.path.dirname(__file__), "..", "data", "cache", "raw")


# ── API helpers ───────────────────────────────────────────────────────────────

def _get(endpoint, params=None, timeout=10):
    resp = requests.get(f"{BASE}{endpoint}", params=params, headers=HEADERS, timeout=timeout)
    resp.raise_for_status()
    text = resp.text.strip()
    if not text or not text.startswith(("[", "{")):
        return []
    return resp.json()


def _get_conferences(season: int) -> list[str]:
    """Load conference names from cached team_stats for this season."""
    path = os.path.join(RAW_DIR, f"team_stats_{season}.csv")
    if not os.path.exists(path):
        return []
    df = pd.read_csv(path)
    return sorted(df["conference"].dropna().unique().tolist())



def get_team_shooting(season: int) -> pd.DataFrame:
    """
    Team shooting breakdown by shot zone for the given season.
    Queries one conference at a time (API requires team or conference param).
    """
    conferences = _get_conferences(season)
    if not conferences:
        print(f"    no team_stats_{season}.csv found — cannot get conference list")
        return pd.DataFrame()

    frames = []
    for conf in conferences:
        try:
            data = _get("/stats/team/shooting/season", {"season": season, "conference": conf}, timeout=20)
        except Exception as e:
            print(f"    WARNING: shooting {conf} → {e}", flush=True)
            continue
        if data:
            frames.append(pd.json_normalize(data, sep="_"))

    if not frames:
        return pd.DataFrame()
    df = pd.concat(frames, ignore_index=True)
    df["season"] = season
    return df


def get_play_by_play(season: int, team: str, shooting_only: bool = False) -> pd.DataFrame:
    return pd.DataFrame(_get(f"/plays/team/?season={season}&team={team}&shootingPlaysOnly={str(shooting_only).lower()}", timeout=60))



# ── Cache helpers ─────────────────────────────────────────────────────────────

def _cache_path(name: str) -> str:
    return os.path.join(RAW_DIR, f"{name}.csv")


def _load_or_fetch(name: str, fetch_fn, season: int, force: bool, play_by_play: bool = False) -> pd.DataFrame:
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
    if play_by_play:
        partial_path = path + ".partial"
        try:
            curr_dir = os.getcwd()
            teams_df = pd.read_csv(os.path.join(curr_dir, "data", "cache", "raw", f"team_stats_{season}.csv"))
            teams = teams_df.dropna(subset=["conference"])["team"].tolist()
        except Exception:
            from data.loader import get_team_stats
            teams_df = get_team_stats(season)
            teams = teams_df.dropna(subset=["conference"])["team"].tolist()

        # resume from partial if it exists
        already_fetched = set()
        if os.path.exists(partial_path):
            try:
                partial_df = pd.read_csv(partial_path)
                if "team" in partial_df.columns:
                    already_fetched = set(partial_df["team"].unique())
                    print(f"  resuming — {len(already_fetched)} teams already cached in partial")
            except Exception:
                pass

        os.makedirs(os.path.dirname(partial_path), exist_ok=True)
        for team in teams:
            if team in already_fetched:
                continue
            print(f"    fetching play-by-play for {season} {team} ...")
            for attempt in range(3):
                try:
                    team_df = fetch_fn(season, team)
                    if not team_df.empty and team_df.notna().any().any():
                        team_df["team"] = team
                        team_df.to_csv(
                            partial_path,
                            mode="a",
                            header=not os.path.exists(partial_path) or os.path.getsize(partial_path) == 0,
                            index=False,
                        )
                    break
                except Exception as team_err:
                    if attempt < 2:
                        wait = 5 * (attempt + 1)
                        print(f"      retry {attempt + 1}/2 for {team} after {wait}s ({team_err})")
                        time.sleep(wait)
                    else:
                        print(f"      WARNING: skipping {team} after 3 failures ({team_err})")

        if os.path.exists(partial_path) and os.path.getsize(partial_path) > 0:
            df = pd.read_csv(partial_path).drop_duplicates().reset_index(drop=True)
        else:
            df = pd.DataFrame()
    else:
        df = fetch_fn(season)
    if len(df) > 0:
        os.makedirs(os.path.dirname(path), exist_ok=True)
        df.to_csv(path, index=False)
        print(f"  saved {path}  ({len(df):,} rows)")
        if play_by_play:
            partial = path + ".partial"
            if os.path.exists(partial):
                os.remove(partial)
    else:
        print(f"  WARNING: no data returned for {name}")
    return df


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="Pull team shooting data.")
    parser.add_argument(
        "--seasons", nargs="+", type=int, required=True,
        help="Seasons to pull (e.g. 2023 2024 2025 2026)"
    )
    parser.add_argument(
        "--force", action="store_true",
        help="Re-fetch even if cache exists"
    )
    args = parser.parse_args()

    for season in args.seasons:
        print(f"\n── {season}: Team Shooting ──────────────────────────")
        _load_or_fetch(f"team_shooting_{season}", get_team_shooting, season, args.force)

    for season in args.seasons:
        print(f"\n── {season}: Play-by-Play ──────────────────────────")
        _load_or_fetch(f"play_by_play_{season}", get_play_by_play, season, args.force, play_by_play=True)

    print("\nDone.")


if __name__ == "__main__":
    main()
