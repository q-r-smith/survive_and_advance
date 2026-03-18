"""
Run this first before building the bracket reconstruction logic.

    python inspect_tournament_data.py

Prints everything we need to know about the tournament games in the cache:
  - What seasonType / gameType / conferenceGame values exist
  - How many tournament games per season
  - What the actual column names are
  - A sample of Round 1 rows so we can see seeds, teams, scores
  - Whether round numbers are available or need to be inferred from dates
"""

import pandas as pd
import glob
import os

CACHE_DIR = "data/cache/raw"

# ── Load all games ────────────────────────────────────────────────────────────

files = sorted(glob.glob(os.path.join(CACHE_DIR, "games_*.csv")))
if not files:
    print(f"ERROR: No games_*.csv files found in {CACHE_DIR}")
    print("Run `python data/pull.py` first.")
    exit(1)

frames = []
for f in files:
    df = pd.read_csv(f)
    frames.append(df)
    print(f"  loaded {os.path.basename(f)}  ({len(df):,} rows)")

games = pd.concat(frames, ignore_index=True)
print(f"\nTotal rows: {len(games):,}")

# ── Column inventory ──────────────────────────────────────────────────────────

print("\n" + "="*60)
print("COLUMNS")
print("="*60)
for col in games.columns:
    n_null = games[col].isna().sum()
    sample = games[col].dropna().iloc[0] if not games[col].dropna().empty else "all null"
    print(f"  {col:<30} nulls={n_null:<6} sample={repr(sample)[:50]}")

# ── seasonType breakdown ──────────────────────────────────────────────────────

print("\n" + "="*60)
print("seasonType VALUE COUNTS")
print("="*60)
print(games["seasonType"].value_counts().to_string())

# ── gameType breakdown ────────────────────────────────────────────────────────

if "gameType" in games.columns:
    print("\n" + "="*60)
    print("gameType VALUE COUNTS")
    print("="*60)
    print(games["gameType"].value_counts().to_string())

# ── conferenceGame breakdown ─────────────────────────────────────────────────

if "conferenceGame" in games.columns:
    print("\n" + "="*60)
    print("conferenceGame VALUE COUNTS")
    print("="*60)
    print(games["conferenceGame"].value_counts().to_string())

# ── tournament filter — try different combinations ───────────────────────────

print("\n" + "="*60)
print("TOURNAMENT GAME FILTERS — trying different combinations")
print("="*60)

# Attempt 1: gameType == TRNMNT, conferenceGame == False
if "gameType" in games.columns and "conferenceGame" in games.columns:
    ncaa = games[
        (games["gameType"] == "TRNMNT") &
        (games["conferenceGame"].astype(str).str.lower().isin(["false", "0"]))
    ]
    print(f"\n  gameType==TRNMNT AND conferenceGame==False: {len(ncaa):,} rows")
    if len(ncaa) > 0:
        print(f"  Seasons: {sorted(ncaa['season'].unique())}")
        print(f"  Per season: {ncaa.groupby('season').size().to_dict()}")

# Attempt 2: gameType == TRNMNT, conferenceGame == True (conf tournaments)
if "gameType" in games.columns and "conferenceGame" in games.columns:
    conf = games[
        (games["gameType"] == "TRNMNT") &
        (games["conferenceGame"].astype(str).str.lower().isin(["true", "1"]))
    ]
    print(f"\n  gameType==TRNMNT AND conferenceGame==True: {len(conf):,} rows")
    if len(conf) > 0:
        print(f"  Seasons: {sorted(conf['season'].unique())}")
        print(f"  Per season: {conf.groupby('season').size().to_dict()}")

# Attempt 3: seasonType == postseason
post = games[games["seasonType"] == "postseason"]
print(f"\n  seasonType==postseason: {len(post):,} rows")
if len(post) > 0:
    print(f"  Seasons: {sorted(post['season'].unique())}")
    print(f"  Per season: {post.groupby('season').size().to_dict()}")

# Attempt 4: tournament field if it exists
if "tournament" in games.columns:
    print(f"\n  tournament field unique values:")
    print(f"  {games['tournament'].value_counts().head(20).to_string()}")

# ── Seed availability ─────────────────────────────────────────────────────────

print("\n" + "="*60)
print("SEED AVAILABILITY")
print("="*60)
seed_cols = [c for c in games.columns if "seed" in c.lower()]
print(f"  Seed columns found: {seed_cols}")
for col in seed_cols:
    n_non_null = games[col].notna().sum()
    print(f"  {col}: {n_non_null:,} non-null values")

# ── Sample NCAA tournament rows (2025) ────────────────────────────────────────

print("\n" + "="*60)
print("SAMPLE 2025 TOURNAMENT ROWS")
print("="*60)

# Best guess filter based on what we found above
if "gameType" in games.columns:
    sample_2025 = games[
        (games["season"] == 2025) &
        (games["gameType"] == "TRNMNT")
    ].sort_values("startDate") if "startDate" in games.columns else games[
        (games["season"] == 2025) &
        (games["gameType"] == "TRNMNT")
    ]
else:
    sample_2025 = games[
        (games["season"] == 2025) &
        (games["seasonType"] == "postseason")
    ]

if len(sample_2025) == 0:
    print("  No 2025 tournament rows found with current filter — check seasonType/gameType values above")
else:
    key_cols = ["startDate", "homeTeam", "homeSeed", "awayTeam", "awaySeed",
                "homePoints", "awayPoints", "homeWinner", "conferenceGame",
                "gameType", "neutralSite", "tournament"]
    available_cols = [c for c in key_cols if c in sample_2025.columns]
    print(f"  Total rows: {len(sample_2025)}")
    print(f"\n  First 10 rows (key columns):")
    print(sample_2025[available_cols].head(10).to_string(index=False))

    # Check round inference — do we have a round column?
    round_cols = [c for c in games.columns if "round" in c.lower()]
    print(f"\n  Round columns found: {round_cols if round_cols else 'NONE — rounds must be inferred from game order/date'}")

    if round_cols:
        for col in round_cols:
            print(f"\n  {col} values in 2025 tournament:")
            print(f"  {sample_2025[col].value_counts().to_string()}")

# ── Check if we can infer rounds from seed matchups ───────────────────────────

print("\n" + "="*60)
print("ROUND INFERENCE — seed matchup patterns in 2025")
print("="*60)

if "homeSeed" in games.columns and "awaySeed" in games.columns:
    seeded_2025 = sample_2025[
        sample_2025["homeSeed"].notna() & sample_2025["awaySeed"].notna()
    ].copy() if len(sample_2025) > 0 else pd.DataFrame()

    if len(seeded_2025) > 0:
        seeded_2025["seed_sum"] = (
            pd.to_numeric(seeded_2025["homeSeed"], errors="coerce") +
            pd.to_numeric(seeded_2025["awaySeed"], errors="coerce")
        )
        # Round 1 seed pairs always sum to 17 (1+16, 2+15, 3+14, etc.)
        r1 = seeded_2025[seeded_2025["seed_sum"] == 17]
        print(f"  Games where seed_sum==17 (Round 1): {len(r1)}")
        print(f"  (expect 64 for a full field, 32 for just the bracket games)")

        print(f"\n  Seed sum distribution:")
        print(seeded_2025["seed_sum"].value_counts().sort_index().to_string())

print("\n" + "="*60)
print("DONE — use this output to configure bracket reconstruction")
print("="*60)
