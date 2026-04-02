"""
analysis/explore_plays_api.py
─────────────────────────────
Exploration-only script: answers five questions about the play-by-play API.
No files are written. Run from project root:

    python analysis/explore_plays_api.py
"""
import requests
import pandas as pd
import numpy as np
import json
import sys
import os
import time
from pathlib import Path

sys.path.insert(0, ".")

from dotenv import load_dotenv
load_dotenv(Path(__file__).parent.parent / ".env", override=True)

API_KEY = os.getenv("COLLEGE_FOOTBALL_API_KEY")
BASE    = "https://api.collegebasketballdata.com"
HEADERS = {"Authorization": f"Bearer {API_KEY}"}

# ── Shared state ──────────────────────────────────────────────────────────────
CALL_COUNT   = 0
WORKING_PARAM = None   # e.g. "gameId" — found in Q2, reused in Q3/Q4
FETCHED_GAMES = {}     # game_id -> {"plays": [...], "bytes": int, "label": str}

SUMMARY = {
    "play_types_available": "UNKNOWN",
    "action_level":         "UNKNOWN",
    "shot_info_pct":        "unknown",
    "on_floor_pct":         "unknown",
    "safe_fetch_rpm":       "unknown",
    "size_conf_ncaa_1yr":   "unknown",
    "blocking_issues":      [],
}

ACTION_KEYWORDS = {
    "pick and roll": ["pick", "roll", "ball handler", "screener"],
    "isolation":     ["iso", "isolation"],
    "post-up":       ["post", "post-up", "post up"],
    "transition":    ["transition", "fast break", "fastbreak"],
    "spot-up":       ["spot", "spot-up", "spot up", "catch and shoot"],
}

EVENT_KEYWORDS = [
    "made", "missed", "foul", "turnover", "rebound", "free throw",
    "block", "steal", "timeout", "substitution", "violation",
]


def hr(char="─", width=66):
    print(char * width)


def _get_raw(endpoint, params=None, timeout=15):
    """Returns (status_code, is_html, data_or_none, bytes_or_0, error_or_none)"""
    global CALL_COUNT
    CALL_COUNT += 1
    try:
        resp = requests.get(
            f"{BASE}{endpoint}", params=params, headers=HEADERS, timeout=timeout
        )
        raw_bytes = len(resp.content)
        is_html   = resp.text.strip().startswith("<")
        if is_html:
            return resp.status_code, True, None, raw_bytes, "HTML response (wrong endpoint or auth redirect)"
        if resp.status_code == 401:
            return 401, False, None, raw_bytes, "Unauthorized — check COLLEGE_FOOTBALL_API_KEY"
        if resp.status_code == 400:
            return 400, False, None, raw_bytes, f"Bad request: {resp.text[:200]}"
        if resp.status_code == 404:
            return 404, False, None, raw_bytes, "Not found"
        if resp.status_code == 429:
            return 429, False, None, raw_bytes, "Rate limited"
        if resp.status_code != 200:
            return resp.status_code, False, None, raw_bytes, resp.text[:200]
        try:
            data = resp.json()
            return resp.status_code, False, data, raw_bytes, None
        except Exception as e:
            return resp.status_code, False, None, raw_bytes, f"JSON parse error: {e}"
    except requests.exceptions.Timeout:
        return 0, False, None, 0, "Timeout"
    except Exception as e:
        return 0, False, None, 0, str(e)


# ══════════════════════════════════════════════════════════════════════════════
# Q1 — Play types taxonomy
# ══════════════════════════════════════════════════════════════════════════════

def question_1():
    print()
    hr("═")
    print("QUESTION 1 — Play types taxonomy")
    hr("═")

    candidates = ["/plays/types", "/play/types", "/plays/type"]

    found_data = None
    found_endpoint = None
    for endpoint in candidates:
        status, is_html, data, _, err = _get_raw(endpoint)
        ok = data is not None
        print(f"  {endpoint:<25}  status={status}  ok={ok}  {'err: ' + err if err else ''}")
        if ok and found_data is None:
            found_data    = data
            found_endpoint = endpoint

    if found_data is None:
        print("\n  ✗ No play types endpoint found.")
        print("    The API may not expose a taxonomy endpoint — play types will")
        print("    appear inline in play records. Will check actual values in Q5.")
        SUMMARY["play_types_available"] = "PARTIAL (inline only — no taxonomy endpoint)"
        return

    items = found_data if isinstance(found_data, list) else []
    print(f"\n  ✓ Got {len(items)} play types from {found_endpoint}")

    if not items:
        print("  Empty list returned.")
        return

    # Normalise: figure out the name/id fields
    sample = items[0] if isinstance(items[0], dict) else {}
    name_key = next((k for k in ("name", "type", "playType", "description") if k in sample), None)
    id_key   = next((k for k in ("id", "typeId", "playTypeId") if k in sample), None)

    sorted_items = sorted(items, key=lambda x: str(x.get(name_key, "")) if isinstance(x, dict) else str(x))

    print(f"\n  Total: {len(sorted_items)}")
    print(f"  {'ID':>6}  Name")
    hr()
    all_names = []
    for item in sorted_items:
        if isinstance(item, dict):
            name = str(item.get(name_key, item))
            id_  = item.get(id_key, "—")
        else:
            name = str(item)
            id_  = "—"
        all_names.append(name.lower())
        print(f"  {str(id_):>6}  {name}")

    # Classify action-level vs event-level
    action_hits = [kw for group in ACTION_KEYWORDS.values() for kw in group
                   if any(kw in n for n in all_names)]
    event_hits  = [kw for kw in EVENT_KEYWORDS if any(kw in n for n in all_names)]

    print()
    if action_hits:
        print(f"  Classification: ACTION-LEVEL — found action keywords: {action_hits[:5]}")
        SUMMARY["action_level"] = "YES (action-level)"
        SUMMARY["play_types_available"] = "YES"
    elif event_hits:
        print(f"  Classification: EVENT-LEVEL — found event keywords: {event_hits[:5]}")
        print("    → No pick-and-roll / isolation / post-up types detected.")
        print("    → Clustering must derive play types from event sequences,")
        print("       not read them directly from playType field.")
        SUMMARY["action_level"] = "NO — event-level only (made shot, foul, etc.)"
        SUMMARY["play_types_available"] = "PARTIAL (event-level taxonomy only)"
        SUMMARY["blocking_issues"].append(
            "Play types are event-level (made/missed/foul), not action-level. "
            "Style clustering must derive play type distributions from event "
            "sequences rather than reading them directly from playType."
        )
    else:
        print("  Classification: UNCLEAR — could not match to action or event keywords")
        SUMMARY["action_level"] = "UNKNOWN (unrecognised type names)"
        SUMMARY["play_types_available"] = "PARTIAL"


# ══════════════════════════════════════════════════════════════════════════════
# Q2 — Single game plays fetch
# ══════════════════════════════════════════════════════════════════════════════

PARAM_VARIATIONS = [
    "gameId",
    "game_id",
    "id",
    "gameID",
]


def _load_2025_ncaa_games():
    """Load the cached 2025 games file and return NCAA postseason rows."""
    cache = Path("data/cache/raw/games_2025.csv")
    if not cache.exists():
        print(f"  Cache not found: {cache}")
        return pd.DataFrame()
    df = pd.read_csv(cache)
    df["margin"] = (pd.to_numeric(df["homePoints"], errors="coerce") -
                    pd.to_numeric(df["awayPoints"], errors="coerce")).abs()
    ncaa = df[
        (df.get("tournament", pd.Series(dtype=str)) == "NCAA") &
        (df.get("seasonType", pd.Series(dtype=str)) == "postseason") &
        df["margin"].notna()
    ].copy()
    return ncaa


def _pick_3_games(ncaa: pd.DataFrame):
    """Return list of (game_id, label) for: blowout R64, close game, championship."""
    if ncaa.empty:
        return []
    picks = []

    # Blowout: largest margin in first round (homeSeed or awaySeed == 16 implies R64)
    r64 = ncaa[
        (pd.to_numeric(ncaa.get("homeSeed", pd.Series(dtype=float)), errors="coerce") >= 9) |
        (pd.to_numeric(ncaa.get("awaySeed", pd.Series(dtype=float)), errors="coerce") >= 9)
    ]
    if r64.empty:
        r64 = ncaa  # fallback: any game
    blowout = r64.loc[r64["margin"].idxmax()]
    picks.append((blowout["id"], f"R64 blowout ({blowout['awayTeam']} @ {blowout['homeTeam']}, margin={blowout['margin']:.0f})"))

    # Close game: margin < 5, prefer from later rounds
    close = ncaa[ncaa["margin"] < 5]
    if not close.empty:
        row = close.iloc[0]
        picks.append((row["id"], f"Close game ({row['awayTeam']} @ {row['homeTeam']}, margin={row['margin']:.0f})"))
    else:
        # Next-smallest margin
        row = ncaa.nsmallest(2, "margin").iloc[-1]
        picks.append((row["id"], f"Closest available ({row['awayTeam']} @ {row['homeTeam']}, margin={row['margin']:.0f})"))

    # Championship: latest date or notes containing "championship"
    champ = ncaa[ncaa.get("gameNotes", pd.Series(dtype=str)).str.lower().str.contains("champion", na=False)]
    if champ.empty:
        champ = ncaa.sort_values("startDate", ascending=False)
    row = champ.iloc[0]
    if row["id"] not in [p[0] for p in picks]:
        picks.append((row["id"], f"Championship ({row['awayTeam']} @ {row['homeTeam']})"))
    else:
        # pick third-latest game
        remaining = ncaa[~ncaa["id"].isin([p[0] for p in picks])].sort_values("startDate", ascending=False)
        if not remaining.empty:
            row = remaining.iloc[0]
            picks.append((row["id"], f"Late round ({row['awayTeam']} @ {row['homeTeam']})"))

    return picks[:3]


BULK_PARAMS = [
    {"season": 2025, "seasonType": "postseason"},
    {"season": 2025, "tournament": "NCAA"},
]


def _fetch_plays_for_game(game_id, label):
    """Try all param variations; return (plays_list, bytes, working_param) or (None, 0, None)."""
    global WORKING_PARAM
    print(f"\n  Game: {label}  (id={game_id})")

    # If we already know which param works, try it first
    ordered = ([WORKING_PARAM] + [p for p in PARAM_VARIATIONS if p != WORKING_PARAM]
               if WORKING_PARAM else PARAM_VARIATIONS)

    for param in ordered:
        status, is_html, data, raw_bytes, err = _get_raw("/plays", {param: game_id})
        ok = data is not None
        print(f"    /plays {{{param}: {game_id}}}  →  status={status}  ok={ok}  {'err: '+err if err else ''}")
        if ok:
            plays = data if isinstance(data, list) else data.get("plays", [])
            if WORKING_PARAM is None:
                WORKING_PARAM = param
                print(f"    ✓ Working param: '{param}'")
            return plays, raw_bytes, param
        if status == 429:
            print("    ⚠ Rate limited — pausing 2s")
            time.sleep(2)

    # Per-game variations failed — try gameId + season combo
    combo = {"gameId": game_id, "season": 2025}
    status, is_html, data, raw_bytes, err = _get_raw("/plays", combo)
    ok = data is not None
    print(f"    /plays {combo}  →  status={status}  ok={ok}  {'err: '+err if err else ''}")
    if ok:
        plays = data if isinstance(data, list) else data.get("plays", [])
        if WORKING_PARAM is None:
            WORKING_PARAM = "gameId+season"
        return plays, raw_bytes, "gameId+season"

    return None, 0, None


def _try_bulk_params():
    """Try bulk (no game ID) param variations — only run once, on the first game."""
    print(f"\n  ── Bulk param variations (no game ID) ──")
    for params in BULK_PARAMS:
        status, is_html, data, raw_bytes, err = _get_raw("/plays", params)
        ok = data is not None
        n = len(data) if isinstance(data, list) else (len(data.get("plays", [])) if isinstance(data, dict) else 0)
        print(f"    /plays {params}  →  status={status}  ok={ok}  n={n}  {'err: '+err if err else ''}")
        if ok and n > 0:
            print(f"    ✓ Bulk fetch works — returns {n} plays across all matching games")


def _analyse_plays(plays, label):
    """Print field inventory and coverage for a fetched play list."""
    n = len(plays)
    if n == 0:
        print(f"    (0 plays returned)")
        return

    df = pd.json_normalize(plays, sep="_") if plays and isinstance(plays[0], dict) else pd.DataFrame()

    print(f"    Plays: {n:,}")
    print(f"    Columns ({len(df.columns)}): {list(df.columns)}")

    # Sample 3 plays
    print(f"\n    ── Sample plays (raw JSON) ──")
    for i, p in enumerate(plays[:3]):
        print(f"    [{i}] {json.dumps(p, indent=6)[:400]}")
        if len(json.dumps(p)) > 400:
            print(f"         ... (truncated)")

    if df.empty:
        return

    # Pagination check
    if isinstance(plays, dict):
        pag_keys = [k for k in plays.keys() if k in ("page", "total", "next", "totalPages", "hasMore")]
        if pag_keys:
            print(f"\n    ⚠ Pagination keys found: {pag_keys}")
        else:
            print(f"\n    Pagination: none detected (flat list response)")
    else:
        print(f"\n    Pagination: none detected (flat list response)")

    # shotInfo
    shot_col = next((c for c in df.columns if "shotInfo" in c or "shot_info" in c.lower()), None)
    pt_col   = next((c for c in df.columns if c in ("playType", "play_type", "type")), None)
    floor_col = next((c for c in df.columns if "onFloor" in c or "on_floor" in c.lower()), None)

    if pt_col:
        shooting_mask = df[pt_col].astype(str).str.lower().str.contains(
            "made|miss|shot|field goal|three", na=False, regex=True
        )
        n_shooting = shooting_mask.sum() or n
    else:
        n_shooting = n
        shooting_mask = pd.Series([True] * n)

    if shot_col:
        populated = df.loc[shooting_mask, shot_col].notna().sum()
        print(f"\n    shotInfo populated: {populated:,} / {n_shooting:,}  ({100*populated/n_shooting:.1f}% of shooting plays)")
    else:
        print(f"\n    shotInfo: no column found")

    # onFloor
    if floor_col:
        def _count(v):
            if isinstance(v, list): return len(v)
            try: return len(json.loads(str(v)))
            except: return 0
        counts = df[floor_col].apply(_count)
        exact10 = (counts == 10).sum()
        print(f"    onFloor populated (10 players): {exact10:,} / {n:,}  ({100*exact10/n:.1f}%)")
    else:
        print(f"    onFloor: no column found")

    # playType distribution
    if pt_col:
        print(f"\n    playType distribution (all unique values):")
        vc = df[pt_col].value_counts(dropna=False)
        for val, cnt in vc.items():
            print(f"      {cnt:>5,}  {val}")
    else:
        print(f"\n    playType: no column found")


def question_2():
    global FETCHED_GAMES

    print()
    hr("═")
    print("QUESTION 2 — Single game plays fetch")
    hr("═")

    ncaa = _load_2025_ncaa_games()
    if ncaa.empty:
        print("  ✗ Could not load 2025 NCAA games from cache.")
        SUMMARY["blocking_issues"].append("games_2025.csv missing or has no NCAA postseason rows")
        return

    print(f"  Loaded {len(ncaa)} NCAA postseason games from cache")
    picks = _pick_3_games(ncaa)
    if not picks:
        print("  ✗ No suitable games found.")
        return

    bulk_tried = False
    for game_id, label in picks:
        plays, raw_bytes, wp = _fetch_plays_for_game(game_id, label)
        if plays is not None:
            FETCHED_GAMES[game_id] = {"plays": plays, "bytes": raw_bytes, "label": label}
            _analyse_plays(plays, label)
        else:
            print(f"    ✗ All param variations failed for game {game_id}")
            SUMMARY["blocking_issues"].append(f"Plays endpoint returned no data for game {game_id} ({label})")

        # Try bulk params once, after the first game attempt
        if not bulk_tried:
            bulk_tried = True
            _try_bulk_params()

    if not FETCHED_GAMES:
        print("\n  ✗ Plays endpoint unreachable for all tested games.")
        print("    What was found:")
        print("      • HTML responses → endpoint path is wrong or auth redirects to login page")
        print("      • 401 → API key missing or lacks play-by-play access tier")
        print("      • 400 → Required parameter not accepted (check API docs for /plays)")
        print("    What to check:")
        print("      1. Confirm the endpoint path via API docs or Swagger UI")
        print("      2. Verify API key has play-by-play access (may be a paid tier)")
        SUMMARY["play_types_available"] = "NO (plays endpoint unreachable)"
        SUMMARY["action_level"]         = "UNKNOWN"


# ══════════════════════════════════════════════════════════════════════════════
# Q3 — Rate limit behavior
# ══════════════════════════════════════════════════════════════════════════════

def question_3():
    print()
    hr("═")
    print("QUESTION 3 — Rate limit behavior")
    hr("═")

    if not FETCHED_GAMES or WORKING_PARAM is None:
        print("  Skipping — no successful plays fetch in Q2.")
        return

    game_id = next(iter(FETCHED_GAMES))

    def _run_batch(n, sleep_s, label):
        times, statuses = [], []
        for i in range(n):
            t0 = time.time()
            status, _, data, _, err = _get_raw("/plays", {WORKING_PARAM: game_id})
            elapsed = time.time() - t0
            times.append(elapsed)
            statuses.append(status)
            marker = "✓" if data is not None else f"✗({status})"
            print(f"    [{label} {i+1}/{n}] {elapsed:.2f}s  {marker}  {err or ''}")
            if sleep_s > 0:
                time.sleep(sleep_s)

        rate_limited = statuses.count(429)
        avg_s = np.mean(times) if times else 0
        return times, rate_limited, avg_s

    print(f"  Using game_id={game_id}  param='{WORKING_PARAM}'")
    print()
    print("  ── Rapid burst (no sleep) ──")
    rapid_times, rapid_429, rapid_avg = _run_batch(3, 0, "rapid")

    print()
    print("  ── Paced (0.5s sleep) ──")
    paced_times, paced_429, paced_avg = _run_batch(3, 0.5, "paced")

    print()
    print(f"  Rapid avg:  {rapid_avg:.2f}s/req  |  429s: {rapid_429}/3")
    print(f"  Paced avg:  {paced_avg:.2f}s/req  |  429s: {paced_429}/3")

    if rapid_429 > 0:
        print(f"\n  ⚠ Rate limiting detected at burst rate.")
        print(f"    Recommended: use 0.5–1.0s sleep between requests.")
        safe_rpm = int(60 / (paced_avg + 0.5))
        SUMMARY["safe_fetch_rpm"] = f"~{safe_rpm} req/min (rate limited without sleep)"
    else:
        safe_rpm = int(60 / max(rapid_avg, 0.1))
        if paced_avg > rapid_avg * 1.5:
            print(f"\n  No rate limiting detected. Response time stable.")
        print(f"    Safe estimate: ~{safe_rpm} req/min (no 429s observed at burst)")
        SUMMARY["safe_fetch_rpm"] = f"~{safe_rpm} req/min (no rate limiting observed)"


# ══════════════════════════════════════════════════════════════════════════════
# Q4 — Response volume estimation
# ══════════════════════════════════════════════════════════════════════════════

def question_4():
    print()
    hr("═")
    print("QUESTION 4 — Response volume estimation")
    hr("═")

    if not FETCHED_GAMES:
        print("  Skipping — no plays data fetched in Q2.")
        return

    print(f"  {'Game':<45}  {'Plays':>6}  {'Bytes':>9}  {'B/play':>7}")
    hr()

    total_bytes = 0
    total_plays = 0
    for gid, info in FETCHED_GAMES.items():
        n     = len(info["plays"])
        b     = info["bytes"]
        bpp   = b / n if n > 0 else 0
        total_bytes += b
        total_plays += n
        print(f"  {info['label']:<45}  {n:>6,}  {b:>9,}  {bpp:>7.0f}")

    if not total_plays:
        print("  No plays data — cannot estimate sizes.")
        return

    avg_bpp = total_bytes / total_plays
    avg_plays_per_game = total_plays / len(FETCHED_GAMES)

    print()
    print(f"  Average: {avg_plays_per_game:.0f} plays/game  |  {avg_bpp:.0f} bytes/play")
    print()

    scopes = [
        ("63 NCAA tournament games",                  63),
        ("400 conf + NCAA postseason games",          400),
        ("650 postseason + last-10 regular season",   650),
        ("5,500 full regular season (1 yr)",         5500),
        ("55,000 full 10-season pull",              55000),
    ]

    print(f"  {'Scope':<45}  {'Games':>6}  {'Est. plays':>12}  {'Est. MB':>9}")
    hr()
    for label, games in scopes:
        est_plays = games * avg_plays_per_game
        est_mb    = (est_plays * avg_bpp) / 1_048_576
        print(f"  {label:<45}  {games:>6,}  {est_plays:>12,.0f}  {est_mb:>9.1f}")

    conf_ncaa_mb = 400 * avg_plays_per_game * avg_bpp / 1_048_576
    SUMMARY["size_conf_ncaa_1yr"] = f"{conf_ncaa_mb:.1f} MB/season"


# ══════════════════════════════════════════════════════════════════════════════
# Q5 — Play type coverage check
# ══════════════════════════════════════════════════════════════════════════════

def question_5():
    print()
    hr("═")
    print("QUESTION 5 — Play type coverage check")
    hr("═")

    if not FETCHED_GAMES:
        print("  Skipping — no plays data fetched in Q2.")
        return

    all_plays = []
    for info in FETCHED_GAMES.values():
        all_plays.extend(info["plays"])

    if not all_plays:
        return

    df = pd.json_normalize(all_plays, sep="_")
    total = len(df)
    print(f"  Total plays across all fetched games: {total:,}")

    pt_col    = next((c for c in df.columns if c in ("playType", "play_type", "type")), None)
    shot_col  = next((c for c in df.columns if "shotInfo" in c or "shot_info" in c.lower()), None)
    floor_col = next((c for c in df.columns if "onFloor" in c or "on_floor" in c.lower()), None)
    part_col  = next((c for c in df.columns if "participant" in c.lower()), None)

    # 1. playType fill rate
    print()
    if pt_col:
        pt_filled = df[pt_col].notna() & (df[pt_col].astype(str).str.strip() != "")
        print(f"  1. playType ('{pt_col}') populated: {pt_filled.sum():,}/{total:,}  ({100*pt_filled.mean():.1f}%)")
    else:
        print(f"  1. playType: no column found")

    # 2. shotInfo on shooting plays
    if pt_col:
        shooting_mask = df[pt_col].astype(str).str.lower().str.contains(
            "made|miss|shot|field goal|three|layup|dunk|jumper", na=False, regex=True
        )
    else:
        shooting_mask = pd.Series([True] * total)
    n_shooting = max(shooting_mask.sum(), 1)

    if shot_col:
        si_filled = df.loc[shooting_mask, shot_col].notna().sum()
        pct = 100 * si_filled / n_shooting
        print(f"  2. shotInfo populated: {si_filled:,}/{n_shooting:,} shooting plays  ({pct:.1f}%)")
        SUMMARY["shot_info_pct"] = f"{pct:.0f}%"
    else:
        print(f"  2. shotInfo: no column found")
        SUMMARY["shot_info_pct"] = "0% (no column)"

    # 3. onFloor with exactly 10 players
    if floor_col:
        def _count(v):
            if isinstance(v, list): return len(v)
            try: return len(json.loads(str(v)))
            except: return 0
        floor_counts = df[floor_col].apply(_count)
        exact10 = (floor_counts == 10).sum()
        pct = 100 * exact10 / total
        print(f"  3. onFloor (10 players): {exact10:,}/{total:,}  ({pct:.1f}%)")
        print(f"     Distribution: {dict(floor_counts.value_counts().head(5))}")
        SUMMARY["on_floor_pct"] = f"{pct:.0f}%"
    else:
        print(f"  3. onFloor: no column found")
        SUMMARY["on_floor_pct"] = "0% (no column)"

    # 4. Plays with empty participants
    if part_col:
        def _empty(v):
            if v is None: return True
            if isinstance(v, list): return len(v) == 0
            try: return len(json.loads(str(v))) == 0
            except: return str(v).strip() in ("", "[]", "None")
        empty_parts = df[part_col].apply(_empty).sum()
        print(f"  4. participants empty: {empty_parts:,}/{total:,}  ({100*empty_parts/total:.1f}%)")
    else:
        print(f"  4. participants: no column found")

    # 5. Top 15 playType values
    print()
    print(f"  5. Top 15 playType values:")
    if pt_col and pt_filled.sum() > 0:
        top15 = df[pt_col].value_counts().head(15)
        for val, cnt in top15.items():
            print(f"     {cnt:>5,}  {val}")

        # 6. Action-level search
        print()
        print(f"  6. Action-level play type check:")
        all_types_lower = {str(t).lower(): t for t in df[pt_col].dropna().unique()}
        any_action = False
        for category, keywords in ACTION_KEYWORDS.items():
            matches = [orig for low, orig in all_types_lower.items()
                       if any(kw in low for kw in keywords)]
            tick = "✓" if matches else "✗"
            print(f"     {tick} {category:<20}  {str(matches[:3]) if matches else '(not found)'}")
            if matches:
                any_action = True

        if any_action:
            SUMMARY["action_level"] = "YES (action-level types found)"
            SUMMARY["play_types_available"] = "YES"
        else:
            SUMMARY["action_level"] = "NO — event-level only"
            if "NO" not in str(SUMMARY["play_types_available"]):
                SUMMARY["play_types_available"] = "PARTIAL (event-level only)"
            if not any(
                "event-level" in issue for issue in SUMMARY["blocking_issues"]
            ):
                SUMMARY["blocking_issues"].append(
                    "Play types are event-level (made/missed/foul), not action-level. "
                    "Style clustering must derive play type distributions from event "
                    "sequences rather than reading them directly from playType."
                )
    else:
        print("     No playType data to analyse.")


# ══════════════════════════════════════════════════════════════════════════════
# Final summary
# ══════════════════════════════════════════════════════════════════════════════

def print_summary():
    print()
    hr("═")
    print("RECOMMENDATION")
    hr("═")

    blocking = SUMMARY["blocking_issues"]
    unreachable = "unreachable" in str(SUMMARY["play_types_available"]).lower()

    if unreachable:
        pilot = "Cannot run pilot — plays endpoint unreachable. Resolve auth/endpoint issue first."
        full  = "Same blocker applies to full pipeline."
    else:
        pilot = (
            "NCAA tournament only, 2–3 seasons (63 games/yr). "
            "Small enough to fully analyse field coverage and clustering signal."
        )
        full = (
            "Conf + NCAA postseason per season (~400 games). "
            "Covers all teams in simulation. Pull 5–10 seasons for robust clusters. "
            "Avoid full regular-season pull unless pace/shot-clock signals are needed."
        )

    print(f"Play types available:       {SUMMARY['play_types_available']}")
    print(f"Action-level plays:         {SUMMARY['action_level']}")
    print(f"shotInfo populated:         {SUMMARY['shot_info_pct']}")
    print(f"onFloor populated:          {SUMMARY['on_floor_pct']}")
    print(f"Safe fetch rate:            {SUMMARY['safe_fetch_rpm']}")
    print(f"Est. size (conf+NCAA, 1yr): {SUMMARY['size_conf_ncaa_1yr']}")
    print()
    print(f"Recommended scope for pilot:")
    print(f"  {pilot}")
    print()
    print(f"Recommended scope for full pipeline:")
    print(f"  {full}")
    print()
    print(f"Blocking issues:")
    if blocking:
        for issue in blocking:
            print(f"  • {issue}")
    else:
        print("  None")
    print()
    print(f"Total API calls made: {CALL_COUNT}")
    hr("═")


# ══════════════════════════════════════════════════════════════════════════════
# Main
# ══════════════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    if not API_KEY or API_KEY in ("YOUR_API_KEY", ""):
        print("ERROR: COLLEGE_FOOTBALL_API_KEY not set or is placeholder.")
        print("       Check .env file in project root.")
        sys.exit(1)

    print(f"API key: ...{API_KEY[-4:]}")

    for q, fn in [(1, question_1), (2, question_2), (3, question_3),
                  (4, question_4), (5, question_5)]:
        try:
            fn()
        except Exception as e:
            print(f"\n  Q{q} crashed unexpectedly: {e}")

    print_summary()
