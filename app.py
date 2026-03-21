# app.py
"""
This Is March — Bracket Explorer

Pages:
  Brackets         — View model-generated bracket scenarios
  My Bracket       — Build your own bracket with model guidance + probability scoring
  Matchup Simulator — Head-to-head win probability for any two teams

Run:
  streamlit run app.py
"""

import os
import sys
import json
import math
from pathlib import Path

import streamlit as st
import pandas as pd
import joblib

sys.path.insert(0, os.path.dirname(__file__))

# ── Page config ───────────────────────────────────────────────────────────────

st.set_page_config(
    page_title="This Is March",
    page_icon="🏀",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── Sidebar navigation ────────────────────────────────────────────────────────

PAGES = ["🏆 Brackets", "🖊️ My Bracket", "⚡ Matchup Simulator"]
page = st.sidebar.radio("Navigation", PAGES)

st.sidebar.divider()
st.sidebar.caption("This Is March — NCAA Bracket Model")

# ── Shared loaders ────────────────────────────────────────────────────────────

@st.cache_resource(show_spinner="Loading model...")
def load_model(model_name: str):
    if model_name == "LogReg (post+conf)":
        from models.logistic_predictor import LogisticPredictor
        return LogisticPredictor.load()
    else:
        from models.game_predictor_model import GamePredictor
        return GamePredictor.load()


@st.cache_resource(show_spinner="Loading features...")
def load_features():
    path = Path("data/cache/features_by_season.pkl")
    if not path.exists():
        return None
    return joblib.load(path)


@st.cache_data(show_spinner=False)
def _bracket_seeds_for_season(season: int) -> dict:
    """Return {team_api_name: seed} for all bracket teams in a given season."""
    from validation.simulation import _normalize
    json_path = Path(f"data/bracket_{season}.json")
    if json_path.exists():
        with open(json_path) as f:
            bj = json.load(f)
        seeds = {}
        for region_data in bj["regions"].values():
            for m in region_data["matchups"]:
                seeds[_normalize(m["team_a"])] = int(m["seed_a"])
                seeds[_normalize(m["team_b"])] = int(m["seed_b"])
        return seeds
    try:
        from data.bracket_builder import build_bracket_from_cache
        bd = build_bracket_from_cache(season)
        return {t: int(s) for t, s in bd["seeds"].items()}
    except Exception:
        return {}


@st.cache_data(show_spinner=False)
def _bracket_win_rates_for_season(season: int) -> dict:
    """Compute regular-season win rates vs. tournament-field opponents for a season."""
    games_path = Path(f"data/cache/raw/games_{season}.csv")
    if not games_path.exists():
        return {}
    try:
        games_df = pd.read_csv(str(games_path))
        seeds = _bracket_seeds_for_season(season)
        if not seeds:
            return {}
        from features.builder import compute_bracket_win_rate
        return compute_bracket_win_rate(games_df, season, set(seeds.keys()))
    except Exception:
        return {}


@st.cache_data(show_spinner="Running simulation...")
def run_simulation(
    model_name: str, season: int, n_sims: int,
    dampen_strength: float = 1.0,
    noise_sigma: float = 0.0,
    chaos_fraction: float = 0.0,
):
    from validation.simulation import BracketSimulator, ROUND_ALPHAS

    features_by_season = load_features()
    if features_by_season is None:
        return None, None, None, None

    features = features_by_season.get(season)
    if features is None:
        return None, None, None, None

    model = load_model(model_name)

    # Scale ROUND_ALPHAS by dampen_strength.
    # dampen_strength=1.0 → full dampening (use ROUND_ALPHAS as-is)
    # dampen_strength=0.0 → no dampening (all alphas = 1.0, raw model probs)
    round_alphas = {r: 1.0 - (1.0 - a) * dampen_strength for r, a in ROUND_ALPHAS.items()}

    sim_kwargs = dict(
        model=model,
        features_2025=features,
        n_simulations=n_sims,
        round_alphas=round_alphas,
        noise_sigma=noise_sigma,
        chaos_fraction=chaos_fraction,
        season=season,
    )

    bracket_path = Path(f"data/bracket_{season}.json")
    if season >= 2026 and bracket_path.exists():
        sim = BracketSimulator(bracket_path=str(bracket_path), **sim_kwargs)
        with open(bracket_path) as f:
            bracket_json = json.load(f)
    else:
        from data.bracket_builder import build_bracket_from_cache
        bracket_data = build_bracket_from_cache(season)
        sim = BracketSimulator(bracket_data=bracket_data, **sim_kwargs)
        bracket_json = bracket_data["bracket"]

    results = sim.simulate()
    return results, sim.prob_matrix, sim.prob_matrix_unseeded, bracket_json


def _ensure_simulation(model_choice, season, n_sims, dampen_strength, noise_sigma, chaos_fraction):
    """Load simulation into session_state if not already present or settings changed."""
    key = (model_choice, season, n_sims, dampen_strength, noise_sigma, chaos_fraction)
    if st.session_state.get("_sim_key") != key or "sim_results" not in st.session_state:
        with st.spinner("Running simulation..."):
            results, pm, pmu, bj = run_simulation(
                model_choice, season, n_sims, dampen_strength, noise_sigma, chaos_fraction
            )
        if results is None:
            st.error("Could not load features or bracket. Run `python data/pull.py` first.")
            st.stop()
        st.session_state["sim_results"]          = results
        st.session_state["prob_matrix"]          = pm
        st.session_state["prob_matrix_unseeded"] = pmu
        st.session_state["bracket_json"]         = bj
        st.session_state["_sim_key"]             = key


# ── Shared rendering helpers ──────────────────────────────────────────────────

ROUND_LABELS = {
    "r64": "Round of 64", "r32": "Round of 32",
    "s16": "Sweet 16",    "e8":  "Elite 8",
}
_CHAMPION_COLOR = "#FFD700"


def _seed_str(team: str, seeds: dict) -> str:
    s = seeds.get(team)
    return f"({s})" if s else ""


def _render_region(region_name: str, region_data: dict, seeds: dict):
    st.markdown(f"#### {region_name}")
    rows = []
    for rnd_key in ["r64", "r32", "s16", "e8"]:
        for team_a, team_b, winner in region_data.get(rnd_key, []):
            loser = team_b if winner == team_a else team_a
            upset = (
                isinstance(seeds.get(winner), int) and isinstance(seeds.get(loser), int)
                and seeds[winner] > seeds[loser]
            )
            rows.append({
                "Round":  ROUND_LABELS[rnd_key],
                "Team A": f"{team_a} {_seed_str(team_a, seeds)}",
                "Team B": f"{team_b} {_seed_str(team_b, seeds)}",
                "Winner": f"{'⚡ ' if upset else ''}{winner} {_seed_str(winner, seeds)}",
            })
    st.dataframe(pd.DataFrame(rows), use_container_width=True, hide_index=True)


def _render_bracket(b: dict, seeds: dict):
    champ, prob, upsets = b["champion"], b.get("champion_prob", 0.0), b.get("upset_count", 0)
    col1, col2, col3 = st.columns(3)
    col1.metric("Champion", champ)
    col2.metric("Model Win Probability", f"{prob:.1%}")
    col3.metric("R64 Upsets", upsets)
    st.divider()
    region_items = list(b["regions"].items())
    for i in range(0, 4, 2):
        ca, cb = st.columns(2)
        with ca: _render_region(region_items[i][0],   region_items[i][1],   seeds)
        with cb: _render_region(region_items[i+1][0], region_items[i+1][1], seeds)
    st.divider()
    st.markdown("#### Final Four")
    st.dataframe(pd.DataFrame([
        {"Matchup": f"{a} {_seed_str(a, seeds)}  vs  {b} {_seed_str(b, seeds)}",
         "Winner": f"{w} {_seed_str(w, seeds)}"}
        for a, b, w in b["final_four"]
    ]), use_container_width=True, hide_index=True)
    st.markdown("#### 🏆 Championship")
    a, b_, w = b["championship"]
    st.markdown(
        f"**{a} {_seed_str(a, seeds)}** vs **{b_} {_seed_str(b_, seeds)}** → "
        f"<span style='color:{_CHAMPION_COLOR};font-weight:700;font-size:1.1em'>"
        f"🏆 {w} {_seed_str(w, seeds)}</span>",
        unsafe_allow_html=True,
    )
    if b.get("upset_details"):
        st.divider()
        st.markdown("#### Upset picks")
        st.dataframe(pd.DataFrame([
            {"Underdog": f"{u['underdog']} (#{u['underdog_seed']})",
             "Over": f"{u['favorite']} (#{u['favorite_seed']})",
             "Model P(upset)": f"{u['p_upset']:.1%}"}
            for u in b["upset_details"]
        ]), use_container_width=True, hide_index=True)


def _render_win_prob_table(sim_results):
    rows = []
    for team, probs in sim_results.win_probs.items():
        rows.append({"Team": team, "Seed": sim_results.team_seeds.get(team, "?"),
                     "R64": probs.get(1, 0), "R32": probs.get(2, 0),
                     "S16": probs.get(3, 0), "E8":  probs.get(4, 0),
                     "F4":  probs.get(5, 0), "Champ": probs.get(6, 0)})
    df = pd.DataFrame(rows).sort_values("Champ", ascending=False).reset_index(drop=True)
    for col in ["R64", "R32", "S16", "E8", "F4", "Champ"]:
        df[col] = df[col].apply(lambda x: f"{x:.1%}")
    return df


# ── My Bracket helpers ────────────────────────────────────────────────────────

REGIONS = ["South", "East", "West", "Midwest"]

# Slot pairing within a region across rounds
# R1 slots G1..G8 → R2 G1..G4 → R3 G1..G2 → R4 G1
_ROUND_SLOT_PAIRS = {
    2: [(1, 2), (3, 4), (5, 6), (7, 8)],
    3: [(1, 2), (3, 4)],
    4: [(1, 2)],
}
_ROUND_PREFIX = {1: "R1", 2: "R2", 3: "R3", 4: "R4"}


def _slot(region: str, rnd: int, game: int) -> str:
    return f"{region[0]}_{_ROUND_PREFIX[rnd]}_G{game}"


def _get_matrix(rnd: int, pm: dict, pmu: dict) -> dict:
    return pmu if rnd >= 4 else pm


def _build_round_matchups(rnd: int, bracket_json: dict, picks: dict, seeds: dict,
                           pm: dict, pmu: dict) -> list[dict]:
    """Return list of game dicts for the given round based on current picks."""
    matchups = []

    if rnd == 1:
        for region in REGIONS:
            for m in bracket_json["regions"][region]["matchups"]:
                from validation.simulation import _normalize
                a = _normalize(m["team_a"])
                b = _normalize(m["team_b"])
                matrix = _get_matrix(1, pm, pmu)
                matchups.append({
                    "slot": m["slot"], "region": region,
                    "team_a": a, "team_b": b,
                    "seed_a": seeds.get(a), "seed_b": seeds.get(b),
                    "prob_a": matrix.get(a, {}).get(b, 0.5),
                })
        return matchups

    if rnd in (2, 3, 4):
        prev_rnd = rnd - 1
        for region in REGIONS:
            for (g1, g2) in _ROUND_SLOT_PAIRS[rnd]:
                slot_prev_a = _slot(region, prev_rnd, g1)
                slot_prev_b = _slot(region, prev_rnd, g2)
                team_a = picks.get(slot_prev_a)
                team_b = picks.get(slot_prev_b)
                if not team_a or not team_b:
                    continue
                game_num = _ROUND_SLOT_PAIRS[rnd].index((g1, g2)) + 1
                slot = _slot(region, rnd, game_num)
                matrix = _get_matrix(rnd, pm, pmu)
                matchups.append({
                    "slot": slot, "region": region,
                    "team_a": team_a, "team_b": team_b,
                    "seed_a": seeds.get(team_a), "seed_b": seeds.get(team_b),
                    "prob_a": matrix.get(team_a, {}).get(team_b, 0.5),
                })
        return matchups

    if rnd == 5:  # Final Four
        for sf_slot, reg_a, reg_b in [("FF_G1", "South", "East"), ("FF_G2", "West", "Midwest")]:
            team_a = picks.get(_slot(reg_a, 4, 1))
            team_b = picks.get(_slot(reg_b, 4, 1))
            if not team_a or not team_b:
                continue
            matrix = _get_matrix(5, pm, pmu)
            matchups.append({
                "slot": sf_slot, "region": "Final Four",
                "team_a": team_a, "team_b": team_b,
                "seed_a": seeds.get(team_a), "seed_b": seeds.get(team_b),
                "prob_a": matrix.get(team_a, {}).get(team_b, 0.5),
            })
        return matchups

    if rnd == 6:  # Championship
        team_a = picks.get("FF_G1")
        team_b = picks.get("FF_G2")
        if not team_a or not team_b:
            return []
        matrix = _get_matrix(6, pm, pmu)
        return [{
            "slot": "CHAMP", "region": "Championship",
            "team_a": team_a, "team_b": team_b,
            "seed_a": seeds.get(team_a), "seed_b": seeds.get(team_b),
            "prob_a": matrix.get(team_a, {}).get(team_b, 0.5),
        }]

    return []


def _bracket_probability(picks: dict, pick_probs: dict) -> tuple[float, dict]:
    """
    Returns (overall_prob, {round: prob}) for all completed picks.
    Probability = product of P(picked winner wins their specific game).
    """
    round_probs = {}
    overall = 1.0
    for rnd in range(1, 7):
        rnd_picks = {k: v for k, v in pick_probs.items()
                     if k.startswith(("S_", "E_", "W_", "M_")) and picks.get(k)
                     or k in ("FF_G1", "FF_G2", "CHAMP") and picks.get(k)}
    # group by round
    for rnd in range(1, 7):
        rnd_slots = [s for s in pick_probs if picks.get(s) and _slot_round(s) == rnd]
        if not rnd_slots:
            continue
        p = 1.0
        for s in rnd_slots:
            p *= pick_probs[s]
        round_probs[rnd] = p
        overall *= p
    return overall, round_probs


def _slot_round(slot: str) -> int:
    if "R1" in slot: return 1
    if "R2" in slot: return 2
    if "R3" in slot: return 3
    if "R4" in slot: return 4
    if slot.startswith("FF"):  return 5
    if slot == "CHAMP":        return 6
    return 0


def _render_game(game: dict, picks: dict, pick_probs: dict, seeds: dict, rnd: int):
    """Render a single matchup with pick buttons and model probability."""
    slot  = game["slot"]
    a, b  = game["team_a"], game["team_b"]
    pa    = game["prob_a"]
    pb    = 1.0 - pa
    sa    = game.get("seed_a")
    sb    = game.get("seed_b")
    picked = picks.get(slot)

    label_a = f"#{sa} {a}" if sa else a
    label_b = f"#{sb} {b}" if sb else b

    # Determine favorite styling
    fav_a = pa >= 0.5

    col_a, col_mid, col_b = st.columns([5, 2, 5])

    with col_a:
        prob_color_a = "#1a7f37" if fav_a else "#888"
        st.markdown(
            f"<div style='text-align:right;'>"
            f"<span style='font-size:1.05em;font-weight:600'>{label_a}</span><br>"
            f"<span style='color:{prob_color_a};font-size:0.95em'>{pa:.1%}</span>"
            f"</div>",
            unsafe_allow_html=True,
        )
        if st.button("Pick ▶", key=f"pick_a_{slot}", use_container_width=True,
                     type="primary" if picked == a else "secondary"):
            picks[slot]      = a
            pick_probs[slot] = pa
            # Clear downstream picks that depended on this slot
            _clear_downstream(slot, picks, pick_probs)
            st.rerun()

    with col_mid:
        st.markdown(
            "<div style='text-align:center;padding-top:8px;color:#aaa;font-size:0.85em'>vs</div>",
            unsafe_allow_html=True,
        )
        if picked:
            winner_label = "✅ " + (a if picked == a else b)
            st.markdown(
                f"<div style='text-align:center;font-size:0.8em;color:#555'>{winner_label}</div>",
                unsafe_allow_html=True,
            )

    with col_b:
        prob_color_b = "#1a7f37" if not fav_a else "#888"
        st.markdown(
            f"<div style='text-align:left;'>"
            f"<span style='font-size:1.05em;font-weight:600'>{label_b}</span><br>"
            f"<span style='color:{prob_color_b};font-size:0.95em'>{pb:.1%}</span>"
            f"</div>",
            unsafe_allow_html=True,
        )
        if st.button("◀ Pick", key=f"pick_b_{slot}", use_container_width=True,
                     type="primary" if picked == b else "secondary"):
            picks[slot]      = b
            pick_probs[slot] = pb
            _clear_downstream(slot, picks, pick_probs)
            st.rerun()


def _clear_downstream(slot: str, picks: dict, pick_probs: dict):
    """When a pick changes, remove any later picks that depended on that team advancing."""
    # Identify which slots feed from this one and clear transitively
    to_clear = _downstream_slots(slot)
    for s in to_clear:
        picks.pop(s, None)
        pick_probs.pop(s, None)


def _downstream_slots(slot: str) -> list[str]:
    """Return all slots downstream of this one (not including the slot itself)."""
    rnd = _slot_round(slot)
    if rnd == 0:
        return []

    affected = []

    if rnd <= 4:
        parts = slot.split("_")
        region_prefix = parts[0]  # S, E, W, M
        game_num = int(parts[-1].replace("G", ""))

        # Walk forward through later rounds within the region (rnd+1 through R4)
        for next_rnd in range(rnd + 1, 5):
            next_game = math.ceil(game_num / (2 ** (next_rnd - rnd)))
            next_slot = f"{region_prefix}_{_ROUND_PREFIX[next_rnd]}_G{next_game}"
            affected.append(next_slot)
            game_num = next_game

        # E8 winner always feeds into a Final Four semi — add FF and Championship
        ff_slot = "FF_G1" if region_prefix in ("S", "E") else "FF_G2"
        affected.append(ff_slot)
        affected.append("CHAMP")

    elif rnd == 5:
        affected.append("CHAMP")

    return list(dict.fromkeys(affected))


# ══════════════════════════════════════════════════════════════════════════════
# PAGE: BRACKETS
# ══════════════════════════════════════════════════════════════════════════════

if page == "🏆 Brackets":
    st.title("🏀 This Is March — Bracket Explorer")

    st.sidebar.header("Settings")
    model_choice = st.sidebar.selectbox("Model", ["LogReg (post+conf)", "HistGBT (trimmed post+conf)"])
    season       = st.sidebar.selectbox("Season", [2026, 2025, 2024, 2023, 2022])
    n_sims       = st.sidebar.select_slider(
        "Simulations", options=[1_000, 5_000, 10_000, 25_000, 50_000], value=10_000
    )

    with st.sidebar.expander("Simulation Tuning", expanded=False):
        dampen_strength = st.slider(
            "Dampening", 0.0, 1.0, 1.0, 0.05,
            help="Compresses win probabilities toward 50/50 in later rounds. "
                 "1.0 = default round-aware dampening. 0.0 = raw model probabilities.",
        )
        noise_sigma = st.slider(
            "Noise σ", 0.0, 0.15, 0.0, 0.01,
            help="Gaussian noise added to each game's probability before resolving. "
                 "Adds game-day variance. 0.0 = off.",
        )
        chaos_fraction = st.slider(
            "Chaos Fraction", 0.0, 0.50, 0.0, 0.05,
            help="Fraction of simulations run with aggressive dampening + noise (chaos mode). "
                 "Blends the chaos results in with normal sims to fatten the upset tail. 0.0 = off.",
        )

    run_btn = st.sidebar.button("Run Simulation", type="primary", use_container_width=True)

    if run_btn:
        st.session_state.pop("_sim_key", None)  # force refresh

    _ensure_simulation(model_choice, season, n_sims, dampen_strength, noise_sigma, chaos_fraction)

    results              = st.session_state["sim_results"]
    prob_matrix          = st.session_state["prob_matrix"]
    prob_matrix_unseeded = st.session_state["prob_matrix_unseeded"]
    bracket_json         = st.session_state["bracket_json"]
    seeds                = results.team_seeds

    from brackets.generator import generate_all
    all_brackets = generate_all(results, prob_matrix, bracket_json, prob_matrix_unseeded)

    tab_names = [b["name"] for b in all_brackets] + ["📊 Win Probabilities"]
    tabs = st.tabs(tab_names)

    for tab, b in zip(tabs[:-1], all_brackets):
        with tab:
            st.markdown(f"*{b['description']}*")
            st.divider()
            _render_bracket(b, seeds)

    with tabs[-1]:
        st.markdown("### Model win probabilities — all 64 teams")
        st.markdown(f"*{n_sims:,} Monte Carlo simulations — {model_choice}*")
        st.dataframe(_render_win_prob_table(results),
                     use_container_width=True, hide_index=True, height=600)


# ══════════════════════════════════════════════════════════════════════════════
# PAGE: MY BRACKET
# ══════════════════════════════════════════════════════════════════════════════

elif page == "🖊️ My Bracket":
    st.title("🖊️ My Bracket")
    st.markdown(
        "Pick your winners round by round. "
        "The model shows win probability for each matchup. "
        "Your bracket's probability score updates as you pick."
    )

    st.sidebar.header("Settings")
    model_choice = st.sidebar.selectbox("Model", ["LogReg (post+conf)", "HistGBT (trimmed post+conf)"])
    season       = st.sidebar.selectbox("Season", [2026, 2025, 2024, 2023, 2022])
    n_sims       = st.sidebar.select_slider(
        "Simulations", options=[1_000, 5_000, 10_000, 25_000, 50_000], value=10_000
    )

    with st.sidebar.expander("Simulation Tuning", expanded=False):
        dampen_strength = st.slider(
            "Dampening", 0.0, 1.0, 1.0, 0.05,
            help="Compresses win probabilities toward 50/50 in later rounds. "
                 "1.0 = default round-aware dampening. 0.0 = raw model probabilities.",
        )
        noise_sigma = st.slider(
            "Noise σ", 0.0, 0.15, 0.0, 0.01,
            help="Gaussian noise added to each game's probability before resolving. "
                 "Adds game-day variance. 0.0 = off.",
        )
        chaos_fraction = st.slider(
            "Chaos Fraction", 0.0, 0.50, 0.0, 0.05,
            help="Fraction of simulations run with aggressive dampening + noise (chaos mode). "
                 "Blends the chaos results in with normal sims to fatten the upset tail. 0.0 = off.",
        )

    _ensure_simulation(model_choice, season, n_sims, dampen_strength, noise_sigma, chaos_fraction)

    results      = st.session_state["sim_results"]
    pm           = st.session_state["prob_matrix"]
    pmu          = st.session_state["prob_matrix_unseeded"]
    bracket_json = st.session_state["bracket_json"]
    seeds        = results.team_seeds

    # Per-page pick state (reset clears picks but not simulation)
    if "my_picks" not in st.session_state:
        st.session_state["my_picks"]      = {}
        st.session_state["my_pick_probs"] = {}

    picks      = st.session_state["my_picks"]
    pick_probs = st.session_state["my_pick_probs"]

    if st.sidebar.button("🗑️ Clear All Picks", use_container_width=True):
        st.session_state["my_picks"]      = {}
        st.session_state["my_pick_probs"] = {}
        st.rerun()

    # ── Sidebar: bracket probability ─────────────────────────────────────────
    st.sidebar.divider()
    st.sidebar.markdown("### Bracket Probability")

    n_picked = len(picks)
    if n_picked == 0:
        st.sidebar.caption("Make picks to see probability.")
    else:
        ROUND_NAMES = {1: "R64", 2: "R32", 3: "S16", 4: "E8", 5: "F4", 6: "Champ"}
        # Compute per-round probability
        round_slot_map = {r: [] for r in range(1, 7)}
        for s in picks:
            r = _slot_round(s)
            if r > 0:
                round_slot_map[r].append(s)

        overall_log = 0.0
        for rnd in range(1, 7):
            slots = round_slot_map[rnd]
            if not slots:
                continue
            p_rnd = 1.0
            for s in slots:
                p_rnd *= pick_probs.get(s, 1.0)
            overall_log += math.log10(p_rnd) if p_rnd > 0 else -99
            st.sidebar.metric(
                ROUND_NAMES[rnd],
                f"{p_rnd:.2%}",
                help=f"Probability all {len(slots)} {ROUND_NAMES[rnd]} picks are correct"
            )

        st.sidebar.divider()
        overall = 10 ** overall_log if overall_log > -90 else 0.0
        st.sidebar.metric(
            "Overall Bracket",
            f"1 in {int(1/overall):,}" if overall > 0 else "—",
            help="Probability this exact bracket occurs end-to-end",
        )
        st.sidebar.caption(f"{n_picked}/63 games picked")

    # ── Main: rounds as tabs ──────────────────────────────────────────────────
    ROUND_TAB_NAMES = ["R64", "R32 (Sweet 32)", "Sweet 16", "Elite 8", "Final Four", "Championship"]
    round_tabs = st.tabs(ROUND_TAB_NAMES)

    for rnd, tab in enumerate(round_tabs, start=1):
        with tab:
            matchups = _build_round_matchups(rnd, bracket_json, picks, seeds, pm, pmu)

            if not matchups:
                st.info(f"Complete the previous round to unlock {ROUND_TAB_NAMES[rnd-1]} matchups.")
                continue

            # Group by region for display
            by_region: dict[str, list] = {}
            for g in matchups:
                by_region.setdefault(g["region"], []).append(g)

            for region_name, games in by_region.items():
                if len(by_region) > 1:
                    st.markdown(f"**{region_name}**")
                for game in games:
                    _render_game(game, picks, pick_probs, seeds, rnd)
                    st.markdown("---")


# ══════════════════════════════════════════════════════════════════════════════
# PAGE: MATCHUP SIMULATOR
# ══════════════════════════════════════════════════════════════════════════════

elif page == "⚡ Matchup Simulator":
    import numpy as _np
    from features.builder import build_matchup_features
    from validation.simulation import (
        scale_matchup, ROUND_FEATURE_SCALES, ROUND_ALPHAS, CHAOS_ALPHAS, dampen,
    )

    st.title("⚡ Matchup Simulator")
    st.markdown(
        "Head-to-head win probability for any two teams. "
        "Seeds are pulled from the bracket when available, otherwise inferred from efficiency ratio. "
        "Round selection applies round-specific feature scaling."
    )

    # ── Sidebar ───────────────────────────────────────────────────────────────
    st.sidebar.header("Settings")
    model_choice = st.sidebar.selectbox("Model", ["LogReg (post+conf)", "HistGBT (trimmed post+conf)"])
    season       = st.sidebar.selectbox("Season", [2026, 2025, 2024, 2023, 2022])

    _ROUND_OPTS = [
        ("Round of 64",   1), ("Round of 32", 2), ("Sweet 16",     3),
        ("Elite 8",       4), ("Final Four",  5), ("Championship", 6),
    ]
    round_label = st.sidebar.selectbox("Round", [r[0] for r in _ROUND_OPTS])
    round_num   = dict(_ROUND_OPTS)[round_label]

    neutral_site = st.sidebar.toggle("Neutral Site", value=True)

    st.sidebar.divider()
    ms_noise_sigma = st.sidebar.slider(
        "Noise σ", 0.0, 0.15, 0.0, 0.01,
        help="Gaussian noise added to the probability before resolving. "
             "Shows game-day variance as an 80% confidence interval. 0.0 = point estimate.",
    )
    ms_chaos_fraction = st.sidebar.slider(
        "Chaos Fraction", 0.0, 1.0, 0.0, 0.05,
        help="Blend aggressive late-round dampening into the probability. "
             "0.0 = standard dampening; 1.0 = full chaos mode.",
    )

    # ── Load data ─────────────────────────────────────────────────────────────
    features_by_season = load_features()
    if features_by_season is None or season not in features_by_season:
        st.error("Features not found. Run `python data/pull.py` first.")
        st.stop()

    season_feats   = features_by_season[season]
    all_teams      = sorted(season_feats.keys())
    bracket_seeds  = _bracket_seeds_for_season(season)
    bracket_win_rates = _bracket_win_rates_for_season(season)

    # Build seed → median efficiency_ratio lookup for inference
    _seed_er_map: dict[int, list] = {}
    for _t, _s in bracket_seeds.items():
        _er = season_feats.get(_t, {}).get("efficiency_ratio")
        if _er is not None and not (isinstance(_er, float) and math.isnan(_er)):
            _seed_er_map.setdefault(int(_s), []).append(float(_er))
    _seed_medians = {s: float(_np.median(ers)) for s, ers in _seed_er_map.items() if ers}

    def _infer_seed(team: str) -> int:
        if team in bracket_seeds:
            return int(bracket_seeds[team])
        er = season_feats.get(team, {}).get("efficiency_ratio")
        if er is None or (isinstance(er, float) and math.isnan(er)) or not _seed_medians:
            return 8
        return min(_seed_medians, key=lambda s: abs(_seed_medians[s] - float(er)))

    # ── Team selectors ────────────────────────────────────────────────────────
    col1, col2 = st.columns(2)

    with col1:
        st.subheader("Team A")
        team_a = st.selectbox("Select Team A", all_teams, key="ms_team_a")
        inf_a  = _infer_seed(team_a)
        seed_a = st.number_input(
            "Seed", min_value=1, max_value=16, value=inf_a,
            key=f"ms_seed_a_{team_a}_{season}",
        )
        if team_a in bracket_seeds:
            st.caption(f"✓ Bracket seed #{bracket_seeds[team_a]}")
        else:
            st.caption(f"Not in bracket — seed inferred from efficiency ratio (≈ #{inf_a})")

    with col2:
        st.subheader("Team B")
        default_b_idx = min(1, len(all_teams) - 1)
        team_b = st.selectbox("Select Team B", all_teams, index=default_b_idx, key="ms_team_b")
        inf_b  = _infer_seed(team_b)
        seed_b = st.number_input(
            "Seed", min_value=1, max_value=16, value=inf_b,
            key=f"ms_seed_b_{team_b}_{season}",
        )
        if team_b in bracket_seeds:
            st.caption(f"✓ Bracket seed #{bracket_seeds[team_b]}")
        else:
            st.caption(f"Not in bracket — seed inferred from efficiency ratio (≈ #{inf_b})")

    st.divider()

    if team_a == team_b:
        st.warning("Select two different teams.")
        st.stop()

    feats_a = season_feats.get(team_a)
    feats_b = season_feats.get(team_b)
    if feats_a is None or feats_b is None:
        st.error("Feature data missing for one or both teams.")
        st.stop()

    # ── Compute matchup ───────────────────────────────────────────────────────
    model = load_model(model_choice)

    # Inject bracket_win_rate (same as _build_prob_matrices does at sim time)
    feats_a_bwr = {**feats_a, "bracket_win_rate": bracket_win_rates.get(team_a, float("nan"))}
    feats_b_bwr = {**feats_b, "bracket_win_rate": bracket_win_rates.get(team_b, float("nan"))}

    # Seeds active only for rounds 1–3; zeroed for E8+
    use_seed_a = int(seed_a) if round_num <= 3 else None
    use_seed_b = int(seed_b) if round_num <= 3 else None

    matchup = build_matchup_features(
        feats_a_bwr, feats_b_bwr,
        neutral_site=neutral_site,
        seed_a=use_seed_a,
        seed_b=use_seed_b,
    )
    if round_num >= 4:
        matchup["seed_diff"] = 0

    matchup_scaled = scale_matchup(matchup, round_num, ROUND_FEATURE_SCALES)
    X = pd.DataFrame([matchup_scaled])
    p_raw = float(model.predict_proba(X)[0])

    # Dampen: blend normal + chaos
    p_normal = dampen(p_raw, round_num, ROUND_ALPHAS)
    p_chaos  = dampen(p_raw, round_num, CHAOS_ALPHAS)
    p_damp   = (1.0 - ms_chaos_fraction) * p_normal + ms_chaos_fraction * p_chaos

    # Noise: Monte Carlo samples for confidence interval
    p_lo = p_hi = None
    if ms_noise_sigma > 0:
        rng     = _np.random.default_rng(42)
        samples = _np.clip(p_damp + rng.normal(0, ms_noise_sigma, 2000), 0.05, 0.95)
        p_final = float(samples.mean())
        p_lo    = float(_np.percentile(samples, 10))
        p_hi    = float(_np.percentile(samples, 90))
    else:
        p_final = p_damp

    p_b = 1.0 - p_final

    # ── Display: win probability ───────────────────────────────────────────────
    c1, c2, c3 = st.columns([5, 2, 5])
    color_a = "#1a7f37" if p_final >= 0.5 else "#888"
    color_b = "#1a7f37" if p_b    >= 0.5 else "#888"

    with c1:
        st.markdown(
            f"<div style='text-align:center'>"
            f"<div style='font-size:1.2em;font-weight:600'>#{int(seed_a)} {team_a}</div>"
            f"<div style='font-size:3em;font-weight:800;color:{color_a};line-height:1.1'>"
            f"{p_final:.1%}</div>"
            f"</div>",
            unsafe_allow_html=True,
        )
    with c2:
        st.markdown(
            "<div style='text-align:center;padding-top:32px;"
            "font-size:1.2em;color:#aaa'>vs</div>",
            unsafe_allow_html=True,
        )
    with c3:
        st.markdown(
            f"<div style='text-align:center'>"
            f"<div style='font-size:1.2em;font-weight:600'>#{int(seed_b)} {team_b}</div>"
            f"<div style='font-size:3em;font-weight:800;color:{color_b};line-height:1.1'>"
            f"{p_b:.1%}</div>"
            f"</div>",
            unsafe_allow_html=True,
        )

    # Context line
    ctx_parts = [f"Raw model: {p_raw:.1%}", round_label]
    if round_num <= 3 and "seed_diff" in matchup:
        ctx_parts.append(f"Seed diff: {int(matchup['seed_diff']):+d}")
    ctx_parts.append("Neutral" if neutral_site else "Home/Away")
    if ms_noise_sigma > 0 and p_lo is not None:
        ctx_parts.append(f"80% CI: {p_lo:.1%}–{p_hi:.1%}")
    if ms_chaos_fraction > 0:
        ctx_parts.append(f"Chaos: {ms_chaos_fraction:.0%}")

    st.markdown(
        f"<div style='text-align:center;color:#777;font-size:0.85em;margin-top:4px'>"
        + "  ·  ".join(ctx_parts) + "</div>",
        unsafe_allow_html=True,
    )

    st.divider()

    # ── Display: feature comparison ───────────────────────────────────────────
    st.markdown("### Feature Breakdown")
    st.caption(
        f"Positive values favor {team_a}; negative favor {team_b}. "
        f"'Scaled' reflects round-specific weighting applied before model prediction."
    )

    _FEAT_LABELS = {
        "diff_efficiency_ratio": "Efficiency Ratio",
        "diff_adj_off_rating":   "Adj. Off. Rating",
        "diff_adj_def_rating":   "Adj. Def. Rating",
        "diff_net_rating":       "Net Rating",
        "diff_elo":              "ELO",
        "diff_srs":              "SRS",
        "diff_star_power":       "Star Power (PORPAG)",
        "diff_hot_streak":       "Hot Streak",
        "diff_experience":       "Experience",
        "diff_bracket_win_rate": "Bracket Win Rate",
        "diff_conf_strength":    "Conf. Strength",
        "diff_road_win_pct":     "Road Win %",
        "diff_close_game_pct":   "Close Game %",
        "diff_upset_propensity": "Upset Propensity",
        "diff_pace":             "Pace",
        "diff_adj_off_rank":     "Adj. Off. Rank",
        "diff_adj_def_rank":     "Adj. Def. Rank",
        "diff_non_conf_sos":     "Non-Conf. SOS",
    }

    feat_rows = []
    for fkey, flabel in _FEAT_LABELS.items():
        raw_v    = matchup.get(fkey)
        scaled_v = matchup_scaled.get(fkey)
        if raw_v is None or (isinstance(raw_v, float) and math.isnan(raw_v)):
            continue
        raw_f    = float(raw_v)
        scaled_f = float(scaled_v) if scaled_v is not None and not math.isnan(float(scaled_v)) else raw_f
        scale_mult = round(scaled_f / raw_f, 2) if abs(raw_f) > 1e-9 else 1.0
        feat_rows.append({
            "Feature":      flabel,
            "A − B":        round(raw_f, 3),
            "Scaled":       round(scaled_f, 3),
            "Scale ×":      scale_mult,
            "Favors":       team_a if raw_f > 0.005 else (team_b if raw_f < -0.005 else "—"),
        })

    if feat_rows:
        feat_df = (
            pd.DataFrame(feat_rows)
            .assign(_abs=lambda d: d["Scaled"].abs())
            .sort_values("_abs", ascending=False)
            .drop(columns=["_abs"])
            .reset_index(drop=True)
        )
        st.dataframe(feat_df, use_container_width=True, hide_index=True)
