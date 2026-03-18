# brackets/generator.py
"""
Generates named bracket picks from a SimulationResults + prob_matrix.

Each bracket is a dict:
  {
    "name":          str,
    "description":   str,
    "champion":      str,
    "champion_prob": float,       # from Monte Carlo
    "upset_count":   int,         # R64 picks where higher seed number wins
    "regions": {
      "South": {
        "r64": [(team_a, team_b, winner), ...],   # 4 games
        "r32": [(team_a, team_b, winner), ...],   # 4 games
        "s16": [(team_a, team_b, winner), ...],   # 2 games
        "e8":  [(team_a, team_b, winner)],        # 1 game
      }, ...
    },
    "final_four":   [(team_a, team_b, winner), (team_a, team_b, winner)],
    "championship": (team_a, team_b, winner),
  }
"""

from __future__ import annotations
from typing import Optional

import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from validation.simulation import _normalize


REGIONS = ["South", "East", "West", "Midwest"]
REGION_SEMIFINAL_PAIRS = [("South", "East"), ("West", "Midwest")]

# Historical seed win-rate lookup (used only for upset counting label)
_LOWER_SEED_IS_FAVORITE = {
    (1, 16), (2, 15), (3, 14), (4, 13),
    (5, 12), (6, 11), (7, 10), (8, 9),
}


def _is_upset(winner: str, team_a: str, team_b: str, seeds: dict) -> bool:
    """True if winner is the higher-seeded (worse) team."""
    sa = seeds.get(team_a)
    sb = seeds.get(team_b)
    if sa is None or sb is None:
        return False
    loser = team_b if winner == team_a else team_a
    winner_seed = seeds[winner]
    loser_seed  = seeds[loser]
    return winner_seed > loser_seed  # higher number = worse seed = upset


def _play(team_a: str, team_b: str, prob_matrix: dict,
          force_winner: Optional[str] = None) -> tuple[str, float]:
    """Return (winner, prob_a_wins). force_winner overrides the model."""
    if force_winner in (team_a, team_b):
        return force_winner, prob_matrix[team_a][team_b]
    p = prob_matrix[team_a][team_b]
    return (team_a if p >= 0.5 else team_b), p


def _run_bracket(
    bracket_json: dict,
    prob_matrix: dict,
    team_seeds: dict,
    forced_champion: Optional[str] = None,
    prob_matrix_unseeded: dict = None,
) -> dict:
    """E8, F4, Championship use prob_matrix_unseeded when provided."""
    """
    Run a single deterministic bracket.
    If forced_champion is set, that team always wins their games.
    Returns the full bracket result dict.
    """
    late = prob_matrix_unseeded if prob_matrix_unseeded is not None else prob_matrix

    games_r64 = {}   # slot → winner
    region_results = {}

    for region in REGIONS:
        prefix   = region[0]
        matchups = bracket_json["regions"][region]["matchups"]
        r1_slots = [m["slot"] for m in matchups]

        # R64
        r64_games = []
        for m in matchups:
            a, b   = _normalize(m["team_a"]), _normalize(m["team_b"])
            w, _   = _play(a, b, prob_matrix, forced_champion)
            games_r64[m["slot"]] = w
            r64_games.append((a, b, w))

        # R32
        r32_games = []
        r32_winners = []
        for i in range(0, 8, 2):
            a = games_r64[r1_slots[i]]
            b = games_r64[r1_slots[i + 1]]
            w, _ = _play(a, b, prob_matrix, forced_champion)
            r32_games.append((a, b, w))
            r32_winners.append(w)

        # S16
        s16_games = []
        s16_winners = []
        for i in range(0, 4, 2):
            a, b = r32_winners[i], r32_winners[i + 1]
            w, _ = _play(a, b, prob_matrix, forced_champion)
            s16_games.append((a, b, w))
            s16_winners.append(w)

        # E8 — unseeded
        a, b = s16_winners[0], s16_winners[1]
        w, _ = _play(a, b, late, forced_champion)
        e8_games = [(a, b, w)]
        region_winner = w

        region_results[region] = {
            "r64":    r64_games,
            "r32":    r32_games,
            "s16":    s16_games,
            "e8":     e8_games,
            "winner": region_winner,
        }

    # Final Four — unseeded
    ff_games = []
    ff_winners = []
    for reg_a, reg_b in REGION_SEMIFINAL_PAIRS:
        a = region_results[reg_a]["winner"]
        b = region_results[reg_b]["winner"]
        w, _ = _play(a, b, late, forced_champion)
        ff_games.append((a, b, w))
        ff_winners.append(w)

    # Championship — unseeded
    a, b = ff_winners[0], ff_winners[1]
    w, _ = _play(a, b, late, forced_champion)
    champion = w

    # Count R64 upsets
    upset_count = sum(
        1 for region in REGIONS
        for (ta, tb, tw) in region_results[region]["r64"]
        if _is_upset(tw, ta, tb, team_seeds)
    )

    return {
        "champion":    champion,
        "upset_count": upset_count,
        "regions":     {r: {k: v for k, v in region_results[r].items() if k != "winner"}
                        for r in REGIONS},
        "final_four":  ff_games,
        "championship": (a, b, w),
    }


# ── Public generators ─────────────────────────────────────────────────────────

def chalk_bracket(sim_results, prob_matrix: dict, bracket_json: dict,
                  prob_matrix_unseeded: dict = None) -> dict:
    """Always picks the higher-probability team. E8+ uses unseeded matrix."""
    result = _run_bracket(bracket_json, prob_matrix, sim_results.team_seeds,
                          prob_matrix_unseeded=prob_matrix_unseeded)
    champ  = result["champion"]
    return {
        "name":          "Chalk",
        "description":   "Model's best pick at every game — highest win probability always advances.",
        "champion_prob": sim_results.win_probs.get(champ, {}).get(6, 0.0),
        **result,
    }


def top_champion_brackets(
    sim_results, prob_matrix: dict, bracket_json: dict, n: int = 5,
    prob_matrix_unseeded: dict = None,
) -> list[dict]:
    """
    For each of the top-N most likely champions, build the most probable bracket
    that ends with that champion winning. Non-champion games use chalk picks.
    """
    champ_probs = sorted(
        sim_results.win_probs.items(),
        key=lambda kv: kv[1].get(6, 0.0),
        reverse=True,
    )[:n]

    brackets = []
    for i, (champ, probs) in enumerate(champ_probs):
        result = _run_bracket(
            bracket_json, prob_matrix, sim_results.team_seeds,
            forced_champion=champ,
            prob_matrix_unseeded=prob_matrix_unseeded,
        )
        seed = sim_results.team_seeds.get(champ, "?")
        brackets.append({
            "name":          f"#{i+1} — {champ} (#{seed} seed)",
            "description":   f"Most probable bracket with {champ} winning the championship.",
            "champion_prob": probs.get(6, 0.0),
            **result,
        })
    return brackets


def upset_bracket(sim_results, prob_matrix: dict, bracket_json: dict,
                  n_upsets: int = 5, prob_matrix_unseeded: dict = None) -> dict:
    """
    Chalk bracket but swaps in the model's top-N most confident upsets in R64.
    Champion is forced to be the chalk winner (top champion prob), keeping it credible.
    """
    seeds = sim_results.team_seeds

    # Find all R64 matchups and compute upset probability (higher seed wins)
    r64_upsets = []
    for region in REGIONS:
        for m in bracket_json["regions"][region]["matchups"]:
            a, b = _normalize(m["team_a"]), _normalize(m["team_b"])
            sa, sb = seeds.get(a, 8), seeds.get(b, 8)
            # Determine which is the "upset" team (higher seed number)
            if sa > sb:
                underdog, favorite = a, b
            else:
                underdog, favorite = b, a
            p_upset = prob_matrix[underdog][favorite]
            r64_upsets.append({
                "slot": m["slot"], "region": region,
                "underdog": underdog, "favorite": favorite,
                "p_upset": p_upset,
                "underdog_seed": seeds.get(underdog, "?"),
                "favorite_seed": seeds.get(favorite, "?"),
            })

    # Sort by upset probability, take top-N where model actually likes the underdog
    top_upsets = sorted(
        [u for u in r64_upsets if u["p_upset"] > 0.35],
        key=lambda u: u["p_upset"],
        reverse=True,
    )[:n_upsets]

    forced_upsets = {u["slot"]: u["underdog"] for u in top_upsets}

    # Build a modified prob_matrix that forces upsets in those slots only
    # We do this by wrapping _run_bracket with slot-aware forced winners
    result = _run_bracket_with_r64_overrides(
        bracket_json, prob_matrix, seeds, forced_upsets,
        prob_matrix_unseeded=prob_matrix_unseeded,
    )

    champ = result["champion"]
    upset_summary = ", ".join(u["underdog"] + " over " + u["favorite"] for u in top_upsets)
    return {
        "name":          f"Upset Special ({len(top_upsets)} upsets)",
        "description":   (
            f"Model's top {len(top_upsets)} R64 upsets applied, chalk elsewhere. "
            f"Upsets: {upset_summary}."
        ),
        "champion_prob": sim_results.win_probs.get(champ, {}).get(6, 0.0),
        "upset_details": top_upsets,
        **result,
    }


def _run_bracket_with_r64_overrides(
    bracket_json: dict,
    prob_matrix: dict,
    team_seeds: dict,
    r64_overrides: dict,
    prob_matrix_unseeded: dict = None,
) -> dict:
    """Like _run_bracket but R64 winners can be overridden per slot."""
    late = prob_matrix_unseeded if prob_matrix_unseeded is not None else prob_matrix

    games_r64 = {}
    region_results = {}

    for region in REGIONS:
        prefix   = region[0]
        matchups = bracket_json["regions"][region]["matchups"]
        r1_slots = [m["slot"] for m in matchups]

        r64_games = []
        for m in matchups:
            a, b = _normalize(m["team_a"]), _normalize(m["team_b"])
            if m["slot"] in r64_overrides:
                w = r64_overrides[m["slot"]]
            else:
                w, _ = _play(a, b, prob_matrix)
            games_r64[m["slot"]] = w
            r64_games.append((a, b, w))

        r32_games, r32_winners = [], []
        for i in range(0, 8, 2):
            a = games_r64[r1_slots[i]]
            b = games_r64[r1_slots[i + 1]]
            w, _ = _play(a, b, prob_matrix)
            r32_games.append((a, b, w))
            r32_winners.append(w)

        s16_games, s16_winners = [], []
        for i in range(0, 4, 2):
            a, b = r32_winners[i], r32_winners[i + 1]
            w, _ = _play(a, b, prob_matrix)
            s16_games.append((a, b, w))
            s16_winners.append(w)

        # E8 — unseeded
        a, b = s16_winners[0], s16_winners[1]
        w, _ = _play(a, b, late)
        region_results[region] = {
            "r64": r64_games, "r32": r32_games,
            "s16": s16_games, "e8": [(a, b, w)],
            "winner": w,
        }

    # FF + Championship — unseeded
    ff_games, ff_winners = [], []
    for reg_a, reg_b in REGION_SEMIFINAL_PAIRS:
        a = region_results[reg_a]["winner"]
        b = region_results[reg_b]["winner"]
        w, _ = _play(a, b, late)
        ff_games.append((a, b, w))
        ff_winners.append(w)

    a, b = ff_winners[0], ff_winners[1]
    w, _ = _play(a, b, late)

    upset_count = sum(
        1 for region in REGIONS
        for (ta, tb, tw) in region_results[region]["r64"]
        if _is_upset(tw, ta, tb, team_seeds)
    )

    return {
        "champion":     w,
        "upset_count":  upset_count,
        "regions":      {r: {k: v for k, v in region_results[r].items() if k != "winner"}
                         for r in REGIONS},
        "final_four":   ff_games,
        "championship": (a, b, w),
    }


def generate_all(sim_results, prob_matrix: dict, bracket_json: dict,
                 prob_matrix_unseeded: dict = None) -> list[dict]:
    """Return all bracket variants in display order."""
    brackets = []
    brackets.append(chalk_bracket(sim_results, prob_matrix, bracket_json, prob_matrix_unseeded))
    brackets.extend(top_champion_brackets(sim_results, prob_matrix, bracket_json, n=5, prob_matrix_unseeded=prob_matrix_unseeded))
    brackets.append(upset_bracket(sim_results, prob_matrix, bracket_json, prob_matrix_unseeded=prob_matrix_unseeded))
    return brackets
