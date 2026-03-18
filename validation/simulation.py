# validation/simulation.py
"""
NCAA tournament bracket simulation.

1. Monte Carlo simulation — run N simulations through the bracket,
   accumulating per-team win probabilities at each round.
2. Chalk bracket generation — always pick the higher-probability team,
   propagating winners forward correctly.
3. Scoring — compare simulation win probs against actual tournament results.

Usage:
    python -m validation.simulation                           # LogReg, 10k sims
    python -m validation.simulation --model histgbt           # HistGBT baseline
    python -m validation.simulation --n-sims 50000            # more simulations
    python -m validation.simulation --season 2023 --score     # score vs actuals
    python -m validation.simulation --save results/sim.pkl    # save results
"""

import os
import sys
import json
import numpy as np
import pandas as pd
import joblib
from dataclasses import dataclass

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

# ── Name normalization ────────────────────────────────────────────────────────
# Maps bracket display names → API names used in features_by_season

NAME_FIXES = {
    # State school abbreviations
    "Alabama St":      "Alabama State",
    "Colorado St":     "Colorado State",
    "Iowa St":         "Iowa State",
    "Kansas St":       "Kansas State",
    "Kennesaw St":     "Kennesaw State",
    "Michigan St":     "Michigan State",
    "Mississippi St":  "Mississippi State",
    "Missouri St":     "Missouri State",
    "Montana St":      "Montana State",
    "Mount St Mary's": "Mount St. Mary's",
    "Norfolk St":      "Norfolk State",
    "Ohio St":         "Ohio State",
    "Oklahoma St":     "Oklahoma State",
    "Oregon St":       "Oregon State",
    "Penn St":         "Penn State",
    "San Diego St":    "San Diego State",
    "Tennessee St":    "Tennessee State",
    "Utah St":         "Utah State",
    "Weber St":        "Weber State",
    "Wright St":       "Wright State",
    # Common abbreviations / bracket display names
    "SIUE":            "SIU Edwardsville",
    "St John's":       "St. John's",
    "Cal Baptist":     "California Baptist",
    "Long Island":     "Long Island University",
    # Typos in bracket
    "Sienna":          "Siena",
}


def _normalize(name: str) -> str:
    return NAME_FIXES.get(name, name)


# ── Seed-based fallback ───────────────────────────────────────────────────────

SEED_WIN_RATES = {
    (1, 16): 0.987, (2, 15): 0.938, (3, 14): 0.850,
    (4, 13): 0.794, (5, 12): 0.647, (6, 11): 0.620,
    (7, 10): 0.601, (8,  9): 0.491,
}


def seed_prob_fallback(seed_a: int, seed_b: int) -> float:
    if seed_a is None or seed_b is None:
        return 0.5
    key = (min(seed_a, seed_b), max(seed_a, seed_b))
    base = SEED_WIN_RATES.get(key, 0.5)
    return base if seed_a < seed_b else 1.0 - base


# ── Startup name check ────────────────────────────────────────────────────────

def check_team_names(bracket: dict, features: dict) -> dict:
    """
    Print every bracket team with found/not-found status against features keys.
    Returns {display_name: api_name} for all teams in the bracket.
    """
    found, missing = [], []
    team_map: dict[str, str] = {}

    for region, data in bracket["regions"].items():
        for game in data["matchups"]:
            for key in ("team_a", "team_b"):
                display  = game[key]
                api_name = _normalize(display)
                if api_name not in team_map:
                    team_map[display] = api_name
                    if api_name in features:
                        found.append((display, api_name))
                    else:
                        missing.append((display, api_name))

    print(f"\n  Team name check ({len(found) + len(missing)} teams):")
    for display, api in sorted(found):
        tag = f"  [normalized from '{display}']" if display != api else ""
        print(f"    ✓  {api}{tag}")
    for display, api in sorted(missing):
        tag = f"  [tried normalizing '{display}']" if display != api else ""
        print(f"    ✗  {api}  (features missing — seed fallback){tag}")

    if missing:
        print(f"\n  WARNING: {len(missing)} team(s) will use seed-based fallback.")
    else:
        print(f"\n  All {len(found)} teams found in features.")

    return team_map


# ── SimulationResults ─────────────────────────────────────────────────────────

@dataclass
class SimulationResults:
    n_simulations: int
    win_probs: dict        # {team: {round: probability}}
    team_seeds: dict       # {team: seed}

    @property
    def champion_probs(self) -> dict:
        return {t: self.win_probs[t][6] for t in self.win_probs}

    @property
    def final_four_probs(self) -> dict:
        return {t: self.win_probs[t][5] for t in self.win_probs}

    def to_dataframe(self) -> pd.DataFrame:
        rows = []
        for team, rounds in self.win_probs.items():
            rows.append({
                "team":  team,
                "seed":  self.team_seeds.get(team, "?"),
                "r64":   rounds.get(1, 0),
                "r32":   rounds.get(2, 0),
                "s16":   rounds.get(3, 0),
                "e8":    rounds.get(4, 0),
                "f4":    rounds.get(5, 0),
                "champ": rounds.get(6, 0),
            })
        df = pd.DataFrame(rows).sort_values("champ", ascending=False).reset_index(drop=True)
        for col in ["r64", "r32", "s16", "e8", "f4", "champ"]:
            df[col] = df[col].apply(lambda x: f"{x:.1%}")
        return df

    def print_summary(self, top_n: int = 20):
        df = self.to_dataframe()
        print(f"\n{'='*72}")
        print(f" Tournament win probabilities — {self.n_simulations:,} simulations")
        print(f"{'='*72}")
        print(f"  {'Team':<22} {'Seed':>4}  {'R64':>6}  {'R32':>6}  {'S16':>6}  "
              f"{'E8':>6}  {'F4':>6}  {'Champ':>6}")
        print(f"  {'-'*68}")
        for _, row in df.head(top_n).iterrows():
            print(f"  {row['team']:<22} {str(row['seed']):>4}  {row['r64']:>6}  "
                  f"{row['r32']:>6}  {row['s16']:>6}  {row['e8']:>6}  "
                  f"{row['f4']:>6}  {row['champ']:>6}")
        print(f"{'='*72}")

        champ_sum = sum(self.win_probs[t][6] for t in self.win_probs)
        status = "✓" if abs(champ_sum - 1.0) < 0.01 else "✗ PROBLEM"
        print(f"\n  Sanity check: champion probs sum = {champ_sum:.4f} {status}")

    def save(self, path: str):
        os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
        joblib.dump(self, path)
        print(f"  saved → {path}")

    @classmethod
    def load(cls, path: str) -> "SimulationResults":
        return joblib.load(path)


# ── BracketSimulator ──────────────────────────────────────────────────────────

class BracketSimulator:
    def __init__(
        self,
        model,
        features_2025: dict,
        bracket_path: str = None,
        bracket_data: dict = None,
        n_simulations: int = 10_000,
        random_state: int = 42,
    ):
        self.model         = model
        self.features_2025 = features_2025
        self.n_simulations = n_simulations
        self.random_state  = random_state

        if bracket_data is not None:
            # Use reconstructed bracket from build_bracket_from_cache()
            self._init_from_bracket_data(bracket_data)
        else:
            # Load from JSON file
            if bracket_path is None:
                bracket_path = os.path.join(
                    os.path.dirname(__file__), "..", "data", "bracket_2025.json"
                )
            with open(bracket_path) as f:
                self._init_from_json(json.load(f))

        print(f"\n  Building {len(self.bracket_teams)}×{len(self.bracket_teams)} "
              f"probability matrix ({len(self.bracket_teams)**2} lookups) ...")
        self.prob_matrix = self._build_prob_matrix()
        print(f"  Probability matrix complete.")

    def _init_from_json(self, bracket_json: dict):
        """Initialize from bracket_2025.json — applies NAME_FIXES normalization."""
        self.bracket    = bracket_json
        self._team_map  = check_team_names(self.bracket, self.features_2025)

        self.team_seeds: dict[str, int] = {}
        self.bracket_teams: list[str]   = []

        for region, data in self.bracket["regions"].items():
            for game in data["matchups"]:
                for t_key, s_key in (("team_a", "seed_a"), ("team_b", "seed_b")):
                    api = _normalize(game[t_key])
                    if api not in self.team_seeds:
                        self.team_seeds[api] = game[s_key]
                        self.bracket_teams.append(api)

    def _init_from_bracket_data(self, bracket_data: dict):
        """
        Initialize from build_bracket_from_cache() output.
        Team names are already API names — no NAME_FIXES normalization needed.
        Uses the pre-reconstructed bracket JSON from bracket_data["bracket"].
        """
        self.bracket       = bracket_data["bracket"]
        self.team_seeds    = {t: s for t, s in bracket_data["seeds"].items()}
        self.bracket_teams = []

        # Collect teams from R1 matchups (bracket already in JSON format)
        for region, data in self.bracket["regions"].items():
            for game in data["matchups"]:
                for t_key in ("team_a", "team_b"):
                    team = game[t_key]
                    if team not in self.bracket_teams:
                        self.bracket_teams.append(team)

        # Name check (informational — teams already API names)
        found   = [t for t in self.bracket_teams if t in self.features_2025]
        missing = [t for t in self.bracket_teams if t not in self.features_2025]
        print(f"\n  Team check: {len(found)} found, {len(missing)} missing in features")
        if missing:
            print(f"  Missing (seed fallback): {missing}")
        self._team_map = {t: t for t in self.bracket_teams}

    def _build_prob_matrix(self) -> dict:
        from features.builder import build_matchup_features

        matrix: dict[str, dict[str, float]] = {}
        teams = self.bracket_teams

        for team_a in teams:
            matrix[team_a] = {}
            feats_a = self.features_2025.get(team_a)
            seed_a  = self.team_seeds.get(team_a)

            for team_b in teams:
                if team_a == team_b:
                    matrix[team_a][team_b] = 0.5
                    continue

                feats_b = self.features_2025.get(team_b)
                seed_b  = self.team_seeds.get(team_b)

                if feats_a is None or feats_b is None:
                    matrix[team_a][team_b] = seed_prob_fallback(seed_a, seed_b)
                    continue

                matchup = build_matchup_features(
                    feats_a, feats_b,
                    neutral_site=True,
                    seed_a=seed_a,
                    seed_b=seed_b,
                )
                X = pd.DataFrame([matchup])
                matrix[team_a][team_b] = float(self.model.predict_proba(X)[0])

        return matrix

    def simulate(self) -> SimulationResults:
        rng = np.random.default_rng(self.random_state)

        win_counts = {team: {r: 0 for r in range(1, 7)} for team in self.bracket_teams}

        # Pre-build R1 game list (API names) for fast loop iteration
        r1_games = []
        for region, data in self.bracket["regions"].items():
            for game in data["matchups"]:
                team_a = _normalize(game["team_a"])
                team_b = _normalize(game["team_b"])
                r1_games.append({
                    "slot":   game["slot"],
                    "team_a": team_a,
                    "team_b": team_b,
                    "region": region,
                })

        for _ in range(self.n_simulations):
            survivors: dict[str, str] = {}

            # Round 1 — R64
            for game in r1_games:
                team_a, team_b = game["team_a"], game["team_b"]
                p      = self.prob_matrix[team_a][team_b]
                winner = team_a if rng.random() < p else team_b
                survivors[game["slot"]] = winner
                win_counts[winner][1] += 1

            # Rounds 2–4 within each region
            for region in ["South", "East", "West", "Midwest"]:
                prefix   = region[0]
                r1_slots = [f"{prefix}_R1_G{i}" for i in range(1, 9)]

                # Round 2 — R32
                r2_winners = []
                for i in range(0, 8, 2):
                    team_a = survivors[r1_slots[i]]
                    team_b = survivors[r1_slots[i + 1]]
                    p      = self.prob_matrix[team_a][team_b]
                    winner = team_a if rng.random() < p else team_b
                    r2_winners.append(winner)
                    win_counts[winner][2] += 1

                # Round 3 — Sweet 16
                r3_winners = []
                for i in range(0, 4, 2):
                    team_a, team_b = r2_winners[i], r2_winners[i + 1]
                    p      = self.prob_matrix[team_a][team_b]
                    winner = team_a if rng.random() < p else team_b
                    r3_winners.append(winner)
                    win_counts[winner][3] += 1

                # Round 4 — Elite 8
                team_a, team_b = r3_winners[0], r3_winners[1]
                p     = self.prob_matrix[team_a][team_b]
                champ = team_a if rng.random() < p else team_b
                survivors[f"{region}_winner"] = champ
                win_counts[champ][4] += 1

            # Round 5 — Final Four
            f4_matchups = [
                (survivors["South_winner"], survivors["East_winner"]),
                (survivors["West_winner"],  survivors["Midwest_winner"]),
            ]
            f4_winners = []
            for team_a, team_b in f4_matchups:
                p      = self.prob_matrix[team_a][team_b]
                winner = team_a if rng.random() < p else team_b
                f4_winners.append(winner)
                win_counts[winner][5] += 1

            # Round 6 — Championship
            team_a, team_b = f4_winners[0], f4_winners[1]
            p     = self.prob_matrix[team_a][team_b]
            champ = team_a if rng.random() < p else team_b
            win_counts[champ][6] += 1

        win_probs = {
            team: {r: count / self.n_simulations for r, count in rounds.items()}
            for team, rounds in win_counts.items()
        }
        return SimulationResults(
            n_simulations=self.n_simulations,
            win_probs=win_probs,
            team_seeds=self.team_seeds,
        )


# ── Chalk bracket ─────────────────────────────────────────────────────────────

def generate_chalk_bracket(
    sim_results: SimulationResults,
    bracket: dict,
    prob_matrix: dict,
) -> tuple[dict, dict]:
    """
    Always picks the higher-probability team at every slot, propagating winners
    forward. Returns (picks_by_round, chalk) where chalk maps slot_id → winner.
    """
    chalk: dict[str, str] = {}
    picks_by_round = {r: [] for r in range(1, 7)}

    # Round 1
    for region, data in bracket["regions"].items():
        for game in data["matchups"]:
            team_a = _normalize(game["team_a"])
            team_b = _normalize(game["team_b"])
            p      = prob_matrix[team_a][team_b]
            winner = team_a if p >= 0.5 else team_b
            chalk[game["slot"]] = winner
            picks_by_round[1].append(winner)

    # Rounds 2–4 within each region
    for region in ["South", "East", "West", "Midwest"]:
        prefix   = region[0]
        r1_slots = [f"{prefix}_R1_G{i}" for i in range(1, 9)]

        r2_winners = []
        for i in range(0, 8, 2):
            team_a, team_b = chalk[r1_slots[i]], chalk[r1_slots[i + 1]]
            p      = prob_matrix[team_a][team_b]
            winner = team_a if p >= 0.5 else team_b
            r2_winners.append(winner)
            picks_by_round[2].append(winner)

        r3_winners = []
        for i in range(0, 4, 2):
            team_a, team_b = r2_winners[i], r2_winners[i + 1]
            p      = prob_matrix[team_a][team_b]
            winner = team_a if p >= 0.5 else team_b
            r3_winners.append(winner)
            picks_by_round[3].append(winner)

        team_a, team_b = r3_winners[0], r3_winners[1]
        p              = prob_matrix[team_a][team_b]
        regional_champ = team_a if p >= 0.5 else team_b
        chalk[f"{region}_winner"] = regional_champ
        picks_by_round[4].append(regional_champ)

    # Round 5 — Final Four
    f4_matchups = [
        (chalk["South_winner"], chalk["East_winner"]),
        (chalk["West_winner"],  chalk["Midwest_winner"]),
    ]
    f4_winners = []
    for team_a, team_b in f4_matchups:
        p      = prob_matrix[team_a][team_b]
        winner = team_a if p >= 0.5 else team_b
        f4_winners.append(winner)
        picks_by_round[5].append(winner)

    # Round 6 — Championship
    team_a, team_b = f4_winners[0], f4_winners[1]
    p        = prob_matrix[team_a][team_b]
    champion = team_a if p >= 0.5 else team_b
    picks_by_round[6].append(champion)

    return picks_by_round, chalk


def print_chalk_bracket(picks_by_round: dict, prob_matrix: dict):
    round_names = {
        1: "Round of 64", 2: "Round of 32", 3: "Sweet 16",
        4: "Elite 8",     5: "Final Four",  6: "Champion",
    }
    print("\n" + "=" * 50)
    print(" Chalk bracket — model's best picks")
    print("=" * 50)
    for r in range(1, 7):
        print(f"\n  {round_names[r]}:")
        for team in picks_by_round[r]:
            print(f"    {team}")


# ── Scoring ───────────────────────────────────────────────────────────────────

ROUND_NAMES = {
    0: "First Four", 1: "R64", 2: "R32",
    3: "S16",        4: "E8",  5: "F4", 6: "Champ",
}


def _resolve_name(team: str, win_probs: dict) -> str | None:
    """Resolve a bracket team name to its simulation name. Returns None if not found."""
    if team in win_probs:
        return team

    ALIASES = {
        "Connecticut":           "UConn",
        "UConn":                 "Connecticut",
        "St John's":             "St. John's",
        "St. John's":            "St John's",
        "Saint John's":          "St. John's",
        "SIU Edwardsville":      "SIUE",
        "SIUE":                  "SIU Edwardsville",
        "NC State":              "North Carolina State",
        "North Carolina State":  "NC State",
        "North Carolina St.":    "NC State",
        "VCU":                   "Virginia Commonwealth",
        "Virginia Commonwealth": "VCU",
        "Miami FL":              "Miami",
        "Miami":                 "Miami FL",
        "USC":                   "Southern California",
        "Southern California":   "USC",
        "LSU":                   "Louisiana State",
        "Louisiana State":       "LSU",
    }
    alias = ALIASES.get(team)
    if alias and alias in win_probs:
        return alias

    # Fuzzy match: normalize punctuation/case
    def norm(s):
        return s.lower().replace(".", "").replace("'", "").replace("-", " ").strip()

    team_norm = norm(team)
    for sim_team in win_probs:
        if norm(sim_team) == team_norm:
            return sim_team

    return None


def score_simulation(
    sim_results: SimulationResults,
    bracket_data: dict,
    verbose: bool = True,
) -> dict:
    """
    Scores simulation win probabilities against actual tournament results.

    For each team×round outcome in the bracket, compares the simulation's
    predicted win probability against the actual outcome (1 = advanced, 0 = lost).

    Metrics per round and overall:
      - Brier score (primary)
      - Accuracy (did the chalk pick — prob >= 0.5 — match actual outcome?)
      - Calibration gap (mean predicted prob vs actual win rate)

    Returns a dict of scoring metrics.
    """
    from sklearn.metrics import brier_score_loss

    actual   = bracket_data["results"]
    champion = bracket_data["champion"]
    season   = bracket_data["season"]

    all_probs: list[float]  = []
    all_outcomes: list[int] = []
    round_data: dict        = {r: {"probs": [], "outcomes": [], "correct": []}
                                for r in range(1, 7)}
    name_misses: list[str]  = []

    for team, rounds in actual.items():
        if not rounds:
            continue  # First Four losers have no round records — skip silently
        sim_team = _resolve_name(team, sim_results.win_probs)
        if sim_team is None:
            name_misses.append(team)
            continue

        for round_num, outcome in rounds.items():
            prob        = sim_results.win_probs[sim_team].get(round_num, 0.0)
            chalk_right = int((prob >= 0.5) == bool(outcome))

            all_probs.append(prob)
            all_outcomes.append(outcome)
            round_data[round_num]["probs"].append(prob)
            round_data[round_num]["outcomes"].append(outcome)
            round_data[round_num]["correct"].append(chalk_right)

    if name_misses:
        print(f"  WARNING: {len(name_misses)} bracket teams not in simulation: "
              f"{name_misses}")

    round_metrics: dict[int, dict] = {}
    for r in range(1, 7):
        rd = round_data[r]
        if not rd["probs"]:
            continue
        round_metrics[r] = {
            "brier":       float(brier_score_loss(rd["outcomes"], rd["probs"])),
            "accuracy":    float(np.mean(rd["correct"])),
            "mean_prob":   float(np.mean(rd["probs"])),
            "actual_rate": float(np.mean(rd["outcomes"])),
            "n":           len(rd["probs"]),
        }

    overall_brier = float(brier_score_loss(all_outcomes, all_probs)) if all_probs else None

    champ_sim  = _resolve_name(champion, sim_results.win_probs) if champion else None
    champ_prob = (sim_results.win_probs[champ_sim].get(6, 0.0)
                  if champ_sim else 0.0)

    if verbose:
        _print_score_report(season, round_metrics, overall_brier,
                            champion, champ_prob, name_misses)

    return {
        "season":        season,
        "overall_brier": overall_brier,
        "round_metrics": round_metrics,
        "champion":      champion,
        "champion_prob": champ_prob,
        "n_games":       len(all_probs),
        "n_mismatches":  len(name_misses),
    }


def _print_score_report(season, round_metrics, overall_brier,
                        champion, champ_prob, name_misses):
    print(f"\n{'='*65}")
    print(f" Simulation scoring — {season} NCAA Tournament")
    print(f"{'='*65}")
    print(f"  {'Round':<8}  {'Brier':>7}  {'Acc':>6}  "
          f"{'Mean P':>7}  {'Actual%':>8}  {'n':>4}")
    print(f"  {'-'*55}")
    for r in range(1, 7):
        if r not in round_metrics:
            continue
        m = round_metrics[r]
        print(f"  {ROUND_NAMES[r]:<8}  {m['brier']:>7.4f}  "
              f"{m['accuracy']:>6.1%}  {m['mean_prob']:>7.3f}  "
              f"{m['actual_rate']:>8.1%}  {m['n']:>4}")
    print(f"  {'-'*55}")
    if overall_brier is not None:
        print(f"  {'Overall':<8}  {overall_brier:>7.4f}")

    label = ("model had them as favorite" if champ_prob > 0.20
             else "upset" if champ_prob < 0.10 else "plausible pick")
    print(f"\n  Champion:       {champion}")
    print(f"  Sim champ prob: {champ_prob:.1%}  ({label})")
    if name_misses:
        print(f"\n  Unmatched teams ({len(name_misses)}): {name_misses}")
    print(f"{'='*65}")


# ── CLI ───────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    import argparse
    from pathlib import Path
    from data.bracket_builder import build_bracket_from_cache

    parser = argparse.ArgumentParser(description="NCAA tournament bracket simulator")
    parser.add_argument("--n-sims",  type=int, default=10_000)
    parser.add_argument("--model",   choices=["logistic", "histgbt"], default="logistic")
    parser.add_argument("--season",  type=int, default=2025)
    parser.add_argument("--top-n",   type=int, default=20)
    parser.add_argument("--score",   action="store_true",
                        help="Score simulation against actual results")
    parser.add_argument("--save",    type=str, default=None,
                        help="Save SimulationResults to this path")
    args = parser.parse_args()

    # Load features
    feat_path = Path("data/cache/features_by_season.pkl")
    if not feat_path.exists():
        print("ERROR: data/cache/features_by_season.pkl not found.")
        print("Run `python data/pull.py` first to build the feature cache.")
        sys.exit(1)

    features_by_season = joblib.load(feat_path)
    features = features_by_season.get(args.season)
    if features is None:
        print(f"ERROR: No features for season {args.season}. "
              f"Available: {sorted(features_by_season.keys())}")
        sys.exit(1)
    print(f"  Loaded features_by_season — {len(features)} teams in {args.season}")

    # Load model
    if args.model == "logistic":
        from models.logistic_predictor import LogisticPredictor
        model = LogisticPredictor.load()
        print("  Loaded: LogisticPredictor (post+conf+weighted — A3 best)")
    else:
        from models.game_predictor_model import GamePredictor
        model = GamePredictor.load()
        print("  Loaded: GamePredictor (HistGBT baseline)")

    # Determine bracket source:
    #   >= 2026 → use manually entered bracket_{season}.json (in-progress / future)
    #   <= 2025 → reconstruct from game cache (historical, scoreable)
    bracket_data = None
    json_path = Path(f"data/bracket_{args.season}.json")

    if args.season >= 2026:
        if not json_path.exists():
            print(f"ERROR: No bracket JSON at {json_path}.")
            print("Create data/bracket_{season}.json with the tournament matchups.")
            sys.exit(1)
        sim = BracketSimulator(
            model=model,
            features_2025=features,
            bracket_path=str(json_path),
            n_simulations=args.n_sims,
        )
        if args.score:
            print("NOTE: --score not available for in-progress seasons (no complete results).")
    else:
        bracket_data = build_bracket_from_cache(args.season)
        sim = BracketSimulator(
            model=model,
            features_2025=features,
            bracket_data=bracket_data,
            n_simulations=args.n_sims,
        )

    # Run simulation
    results = sim.simulate()
    results.print_summary(top_n=args.top_n)

    # Chalk bracket
    picks, chalk = generate_chalk_bracket(results, sim.bracket, sim.prob_matrix)
    print_chalk_bracket(picks, sim.prob_matrix)

    # Score against actual results (historical seasons only)
    if args.score and bracket_data is not None:
        metrics = score_simulation(results, bracket_data, verbose=True)

    if args.save:
        results.save(args.save)

# TODO: pool optimization — see future task
