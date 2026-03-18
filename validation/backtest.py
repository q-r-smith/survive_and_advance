# validation/backtest.py
"""
Backtest simulation across multiple historical NCAA tournament seasons.

Usage:
    python -m validation.backtest                           # 2022-2025
    python -m validation.backtest --seasons 2019 2021 2022 2023 2024 2025
    python -m validation.backtest --model histgbt
    python -m validation.backtest --n-sims 5000
"""

import os
import sys
import argparse
import numpy as np
import joblib

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from data.bracket_builder import build_bracket_from_cache
from validation.simulation import BracketSimulator, score_simulation

DEFAULT_SEASONS = [2016, 2017, 2018, 2019, 2021, 2022, 2023, 2024, 2025]  # 2020 skipped (no tournament)


def backtest_season(
    season: int,
    model,
    features_by_season: dict,
    n_simulations: int = 10_000,
) -> dict | None:
    if season not in features_by_season:
        print(f"  {season}: no features — skipping")
        return None

    try:
        bracket_data = build_bracket_from_cache(season)
    except (FileNotFoundError, ValueError) as e:
        print(f"  {season}: {e}")
        return None

    features = features_by_season[season]

    sim = BracketSimulator(
        model=model,
        features_2025=features,
        bracket_data=bracket_data,
        n_simulations=n_simulations,
        random_state=42,
    )
    results = sim.simulate()
    metrics = score_simulation(results, bracket_data, verbose=False)

    r64_acc = metrics["round_metrics"].get(1, {}).get("accuracy", float("nan"))
    print(f"  {season}:  Brier={metrics['overall_brier']:.4f}  "
          f"Acc={r64_acc:.1%} R64  "
          f"Champion={bracket_data['champion']}  "
          f"(sim prob={metrics['champion_prob']:.1%})")

    return metrics


def main():
    parser = argparse.ArgumentParser(description="Backtest simulation across seasons")
    parser.add_argument("--seasons",  type=int, nargs="+", default=DEFAULT_SEASONS)
    parser.add_argument("--model",    choices=["logistic", "histgbt"], default="logistic")
    parser.add_argument("--n-sims",   type=int, default=10_000)
    parser.add_argument("--verbose",  action="store_true",
                        help="Print full scoring report for each season")
    args = parser.parse_args()

    # Load features
    feat_path = "data/cache/features_by_season.pkl"
    if not os.path.exists(feat_path):
        print("ERROR: data/cache/features_by_season.pkl not found.")
        print("Run `python data/pull.py` first.")
        sys.exit(1)

    features_by_season = joblib.load(feat_path)
    print(f"Loaded features for seasons: {sorted(features_by_season.keys())}")

    # Load model
    if args.model == "logistic":
        from models.logistic_predictor import LogisticPredictor
        model = LogisticPredictor.load()
        print("Model: LogisticPredictor (post+conf+weighted — A3 best)\n")
    else:
        from models.game_predictor_model import GamePredictor
        model = GamePredictor.load()
        print("Model: GamePredictor (HistGBT baseline)\n")

    # Run backtest
    all_metrics = {}
    print(f"{'='*65}")
    print(f" Backtest — {args.n_sims:,} simulations per season")
    print(f"{'='*65}")

    for season in sorted(args.seasons):
        metrics = backtest_season(
            season=season,
            model=model,
            features_by_season=features_by_season,
            n_simulations=args.n_sims,
        )
        if metrics is not None:
            all_metrics[season] = metrics

    # Summary table
    if len(all_metrics) > 1:
        # Per-season overall row
        print(f"\n{'='*80}")
        print(f" Summary — {args.model} model")
        print(f"{'='*80}")
        print(f"  {'Season':>6}  {'Overall':>7}  {'R64':>6}  {'R32':>6}  "
              f"{'S16':>6}  {'E8':>6}  {'F4':>6}  {'Champ':>6}  {'Champion':<18}  {'Prob':>6}")
        print(f"  {'-'*76}")

        col_briers = {r: [] for r in range(1, 7)}
        overall_briers = []
        for season in sorted(all_metrics):
            m  = all_metrics[season]
            b  = m["overall_brier"]
            overall_briers.append(b)
            round_cols = ""
            for r in range(1, 7):
                rb = m["round_metrics"].get(r, {}).get("brier", float("nan"))
                col_briers[r].append(rb)
                round_cols += f"  {rb:>6.4f}" if not np.isnan(rb) else f"  {'—':>6}"
            print(f"  {season:>6}  {b:>7.4f}{round_cols}  "
                  f"{m['champion']:<18}  {m['champion_prob']:>6.1%}")

        print(f"  {'-'*76}")
        mean_cols = ""
        for r in range(1, 7):
            vals = [v for v in col_briers[r] if not np.isnan(v)]
            mean_cols += f"  {np.mean(vals):>6.4f}" if vals else f"  {'—':>6}"
        print(f"  {'Mean':>6}  {np.mean(overall_briers):>7.4f}{mean_cols}")
        print(f"{'='*80}")


if __name__ == "__main__":
    main()
