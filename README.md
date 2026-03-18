# March Madness Model — Terminal Runbook

Complete command reference for running the pipeline start to finish.
Run all commands from the project root directory.

---

## 0. First-time setup

```bash
# Create and activate virtual environment
python -m venv venv
source venv/bin/activate          # Mac/Linux
# venv\Scripts\activate           # Windows

# Install dependencies
pip install -r requirements.txt

# Confirm API key is set
cat .env                           # should show COLLEGE_FOOTBALL_API_KEY=...
```

---

## 1. Pull raw data + build features

```bash
# Full pull — fetches all seasons from API, builds features, saves train/val splits
# Takes ~10–20 min on first run (API calls). Subsequent runs use cache.
python data/pull.py

# Force re-fetch from API even if cache exists (use sparingly)
python data/pull.py --force

# What this produces:
#   data/cache/raw/games_{year}.csv          ← raw game data per season
#   data/cache/raw/team_stats_{year}.csv     ← raw team stats
#   data/cache/raw/player_stats_{year}.csv   ← raw player stats
#   data/cache/raw/rosters_{year}.csv        ← roster data
#   data/cache/raw/srs_{year}.csv            ← SRS ratings
#   data/cache/raw/adjusted_ratings_{year}.csv
#   data/cache/raw/elo_{year}.csv
#   data/cache/X_train.csv                   ← training features (2015-2024)
#   data/cache/y_train.csv                   ← training labels
#   data/cache/X_val.csv                     ← validation features (2025)
#   data/cache/y_val.csv                     ← validation labels
#   data/cache/features_by_season.pkl        ← per-team features dict (needed for simulation)
```

---

## 2. Hyperparameter tuning (optional — run when features change)

```bash
# Tune HistGBT hyperparameters (default 40 iterations)
python -m validation.calibration

# More thorough search
python -m validation.calibration --n-iter 80

# Weight recent seasons 2x in cross-validation
python -m validation.calibration --weighted-cv

# Tune logistic regression separately
python -m validation.calibration --logistic

# Run both searches
python -m validation.calibration --logistic --histgbt --n-iter 60

# What this produces:
#   models/best_params.json           ← best HistGBT params
#   models/top5_params.json           ← top 5 HistGBT param sets
#   models/best_logistic_params.json  ← best logistic params
```

---

## 3. Train models

```bash
# Baseline only — HistGBT on all games (fastest, sanity check)
python train.py

# Full comparison — all model types x training sets (primary command)
python train.py --all

# Individual flags (combinable):
python train.py --logistic            # add logistic regression
python train.py --tournament-only     # also train on postseason-only data
python train.py --calibrate           # add isotonic calibration wrapper
python train.py --ensemble            # add LightGBM + HistGBT ensemble

# Most useful single command — runs everything and prints comparison table
python train.py --all

# What this produces:
#   models/game_predictor.pkl         ← HistGBT model
#   models/logistic_predictor.pkl     ← LogReg model (if --logistic)
#   models/calibrated_predictor.pkl   ← calibrated model (if --calibrate)
#   models/ensemble_predictor.pkl     ← ensemble model (if --ensemble)

# Current best result to expect:
#   LogReg  full  post+conf (unweighted)  Brier=0.1394  Acc=84.3%  n=121  ← current best
#   LogReg  full  post+conf+weighted      Brier=0.1404  Acc=82.6%  n=121
#   HistGBT trimmed  post+conf            Brier=0.1521  Acc=78.5%  n=121
```

---

## 4. Inspect tournament data (run once to validate cache)

```bash
# Confirm gameNotes filter and round inference work correctly
python inspect_tournament_data.py

# Quick sanity check on bracket reconstruction for a single year
python -c "
from data.bracket_builder import build_bracket_from_cache
b = build_bracket_from_cache(2025)
print('Champion:', b['champion'])
print('Seeds sample:', dict(list(b['seeds'].items())[:5]))
print('R1 games:', len(b['rounds'][1]))
"
```

---

## 5. Run tournament simulation

```bash
# Default — LogReg model, 10k simulations, top 20 teams, no randomness adjustments
python validation/simulation.py

# Score simulation against actual 2025 results (bracket reconstructed from cache)
python validation/simulation.py --score

# More simulations for stable probabilities
python validation/simulation.py --n-sims 50000 --score

# Use HistGBT instead
python validation/simulation.py --model histgbt --score

# Show top 30 teams
python validation/simulation.py --top-n 30

# Run for a historical season
python validation/simulation.py --season 2024 --score

# Save results for later use
python validation/simulation.py --save data/cache/sim_results_2025.pkl

# Load and inspect saved results
python -c "
import joblib
results = joblib.load('data/cache/sim_results_2025.pkl')
results.print_summary(top_n=20)
"
```

### Randomness adjustment flags

The simulator supports three independent levers for reducing chalk bias — the
tendency of the model to over-represent 1-seeds in Final Four outcomes. These
flags can be combined freely. See Section 9 for background on how each works.

```bash
# Round-specific win probability dampening
# Compresses probabilities toward 50/50 with a per-round alpha (6 values: R64 → Champ)
# Default alphas: 1.0 0.95 0.85 0.75 0.65 0.60
python validation/simulation.py --round-alphas 1.0 0.95 0.85 0.75 0.65 0.60

# Per-game Gaussian noise injection (sigma=0.0 disables — default)
# Adds N(0, sigma) to each game probability before resolving, clipped to [0.05, 0.95]
python validation/simulation.py --noise-sigma 0.05

# Chaos mode — blend of normal + aggressive simulations (fraction=0.0 disables — default)
# Runs chaos_fraction of sims with CHAOS_ALPHAS + chaos_sigma, blends into output
python validation/simulation.py --chaos-fraction 0.3
python validation/simulation.py --chaos-fraction 0.3 --chaos-sigma 0.08  # default sigma

# Recommended combination — matches historical Final Four seed distribution well
python validation/simulation.py --n-sims 10000 --noise-sigma 0.05 --chaos-fraction 0.3

# Push harder toward historical rates (1-seeds closer to ~40%)
python validation/simulation.py --n-sims 10000 --noise-sigma 0.05 --chaos-fraction 0.5

# Full production run — 50k sims, all adjustments, save output
python validation/simulation.py --n-sims 50000 --noise-sigma 0.05 --chaos-fraction 0.3 \
    --save data/cache/sim_results_2026.pkl
```

---

## 6. Backtest simulation against historical tournaments

```bash
# Backtest all available years (2016-2025, skips 2020)
python validation/backtest.py

# Specific years only
python validation/backtest.py --seasons 2022 2023 2024

# Use HistGBT instead of LogReg
python validation/backtest.py --model histgbt

# More simulations per year (slower but more stable)
python validation/backtest.py --n-sims 50000

# Both models for comparison
python validation/backtest.py --model logistic --seasons 2022 2023 2024
python validation/backtest.py --model histgbt  --seasons 2022 2023 2024
```

---

## 7. Full pipeline — run everything fresh

```bash
# Step 1: Pull data (skip if cache is fresh)
python data/pull.py

# Step 2: Tune hyperparameters (skip if params files exist and features unchanged)
python -m validation.calibration --logistic --histgbt --n-iter 60 --weighted-cv

# Step 3: Train all models
python train.py --all

# Step 4: Run 2026 simulation with recommended randomness settings
python validation/simulation.py --n-sims 50000 --noise-sigma 0.05 --chaos-fraction 0.3 \
    --save data/cache/sim_results_2026.pkl

# Step 5: Backtest
python validation/backtest.py
```

---

## 8. Typical iteration loop (after initial setup)

When you've changed features or model code:

```bash
# 1. Regenerate feature matrices (features changed)
python data/pull.py

# 2. Re-train and compare
python train.py --all

# 3. If Brier improved and features changed, re-tune
python -m validation.calibration --logistic --n-iter 60

# 4. Re-train with new tuned params
python train.py --all

# 5. Re-run simulation if model improved
python validation/simulation.py --n-sims 10000 --noise-sigma 0.05 --chaos-fraction 0.3
```

---

## 9. Chalk bias — background and what was done

Out of the box, the simulation produced too many 1-seeds in the Final Four
(~48.6% of Final Four slots) relative to the historical tournament rate (~40%).
This is a compound effect: even a small per-game edge for top seeds multiplies
across six rounds.

Four changes were made to address this:

### Feature change: momentum decay (features/builder.py)

`compute_hot_streak` was updated to use a split-window recency weight rather
than a flat exponential over the last 10 games. The last 5 games are weighted
2x; games 6-15 are weighted 1x. This makes the feature more sensitive to teams
genuinely peaking in March vs. teams that were hot in December.

This was a model-level change — it required `python data/pull.py` to rebuild
the feature cache and `python train.py --all` to retrain. The result was a
meaningful Brier improvement (see Section 10). A secondary effect: the
unweighted LogReg variant (A2) became the new best model, narrowly beating
the previously top-ranked weighted variant (A3). The momentum decay feature
now does the recency work that sample weighting was previously compensating for.

### Simulation change: seed_diff removed from Elite 8 onward

`seed_diff` is excluded from the matchup features for rounds 4-6 (Elite 8,
Final Four, Championship). In late rounds, all remaining teams are strong —
1-seeds should not receive a name-based model advantage against a 4-seed that
has earned its spot.

### Simulation change: round-specific probability dampening

A `dampen(p, round_num)` function applies a per-round alpha to compress all
win probabilities toward 50/50. The formula is:

```
p_adjusted = 0.5 + (p_raw - 0.5) * alpha
```

Default alphas: `{1: 1.0, 2: 0.95, 3: 0.85, 4: 0.75, 5: 0.65, 6: 0.60}`

R64 is untouched (seeds matter most early). The compression tightens each
round, reflecting that late-round games are historically less predictable
regardless of team quality. Override with `--round-alphas`.

### Simulation change: per-game Gaussian noise injection

Before resolving each game, a random perturbation is added to the probability:

```
p_noisy = clip(p_adjusted + N(0, sigma), 0.05, 0.95)
```

This simulates game-day variance — foul trouble, shooting streaks, pace
mismatches — that the model cannot capture from season averages alone.
Default sigma is 0.0 (disabled). Recommended: `--noise-sigma 0.05`.

### Simulation change: chaos mode

A fraction of simulations run with more aggressive alphas (`CHAOS_ALPHAS`:
`{1: 0.90, 2: 0.80, 3: 0.70, 4: 0.60, 5: 0.50, 6: 0.45}`) and a higher
noise sigma (default 0.08). The results are blended into the main output,
producing a fatter tail on upset outcomes without fully discarding the model's
signal. Recommended: `--chaos-fraction 0.3`.

### Results

| Seed group | Baseline | Dampened+Noise+Chaos | Historical target |
| ---------- | -------- | -------------------- | ----------------- |
| 1-seeds    | 48.6%    | 44.3%                | ~40%              |
| 2-seeds    | 18.7%    | 18.6%                | ~20%              |
| 3-seeds    | 10.2%    | 10.4%                | ~10%              |
| 4-12+      | 22.5%    | 26.7%                | ~30%              |

Settings used: `--noise-sigma 0.05 --chaos-fraction 0.3`

The 1-seed rate moved from 48.6% → 44.3% and the 4-12+ rate increased from
22.5% → 26.7%, moving meaningfully toward historical norms. To push 1-seeds
closer to the ~40% target, increase `--chaos-fraction` to 0.5.

---

## 10. File structure reference

```
project root/
├── data/
│   ├── loader.py               ← API calls
│   ├── pull.py                 ← orchestrates data pull + feature build
│   ├── cache/
│   │   ├── raw/                ← one CSV per data type per season
│   │   ├── X_train.csv         ← training feature matrix
│   │   ├── y_train.csv         ← training labels
│   │   ├── X_val.csv           ← validation feature matrix
│   │   ├── y_val.csv           ← validation labels
│   │   └── features_by_season.pkl  ← per-team features dict (needed for simulation)
│   └── brackets/
│       ├── bracket_2025.json   ← 2025 tournament bracket
│       └── bracket_{year}.json ← historical brackets (for backtest)
├── features/
│   └── builder.py              ← all feature engineering (incl. momentum decay)
├── models/
│   ├── game_predictor_model.py     ← HistGBT wrapper
│   ├── logistic_predictor.py       ← LogReg wrapper
│   ├── calibrated_predictor.py     ← isotonic calibration wrapper
│   ├── ensemble_predictor.py       ← HistGBT + LightGBM ensemble
│   ├── best_params.json            ← tuned HistGBT params
│   ├── best_logistic_params.json   ← tuned LogReg params
│   ├── game_predictor.pkl          ← saved HistGBT model
│   └── logistic_predictor.pkl      ← saved LogReg model
├── validation/
│   ├── calibration.py          ← hyperparameter search
│   ├── simulation.py           ← Monte Carlo bracket simulator (+ dampening/noise/chaos)
│   └── backtest.py             ← historical simulation backtesting
├── train.py                    ← main training + evaluation script
├── requirements.txt
└── .env                        ← COLLEGE_FOOTBALL_API_KEY=...
```

---

## 11. Current benchmarks — what good looks like

```
Prediction model (train.py --all):
  LogReg  full  post+conf (unweighted)  Brier=0.1394  Acc=84.3%  n=121  ← current best
  LogReg  full  post+conf+weighted      Brier=0.1404  Acc=82.6%  n=121
  HistGBT trimmed  post+conf            Brier=0.1521  Acc=78.5%  n=121

Simulation (validation/simulation.py):
  R64 Brier should be close to ~0.139 (same as prediction model)
  Overall Brier will be higher (~0.17-0.19) due to round compounding
  Calibration max gap should be < 0.05 across all probability bins

Final Four seed distribution (10k sims, --noise-sigma 0.05 --chaos-fraction 0.3):
  1-seeds: ~44%   (historical target ~40%)
  2-seeds: ~19%   (historical target ~20%)
  3-seeds: ~10%   (historical target ~10%)
  4-12+:   ~27%   (historical target ~30%)

Backtest (validation/backtest.py):
  Average Brier across 2016-2025: target < 0.18
  Year-to-year variance: expected — some years are upset-heavy
```
