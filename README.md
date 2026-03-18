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
#   LogReg  full  post+conf+weighted  Brier=0.1441  Acc=81.0%  n=121
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

## 4. Run tournament simulation (2025)

```bash
# Default — LogReg model, 10k simulations, top 20 teams
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

---

## 5. Backtest simulation against historical tournaments

```bash
# Backtest all available years (2019–2024, skips 2020)
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

## 6. Full pipeline — run everything fresh

```bash
# Step 1: Pull data (skip if cache is fresh)
python data/pull.py

# Step 2: Tune hyperparameters (skip if params files exist and features unchanged)
python -m validation.calibration --logistic --histgbt --n-iter 60 --weighted-cv

# Step 3: Train all models
python train.py --all

# Step 4: Run 2025 simulation
python validation/simulation.py --n-sims 10000 --save data/cache/sim_results_2025.pkl

# Step 5: Backtest
python validation/backtest.py --seasons 2022 2023 2024
```

---

## 7. Typical iteration loop (after initial setup)

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
python validation/simulation.py --n-sims 10000
```

---

## 8. Useful one-liners for inspection

```bash
# Check what's in the feature matrix
python -c "
import pandas as pd
X = pd.read_csv('data/cache/X_train.csv')
print('Shape:', X.shape)
print('Columns:', X.columns.tolist())
print('Season types:', X['season_type'].value_counts().to_dict())
"

# Check available team names in 2025 features (for bracket alignment)
python -c "
import joblib
fs = joblib.load('data/cache/features_by_season.pkl')
print(sorted(fs[2025].keys()))
"

# Print current best params
python -c "
import json
print('HistGBT:')
print(json.dumps(json.load(open('models/best_params.json')), indent=2))
print('LogReg:')
print(json.dumps(json.load(open('models/best_logistic_params.json')), indent=2))
"

# Quick feature importance check without retraining
python -c "
import joblib, pandas as pd
model = joblib.load('models/logistic_predictor.pkl')
coefs = model.pipeline.named_steps['clf'].coef_[0]
feat_df = pd.DataFrame({'feature': model.feature_cols, 'coef': coefs})
feat_df['abs'] = feat_df['coef'].abs()
print(feat_df.sort_values('abs', ascending=False).to_string(index=False))
"

# Check which seasons are in features cache
python -c "
import joblib
fs = joblib.load('data/cache/features_by_season.pkl')
print('Available seasons:', sorted(fs.keys()))
print('Teams in 2025:', len(fs.get(2025, {})))
"

# Count postseason rows per season in training data
python -c "
import pandas as pd
X = pd.read_csv('data/cache/X_train.csv')
post = X[X['season_type'] == 'postseason']
print(post.groupby('season').size().to_string())
"
```

---

## 9. File structure reference

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
│   │   └── features_by_season.pkl  ← per-team features for simulation
│   └── brackets/
│       ├── bracket_2025.json   ← 2025 tournament bracket
│       └── bracket_{year}.json ← historical brackets (for backtest)
├── features/
│   └── builder.py              ← all feature engineering
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
│   ├── simulation.py           ← Monte Carlo bracket simulator
│   └── backtest.py             ← historical simulation backtesting
├── train.py                    ← main training + evaluation script
├── requirements.txt
└── .env                        ← COLLEGE_FOOTBALL_API_KEY=...
```

---

## 10. Current benchmark — what good looks like

```
Prediction model (train.py --all):
  LogReg  full  post+conf+weighted   Brier=0.1438  Acc=80.7%  n=121  ← current best
  HistGBT trimmed  post+conf         Brier=0.1478  Acc=77.7%  n=121

Simulation (validation/simulation.py):
  R64 Brier should be close to ~0.144 (same as prediction model)
  Overall Brier will be higher (~0.17-0.19) due to round compounding
  Calibration max gap should be < 0.05 across all probability bins

Backtest (validation/backtest.py):
  Average Brier across 2019-2026: target < 0.18
  Year-to-year variance: expected, some years are upset-heavy
```
