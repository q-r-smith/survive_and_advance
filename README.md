
# March Madness Simulation Model

A personal data science project — end-to-end NCAA tournament bracket prediction
using machine learning, Monte Carlo simulation, and historical backtesting.

The model predicts the probability of each team winning any given matchup, then
simulates the full 64-team bracket 10,000+ times to produce win probabilities
at every round. It was built from scratch, covers the complete ML pipeline, and
has been iteratively improved through several rounds of experimentation.

Run all commands from the project root directory.

---

## Table of Contents

1. [Project Overview](https://claude.ai/chat/a16475b9-efdd-404a-8f92-f119a584d261#1-project-overview)
2. [How It Works — Architecture](https://claude.ai/chat/a16475b9-efdd-404a-8f92-f119a584d261#2-how-it-works--architecture)
3. [Feature Engineering](https://claude.ai/chat/a16475b9-efdd-404a-8f92-f119a584d261#3-feature-engineering)
4. [Models &amp; Training](https://claude.ai/chat/a16475b9-efdd-404a-8f92-f119a584d261#4-models--training)
5. [Simulation](https://claude.ai/chat/a16475b9-efdd-404a-8f92-f119a584d261#5-simulation)
6. [Chalk Bias — The Problem &amp; What We Did](https://claude.ai/chat/a16475b9-efdd-404a-8f92-f119a584d261#6-chalk-bias--the-problem--what-we-did)
7. [Development History &amp; Key Decisions](https://claude.ai/chat/a16475b9-efdd-404a-8f92-f119a584d261#7-development-history--key-decisions)
8. [Current Benchmarks](https://claude.ai/chat/a16475b9-efdd-404a-8f92-f119a584d261#8-current-benchmarks)
9. [Commands — Setup &amp; Data](https://claude.ai/chat/a16475b9-efdd-404a-8f92-f119a584d261#9-commands--setup--data)
10. [Commands — Training](https://claude.ai/chat/a16475b9-efdd-404a-8f92-f119a584d261#10-commands--training)
11. [Commands — Simulation](https://claude.ai/chat/a16475b9-efdd-404a-8f92-f119a584d261#11-commands--simulation)
12. [Commands — Backtesting](https://claude.ai/chat/a16475b9-efdd-404a-8f92-f119a584d261#12-commands--backtesting)
13. [Commands — Full Pipeline](https://claude.ai/chat/a16475b9-efdd-404a-8f92-f119a584d261#13-commands--full-pipeline)
14. [Commands — Iteration Loop](https://claude.ai/chat/a16475b9-efdd-404a-8f92-f119a584d261#14-commands--iteration-loop)
15. [Useful One-Liners](https://claude.ai/chat/a16475b9-efdd-404a-8f92-f119a584d261#15-useful-one-liners)
16. [File Structure](https://claude.ai/chat/a16475b9-efdd-404a-8f92-f119a584d261#16-file-structure)

---

## 1. Project Overview

This is a personal, non-work project. The goal was to build a bracket prediction
system that goes beyond seed-based heuristics and actually learns from historical
tournament and regular-season game data. The project is fully operational and has
been used to generate 2025 and 2026 tournament predictions.

**Stack:** Python, scikit-learn, numpy, pandas, HistGradientBoostingClassifier,
LogisticRegression, College Basketball Data API (CBBD).

**Data:** 2015–2026 seasons, including raw game results, team stats, player stats,
rosters, SRS ratings, adjusted efficiency ratings, and ELO ratings. All data is
cached locally as CSVs after the first pull to avoid redundant API calls.

**Approach in one sentence:** Train a binary classifier to predict game outcomes
using end-of-season team features represented as differentials, then use that
classifier to drive a Monte Carlo bracket simulation.

---

## 2. How It Works — Architecture

### The pipeline has five stages:

**1. Data collection** — `data/pull.py` fetches all raw data from the CFBD API
and caches it locally. This takes 10–20 minutes on the first run; subsequent
runs use the cache.

**2. Feature engineering** — `features/builder.py` computes per-team,
per-season features from the raw data. These become the columns in `X_train`
and `X_val`.

**3. Model training** — `train.py` builds matchup features (team A stats minus
team B stats) and trains classifiers to predict which team wins. The primary
metric is Brier score on the postseason validation set.

**4. Simulation** — `validation/simulation.py` pre-computes a full 64×64
probability matrix (one entry per possible matchup), then runs N independent
bracket simulations. Each simulation draws from those probabilities to resolve
each game. The result is a distribution of outcomes: for every team, the
probability of reaching R32, Sweet 16, Elite 8, Final Four, Championship, and
winning it all.

**5. Backtesting** — `validation/backtest.py` reconstructs historical brackets
from the game cache and scores the simulation against actual results, producing
round-by-round Brier scores for each historical season.

### Key design decisions:

**Matchup features as differentials.** Every feature is computed as
(Team A value) − (Team B value). The model sees relative strength, not absolute
numbers, and positive values always favor Team A. This halves the feature space
and makes the model naturally symmetric.

**Feature vintage to prevent leakage.** Regular season games use prior-season
features; postseason and conference tournament games use current-season
end-of-regular-season features. This mirrors real deployment — you're always
predicting with what you'd know on Selection Sunday. Validated at build time
with an assertion that 0 postseason 2025 games appear in the training set.

**Pre-built probability matrix.** The 64×64 matrix is computed once before the
simulation loop starts. Each simulation is then just matrix lookups and random
draws — 10,000 simulations run in seconds with no repeated model calls.

**Seed-based fallback.** For any team missing from the feature cache, the
simulator falls back to historical matchup win rates by seed pair. This prevents
crashes and keeps the simulation runnable even with incomplete data.

---

## 3. Feature Engineering

All features are computed at team-season level in `features/builder.py` and
differenced at matchup time in `build_matchup_features()`.

| Feature                   | Description                                                          | LogReg importance (rank) |
| ------------------------- | -------------------------------------------------------------------- | ------------------------ |
| `diff_elo`              | ELO rating differential — strength-of-record with recency weighting | #1                       |
| `diff_hot_streak`       | Recency-weighted win rate, split-window (last 5 games 2x, 6-15 1x)   | #2                       |
| `diff_srs`              | Simple Rating System — margin of victory adjusted for schedule      | #3                       |
| `diff_conf_strength`    | Leave-one-out avg SRS of conference peers                            | #4                       |
| `neutral_site`          | Always 1 in tournament (all games neutral)                           | #5                       |
| `diff_road_win_pct`     | Win rate in true road games — pressure performance proxy            | #6                       |
| `diff_adj_def_rank`     | Adjusted defensive efficiency rank (normalized 0–1)                 | #7                       |
| `diff_close_game_pct`   | Pct of games decided by ≤5 points — clutch signal                  | #8                       |
| `diff_adj_off_rating`   | Adjusted offensive efficiency (pts per 100 possessions)              | #9                       |
| `diff_upset_propensity` | Efficiency ratio deviation from seed-group mean                      | #11                      |
| `diff_experience`       | % of win shares from players 21+ at season start                     | low                      |
| `diff_star_power`       | Max PORPAG on roster — peak individual impact                       | low                      |
| `diff_pace`             | Possessions per game                                                 | low                      |
| `diff_non_conf_sos`     | Avg SRS of non-conference opponents                                  | low                      |
| `seed_diff`             | Seed number differential (dropped Elite 8 onward)                    | near-zero                |

**On `hot_streak`:** Uses a split-window recency weight — last 5 games at 2x,
games 6–15 at 1x. An earlier version used flat exponential decay over 10 games.
The split-window change produced a measurable Brier improvement (see Section 7).

**On the HistGBT trimmed feature set:** When HistGBT was the primary model,
permutation importance was used to identify and drop 8 features with zero or
negative importance (`diff_adj_off_rating`, `diff_net_rating`, `diff_experience`,
`diff_adj_off_rank`, `diff_adj_def_rank`, `diff_close_game_pct`,
`diff_road_win_pct`, `diff_non_conf_sos`). LogReg uses the full feature set —
it handles the extra features without overfitting due to its stronger built-in
regularization.

---

## 4. Models & Training

### Why Logistic Regression beats Gradient Boosting here

This is counterintuitive but well-established in tournament prediction
literature. The Klemm benchmark — a published study where logistic regression
trained on tournament-only data with ~6 features hit the top 99.5th percentile
on ESPN Tournament Challenge — validated this approach before the first line of
code was written here.

The reason: tournament games have fundamentally different dynamics than regular
season games. Pace, pressure, single-elimination survival, and coaching all
matter more. HistGBT fits non-linear patterns better in large training sets,
but with only ~600 NCAA tournament games in 10 years of training data, it
overfits. LogReg is more regularized by design. HistGBT's train Brier (~0.11)
vs. val Brier (~0.15–0.17) confirms the diagnosis.

### Training data strategy

Three regimes were evaluated:

* **All games** — every regular season + postseason game 2015–2024 (~61,000
  rows). Most data, but regular season noise dilutes tournament signal.
* **High-signal (post+conf)** — NCAA tournament + conference tournament games
  only (~4,300 rows). Less data, higher signal. Current best for LogReg.
* **Weighted** — all games with tiered sample weights (NCAA=10x, conf=5x,
  regular=1x for HistGBT; NCAA=3x, conf=1x for LogReg).

### Current model comparison

```
  Model      Features   Training set            Brier      Acc      n
  ──────────────────────────────────────────────────────────────────
  HistGBT    full       all games               0.1706   75.2%    121
  HistGBT    trimmed    all games               0.1698   73.6%    121
  HistGBT    trimmed    all+weighted            0.1583   75.2%    121
  HistGBT    trimmed    post+conf               0.1521   78.5%    121
  HistGBT    trimmed    post+conf+wtd           0.1537   76.9%    121
  HistGBT    full       post only (NCAA)        0.1652   75.2%    121
  LogReg     full       post only (NCAA)        0.1557   77.7%    121
  LogReg     full       post+conf               0.1394   84.3%    121  ← best ★
  LogReg     full       post+conf+wtd           0.1404   82.6%    121
```

Brier score: lower = better. Coin-flip baseline = 0.25. Perfect model = 0.0.
Target: sub-0.15 on the 121-game postseason validation set.

### Hyperparameter tuning

LogReg tuned via GridSearchCV over C, penalty (L1/L2/elasticnet), and solver.
HistGBT tuned via RandomizedSearchCV (40–80 iterations) over learning rate,
max_iter, max_depth, min_samples_leaf, l2_regularization, and max_bins. Best
params auto-saved to `models/best_params.json` and
`models/best_logistic_params.json`, auto-loaded by `train.py`.

Current best LogReg params: `C=0.1, penalty=l1, solver=liblinear`

---

## 5. Simulation

### Monte Carlo bracket simulation

The simulator pre-builds a 64×64 probability matrix before the loop starts,
then runs N independent bracket simulations (default 10,000). Each simulation
resolves matchups probabilistically using matrix lookups and numpy random draws.
Win counts accumulate across all simulations; dividing by N gives per-team win
probabilities at each round.

The simulator also generates a **chalk bracket** — always picking the
higher-probability team, propagating winners forward correctly. This is the
model's single-best bracket pick.

### Scoring

Against historical seasons (`--score`), the simulator reports:

* **Brier score** per round and overall (primary metric)
* **Accuracy** — did the chalk pick match the actual winner?
* **Calibration gap** — mean predicted probability vs. actual win rate per bin

R64 Brier should match the single-game model Brier (~0.139). Overall Brier
across all rounds will be higher (~0.17–0.19) due to compounding.

### Randomness adjustment flags

Three simulation-level levers reduce chalk bias (see Section 6). They combine
freely:

```bash
--round-alphas 1.0 0.95 0.85 0.75 0.65 0.60   # round-specific dampening
--noise-sigma 0.05                              # per-game Gaussian noise
--chaos-fraction 0.3                            # chaos mode blend fraction
--chaos-sigma 0.08                              # chaos-mode noise sigma (default)
```

**Recommended production settings:** `--noise-sigma 0.05 --chaos-fraction 0.3`

---

## 6. Chalk Bias — The Problem & What We Did

### The problem

A well-calibrated single-game model can still produce unrealistic bracket
distributions. Small per-game edges compound across six rounds: a 1-seed at
72% per game reaches the Final Four in ~26% of simulations per region, and
with four regions, 1-seeds dominate at well above historical rates.

Baseline simulation: **48.6%** of Final Four slots were 1-seeds.
Historical rate:  **~40%** .

### Four changes made

**1. Removed `seed_diff` from Elite 8 onward (model-level)**

`seed_diff` is excluded from matchup features for rounds 4–6. Early-round seed
gaps carry real information; by the Elite 8, every team has earned their spot
and a 1-seed should not receive a number-based model advantage.

**2. Momentum decay in `hot_streak` (feature-level)**

Split-window recency weighting (last 5 games 2x, games 6–15 1x) rather than
flat exponential decay over 10 games. Required rebuilding the feature cache and
retraining. Produced Brier 0.1438 → 0.1394, accuracy 80.7% → 84.3%.

**3. Round-specific probability dampening (simulation-level)**

Formula: `p_adjusted = 0.5 + (p_raw - 0.5) * alpha`

| Round        | Alpha | Example: raw p=0.78 → adjusted |
| ------------ | ----- | ------------------------------- |
| R64          | 1.00  | 78.0% (unchanged)               |
| R32          | 0.95  | 77.1%                           |
| Sweet 16     | 0.85  | 74.3%                           |
| Elite 8      | 0.75  | 71.5%                           |
| Final Four   | 0.65  | 68.7%                           |
| Championship | 0.60  | 67.0%                           |

Override with `--round-alphas`. No retraining required.

**4. Per-game Gaussian noise injection (simulation-level)**

`p_noisy = clip(p_adjusted + N(0, sigma), 0.05, 0.95)`

Simulates game-day variance the model cannot capture from season averages.
Default sigma=0.0 (disabled). Recommended: `--noise-sigma 0.05`.

**5. Chaos mode — simulation blending (simulation-level)**

A fraction of simulations run with aggressive dampening (`CHAOS_ALPHAS`:
R64=0.90, R32=0.80, S16=0.70, E8=0.60, F4=0.50, Champ=0.45) and higher
noise (default sigma=0.08). Results blend into the main output, producing
a fatter tail on upset outcomes. Default fraction=0.0. Recommended: `--chaos-fraction 0.3`.

### Results

| Seed group | Baseline | Dampened+Noise+Chaos | Historical target |
| ---------- | -------- | -------------------- | ----------------- |
| 1-seeds    | 48.6%    | 44.3%                | ~40%              |
| 2-seeds    | 18.7%    | 18.6%                | ~20%              |
| 3-seeds    | 10.2%    | 10.4%                | ~10%              |
| 4-12+      | 22.5%    | 26.7%                | ~30%              |

Settings: `--noise-sigma 0.05 --chaos-fraction 0.3`

1-seeds moved from 48.6% → 44.3%; 4-12+ seeds increased from 22.5% → 26.7%.
To push 1-seeds closer to ~40%, increase `--chaos-fraction` to 0.5.

---

## 7. Development History & Key Decisions

This section documents the major decisions, dead ends, and iterations that got
the project to where it is. The path was not linear.

### Starting point — the problem to solve

The naive seed-based approach (just pick by seed) gets Brier ~0.18–0.20 on
tournament games. The question was how much better a data-driven model could do,
and what features and training approaches actually moved the needle.

### Data source and API integration

The College Basketball Data API was chosen for its depth — game results, team
stats, player stats, rosters, SRS, ELO, and adjusted efficiency ratings all in
one place. An early challenge: the API returns game data without clean
tournament-round labels. Rounds had to be inferred from seed-pair matchups
(Round 1 matchups always sum to seed 17: 1+16, 2+15, etc.) and game dates.
`inspect_tournament_data.py` was written specifically to validate this round
inference before any model training began.

### Feature look-ahead leakage — a subtle but critical early bug

An early version of the feature pipeline used full-season stats for all games,
including regular season games. This introduced look-ahead leakage: a January
game was being predicted using statistics from March. The model was cheating.

The fix was the feature vintage system in `build_training_set()`: regular season
games use prior-season features; postseason and conference tournament games use
current-season end-of-regular-season features. This is explicitly validated at
build time with an assertion that 0 postseason 2025 games appear in training.
Plugging this leak reduced Brier — the earlier inflated metrics were not real.

### HistGBT as the baseline — and why it underperformed

The first real model was HistGradientBoostingClassifier trained on all games.
It's fast, handles missing values natively, and generally performs well on
tabular data. On the full regular-season training set, it achieved good training
Brier (~0.11) but mediocre validation Brier on postseason games (~0.15–0.17).
The train/val gap was a clear sign of overfitting.

Several things were tried to fix this:

* **Trimming features** to only those with positive permutation importance — the
  `HISTGBT_FEATURES` list dropped 8 features (`diff_adj_off_rating`,
  `diff_net_rating`, `diff_experience`, `diff_adj_off_rank`, `diff_adj_def_rank`,
  `diff_close_game_pct`, `diff_road_win_pct`, `diff_non_conf_sos`). Small gain.
* **Training on postseason + conference tournament games only** (~4,300 rows vs.
  61,000). Surprising improvement despite 93% less data — see below.
* **Sample weighting** (NCAA=10x, conf=5x, regular=1x). Helped for the
  all-games regime, less so when already training on high-signal data only.
* **Hyperparameter tuning** via RandomizedSearchCV (40–80 iterations). Marginal
  gains once the training set was right.

The best HistGBT result was Brier=0.1521 — competitive but stubbornly above
the sub-0.15 target.

### The Klemm benchmark — why we tried logistic regression

Digging into the tournament prediction literature, the Klemm benchmark stood
out: logistic regression trained on tournament-only data with ~6 features hit
the top 99.5th percentile on ESPN Tournament Challenge. The argument was that
tournament games are a different distribution than regular season, and that a
linear model generalizes better in this low-data regime.

Trying LogReg felt like going backwards — simpler model, fewer parameters. It
was not. The first run of LogReg on postseason-only data beat HistGBT's best
result immediately. Adding conference tournament games pushed it further. The
weighted variant (A3: NCAA=3x, conf=1x) became the first model to clearly break
sub-0.15.

The lesson: model complexity is not always an advantage. When the training set
is small and the distribution shift between training and inference is large,
regularization matters more than expressiveness.

### Training data composition — the high-signal discovery

The counterintuitive finding of the project: filtering from 61,000 rows to
4,300 rows (a 93% reduction) improved Brier. This was double-checked multiple
times.

The explanation: regular season games contain noise that doesn't transfer to
tournament dynamics. Home court advantage, back-to-back fatigue, teams resting
starters, early-season schedule mismatch — all captured in regular season
outcomes but irrelevant to neutral-site, single-elimination, high-stakes games.

Conference tournament games were a particularly valuable addition: they share
pressure dynamics with the NCAA tournament, occur after the full regular season
(so current features are fully available and leak-free), and add meaningful
signal without regular-season noise. Adding them to the training set pushed
LogReg to its best pre-decay result.

### Sample weighting — useful for HistGBT, partially redundant for LogReg

For HistGBT trained on all games, sample weighting (NCAA=10x, conf=5x,
regular=1x) helped significantly by forcing the model to prioritize
getting tournament-like games right. For LogReg already trained on only
post+conf data, the additional NCAA=3x weight gave a small improvement —
Brier 0.1394 (unweighted) vs. 0.1404 (weighted), with the unweighted
variant narrowly winning after the momentum decay change (see below).

This reversal is a good example of how feature improvements can make other
hyperparameter choices less important.

### Momentum decay — the most impactful feature change

The original `compute_hot_streak` used exponential decay over the last 10 games.
Already recency-weighted, but with no explicit emphasis on games immediately
before the tournament.

The change used a split-window approach: last 5 games at 2x, games 6–15 at 1x.
The intuition: teams genuinely peaking in late February/March are more dangerous
in the tournament than teams that dominated in November and faded.

Before retraining, the correlation between old and new `hot_streak` values was
examined. The values diverged enough — particularly for teams with uneven
season arcs — to suggest a real improvement was possible.

After rebuilding the feature cache (`python data/pull.py`) and retraining
(`python train.py --all`):

* Brier improved from **0.1438 → 0.1394** — clearing sub-0.15 with margin
* Accuracy jumped from **80.7% → 84.3%** — roughly 4–5 more correct picks
  across the 121-game val set
* The unweighted LogReg (A2) became the new best variant, narrowly beating
  the previously top-ranked weighted variant (A3, Brier 0.1404)

The weighting flip was the secondary finding: momentum decay absorbed the
recency work that sample weighting had been compensating for.

### Chalk bias — discovering the simulation is a separate problem

After getting model Brier below 0.15, running the simulation against the 2026
bracket made it clear that good single-game predictions don't automatically
produce realistic bracket distributions. The model was outputting 1-seeds in
48.6% of Final Four slots — historically it's ~40%.

The root cause: the probability matrix is static. Every round uses the same
raw win probability. A 1-seed that is 75% to win in Round 1 is treated as 75%
in the Final Four too, even though late-round games are empirically much less
predictable.

The fix was a set of simulation-level adjustments — dampening, noise injection,
and chaos mode — applied on top of model output without retraining anything.
The first step (removing `seed_diff` from Elite 8 onward) was a model-level
change that addressed a separate but related issue: 1-seeds were getting a
name-based feature advantage in late rounds even against other strong teams.
See Section 6 for full details and results.

---

## 8. Current Benchmarks

```
Prediction model (train.py --all):
  LogReg  full  post+conf (unweighted)   Brier=0.1394  Acc=84.3%  n=121  ← best
  LogReg  full  post+conf+weighted       Brier=0.1404  Acc=82.6%  n=121
  HistGBT trimmed  post+conf             Brier=0.1521  Acc=78.5%  n=121

Simulation (validation/simulation.py):
  R64 Brier should be close to ~0.139 (matches prediction model)
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

---

## 9. Commands — Setup & Data

```bash
# First-time setup
python -m venv venv
source venv/bin/activate          # Mac/Linux
# venv\Scripts\activate           # Windows
pip install -r requirements.txt
cat .env                          # confirm COLLEGE_BASKETBALL_API_KEY=... is set

# Pull all data and build feature cache (10–20 min first run, cached after)
python data/pull.py

# Force re-fetch from API even if cache exists (use sparingly)
python data/pull.py --force

# Validate tournament data and round inference
python inspect_tournament_data.py

# Quick bracket sanity check
python -c "
from data.bracket_builder import build_bracket_from_cache
b = build_bracket_from_cache(2025)
print('Champion:', b['champion'])
print('Seeds sample:', dict(list(b['seeds'].items())[:5]))
print('R1 games:', len(b['rounds'][1]))
"
```

---

## 10. Commands — Training

```bash
# Full comparison — all model types x training sets (primary command)
python train.py --all

# Baseline HistGBT only (fastest sanity check)
python train.py

# Tune hyperparameters (run when features change)
python -m validation.calibration --logistic --histgbt --n-iter 60 --weighted-cv

# Logistic only
python -m validation.calibration --logistic

# More thorough HistGBT search
python -m validation.calibration --n-iter 80
```

---

## 11. Commands — Simulation

```bash
# Default — LogReg, 10k sims, no randomness adjustments
python validation/simulation.py

# Score against actual 2025 results
python validation/simulation.py --score

# Recommended settings — reduces chalk bias toward historical norms
python validation/simulation.py --n-sims 10000 --noise-sigma 0.05 --chaos-fraction 0.3

# Push 1-seeds harder toward ~40% historical rate
python validation/simulation.py --n-sims 10000 --noise-sigma 0.05 --chaos-fraction 0.5

# Full production run — save output
python validation/simulation.py --n-sims 50000 --noise-sigma 0.05 --chaos-fraction 0.3 \
    --save data/cache/sim_results_2026.pkl

# Score recommended settings against actuals
python validation/simulation.py --n-sims 10000 --noise-sigma 0.05 --chaos-fraction 0.3 --score

# Override round alphas manually (6 values: R64 R32 S16 E8 F4 Champ)
python validation/simulation.py --round-alphas 1.0 0.95 0.85 0.75 0.65 0.60

# Historical season
python validation/simulation.py --season 2024 --score

# HistGBT model instead of LogReg
python validation/simulation.py --model histgbt

# Show more teams in output
python validation/simulation.py --top-n 30

# Load and inspect saved results
python -c "
import joblib
results = joblib.load('data/cache/sim_results_2026.pkl')
results.print_summary(top_n=20)
"
```

---

## 12. Commands — Backtesting

```bash
# Backtest all available seasons (2016-2025, skips 2020)
python validation/backtest.py

# Specific seasons only
python validation/backtest.py --seasons 2022 2023 2024

# More sims per season for stability
python validation/backtest.py --n-sims 50000

# HistGBT model
python validation/backtest.py --model histgbt
```

---

## 13. Commands — Full Pipeline

```bash
# Run everything from scratch
python data/pull.py
python -m validation.calibration --logistic --histgbt --n-iter 60 --weighted-cv
python train.py --all
python validation/simulation.py --n-sims 50000 --noise-sigma 0.05 --chaos-fraction 0.3 \
    --save data/cache/sim_results_2026.pkl
python validation/backtest.py
```

---

## 14. Commands — Iteration Loop

When you've changed features or model code:

```bash
# 1. Rebuild feature cache (required if features/builder.py changed)
python data/pull.py

# 2. Retrain and compare
python train.py --all

# 3. If Brier improved and you want to re-tune hyperparams
python -m validation.calibration --logistic --n-iter 60

# 4. Retrain with new tuned params
python train.py --all

# 5. Re-run simulation
python validation/simulation.py --n-sims 10000 --noise-sigma 0.05 --chaos-fraction 0.3
```

---

## 15. Useful One-Liners

```bash
# Check feature matrix shape and columns
python -c "
import pandas as pd
X = pd.read_csv('data/cache/X_train.csv')
print('Shape:', X.shape)
print('Columns:', X.columns.tolist())
print('Season types:', X['season_type'].value_counts().to_dict())
"

# Check available team names in 2026 features
python -c "
import joblib
fs = joblib.load('data/cache/features_by_season.pkl')
print(sorted(fs[2026].keys()))
"

# Print current best hyperparameters
python -c "
import json
print('HistGBT:')
print(json.dumps(json.load(open('models/best_params.json')), indent=2))
print('LogReg:')
print(json.dumps(json.load(open('models/best_logistic_params.json')), indent=2))
"

# Quick LogReg feature importance without retraining
python -c "
import joblib, pandas as pd
model = joblib.load('models/logistic_predictor.pkl')
coefs = model.pipeline.named_steps['clf'].coef_[0]
feat_df = pd.DataFrame({'feature': model.feature_cols, 'coef': coefs})
feat_df['abs'] = feat_df['coef'].abs()
print(feat_df.sort_values('abs', ascending=False).to_string(index=False))
"

# Count postseason rows per season in training data
python -c "
import pandas as pd
X = pd.read_csv('data/cache/X_train.csv')
post = X[X['season_type'] == 'postseason']
print(post.groupby('season').size().to_string())
"

# Check available seasons and team counts in feature cache
python -c "
import joblib
fs = joblib.load('data/cache/features_by_season.pkl')
for s in sorted(fs.keys()):
    print(f'  {s}: {len(fs[s])} teams')
"
```

---

## 16. File Structure

```
project root/
├── data/
│   ├── loader.py                    ← API calls (CFBD)
│   ├── pull.py                      ← orchestrates data pull + feature build
│   ├── bracket_builder.py           ← reconstructs historical brackets from cache
│   ├── cache/
│   │   ├── raw/                     ← one CSV per data type per season
│   │   │   ├── games_{year}.csv
│   │   │   ├── team_stats_{year}.csv
│   │   │   ├── player_stats_{year}.csv
│   │   │   ├── rosters_{year}.csv
│   │   │   ├── srs_{year}.csv
│   │   │   ├── adjusted_ratings_{year}.csv
│   │   │   └── elo_{year}.csv
│   │   ├── X_train.csv              ← training feature matrix (2015-2024)
│   │   ├── y_train.csv              ← training labels
│   │   ├── X_val.csv                ← validation feature matrix (2025)
│   │   ├── y_val.csv                ← validation labels
│   │   └── features_by_season.pkl  ← per-team features dict (used by simulation)
│   └── bracket_2026.json            ← 2026 tournament bracket (manually entered)
├── features/
│   └── builder.py                   ← all feature engineering, incl. momentum decay
├── models/
│   ├── game_predictor_model.py      ← HistGBT wrapper
│   ├── logistic_predictor.py        ← LogReg wrapper (current best)
│   ├── calibrated_predictor.py      ← isotonic calibration wrapper
│   ├── ensemble_predictor.py        ← HistGBT + LightGBM ensemble
│   ├── best_params.json             ← tuned HistGBT hyperparameters
│   ├── best_logistic_params.json    ← tuned LogReg hyperparameters
│   ├── eval_results.json            ← full metrics from last train.py --all run
│   ├── game_predictor.pkl           ← saved HistGBT model
│   └── logistic_predictor.pkl       ← saved LogReg model (current best)
├── validation/
│   ├── calibration.py               ← hyperparameter search
│   ├── simulation.py                ← Monte Carlo simulator + dampening/noise/chaos
│   └── backtest.py                  ← historical simulation backtesting
├── inspect_tournament_data.py       ← validates round inference from API data
├── train.py                         ← main training + evaluation script
├── requirements.txt
└── .env                             ← COLLEGE_BASKETBALL_API_KEY=...
```
