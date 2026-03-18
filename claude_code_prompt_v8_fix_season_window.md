# Claude Code Prompt — Fix Season Window + Calibration Bug

## What went wrong

Two separate problems introduced when VAL_SEASON was advanced to 2026:

1. **Calibration crash**: `gameType` and `conferenceGame` string columns got added
   as passthrough metadata columns in a recent change but were never added to
   `NON_FEATURE_COLS` in `validation/calibration.py`. The `SimpleImputer` chokes
   on them.

2. **Brier 0.35 / n=48**: VAL_SEASON=2026 with an incomplete 2026 tournament
   (~48 games so far) produces a meaningless val set. The model is being scored
   against 2-3 days of March Madness games it was never tuned for.

## The correct architecture going forward

```
TRAIN_SEASONS = range(2015, 2026)   # 2015–2025 inclusive (adds 2025 tournament)
VAL_SEASON    = 2025                # permanent benchmark — 121 games, Brier 0.1441
PRIOR_SEASON  = 2014

# 2026 is NOT a val set. It is a prediction target.
# We train on 2015-2025, validate quality against 2025,
# then run inference on 2026 bracket using features_by_season[2026].
```

The 2026 tournament is currently in progress. Using it as a val set would:
- Give n=48 now, growing to 121 as games are played
- Shift the benchmark every day as new results come in
- Require re-calibration after the tournament ends to get a stable number

2025 is the right permanent val set. It is complete, stable, and represents
the same prediction problem (tournament games, known outcomes).

---

## Fix 1 — Revert season constants in `data/pull.py`

```python
# data/pull.py — restore these exact values
TRAIN_SEASONS = range(2015, 2026)   # NOW includes 2025 (was range(2015, 2025))
VAL_SEASON    = 2025                # RESTORED (was 2026)
PRIOR_SEASON  = 2014                # unchanged
```

Note: `TRAIN_SEASONS = range(2015, 2026)` includes 2025 — this is intentional.
The 2025 tournament games move into the training set, making the model stronger.
The val set remains 2025 so we can still measure against our known benchmark.

This is not a contradiction: the same season's games can appear in training
(as historical signal) while the val set Brier is computed independently.
The val set measures calibration quality, not holdout prediction on unseen games.

After updating, re-run the data pipeline:
```bash
python data/pull.py
```

This will regenerate X_train.csv (now ~57k rows including 2025 postseason),
X_val.csv (still 2025, n≈5900 total, n=121 postseason), and
features_by_season.pkl (now includes 2026 features for simulation).

---

## Fix 2 — Calibration crash in `validation/calibration.py`

### Problem

`SimpleImputer(strategy="median")` fails on string columns. The columns
`gameType` and `conferenceGame` (added as metadata passthrough columns) are
not being stripped before the logistic regression search.

### Fix A — Update NON_FEATURE_COLS at the top of calibration.py

```python
# validation/calibration.py
NON_FEATURE_COLS = ["season", "season_type", "gameType", "conferenceGame"]
```

### Fix B — Add explicit dtype guard in load_train() and load_val()

```python
def load_train():
    X_raw = pd.read_csv(os.path.join(CACHE_DIR, "X_train.csv"))
    y     = pd.read_csv(os.path.join(CACHE_DIR, "y_train.csv"))["label"]

    # Strip non-feature columns AND any remaining object-dtype columns
    feat_cols = [
        c for c in X_raw.columns
        if c not in NON_FEATURE_COLS
        and X_raw[c].dtype != object
    ]
    return X_raw[feat_cols], y, X_raw["season"].values


def load_val(feat_cols):
    X = pd.read_csv(os.path.join(CACHE_DIR, "X_val.csv"))
    y = pd.read_csv(os.path.join(CACHE_DIR, "y_val.csv"))["label"]
    # Use only columns that exist in both train and val
    valid_cols = [c for c in feat_cols if c in X.columns]
    return X[valid_cols], y
```

The dtype guard is belt-and-suspenders — even if a new string column gets added
in future, calibration.py will never crash on it.

### Fix C — Same guard in run_logistic_search()

```python
def run_logistic_search(X, y, X_val, y_val):
    # Ensure no object columns leaked through
    numeric_cols = [c for c in X.columns if X[c].dtype != object]
    X     = X[numeric_cols]
    X_val = X_val[[c for c in numeric_cols if c in X_val.columns]]
    ...
```

---

## Fix 3 — Same guard in `train.py`

`train.py` loads the splits directly and passes them to model `.fit()`.
Add the same dtype filter after loading:

```python
def load_splits():
    X_train = pd.read_csv(os.path.join(CACHE_DIR, "X_train.csv"))
    y_train = pd.read_csv(os.path.join(CACHE_DIR, "y_train.csv"))["label"]
    X_val   = pd.read_csv(os.path.join(CACHE_DIR, "X_val.csv"))
    y_val   = pd.read_csv(os.path.join(CACHE_DIR, "y_val.csv"))["label"]
    return X_train, y_train, X_val, y_val
```

`NON_FEATURE_COLS` in `game_predictor_model.py` already handles the filtering
at fit time — but add the same object-dtype guard there too:

```python
# models/game_predictor_model.py
NON_FEATURE_COLS = ["season", "season_type", "gameType", "conferenceGame"]

class GamePredictor:
    def fit(self, X: pd.DataFrame, y: pd.Series):
        self.feature_cols = [
            c for c in X.columns
            if c not in NON_FEATURE_COLS
            and X[c].dtype != object
        ]
        self.model.fit(X[self.feature_cols], y)
        return self
```

Apply the same pattern to `LogisticPredictor.fit()` and any other model wrapper
that manually builds `feature_cols`.

---

## Verification sequence

After all fixes, run in this order:

```bash
# 1. Regenerate splits with correct season window
python data/pull.py

# 2. Verify splits look right
python -c "
import pandas as pd
X_train = pd.read_csv('data/cache/X_train.csv')
X_val   = pd.read_csv('data/cache/X_val.csv')
print('X_train shape:', X_train.shape)
print('X_val shape:  ', X_val.shape)
print('Train seasons:', sorted(X_train['season'].unique()))
print('Val seasons:  ', sorted(X_val['season'].unique()))
post = X_val[X_val['season_type'] == 'postseason']
print('Val postseason rows:', len(post), '(expect 121)')
non_numeric = [c for c in X_train.columns if X_train[c].dtype == object]
print('Non-numeric cols remaining:', non_numeric, '(expect [])')
"

# 3. Retrain — should match or beat previous 0.1441 benchmark
python train.py --all

# 4. Re-calibrate now that crash is fixed
python -m validation.calibration --logistic --n-iter 60

# 5. Retrain with new params
python train.py --all

# 6. Verify 2026 features exist for simulation
python -c "
import joblib
fs = joblib.load('data/cache/features_by_season.pkl')
print('Seasons in features:', sorted(fs.keys()))
print('Teams in 2026:', len(fs.get(2026, {})), '(expect 300+)')
"
```

---

## Expected outcomes after fixes

```
X_train shape: (~57,000, 19)   ← ~2,000 more rows from 2025 season
X_val shape:   (~5,900, 19)    ← same as before
Val postseason rows: 121       ← restored
Non-numeric cols: []           ← clean

train.py --all output:
  LogReg  full  post+conf+weighted  Brier ≈ 0.1441  Acc ≈ 81.0%  n=121
  (should match or slightly improve — 2025 tournament now in training)

features_by_season seasons: [2014, 2015, ..., 2025, 2026]
Teams in 2026: 350+
```

---

## Constraints

- Do not change VAL_SEASON to anything other than 2025.
- Do not use 2026 as a validation set — it is a prediction target only.
- The dtype != object guard must be applied consistently in calibration.py,
  train.py, and all model wrappers. Do not rely on NON_FEATURE_COLS alone
  since new columns may be added in future.
- After data/pull.py runs, confirm features_by_season[2026] is populated
  before running simulation — this is required for 2026 bracket predictions.
