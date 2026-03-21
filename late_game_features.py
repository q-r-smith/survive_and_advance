import pandas as pd
import numpy as np

X = pd.read_csv('data/cache/X_train.csv')
y = pd.read_csv('data/cache/y_train.csv')['label']

# Postseason only, attach outcome
post = X[X['season_type'] == 'postseason'].copy()
post['outcome'] = y[post.index]

feature_cols = [c for c in post.columns if c.startswith('diff_')]

# Round-by-round Pearson correlation with win outcome
round_names = {1: 'R64', 2: 'R32', 3: 'S16', 4: 'E8', 5: 'F4', 6: 'Champ'}
results = {}

for rnd, name in round_names.items():
    subset = post[post['round_num'] == rnd]
    if len(subset) < 10:
        continue
    corr = subset[feature_cols].corrwith(subset['outcome'])
    results[name] = corr
    print(f"\n{name}  (n={len(subset)})")
    print(corr.abs().sort_values(ascending=False).head(8).to_string())

# Build a comparison DataFrame — how does each feature's importance shift by round?
df = pd.DataFrame(results).T
print("\n\n=== Feature importance shift across rounds (absolute correlation) ===")
print(df.abs().to_string())