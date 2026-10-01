"""Step 3. Rate every 2026 defense (FBS and FCS) on each metric, starting from regressed 2025 ratings.
Writes work/def_effects.json and work/off_effects.json."""
import paths  # noqa: F401  (sets the working directory to work/)
import pandas as pd, numpy as np, json, pickle
from common import *
R = pickle.load(open('seasons.pkl', 'rb')); Y = pd.read_csv('persistence.csv')
T = json.load(open('teams.json')); FBS = set(T['fbs'])
s = pd.read_parquet('plays.parquet')
fbs25 = set(R[2025]['_fbs'])
prior = {}
for k in [m for m in R[2025] if not m.startswith('_')]:
    r25 = R[2025][k]; tau2_full = r25['s2'] / r25['lam0']
    pk = {'tau2_default': tau2_full}
    for side, key in (('off', 'off'), ('def', 'dfn')):
        row = Y[(Y.metric == k) & (Y.side == key)].iloc[0]
        pk['tau2_' + side] = max(row.resid_sd ** 2, 0.25 * tau2_full)
        pk['mu_' + side] = {t: row.slope * v for t, v in r25[key].items() if t in fbs25}
    prior[k] = pk
res = fit(s, FBS, prior)
out = {k: v['dfn'] for k, v in res.items()}
json.dump(out, open('def_effects.json', 'w'))
json.dump({k: v['off'] for k, v in res.items()}, open('off_effects.json', 'w'))
e = pd.Series(out['epa']); fb = e[e.index.isin(FBS)].sort_values()
print('FBS defense spread (sd):', {k: round(float(pd.Series(v)[pd.Series(v).index.isin(FBS)].std()), 4) for k, v in out.items()})
print('FCS avg:', {k: round(float(pd.Series(v)[~pd.Series(v).index.isin(FBS)].mean()), 3) for k, v in out.items()})
print('\nBest FBS defenses (EPA/play allowed vs avg):'); print(fb.head(15).round(3).to_string())
print('\nWorst:'); print(fb.tail(8).round(3).to_string())
