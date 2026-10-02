"""Step 2. Fit defense ratings on the 2024 and 2025 seasons and measure how much they carry over year to
year; the carryover sets the strength of the 2026 priors. Also splits the carryover into a conference-average part and a
within-conference part, used by the game projections. Writes work/seasons.pkl, work/persistence.csv and
work/persistence_conf.csv."""
import paths  # noqa: F401  (sets the working directory to work/)
import pandas as pd, numpy as np, json, pickle
from common import *
R = {}
for y in (2024, 2025):
    p = pd.read_parquet(f'../data/pbp_{y}.parquet', columns=COLS)
    FBS = fbs_teams(p); s = outcomes(p)
    R[y] = fit(s, FBS); R[y]['_fbs'] = sorted(FBS)
    g = p.drop_duplicates('game_id')
    R[y]['_conf'] = {**dict(zip(g.away_team, g.away_team_conference)), **dict(zip(g.home_team, g.home_team_conference))}
    print(y, 'plays', len(s), 'FBS', len(FBS), {k: round(v['lam0']) for k, v in R[y].items() if not k.startswith('_')})
pickle.dump(R, open('seasons.pkl', 'wb'))
# year-to-year persistence per metric (FBS teams in both seasons)
both = sorted(set(R[2024]['_fbs']) & set(R[2025]['_fbs']))
rows = []
for k in [m for m in R[2025] if not m.startswith('_')]:
    for side in ('off', 'dfn'):
        a = np.array([R[2024][k][side][t] for t in both]); b = np.array([R[2025][k][side][t] for t in both])
        slope = np.cov(a, b)[0, 1] / a.var(ddof=1); resid = b - slope * a
        rows.append(dict(metric=k, side=side, r=np.corrcoef(a, b)[0, 1], slope=slope, sd25=b.std(), resid_sd=resid.std()))
Y = pd.DataFrame(rows); print(Y.round(3).to_string(index=False)); Y.to_csv('persistence.csv', index=False)
# the same carryover, split: conference average (teams grouped by their 2025 conference) and standing within it
rows = []
for k in [m for m in R[2025] if not m.startswith('_')]:
    for side in ('off', 'dfn'):
        cm = conf_means(R[2024][k][side], both, R[2025]['_conf'])
        xc = np.array([cm[t] for t in both]); xd = np.array([R[2024][k][side][t] for t in both]) - xc
        b = np.array([R[2025][k][side][t] for t in both])
        (sc, sd), *_ = np.linalg.lstsq(np.c_[xc, xd], b, rcond=None)
        rows.append(dict(metric=k, side=side, s_conf=sc, s_dev=sd, resid_sd=(b - sc * xc - sd * xd).std()))
Yc = pd.DataFrame(rows); print(Yc.round(3).to_string(index=False)); Yc.to_csv('persistence_conf.csv', index=False)
