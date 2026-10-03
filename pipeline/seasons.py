"""Step 2. Fit offense and defense ratings on the 2024 and 2025 seasons (every play metric plus points per drive) and
measure how much they carry over year to year; the carryover sets the 2026 starting points. Also splits the carryover
into a conference-average part and a within-conference part, plus a roster-talent term (247 talent composite), used by
the game and player projections. Writes work/seasons.pkl, work/persistence.csv and work/persistence_conf.csv."""
import paths  # noqa: F401  (sets the working directory to work/)
import pandas as pd, numpy as np, json, pickle
from common import *
R = {}
for y in (2024, 2025):
    p = pd.read_parquet(f'../data/pbp_{y}.parquet', columns=COLS)
    FBS = fbs_teams(p); s = outcomes(p)
    R[y] = fit(s, FBS, drives=drive_table(s)); R[y]['_fbs'] = sorted(FBS)
    g = p.drop_duplicates('game_id')
    R[y]['_conf'] = {**dict(zip(g.away_team, g.away_team_conference)), **dict(zip(g.home_team, g.home_team_conference))}
    R[y]['_tid'] = team_ids(p)
    print(y, 'plays', len(s), 'FBS', len(FBS), {k: round(v['lam0']) for k, v in R[y].items() if not k.startswith('_')})
pickle.dump(R, open('seasons.pkl', 'wb'))
# year-to-year persistence per metric (FBS teams in both seasons)
both = sorted(set(R[2024]['_fbs']) & set(R[2025]['_fbs']))
METRICS = [m for m in R[2025] if not m.startswith('_')]
rows = []
for k in METRICS:
    for side in ('off', 'dfn'):
        a = np.array([R[2024][k][side][t] for t in both]); b = np.array([R[2025][k][side][t] for t in both])
        slope = np.cov(a, b)[0, 1] / a.var(ddof=1); resid = b - slope * a
        rows.append(dict(metric=k, side=side, r=np.corrcoef(a, b)[0, 1], slope=slope, sd25=b.std(), resid_sd=resid.std()))
Y = pd.DataFrame(rows); print(Y.round(3).to_string(index=False)); Y.to_csv('persistence.csv', index=False)
# the same carryover, split: conference average (teams grouped by their 2025 conference), standing within it, and 2025 roster talent
tz = talent_z('../data/talent_2025.parquet', R[2025]['_tid'], set(R[2025]['_fbs']))
rows = []
for k in METRICS:
    for side in ('off', 'dfn'):
        cm = conf_means(R[2024][k][side], both, R[2025]['_conf'])
        xc = np.array([cm[t] for t in both]); xd = np.array([R[2024][k][side][t] for t in both]) - xc
        xt = np.array([tz.get(t, 0.0) for t in both])
        b = np.array([R[2025][k][side][t] for t in both])
        (sc, sd, st), *_ = np.linalg.lstsq(np.c_[xc, xd, xt], b, rcond=None)
        rows.append(dict(metric=k, side=side, s_conf=sc, s_dev=sd, s_tal=st, resid_sd=(b - sc * xc - sd * xd - st * xt).std()))
Yc = pd.DataFrame(rows); print(Yc.round(3).to_string(index=False)); Yc.to_csv('persistence_conf.csv', index=False)
