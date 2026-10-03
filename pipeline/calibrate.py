"""Step 3b. Calibrate the game model on last season. Walks 2025 forward week by week the way the 2026 projections work:
ratings start from 2024 ratings (conference average, within-conference standing and roster talent carried over
separately, held loosely) and only use plays from earlier weeks. Each team's edge is measured three ways, EPA per play,
points per drive and success rate, each scaled to the game's expected plays or drives; least squares on 2025 margins and
totals sets how much each counts, plus league-tier terms (scrimmage stats understate how far apart Power 4, Group of 5 and
FCS teams are) and the spread of actual margins around the projection. Writes work/calib.json."""
import paths  # noqa: F401  (sets the working directory to work/)
import pandas as pd, numpy as np, json, pickle
from common import *
R = pickle.load(open('seasons.pkl', 'rb')); Yc = pd.read_csv('persistence_conf.csv')
p = pd.read_parquet('../data/pbp_2025.parquet', columns=COLS)
FBS = fbs_teams(p); s = outcomes(p); conf25 = R[2025]['_conf']
tz = talent_z('../data/talent_2025.parquet', R[2025]['_tid'], FBS)
prior = season_prior(R, Yc, base=2024, conf=conf25, tz=tz, loosen=LOOSEN)
sched = pd.read_parquet('../data/schedules_2025.parquet'); sched = sched[sched.season_type == 'regular']
s = s[s.game_id.astype(int).isin(set(sched.game_id.astype(int)))]   # play-by-play labels bowls week 1; regular season only
dr = drive_table(s)
pts = points_lookup(sched)
gw = s.groupby(s.game_id.astype(int)).week.first()
games = sched[sched.game_id.astype(int).isin(gw.index)]
rows = []
for w in range(2, int(gw.max()) + 1):
    tr = s[s.week < w]; dw = dr[dr.week < w]
    r = fit(tr, FBS, prior, keys=('epa', 'sr'), drives=dw)
    L, ppp, po, pdf = pace_points(tr, FBS, pts); Ld, dpo, dpd = drive_pace(dw)
    for z in games[games.game_id.astype(int).map(gw) == w].itertuples(index=False):
        g = int(z.game_id)
        if (g, z.home_team) not in pts or not (z.home_team in FBS or z.away_team in FBS): continue
        hf = 0 if z.neutral_site else 1
        fh = side_feats(z.home_team, z.away_team, hf, r, FBS, conf25, L, ppp, po, pdf, Ld, dpo, dpd)
        fa = side_feats(z.away_team, z.home_team, -hf, r, FBS, conf25, L, ppp, po, pdf, Ld, dpo, dpd)
        xm, xt, b0 = game_x(fh, fa, hf)
        rows.append(dict(g=g, w=w, xm=xm, xt=xt, b0=b0, am=pts[(g, z.home_team)] - pts[(g, z.away_team)], at=pts[(g, z.home_team)] + pts[(g, z.away_team)]))
D = pd.DataFrame(rows)
Xm = np.vstack(D.xm); Xt = np.vstack(D.xt)
bm, *_ = np.linalg.lstsq(Xm, (D.am - D.b0).values, rcond=None); bt, *_ = np.linalg.lstsq(Xt, D['at'].values, rcond=None)
D['pm'] = D.b0 + Xm @ bm; D['pt'] = Xt @ bt
sigma = float(np.sqrt(((D.pm - D.am) ** 2).mean()))
mae, tmae = float((D.pm - D.am).abs().mean()), float((D.pt - D['at']).abs().mean())
print(f'2025 walk-forward: {len(D)} games; margin MAE {mae:.1f}, RMSE (sigma) {sigma:.1f}; total MAE {tmae:.1f}; winners {((D.pm > 0) == (D.am > 0))[D.am != 0].mean() * 100:.1f}%')
print('margin terms', dict(zip(MCOLS, bm.round(3))), '\ntotal terms', dict(zip(TCOLS, bt.round(3))))
print('margin MAE by week:', D.assign(e=(D.pm - D.am).abs()).groupby('w').e.mean().round(1).to_dict())
json.dump(dict(m=bm.tolist(), t=bt.tolist(), mcols=MCOLS, tcols=TCOLS, sigma=sigma, mae=mae, tmae=tmae, n=len(D)), open('calib.json', 'w'))
