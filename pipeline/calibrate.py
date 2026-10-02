"""Step 3b. Calibrate the game model on last season. Walks 2025 forward week by week the way the 2026 projections work:
ratings start from regressed 2024 ratings (conference average and within-conference standing
carried over separately) and only use plays from earlier weeks. Fits how many points one point of
EPA-per-play rating is worth over a game (ridge ratings are shrunk, so it is more than 1), a league-tier term (scrimmage
EPA understates how far apart Power 4, Group of 5 and FCS teams are) and the spread of actual margins around the
projection. Writes work/calib.json."""
import paths  # noqa: F401  (sets the working directory to work/)
import pandas as pd, numpy as np, json, pickle
from common import *
R = pickle.load(open('seasons.pkl', 'rb')); Yc = pd.read_csv('persistence_conf.csv')
prior = season_prior(R, Yc, base=2024, conf=R[2025]['_conf'])
p = pd.read_parquet('../data/pbp_2025.parquet', columns=COLS)
FBS = fbs_teams(p); s = outcomes(p); conf25 = R[2025]['_conf']
sched = pd.read_parquet('../data/schedules_2025.parquet'); sched = sched[sched.season_type == 'regular']
s = s[s.game_id.astype(int).isin(set(sched.game_id.astype(int)))]   # play-by-play labels bowls week 1; regular season only
pts = points_lookup(sched)
gw = s.groupby(s.game_id.astype(int)).week.first()
games = sched[sched.game_id.astype(int).isin(gw.index)]
rows = []
for w in range(2, 15):
    tr = s[s.week < w].copy()
    r = fit(tr, FBS, prior)['epa']
    L, ppp, po, pdf = pace_points(tr, FBS, pts)
    for z in games[games.game_id.astype(int).map(gw) == w].itertuples(index=False):
        g = int(z.game_id)
        if (g, z.home_team) not in pts or not (z.home_team in FBS or z.away_team in FBS): continue
        hf = 0 if z.neutral_site else 1
        for t, o, h in ((z.home_team, z.away_team, hf), (z.away_team, z.home_team, -hf)):
            plays, epa, _ = team_points(t, o, h, r, FBS, L, ppp, po, pdf, dict(scale=1.0), conf25)
            p4, fcs = tiers(t, conf25, FBS)
            rows.append(dict(g=g, w=w, t=t, plays=plays, epa=epa, ppp=ppp, y=pts[(g, t)], p4=p4, fcs=fcs))
D = pd.DataFrame(rows)
# points = plays * (ppp + scale * epa) + tier terms; fit on game margins, which is what spreads and win chances rest on
H = D.groupby('g').nth(0).set_index('g'); A = D.groupby('g').nth(1).set_index('g')
X = np.c_[H.plays * H.epa - A.plays * A.epa, H.p4 - A.p4, H.fcs - A.fcs]; yy = (H.y - A.y) - (H.plays - A.plays) * H.ppp
(scale, p4, fcs), *_ = np.linalg.lstsq(X, yy.values, rcond=None)
scale, p4, fcs = float(scale), float(p4), float(fcs)
D['tier'] = p4 * D.p4 + fcs * D.fcs; D['tier'] -= D.groupby('g').tier.transform('mean')    # split evenly: totals unchanged
D['proj'] = (D.plays * (D.ppp + scale * D.epa) + D.tier).clip(lower=0)
M = D.groupby('g').apply(lambda d: pd.Series(dict(pm=d.proj.iloc[0] - d.proj.iloc[1], am=d.y.iloc[0] - d.y.iloc[1], w=d.w.iloc[0],
                                                  pt=d.proj.sum(), at=d.y.sum())), include_groups=False)
sigma = float(np.sqrt(((M.pm - M.am) ** 2).mean()))
byw = M.assign(e=(M.pm - M.am).abs()).groupby('w').e.mean().round(1).to_dict()
print(f'2025 walk-forward: {len(M)} games; points per EPA/play-rating point scale {scale:.2f}; tier terms P4 {p4:+.1f}, FCS {fcs:+.1f}; margin RMSE (sigma) {sigma:.1f}; '
      f'margin MAE {(M.pm - M.am).abs().mean():.1f}; total MAE {(M.pt - M["at"]).abs().mean():.1f}; slope after scaling {np.polyfit(M.pm, M.am, 1)[0]:.2f}')
print('margin MAE by week:', byw)
json.dump(dict(scale=scale, p4=p4, fcs=fcs, sigma=sigma, n=len(M)), open('calib.json', 'w'))
