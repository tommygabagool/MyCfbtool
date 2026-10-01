"""Step 4. Raw and opponent-adjusted stats for every FBS starting quarterback.
Writes work/plays_adj.parquet and work/qb_all.pkl."""
import paths  # noqa: F401  (sets the working directory to work/)
import pandas as pd, numpy as np, json
s = pd.read_parquet('plays.parquet'); T = json.load(open('teams.json')); conf = T['conf']; FBS = set(T['fbs'])
E = json.load(open('def_effects.json'))
s['db'] = s.is_att | s.is_sack; s['live'] = (s.db | s.is_rush) & ~s.kneel; s['nk'] = s.is_rush & ~s.kneel
for k, v in E.items(): s['e_' + k] = s.dfn.map(v).fillna(0.0)
# per-play defense effect attached to each outcome (zero where the stat doesn't apply)
s['a_py'] = np.where(s.is_att, s.pyds - s.e_ypa, 0.0)
s['a_cmp'] = np.where(s.is_att, s.cmp - s.e_cmp, 0.0)
s['a_ptd'] = np.where(s.is_att, s.ptd - s.e_ptd, 0.0)
s['a_int'] = np.where(s.is_att, s.intc - s.e_int, 0.0)
s['a_xp'] = np.where(s.is_att, s.xp - s.e_xp, 0.0)
s['a_sk'] = np.where(s.db, s.is_sack - s.e_sk, 0.0)
s['a_ry'] = np.where(s.is_rush, s.ryds - np.where(s.nk, s.e_ry, 0.0), 0.0)
s['a_rtd'] = np.where(s.is_rush, s.rtd - np.where(s.nk, s.e_rtd, 0.0), 0.0)
s['a_xr'] = np.where(s.is_rush, s.xr - np.where(s.nk, s.e_xr, 0.0), 0.0)
s['e_succ'] = np.where(s.db, s.e_dsr, s.e_rsr); s['e_epa2'] = np.where(s.db, s.e_depa, s.e_repa)
s.to_parquet('plays_adj.parquet')

# starters (most attempts in team's latest game)
q = s[s.qb.notna()]
lastg = s[s.off.isin(FBS)].groupby('off').week.max()
att = q[q.is_att].groupby(['off', 'week', 'qb']).size().rename('att').reset_index().merge(lastg.rename('lw'), left_on='off', right_index=True)
st = att[att.week == att.lw].sort_values('att', ascending=False).drop_duplicates('off')

def mean_or_nan(x): return float(x.mean()) if len(x) else float('nan')
rows = []
for t, n in st[['off', 'qb']].itertuples(index=False):
    d = q[(q.off == t) & (q.qb == n)]
    a, k, r, lv = d[d.is_att], d[d.db], d[d.is_rush], d[d.live]
    c = a[a.cmp]; lr = r[~r.kneel]; late = lv[lv.late & lv.conv.notna()]
    ls = lv[lv.succ.notna()]; le = lv[lv.epa.notna()]
    gp = d.game_id.nunique()
    R = dict(py=a.pyds.sum(), ptd=a.ptd.sum(), i=a.intc.sum(), cmp=a.cmp.sum(), xp=a.xp.sum(), sk=d.is_sack.sum(),
             ry=r.ryds.sum(), rtd=r.rtd.sum(), xr=r.xr.sum(),
             sr=mean_or_nan(ls.succ), epa=mean_or_nan(le.epa), ypc=mean_or_nan(c.pyds), conv=mean_or_nan(late.conv),
             dsr=mean_or_nan(ls[ls.db].succ), rsr=mean_or_nan(ls[~ls.db].succ))
    A = dict(py=a.a_py.sum(), ptd=a.a_ptd.sum(), i=a.a_int.sum(), cmp=a.a_cmp.sum(), xp=a.a_xp.sum(), sk=k.a_sk.sum(),
             ry=r.a_ry.sum(), rtd=r.a_rtd.sum(), xr=r.a_xr.sum(),
             sr=mean_or_nan(ls.succ - ls.e_succ), epa=mean_or_nan(le.epa - le.e_epa2), ypc=mean_or_nan(c.pyds - c.e_ypc),
             conv=mean_or_nan(late.conv - late.e_conv), dsr=mean_or_nan(ls[ls.db].succ - ls[ls.db].e_dsr),
             rsr=mean_or_nan(ls[~ls.db].succ - ls[~ls.db].e_rsr))
    base = dict(team=t, qb=n, conf=conf.get(t), gp=gp, att=len(a), db=len(k), car=len(r), carl=len(lr), live=len(lv), comp=len(c),
                latep=len(late), sky=float(k.sky.sum()), sos=mean_or_nan(le.e_epa2))
    rows.append({**base, **{'r_' + x: float(v) for x, v in R.items()}, **{'a_' + x: float(v) for x, v in A.items()}})
Q = pd.DataFrame(rows)
Q.to_pickle('qb_all.pkl')
P = Q[Q.conf.isin(['ACC', 'Big 12', 'Big Ten', 'SEC']) | (Q.team == 'Notre Dame')].copy()
pd.set_option('display.width', 250)
P['d_py'] = P.a_py - P.r_py; P['d_epa'] = P.a_epa - P.r_epa
print(P.sort_values('sos')[['qb', 'team', 'gp', 'sos', 'r_epa', 'a_epa', 'r_py', 'a_py', 'r_ptd', 'a_ptd', 'r_ry', 'a_ry', 'r_sr', 'a_sr']].round(3).head(10).to_string(index=False))
print('...')
print(P.sort_values('sos')[['qb', 'team', 'gp', 'sos', 'r_epa', 'a_epa', 'r_py', 'a_py', 'r_ptd', 'a_ptd', 'r_ry', 'a_ry', 'r_sr', 'a_sr']].round(3).tail(10).to_string(index=False))
print('\nadjust magnitudes: EPA/play', P.d_epa.describe().round(3).to_dict(), '\npass yds', P.d_py.describe().round(0).to_dict())
