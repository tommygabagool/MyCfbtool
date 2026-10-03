"""Step 10. Game and player projections. For every Power 4 / Notre Dame game this season, projects each team's points and
each player's passing, rushing and receiving line. Completed games from Week 2 on get a true pregame projection: ratings,
usage and player rates are rebuilt from only the weeks before that game, so the page can show projected vs. actual and an
honest accuracy record (against Vegas lines and a season-average baseline). Upcoming games use everything to date.
Game model (common.py): margin and total from EPA-per-play, points-per-drive and success-rate edges, starting from the
2025 calibration (calibrate.py) and updated, walk-forward, with this season's completed games. Player model (pmodel.py),
weights from calibrate_players.py. Writes work/proj.pkl."""
import paths  # noqa: F401  (sets the working directory to work/)
import pandas as pd, numpy as np, json, pickle
from math import erf, sqrt
from common import (season_prior, talent_z, team_ids, points_lookup, drive_table, side_feats, game_x, game_points, rating, LOOSEN,
                    MCOLS, TCOLS)
from pmodel import STAT, PK, FAM, Ctx, box, usage_flags, team_volume, apply

s = pd.read_parquet('plays_skill.parquet'); PL = pd.read_pickle('players.pkl')
T = json.load(open('teams.json')); FBS = set(T['fbs']); conf = T['conf']
P4 = {t for t, c in conf.items() if c in {'ACC', 'Big 12', 'Big Ten', 'SEC'}} | {'Notre Dame'}
tid = team_ids(pd.read_parquet('../data/pbp.parquet', columns=['game_id', 'home_team', 'away_team', 'home_team_id', 'away_team_id']))
prior = season_prior(pickle.load(open('seasons.pkl', 'rb')), pd.read_csv('persistence_conf.csv'), conf=conf,
                     tz=talent_z('../data/talent_2026.parquet', tid, FBS), loosen=LOOSEN)
CAL = json.load(open('calib.json')); PC = json.load(open('calib_players.json'))
N0 = 100                 # the 2025 calibration counts as this many games when updating it with 2026 results

# ---------- passer ids (ESPN athlete id), falling back to the name the play text gave
x = pd.read_parquet('../data/pbp.parquet', columns=['completion_player_id', 'incompletion_player_id', 'interception_thrown_player_id',
                                                    'sack_taken_player_id']).loc[s.index].apply(pd.to_numeric, errors='coerce')
s['qid'] = np.where(s.is_att, x.completion_player_id.fillna(x.incompletion_player_id).fillna(x.interception_thrown_player_id),
                    np.where(s.is_sack, x.sack_taken_player_id, np.nan))
nm = pd.concat([s.loc[s.qid.notna(), ['off', 'qb', 'qid']].rename(columns={'qid': 'id'}),
                s.loc[s.is_rush & s.rus_id.notna() & s.qb.notna(), ['off', 'qb', 'rus_id']].rename(columns={'rus_id': 'id'})]).dropna()
name2id = nm.groupby(['off', 'qb']).id.agg(lambda v: v.value_counts().index[0]).to_dict()
miss = (s.is_att | s.is_sack) & s.qid.isna() & s.qb.notna()
s.loc[miss, 'qid'] = [name2id.get(k) for k in zip(s.loc[miss, 'off'], s.loc[miss, 'qb'])]
print('dropbacks without a passer id:', int(((s.is_att | s.is_sack) & s.qid.isna()).sum()), 'of', int((s.is_att | s.is_sack).sum()))
qname = nm.groupby('id').qb.agg(lambda v: v.value_counts().index[0])
s = usage_flags(s)

# ---------- schedule: dates, scores, home/away, neutral sites
S = pd.read_parquet('../data/schedules_2026.parquet')
S['gid'] = S.game_id.astype(int); S['ts'] = pd.to_datetime(S.start_date, utc=True)
S = S.drop_duplicates('gid').set_index('gid')
PTS = points_lookup(S.reset_index())
s['gid'] = s.game_id.astype(int)
# play-by-play labels postseason games week 1; put them after the regular season so walk-forward windows stay in order
REGMAX = int(S[S.season_type == 'regular'].week.max())
POST = set(S.index[S.season_type != 'regular'])
s.loc[s.gid.isin(POST), 'week'] = REGMAX + 1
wk_of = lambda g: REGMAX + 1 if g in POST else int(S.at[g, 'week'])
TS = S.ts.to_dict()
dr = drive_table(s)
POS = PL.pos.replace({'FB': 'RB'}).to_dict()

# ---------- actual box score per player per game
BX = box(s)
BOX = {}
for z in BX.itertuples(index=False):
    BOX.setdefault(int(z.gid), {})[int(z.pid)] = [round(getattr(z, k)) for k in STAT]
# player -> team: the offense he has the most plays for
pt = pd.concat([s[['qid', 'off']].rename(columns={'qid': 'p'}), s[['rus_id', 'off']].rename(columns={'rus_id': 'p'}),
                s[['rec_id', 'off']].rename(columns={'rec_id': 'p'})]).dropna()
pt['p'] = pt.p.astype(int); PTEAM = pt.groupby('p').off.agg(lambda v: v.value_counts().index[0]).to_dict()

# ---------- which games: every scheduled game with a Power 4 team or Notre Dame
G = S[(S.home_team.isin(P4) | S.away_team.isin(P4))].copy()
G['final'] = G.completed.fillna(False).astype(bool) & G.home_points.notna()
G['done'] = G.final & G.index.isin(set(s.gid))            # played and in play-by-play: graded in the backtest
LAST = int(s.week.max())
print('P4+ND games:', len(G), 'final:', int(G.final.sum()), 'with play-by-play:', int(G.done.sum()), 'upcoming:', int((~G.final).sum()))
class Ctxs(dict):
    def __missing__(self, w): self[w] = c = Ctx(s, dr, FBS, conf, prior, PTS, TS, POS, w); return c
ctx = Ctxs()

def feats(c, home, away, neutral):
    hf = 0 if neutral else 1; L = c.lg
    fh = side_feats(home, away, hf, c.r, FBS, conf, L['plays'], L['ppp'], c.pace_o, c.pace_d, c.Ld, c.dpo, c.dpd)
    fa = side_feats(away, home, -hf, c.r, FBS, conf, L['plays'], L['ppp'], c.pace_o, c.pace_d, c.Ld, c.dpo, c.dpd)
    return fh, fa, hf

# ---------- in-season update of the game model, walk-forward: week w uses only games from earlier weeks
FIN = S[S.completed.fillna(False).astype(bool) & S.home_points.notna() & S.index.isin(set(s.gid))
        & (S.home_team.isin(FBS) | S.away_team.isin(FBS))]
MROWS, bm0, bt0 = [], np.array(CAL['m']), np.array(CAL['t'])
def calib(w):
    rows = [r for r in MROWS if r[0] < w]
    if not rows: return dict(CAL, n26=0)
    def upd(X, y, b0):
        lam = N0 * (X ** 2).mean(0) + 1e-9
        return np.linalg.solve(X.T @ X + np.diag(lam), X.T @ y + lam * b0)
    Xm = np.array([r[1] for r in rows]); ym = np.array([r[2] for r in rows]); Xt = np.array([r[3] for r in rows]); yt = np.array([r[4] for r in rows])
    bm, bt = upd(Xm, ym, bm0), upd(Xt, yt, bt0)
    sig = sqrt((N0 * CAL['sigma'] ** 2 + ((ym - Xm @ bm) ** 2).sum()) / (N0 + len(ym)))
    return dict(CAL, m=bm.tolist(), t=bt.tolist(), sigma=sig, n26=len(ym))
for w in range(2, LAST + 2):
    c = ctx[w]; c.cal = calib(w)
    for g, z in FIN.iterrows():
        if wk_of(g) != w: continue
        fh, fa, hf = feats(c, z.home_team, z.away_team, bool(z.neutral_site))
        xm, xt, b0 = game_x(fh, fa, hf)
        MROWS.append((w, xm, z.home_points - z.away_points - b0, xt, z.home_points + z.away_points))
print('game model by week:', {w: dict(n26=ctx[w].cal['n26'], sigma=round(ctx[w].cal['sigma'], 2), **dict(zip(MCOLS, np.round(ctx[w].cal['m'], 2))))
                              for w in range(2, LAST + 2)})
print('league baselines (latest):', {k: round(float(v), 3) for k, v in ctx[LAST + 1].lg.items()})

def wp(margin, sig): return 0.5 * (1 + erf(margin / (sig * sqrt(2))))
MODEL, PROJ, BT, COMPS = {}, {}, [], {}
for g, z in G.iterrows():
    wk = min(wk_of(g), LAST + 1) if z.final else LAST + 1
    if wk < 2: continue                                       # Week 1: no 2026 games to project from
    c = ctx[wk]
    fh, fa, hf = feats(c, z.home_team, z.away_team, bool(z.neutral_site))
    hp, ap, margin, total = game_points(c.cal, fh, fa, hf)
    MODEL[g] = dict(hp=hp, ap=ap, wp=wp(margin, c.cal['sigma']), hpl=fh['plays'], apl=fa['plays'], asof=wk - 1)
    proj = {}
    for t, o, h, pm in ((z.home_team, z.away_team, hf, margin), (z.away_team, z.home_team, -hf, -margin)):
        C = c.comps(t, o, h)
        if C.empty: continue
        tv = team_volume(c.team_inputs(t, o, h), pm, total, PC)
        MODEL[g].setdefault('tv', {})[t] = [tv[k] for k in ('plays', 'att', 'car', 'tgt')]
        pp = {p: v for p, v in apply(C, tv, PC, pm).items() if v['att'] + v['car'] + v['tgt'] >= 0.5}
        proj[t] = pp
        if wk == LAST + 1:                                    # the latest numbers behind each player's projection (matchup panel)
            for p in pp:
                if p not in COMPS: COMPS[p] = C.loc[p]
        if z.done:
            b = c.base
            for p, v in pp.items():
                act = BOX.get(g, {}).get(p)
                if act is None or sum(act[STAT.index(k)] for k in ('att', 'car', 'tgt')) == 0: continue  # didn't play: no line to grade
                BT.append(dict(g=g, wk=wk, p=p, team=t, p4=t in P4, **{'m_' + k: v[k] for k in STAT},
                               **{'a_' + k: act[i] for i, k in enumerate(STAT)},
                               **{'b_' + k: (b.at[p, k] if p in b.index else np.nan) for k in STAT}))
    PROJ[g] = {p: [round(v[k], 2) for k in STAT] for pp in proj.values() for p, v in pp.items()}

# ---------- backtest: games
bet = pd.read_parquet('../data/betting_2026.parquet'); bet['gid'] = bet.game_id.astype(int); bet = bet.drop_duplicates('gid').set_index('gid')
rows = []
for g, m in MODEL.items():
    z = G.loc[g]
    if not z.done: continue
    rows.append(dict(g=g, wk=m['asof'] + 1, pm=m['hp'] - m['ap'], pt=m['hp'] + m['ap'], am=z.home_points - z.away_points, atot=z.home_points + z.away_points,
                     wp=m['wp'], vm=-bet.at[g, 'home_team_spread'] if g in bet.index and bet.at[g, 'game_spread_available'] == True else np.nan,
                     vt=bet.at[g, 'over_under'] if g in bet.index else np.nan))
GB, PB = pd.DataFrame(rows), pd.DataFrame(BT)
acc = None                       # nothing to grade yet early in a season
if len(GB) and len(PB):
    hasv = GB.vm.notna() & GB.vt.notna()
    fin = lambda v: float(round(v, 1))
    win = ((GB.pm > 0) == (GB.am > 0))[GB.am != 0]
    acc = dict(games=len(GB), wk0=int(GB.wk.min()), wk1=int(GB.wk.max()),
               m_mae=fin((GB.pm - GB.am).abs().mean()), t_mae=fin((GB.pt - GB.atot).abs().mean()), su=fin(100 * win.mean()),
               nv=int(hasv.sum()), mv_mae=fin((GB.pm - GB.am).abs()[hasv].mean()), v_mae=fin((GB.vm - GB.am).abs()[hasv].mean()),
               mvt_mae=fin((GB.pt - GB.atot).abs()[hasv].mean()), vt_mae=fin((GB.vt - GB.atot).abs()[hasv].mean()),
               v_su=fin(100 * ((GB.vm > 0) == (GB.am > 0))[hasv & (GB.am != 0)].mean()), m_su_v=fin(100 * ((GB.pm > 0) == (GB.am > 0))[hasv & (GB.am != 0)].mean()),
               rmse=fin(np.sqrt(((GB.pm - GB.am) ** 2).mean())))
    # calibration: when the model said X% the favorite won how often?
    GB['fav'] = np.maximum(GB.wp, 1 - GB.wp); GB['favwon'] = np.where(GB.wp >= 0.5, GB.am > 0, GB.am < 0)
    bins = [(.5, .65), (.65, .8), (.8, .9), (.9, 1.01)]
    acc['cal'] = [dict(lo=lo, hi=min(hi, 1), n=int(((GB.fav >= lo) & (GB.fav < hi)).sum()),
                       exp=fin(100 * GB.fav[(GB.fav >= lo) & (GB.fav < hi)].mean()), won=fin(100 * GB.favwon[(GB.fav >= lo) & (GB.fav < hi)].mean())) for lo, hi in bins]
    slope = np.polyfit(GB.pm, GB.am, 1)[0]
    print(f"\nGAMES ({len(GB)}, weeks {acc['wk0']}-{acc['wk1']}): margin MAE {acc['m_mae']} (RMSE {acc['rmse']}), total MAE {acc['t_mae']}, winners {acc['su']}%, "
          f"actual-on-projected margin slope {slope:.2f}")
    print(f"  with Vegas lines ({acc['nv']}): model margin MAE {acc['mv_mae']} vs Vegas {acc['v_mae']}; total {acc['mvt_mae']} vs {acc['vt_mae']}; winners {acc['m_su_v']}% vs {acc['v_su']}%")
    print('  calibration', acc['cal'])

    # ---------- backtest: players (Power 4 / ND players who played)
    pl = {}
    for key, vol, k, lbl in (('py', 'att', 15, 'QB passing yds (proj 15+ att)'), ('ry', 'car', 8, 'Rushing yds (proj 8+ carries)'),
                             ('recy', 'tgt', 3, 'Receiving yds (proj 3+ targets)'), ('rec', 'tgt', 3, 'Receptions (proj 3+ targets)'),
                             ('ptd', 'att', 15, 'QB passing TDs (proj 15+ att)'), ('rtd', 'car', 8, 'Rushing TDs (proj 8+ carries)'),
                             ('rectd', 'tgt', 3, 'Receiving TDs (proj 3+ targets)')):
        d = PB[PB.p4 & (PB['m_' + vol] >= k) & PB['b_' + key].notna()]
        r2 = lambda v: float(round(v, 2)) if key.endswith('td') else fin(v)
        pl[key] = dict(l=lbl, n=len(d), m=r2((d['m_' + key] - d['a_' + key]).abs().mean()), b=r2((d['b_' + key] - d['a_' + key]).abs().mean()),
                       r=fin(np.corrcoef(d['m_' + key], d['a_' + key])[0, 1] * 100) if len(d) > 2 else None)
        print(f"  {lbl}: n={len(d)} model MAE {pl[key]['m']} vs season-average MAE {pl[key]['b']}  (r={pl[key]['r']})")
    acc['pl'] = pl
if acc is not None: acc['cal26'] = {k: (round(v, 3) if isinstance(v, float) else v) for k, v in ctx[LAST + 1].cal.items() if k in ('sigma', 'n26', 'mae', 'tmae', 'n')}
cl = ctx[LAST + 1]
DEF = {k: {'dfn': cl.r[k]['dfn'], 'home': cl.r[k]['home']} for k in PK}     # defensive ratings behind player rates, latest fit
pickle.dump(dict(MODEL=MODEL, PROJ=PROJ, BOX=BOX, ACC=acc, PTEAM=PTEAM, QNAME=qname.to_dict(), STAT=STAT, LAST=LAST,
                 CAL=cl.cal, LG=cl.lg, PACE=cl.pace_o, DBR=cl.dbr, R=cl.r, COMPS=COMPS, DEF=DEF, PC=PC, DPACE=(cl.Ld, cl.dpo, cl.dpd)),
            open('proj.pkl', 'wb'))
print('wrote proj.pkl:', len(MODEL), 'games,', sum(len(v) for v in PROJ.values()), 'player projections,', len(COMPS), 'players with matchup inputs')
