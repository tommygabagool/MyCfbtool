"""Step 10. Game and player projections. For every Power 4 / Notre Dame game this season, projects each team's points and
each player's passing, rushing and receiving line. Completed games from Week 2 on get a true pregame projection: ratings,
usage and player rates are rebuilt from only the weeks before that game, so the page can show projected vs. actual and an
honest accuracy record (against Vegas lines and a season-average baseline). Upcoming games use everything to date.
The points-per-EPA scale, league-tier terms and margin spread start from the 2025 calibration (calibrate.py) and are
updated, walk-forward, from this season's completed games: the source EPA shifts from season to season (its expected-
points model leans on the pregame spread), so last season's constants alone understate this season's margins.
Writes work/proj.pkl."""
import paths  # noqa: F401  (sets the working directory to work/)
import pandas as pd, numpy as np, json, pickle
from math import erf, sqrt
from common import fit, season_prior, pace_points, points_lookup, rating, team_points, tiers

s = pd.read_parquet('plays_skill.parquet'); PL = pd.read_pickle('players.pkl')
T = json.load(open('teams.json')); FBS = set(T['fbs']); conf = T['conf']
P4 = {t for t, c in conf.items() if c in {'ACC', 'Big 12', 'Big Ten', 'SEC'}} | {'Notre Dame'}
prior = season_prior(pickle.load(open('seasons.pkl', 'rb')), pd.read_csv('persistence_conf.csv'), conf=conf)
CAL = json.load(open('calib.json'))
STAT = ['att', 'cmp', 'py', 'ptd', 'int', 'sk', 'car', 'ry', 'rtd', 'tgt', 'rec', 'recy', 'rectd']
HALF_LIFE = 2.0          # usage weights halve every 2 team games back
K = dict(cmp=60, ypa=100, ptd=150, int=250, ry=50, rtd=120, rcmp=25, rypt=40, rtdt=80)   # shrinkage, in plays of evidence
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

s['db'] = s.is_att | s.is_sack; s['car'] = s.is_rush & ~s.kneel; s['live'] = s.db | s.car
s['tgt'] = s.is_att & s.rec_id.notna()

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

# ---------- actual box score per player per game
def box(d):
    """d: plays for one offense in one game -> {pid: [STAT...]}"""
    out = {}
    def add(pid, k, v):
        if pd.isna(pid): return
        out.setdefault(int(pid), dict.fromkeys(STAT, 0.0))[k] += float(v)
    for r in d.itertuples(index=False):
        if r.is_att:
            add(r.qid, 'att', 1); add(r.qid, 'cmp', r.cmp); add(r.qid, 'py', r.pyds); add(r.qid, 'ptd', r.ptd); add(r.qid, 'int', r.intc)
            if r.tgt:
                add(r.rec_id, 'tgt', 1); add(r.rec_id, 'rec', r.cmp); add(r.rec_id, 'recy', r.pyds); add(r.rec_id, 'rectd', r.ptd)
        elif r.is_sack: add(r.qid, 'sk', 1)
        elif r.car: add(r.rus_id, 'car', 1); add(r.rus_id, 'ry', r.ryds); add(r.rus_id, 'rtd', r.rtd)
    return {p: [v[k] for k in STAT] for p, v in out.items()}
BOX = {}
for (g, t), d in s.groupby(['gid', 'off']):
    BOX.setdefault(g, {}).update({p: [round(v) for v in arr] for p, arr in box(d).items()})
# player -> team: the offense he has the most plays for
pt = pd.concat([s[['qid', 'off']].rename(columns={'qid': 'p'}), s[['rus_id', 'off']].rename(columns={'rus_id': 'p'}),
                s[['rec_id', 'off']].rename(columns={'rec_id': 'p'})]).dropna()
pt['p'] = pt.p.astype(int); PTEAM = pt.groupby('p').off.agg(lambda v: v.value_counts().index[0]).to_dict()

# ---------- long play-level usage tables (one row per player-opportunity)
def usage(tr):
    """Per player per game: attempts, carries, targets; and per team per game: the same totals."""
    a = tr[tr.is_att & tr.qid.notna()].groupby(['gid', 'off', 'qid']).size().rename('att')
    c = tr[tr.car & tr.rus_id.notna()].groupby(['gid', 'off', 'rus_id']).size().rename('car')
    t = tr[tr.tgt].groupby(['gid', 'off', 'rec_id']).size().rename('tgt')
    for z in (a, c, t): z.index = z.index.set_names(['gid', 'off', 'pid'])
    U = pd.concat([a, c, t], axis=1).fillna(0.0).reset_index(); U['pid'] = U.pid.astype(int)
    TG = tr.groupby(['gid', 'off']).agg(att=('is_att', 'sum'), car=('car', 'sum'), tgt=('tgt', 'sum')).reset_index()
    return U, TG

def ratings(w):
    """Ratings fit from plays before week w (w >= 2); w beyond the last week uses everything."""
    r = fit(s[s.week < w], FBS, prior)
    out = {k: {f: v[f] for f in ('off', 'dfn', 'home', 'fcs_off_c', 'fcs_def_c')} for k, v in r.items()}
    for k, v in out.items():           # an FBS team with no plays yet starts from its preseason prior, not from average
        for side, ps in (('off', 'mu_off'), ('dfn', 'mu_def')):
            for t in FBS - set(v[side]): v[side][t] = prior[k][ps].get(t, 0.0)
    return out

class Ctx:
    """Everything the projections need, built from the plays before week w only."""
    def __init__(self, w):
        tr = s[s.week < w].copy(); self.w = w; self.r = r = ratings(w)
        for k in ['cmp', 'ypa', 'ptd', 'int', 'ry', 'rtd', 'sk']:
            tr['e_' + k] = tr.dfn.map(r[k]['dfn']).fillna(0.0)
        self.U, self.TG = usage(tr)
        ng = tr[~tr.garbage]; fo = ng[ng.off.isin(FBS) & ng.dfn.isin(FBS)]
        # league baselines (FBS vs FBS, non-garbage): the fits are centered on these
        self.lg = dict(db=fo.db.sum() / fo.live.sum(), sk=fo.is_sack.sum() / fo.db.sum(), tgt=fo.tgt.sum() / max(1, fo.is_att.sum()),
                       cmp=fo[fo.is_att].cmp.mean(), ypa=fo[fo.is_att].pyds.mean(), ptd=fo[fo.is_att].ptd.mean(), int=fo[fo.is_att].intc.mean(),
                       ry=fo[fo.car].ryds.mean(), rtd=fo[fo.car].rtd.mean())
        self.lg['plays'], self.lg['ppp'], self.pace_o, self.pace_d = pace_points(tr, FBS, PTS)
        # team dropback rate and credited-target rate, non-garbage, shrunk 120 plays / 80 attempts toward average
        tm = ng.groupby('off').agg(db=('db', 'sum'), live=('live', 'sum'), att=('is_att', 'sum'), tgt=('tgt', 'sum'))
        self.dbr = ((tm.db + 120 * self.lg['db']) / (tm.live + 120)).to_dict()
        self.tgr = ((tm.tgt + 80 * self.lg['tgt']) / (tm.att + 80)).to_dict()
        # per-player opponent-adjusted rates (raw minus what that defense typically allows), with shrinkage targets
        A = tr[tr.is_att & tr.qid.notna()].assign(pid=lambda d: d.qid.astype(int))
        C = tr[tr.car & tr.rus_id.notna()].assign(pid=lambda d: d.rus_id.astype(int))
        G = tr[tr.tgt].assign(pid=lambda d: d.rec_id.astype(int))
        A = A.assign(cmp_a=A.cmp - A.e_cmp, ypa_a=A.pyds - A.e_ypa, ptd_a=A.ptd - A.e_ptd, int_a=A.intc - A.e_int)
        C = C.assign(ry_a=C.ryds - C.e_ry, rtd_a=C.rtd - C.e_rtd)
        G = G.assign(rcmp_a=G.cmp - G.e_cmp, rypt_a=G.pyds - G.e_ypa, rtdt_a=G.ptd - G.e_ptd)
        pos = lambda d: d.pid.map(PL.pos).replace({'FB': 'RB'}).where(lambda v: v.isin(['QB', 'RB', 'WR', 'TE']), 'WR')
        C['grp'] = np.where(C.pid.isin(set(A.pid)), 'QB', pos(C).where(lambda v: v != 'QB', 'RB'))
        G['grp'] = pos(G).where(lambda v: v.isin(['WR', 'TE', 'RB']), 'WR')
        def rates(D, cols, by=None):
            fb = D[D.off.isin(FBS)]
            avg = fb.groupby(by)[cols].mean() if by else fb[cols].mean()
            agg = D.groupby('pid').agg(n=(cols[0], 'size'), **{c: (c, 'sum') for c in cols})
            if by: agg['grp'] = D.groupby('pid')[by].agg(lambda v: v.value_counts().index[0])
            return agg, avg
        self.Ar, self.Aavg = rates(A, ['cmp_a', 'ypa_a', 'ptd_a', 'int_a'])
        self.Cr, self.Cavg = rates(C, ['ry_a', 'rtd_a'], 'grp')
        self.Gr, self.Gavg = rates(G, ['rcmp_a', 'rypt_a', 'rtdt_a'], 'grp')
        T0 = pd.Timestamp(0, tz='UTC')
        self.tgames = {t: sorted(gs, key=lambda g: TS.get(g, T0)) for t, gs in tr.groupby('off').gid.unique().items()}
        self.cal = CAL
        self.base = self.baseline(tr)

    def shr(self, tab, avg, pid, col, k, default_grp=None):
        a = avg[col] if not isinstance(avg, pd.DataFrame) else avg.loc[tab.at[pid, 'grp'] if pid in tab.index else default_grp, col]
        if pid not in tab.index: return float(a)
        n = tab.at[pid, 'n']; return float((tab.at[pid, col] + k * a) / (n + k))

    def eff(self, k, t, o, hf, side='dfn'):
        """Rating offset for metric k: opponent o's defense rating plus home edge (plus team t's offense rating if side='both')."""
        r = self.r[k]; return rating(r, 'off', t, FBS) * (side != 'dfn') + rating(r, 'dfn', o, FBS) + r['home'] * hf

    def shares(self, t):
        """Recency-weighted share of team attempts, carries and targets for every player who has played for team t."""
        gs = self.tgames.get(t, [])
        if not gs: return pd.DataFrame(columns=['att', 'car', 'tgt'])
        w = pd.Series([0.5 ** ((len(gs) - 1 - i) / HALF_LIFE) for i in range(len(gs))], index=gs)
        U = self.U[(self.U.off == t) & self.U.gid.isin(gs)].copy(); U['w'] = U.gid.map(w)
        TG = self.TG[(self.TG.off == t) & self.TG.gid.isin(gs)].copy(); TG['w'] = TG.gid.map(w)
        num = U[['att', 'car', 'tgt']].mul(U.w, axis=0).groupby(U.pid).sum()
        den = TG[['att', 'car', 'tgt']].mul(TG.w, axis=0).sum().replace(0, np.nan)
        return (num / den).fillna(0.0)

    def team(self, t, o, hf):
        """Projected team volume and points for offense t against defense o."""
        L = self.lg
        plays, epa, pts = team_points(t, o, hf, self.r['epa'], FBS, L['plays'], L['ppp'], self.pace_o, self.pace_d, self.cal, conf)
        dbr = float(np.clip(self.dbr.get(t, L['db']), .3, .75))
        sk = float(np.clip(L['sk'] + self.eff('sk', t, o, hf, 'both'), .01, .2))
        db = plays * dbr; att = db * (1 - sk)
        return dict(plays=plays, pts=pts, epa=epa, db=db, sk=db * sk, att=att, car=plays - db, tgt=att * self.tgr.get(t, L['tgt']))

    def players(self, t, o, hf, tm):
        """{pid: [STAT...]} projected for offense t vs defense o, given the team projection tm."""
        sh = self.shares(t); L = self.lg; out = {}
        if sh.empty: return out
        e = {k: self.eff(k, t, o, hf) for k in ['cmp', 'ypa', 'ptd', 'int', 'ry', 'rtd']}
        cl = lambda v, lo, hi: float(np.clip(v, lo, hi))
        for p, r in sh.iterrows():
            v = dict.fromkeys(STAT, 0.0)
            if r.att > 0:
                a = tm['att'] * r.att; v['att'] = a
                v['cmp'] = a * cl(self.shr(self.Ar, self.Aavg, p, 'cmp_a', K['cmp']) + e['cmp'], .3, .85)
                v['py'] = a * cl(self.shr(self.Ar, self.Aavg, p, 'ypa_a', K['ypa']) + e['ypa'], 3, 13)
                v['ptd'] = a * cl(self.shr(self.Ar, self.Aavg, p, 'ptd_a', K['ptd']) + e['ptd'], 0, .15)
                v['int'] = a * cl(self.shr(self.Ar, self.Aavg, p, 'int_a', K['int']) + e['int'], .005, .08)
                v['sk'] = tm['sk'] * r.att
            if r.car > 0:
                c = tm['car'] * r.car; v['car'] = c
                v['ry'] = c * cl(self.shr(self.Cr, self.Cavg, p, 'ry_a', K['ry'], 'RB') + e['ry'], -1, 12)
                v['rtd'] = c * cl(self.shr(self.Cr, self.Cavg, p, 'rtd_a', K['rtd'], 'RB') + e['rtd'], 0, .15)
            if r.tgt > 0:
                g = tm['tgt'] * r.tgt; v['tgt'] = g
                v['rec'] = g * cl(self.shr(self.Gr, self.Gavg, p, 'rcmp_a', K['rcmp'], 'WR') + e['cmp'], .25, .95)
                v['recy'] = g * cl(self.shr(self.Gr, self.Gavg, p, 'rypt_a', K['rypt'], 'WR') + e['ypa'], 1, 18)
                v['rectd'] = g * cl(self.shr(self.Gr, self.Gavg, p, 'rtdt_a', K['rtdt'], 'WR') + e['ptd'], 0, .2)
            out[int(p)] = v
        # receivers and passers describe the same throws: meet in the middle so the targets' catches, yards and TDs add up to the QBs'
        for q, rk in (('cmp', 'rec'), ('py', 'recy'), ('ptd', 'rectd')):
            Q = sum(v[q] for v in out.values()); Rr = sum(v[rk] for v in out.values())
            if Q > 0 and Rr > 0:
                m = (Q + Rr) / 2
                for v in out.values(): v[q] *= m / Q; v[rk] *= m / Rr
        return {p: v for p, v in out.items() if v['att'] + v['car'] + v['tgt'] >= 0.5}

    def baseline(self, tr):
        """Naive comparison: each player's season-to-date average per game played."""
        B = {}
        for (g, t), d in tr.groupby(['gid', 'off']):
            for p, arr in box(d).items(): B.setdefault(p, []).append(arr)
        return {p: np.mean(v, axis=0) for p, v in B.items()}

    def game(self, home, away, neutral):
        hf = 0 if neutral else 1
        th, ta = self.team(home, away, hf), self.team(away, home, -hf)
        return th, ta, self.players(home, away, hf, th), self.players(away, home, -hf, ta)

# ---------- which games: every scheduled game with a Power 4 team or Notre Dame
G = S[(S.home_team.isin(P4) | S.away_team.isin(P4))].copy()
G['final'] = G.completed.fillna(False).astype(bool) & G.home_points.notna()
G['done'] = G.final & G.index.isin(set(s.gid))            # played and in play-by-play: graded in the backtest
LAST = int(s.week.max())
print('P4+ND games:', len(G), 'final:', int(G.final.sum()), 'with play-by-play:', int(G.done.sum()), 'upcoming:', int((~G.final).sum()))
class Ctxs(dict):
    def __missing__(self, w): self[w] = c = Ctx(w); return c
ctx = Ctxs()

# ---------- in-season update of the game constants, walk-forward: week w uses only games from earlier weeks
def feats(c, home, away, neutral):
    hf = 0 if neutral else 1; x = {}
    for t, o, h in ((home, away, hf), (away, home, -hf)):
        plays, epa, _ = team_points(t, o, h, c.r['epa'], FBS, c.lg['plays'], c.lg['ppp'], c.pace_o, c.pace_d, dict(scale=1.0), conf)
        x[t] = (plays, epa) + tiers(t, conf, FBS)
    (ph, eh, p4h, fh), (pa, ea, p4a, fa) = x[home], x[away]
    return [ph * eh - pa * ea, p4h - p4a, fh - fa], (ph - pa) * c.lg['ppp']
CROWS, b0 = [], np.array([CAL['scale'], CAL['p4'], CAL['fcs']])
def calib(w):
    rows = [r for r in CROWS if r[0] < w]
    if not rows: return dict(CAL, n26=0)
    X = np.array([r[1] for r in rows]); y = np.array([r[2] for r in rows])
    lam = N0 * (X ** 2).mean(0) + 1e-9
    b = np.linalg.solve(X.T @ X + np.diag(lam), X.T @ y + lam * b0)
    sig = sqrt((N0 * CAL['sigma'] ** 2 + ((y - X @ b) ** 2).sum()) / (N0 + len(y)))
    return dict(scale=float(b[0]), p4=float(b[1]), fcs=float(b[2]), sigma=sig, n26=len(y))
FIN = S[S.completed.fillna(False).astype(bool) & S.home_points.notna() & S.index.isin(set(s.gid))
        & (S.home_team.isin(FBS) | S.away_team.isin(FBS))]
for w in range(2, LAST + 2):
    c = ctx[w]; c.cal = calib(w)
    for g, z in FIN.iterrows():
        if wk_of(g) != w: continue
        x, off = feats(c, z.home_team, z.away_team, bool(z.neutral_site))
        CROWS.append((w, x, z.home_points - z.away_points - off))
print('game constants by week:', {w: {k: round(v, 2) for k, v in ctx[w].cal.items() if k != 'n'} for w in range(2, LAST + 2)})
print('league baselines (latest):', {k: round(float(v), 3) for k, v in ctx[LAST + 1].lg.items()})

def wp(margin, sig): return 0.5 * (1 + erf(margin / (sig * sqrt(2))))
MODEL, PROJ, BT = {}, {}, []
for g, z in G.iterrows():
    wk = min(wk_of(g), LAST + 1) if z.final else LAST + 1
    if wk < 2: continue                                       # Week 1: no 2026 games to project from
    c = ctx[wk]
    neutral = bool(z.neutral_site)
    th, ta, ph, pa = c.game(z.home_team, z.away_team, neutral)
    MODEL[g] = dict(hp=th['pts'], ap=ta['pts'], wp=wp(th['pts'] - ta['pts'], c.cal['sigma']), hpl=th['plays'], apl=ta['plays'], asof=wk - 1)
    PROJ[g] = {p: [round(v[k], 2) for k in STAT] for p, v in {**ph, **pa}.items()}
    if z.done:
        for side, team, pp in (('h', z.home_team, ph), ('a', z.away_team, pa)):
            for p, v in pp.items():
                act = BOX.get(g, {}).get(p)
                if act is None or sum(act[STAT.index(k)] for k in ('att', 'car', 'tgt')) == 0: continue  # didn't play: no line to grade
                b = c.base.get(p)
                BT.append(dict(g=g, wk=wk, p=p, team=team, p4=team in P4, **{'m_' + k: v[k] for k in STAT},
                               **{'a_' + k: act[i] for i, k in enumerate(STAT)}, **{'b_' + k: (b[i] if b is not None else np.nan) for i, k in enumerate(STAT)}))

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
                             ('recy', 'tgt', 3, 'Receiving yds (proj 3+ targets)'), ('rec', 'tgt', 3, 'Receptions (proj 3+ targets)')):
        d = PB[PB.p4 & (PB['m_' + vol] >= k) & PB['b_' + key].notna()]
        pl[key] = dict(l=lbl, n=len(d), m=fin((d['m_' + key] - d['a_' + key]).abs().mean()), b=fin((d['b_' + key] - d['a_' + key]).abs().mean()),
                       r=fin(np.corrcoef(d['m_' + key], d['a_' + key])[0, 1] * 100) if len(d) > 2 else None)
        print(f"  {lbl}: n={len(d)} model MAE {pl[key]['m']} vs season-average MAE {pl[key]['b']}  (r={pl[key]['r']})")
    acc['pl'] = pl
if acc is not None: acc['cal26'] = {k: round(v, 3) for k, v in ctx[LAST + 1].cal.items()}
pickle.dump(dict(MODEL=MODEL, PROJ=PROJ, BOX=BOX, ACC=acc, PTEAM=PTEAM, QNAME=qname.to_dict(), STAT=STAT, LAST=LAST,
                 CAL=ctx[LAST + 1].cal, LG=ctx[LAST + 1].lg, PACE=ctx[LAST + 1].pace_o, DBR=ctx[LAST + 1].dbr, R=ctx[LAST + 1].r), open('proj.pkl', 'wb'))
print('wrote proj.pkl:', len(MODEL), 'games,', sum(len(v) for v in PROJ.values()), 'player projections')
