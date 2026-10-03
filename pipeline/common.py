"""Shared helpers: play outcomes (success, explosives, conversions, garbage time), the metric definitions, the ridge-
regression fit that rates every offense and defense on each metric (and on points per drive), last season's ratings as
this season's starting point, and the game model (projected margin and total)."""
import pandas as pd, numpy as np, scipy.sparse as sp
FBSC = {'ACC','American Athletic','Big 12','Big Ten','Conference USA','FBS Independents',
        'Mid-American','Mountain West','Pac-12','SEC','Sun Belt'}
COLS = ['game_id','week','pos_team','def_pos_team','home','away','neutral_site','play_type','play_text','rush','pass','pass_attempt',
        'completion','sack','int','pass_td','rush_td','fumble_vec','yards_gained','completion_yds','rush_yds','down','distance','EPA',
        'pos_score_diff_start','period','penalty_no_play','home_team_conference','away_team_conference','home_team','away_team',
        'drive_id','drive_result','home_team_id','away_team_id']
LOOSEN = 4.0   # last season's ratings are held 4x more loosely than their year-to-year spread suggests (2024-2026 replays)

def fbs_teams(p):
    g = p.drop_duplicates('game_id')
    a = set(g.loc[g.home_team_conference.isin(FBSC), 'home_team']) | set(g.loc[g.away_team_conference.isin(FBSC), 'away_team'])
    return a

def outcomes(p):
    # fumbled runs (and a few fumbled catches) come through with rush = pass = 0; put them back in
    _t = p.play_text.fillna('')
    _f = (p.play_type.str.contains('Fumble', na=False) & (p['rush'] != 1) & (p['pass'] != 1)
          & ~_t.str.contains(r'\b(?:punt|kickoff|kick|onside)\b|attempt (?:failed|good)|two-point|2pt', case=False))
    p = p.copy()
    p.loc[_f & _t.str.contains(r'\b(?:rush|run|scramble)\b') & ~_t.str.contains(r'\bpass\b|\bsacked\b'), 'rush'] = 1
    p.loc[_f & _t.str.contains('pass complete'), ['pass', 'pass_attempt', 'completion']] = 1
    s = p[((p['rush'] == 1) | (p['pass'] == 1)) & ~p.play_type.str.startswith('Two Point', na=False)].copy()
    s = s[s.penalty_no_play.fillna(False) != True]
    s['is_sack'] = (s.sack == 1); s['is_att'] = (s.pass_attempt == 1); s['is_rush'] = (s.rush == 1) & ~s.is_sack
    s['kneel'] = s.is_rush & s.play_text.str.contains(r'kneel', case=False, na=False)
    yg = s.yards_gained.fillna(0)
    s['cmp'] = (s.completion == 1) & s.is_att
    cy = s.completion_yds
    bad = cy.isna() | ((cy == -s.yards_gained) & (s.yards_gained < 0)) | s.play_text.str.contains('nullified', case=False, na=False)
    s['pyds'] = np.where(s.cmp, np.where(bad, yg, cy), 0.0)   # completion_yds drops the sign on losses and ignores spot fouls
    s['ryds'] = np.where(s.is_rush, s.rush_yds.fillna(yg), 0.0)
    s['sky'] = np.where(s.is_sack, -yg.clip(upper=0), 0.0)
    s['ptd'] = (s.pass_td == 1) & s.is_att; s['rtd'] = (s.rush_td == 1) & s.is_rush
    s['intc'] = (s['int'] == 1) & s.is_att
    s['xp'] = s.cmp & (s.pyds >= 20); s['xr'] = s.is_rush & (s.ryds >= 10); s['stf'] = s.is_rush & (s.ryds <= 0)
    gain = np.where(s.is_att, s.pyds, np.where(s.is_rush, s.ryds, yg)); td = s.ptd | s.rtd
    thr = np.select([s.down == 1, s.down == 2, s.down >= 3], [0.5 * s.distance, 0.7 * s.distance, s.distance], np.nan)
    to = s.intc | ((s.fumble_vec == 1) & s.play_type.str.contains('Opponent', na=False))
    s['succ'] = (((gain >= thr) | td) & ~to).astype(float); s.loc[s.down.isna() | s.distance.isna(), 'succ'] = np.nan
    s['late'] = s.down >= 3
    s['conv'] = np.where(s.late, (((gain >= s.distance) | td) & ~to).astype(float), np.nan)
    s['epa'] = s.EPA
    m_ = s.pos_score_diff_start.abs()
    s['garbage'] = ((s.period == 1) & (m_ > 43)) | ((s.period == 2) & (m_ > 37)) | ((s.period == 3) & (m_ > 27)) | ((s.period >= 4) & (m_ > 21))
    s['off'] = s.pos_team; s['dfn'] = s.def_pos_team
    s['homeflag'] = np.where(s.neutral_site == True, 0, np.where(s.pos_team == s.home, 1, np.where(s.pos_team == s.away, -1, 0)))
    return s

def metric_defs(s):
    s['db'] = s.is_att | s.is_sack; s['live'] = (s.db | s.is_rush) & ~s.kneel; s['yb'] = s.is_sack.astype(float)
    r = s.is_rush & ~s.kneel; s['stf'] = (s.is_rush & (s.ryds <= 0)).astype(float)
    return {'ypa': (s.is_att, 'pyds'), 'cmp': (s.is_att, 'cmp'), 'ptd': (s.is_att, 'ptd'), 'int': (s.is_att, 'intc'), 'xp': (s.is_att, 'xp'),
            'sk': (s.db, 'yb'), 'dsr': (s.db, 'succ'), 'depa': (s.db, 'epa'),
            'ry': (r, 'ryds'), 'rsr': (r, 'succ'), 'xr': (r, 'xr'), 'rtd': (r, 'rtd'), 'repa': (r, 'epa'),
            'ypc': (s.cmp, 'pyds'), 'conv': (s.live & s.late, 'conv'), 'epa': (s.live, 'epa'), 'sr': (s.live, 'succ'),
            'psr': (s.is_att, 'succ'), 'pepa': (s.is_att, 'epa'), 'stf': (r, 'stf')}

DRIVE_PTS = {'TD': 7.0, 'FG': 3.0, 'END OF HALF TD': 7.0, 'END OF GAME TD': 7.0}
DRIVE_SKIP = {'END OF HALF', 'END OF GAME', 'END OF 4TH QUARTER', 'Uncategorized', 'KICKOFF'}   # kneel-downs and clock-outs say little

def drive_table(s):
    """One row per offensive drive (from scrimmage plays carrying drive_id/drive_result): game, week, offense, defense,
    home flag, the offense's points (7 for a touchdown, 3 for a field goal) and whether it began in garbage time."""
    d = s[s.drive_id.notna()]
    D = d.groupby('drive_id').agg(game_id=('game_id', 'first'), week=('week', 'first'), off=('off', 'first'), dfn=('dfn', 'first'),
                                  homeflag=('homeflag', 'first'), res=('drive_result', 'first'), garbage=('garbage', 'first')).reset_index()
    D['gid'] = D.game_id.astype(int); D['pts'] = D.res.map(DRIVE_PTS).fillna(0.0)
    return D

def _ridge(d, y, FBS, pk=None):
    """Ridge: y = mu + off_team + def_team + FCS-group terms + home, with team terms shrunk toward pk's means (else zero)."""
    teams = sorted(set(d.off) | set(d.dfn)); ix = {t: i for i, t in enumerate(teams)}; N = len(teams)
    isF = np.array([t not in FBS for t in teams])
    n = len(d)
    oi = d.off.map(ix).values; di = d.dfn.map(ix).values; rows = np.arange(n)
    X = sp.hstack([sp.csr_matrix((np.ones(n), (rows, oi)), shape=(n, N)), sp.csr_matrix((np.ones(n), (rows, di)), shape=(n, N)),
                   sp.csr_matrix(isF[oi].astype(float)[:, None]), sp.csr_matrix(isF[di].astype(float)[:, None]),
                   sp.csr_matrix(d.homeflag.values.astype(float)[:, None]), sp.csr_matrix(np.ones((n, 1)))]).tocsr()
    s2 = y.var()
    g = pd.DataFrame(dict(t=d.dfn.values, y=y))[d.dfn.isin(FBS).values].groupby('t').y.agg(['mean', 'size'])
    tau2 = max(g['mean'].var() - (s2 / g['size']).mean(), 0.05 * g['mean'].var())
    lam0 = float(np.clip(s2 / tau2, 1, 5000))
    P = np.r_[np.full(2 * N, lam0), 1.0, 1.0, 1.0, 0.0]; mu = np.zeros(2 * N + 4)
    if pk is not None:
        P[:2 * N] = s2 / pk['tau2_default']                      # teams with no usable history: shrink toward average (FCS toward FCS)
        for side, off in (('off', 0), ('def', N)):
            lam = s2 / pk['tau2_' + side]
            for t, m0 in pk['mu_' + side].items():
                if t in ix and t in FBS: mu[off + ix[t]] = m0; P[off + ix[t]] = lam
    A = (X.T @ X).toarray() + np.diag(P); b = X.T @ y + P * mu
    beta = np.linalg.solve(A, b)
    oe = beta[:N] + np.where(isF, beta[2 * N], 0.0); de = beta[N:2 * N] + np.where(isF, beta[2 * N + 1], 0.0)
    oe = oe - oe[~isF].mean(); de = de - de[~isF].mean()
    return dict(off={t: float(oe[ix[t]]) for t in teams}, dfn={t: float(de[ix[t]]) for t in teams},
                lam0=lam0, s2=float(s2), n=n, home=float(beta[2 * N + 2]), fcs_def=float(beta[2 * N + 1]),
                # rating of an FCS team with no plays yet: the FCS group term, on the same centered scale
                fcs_off_c=float(beta[2 * N] - (beta[:N] + np.where(isF, beta[2 * N], 0.0))[~isF].mean()),
                fcs_def_c=float(beta[2 * N + 1] - (beta[N:2 * N] + np.where(isF, beta[2 * N + 1], 0.0))[~isF].mean()))

def fit(s, FBS, prior=None, keys=None, drives=None):
    """Rates every offense and defense, plus home field, on each play metric (blowout plays left out); with drives (from
    drive_table), also on points per drive ('ppd'). keys limits the play metrics. Returns {metric: effects}.
    prior: {metric: {'mu_off': {team: mean}, 'mu_def': {...}, 'tau2_off', 'tau2_def', 'tau2_default'}} shrinks each team
    toward last season's rating instead of zero."""
    M = metric_defs(s)
    res = {}
    for k, (f, col) in M.items():
        if keys is not None and k not in keys: continue
        d = s[f & ~s.garbage & s[col].notna()]
        res[k] = _ridge(d, d[col].astype(float).values, FBS, prior.get(k) if prior else None)
    if drives is not None:
        dd = drives[~drives.garbage & drives.res.notna() & ~drives.res.isin(DRIVE_SKIP)]
        res['ppd'] = _ridge(dd, dd.pts.values, FBS, prior.get('ppd') if prior else None)
    return res

def talent_z(path, tid, FBS):
    """{team: talent composite as a z-score among FBS teams} from a cfb_team_talent file; tid maps team name -> ESPN team id."""
    T = pd.read_parquet(path); by_id = dict(zip(T.team_id.astype(int), T.talent_composite.astype(float)))
    v = pd.Series({t: by_id[int(i)] for t, i in tid.items() if t in FBS and pd.notna(i) and int(i) in by_id}, dtype=float)
    return ((v - v.mean()) / v.std()).to_dict()

def team_ids(p):
    """{team name: ESPN team id} from play-by-play."""
    g = p.drop_duplicates('game_id')
    return {**dict(zip(g.away_team, g.away_team_id)), **dict(zip(g.home_team, g.home_team_id))}

def season_prior(R, Y, base=2025, conf=None, tz=None, loosen=1.0):
    """Prior for fit(): each FBS team starts from its rating in season `base` scaled by the 2024-to-2025 carryover slope,
    held with the year-to-year residual variance. R comes from seasons.pkl, Y from persistence.csv.
    With conf ({team: conference in the season being rated}) and Y from persistence_conf.csv, the carryover is split in two:
    the conference's average rating carries over at one rate and the team's standing within its conference at another.
    With tz ({team: talent z-score this season}) and an s_tal column in Y, roster talent shifts the starting point too
    (teams new to FBS start from talent alone). loosen multiplies the prior variances (> 1: this season's games count more)."""
    fbs25 = set(R[base]['_fbs']); prior = {}
    for k in [m for m in R[base] if not m.startswith('_')]:
        r25 = R[base][k]; tau2_full = r25['s2'] / r25['lam0']
        pk = {'tau2_default': tau2_full * loosen}
        for side, key in (('off', 'off'), ('def', 'dfn')):
            sel = Y[(Y.metric == k) & (Y.side == key)]
            if not len(sel): continue
            row = sel.iloc[0]
            pk['tau2_' + side] = max(row.resid_sd ** 2, 0.25 * tau2_full) * loosen
            if conf is None:
                pk['mu_' + side] = {t: row.slope * v for t, v in r25[key].items() if t in fbs25}
            else:
                cm = conf_means(r25[key], fbs25, conf)
                st = row.s_tal if (tz is not None and 's_tal' in row and pd.notna(row.s_tal)) else 0.0
                pk['mu_' + side] = {t: row.s_conf * cm[t] + row.s_dev * (v - cm[t]) + st * tz.get(t, 0.0)
                                    for t, v in r25[key].items() if t in fbs25}
                if st and tz is not None:
                    for t, z in tz.items():
                        pk['mu_' + side].setdefault(t, st * z)
        if 'mu_off' in pk and 'mu_def' in pk: prior[k] = pk
    return prior

def conf_means(rating, teams, conf):
    """{team: average rating of the teams in its conference}. Independents (and teams with no conference) get 0, the FBS
    average, so their whole rating counts as standing rather than as a conference level that carries over strongly."""
    grp = {t: conf.get(t) for t in teams if t in rating and conf.get(t) not in (None, 'FBS Independents')}
    s = pd.Series({t: rating[t] for t in grp}, dtype=float); m = s.groupby(pd.Series(grp, dtype=object)).mean()
    return {t: float(m[grp[t]]) if t in grp else 0.0 for t in teams if t in rating}

def pace_points(tr, FBS, pts):
    """From plays tr: league scrimmage plays per team-game and points per play in FBS-vs-FBS games (the ratings are centered
    on FBS, and FCS blowouts would inflate both), plus each team's pace (offensive plays per game run and allowed, shrunk 2
    games toward average). pts: {(game_id, team): points scored}."""
    live = (tr.is_att | tr.is_sack | tr.is_rush) & ~tr.kneel
    gid = tr.game_id.astype(int)
    tg = live.groupby([gid, tr.off, tr.dfn]).sum()
    fb = tg[tg.index.get_level_values(1).isin(FBS) & tg.index.get_level_values(2).isin(FBS)].droplevel(2)
    L = float(fb.mean())
    pp = np.array([(n, pts[k]) for k, n in fb.items() if k in pts], dtype=float)
    ppp = pp[:, 1].sum() / pp[:, 0].sum()
    def shr(by):
        g = live.groupby([tr[by], gid]).sum().groupby(level=0).agg(['mean', 'size'])
        return ((g['mean'] * g['size'] + 2 * L) / (g['size'] + 2)).to_dict()
    return L, ppp, shr('off'), shr('dfn')

def drive_pace(dr):
    """League drives per team-game and each team's drives per game on offense and allowed, shrunk 2 games toward average."""
    L = float(dr.groupby(['gid', 'off']).size().mean())
    def shr(by):
        g = dr.groupby([by, 'gid']).size().groupby(level=0).agg(['mean', 'size'])
        return ((g['mean'] * g['size'] + 2 * L) / (g['size'] + 2)).to_dict()
    return L, shr('off'), shr('dfn')

def rating(r, side, t, FBS):
    """Team t's rating from one metric's fit() result; a team with no plays yet gets the FCS group rating (or average FBS)."""
    v = r[side].get(t)
    if v is not None: return v
    return 0.0 if t in FBS else r['fcs_off_c' if side == 'off' else 'fcs_def_c']

P4C = {'ACC', 'Big 12', 'Big Ten', 'SEC'}
def tiers(t, conf, FBS):
    """(Power 4 or Notre Dame, FCS) indicators for the league-tier term."""
    return float(conf.get(t) in P4C or t == 'Notre Dame'), float(t not in FBS)

# ---------- game model: margin and total from three rating edges, each scaled to the game's expected plays or drives
MCOLS = ['E', 'Pd', 'Sr', 'p4', 'fcs', 'hf']     # margin terms (home minus away; hf = 1 at the home team's field)
TCOLS = ['base', 'E', 'Pd', 'Sr', 'c']           # total terms (both teams summed)

def side_feats(t, o, h, R, FBS, conf, L, ppp, pace_o, pace_d, Ld, dpace_o, dpace_d):
    """Offense t against defense o (h = +1 home, -1 away, 0 neutral): expected plays and drives, and the EPA, points-per-
    drive and success-rate edges in points-like units (edge per play or drive x plays or drives)."""
    plays = pace_o.get(t, L) + pace_d.get(o, L) - L
    drives = dpace_o.get(t, Ld) + dpace_d.get(o, Ld) - Ld
    net = lambda k: rating(R[k], 'off', t, FBS) + rating(R[k], 'dfn', o, FBS) + R[k]['home'] * h
    p4, fcs = tiers(t, conf, FBS)
    return dict(plays=plays, drives=drives, base=plays * ppp, E=plays * net('epa'), Pd=drives * net('ppd'), Sr=plays * net('sr'), p4=p4, fcs=fcs)

def game_x(fh, fa, hf):
    """Margin and total design rows for a game from side_feats() of the home and away offenses."""
    xm = np.array([fh['E'] - fa['E'], fh['Pd'] - fa['Pd'], fh['Sr'] - fa['Sr'], fh['p4'] - fa['p4'], fh['fcs'] - fa['fcs'], float(hf)])
    xt = np.array([fh['base'] + fa['base'], fh['E'] + fa['E'], fh['Pd'] + fa['Pd'], fh['Sr'] + fa['Sr'], 1.0])
    return xm, xt, fh['base'] - fa['base']

def game_points(cal, fh, fa, hf):
    """(home points, away points, margin, total). The base margin (plays x league points per play) enters with weight 1."""
    xm, xt, b0 = game_x(fh, fa, hf)
    margin = b0 + xm @ np.asarray(cal['m']); total = xt @ np.asarray(cal['t'])
    return floor0((total + margin) / 2), floor0((total - margin) / 2), margin, total

def floor0(mu, sd=12.0):
    """Expected points when the raw projection mu is the mean of a team score (sd ~12 points) that can't go below zero.
    Matters only in mismatches, where a plain max(0, mu) would project a shutout."""
    from math import erf, exp, pi, sqrt
    z = mu / sd
    return mu * 0.5 * (1 + erf(z / sqrt(2))) + sd * exp(-z * z / 2) / sqrt(2 * pi)

def points_lookup(sched):
    """{(game_id, team): points} from a cfb_schedules table."""
    out = {}
    for z in sched.itertuples(index=False):
        if pd.notna(z.home_points) and pd.notna(z.away_points):
            out[(int(z.game_id), z.home_team)] = float(z.home_points); out[(int(z.game_id), z.away_team)] = float(z.away_points)
    return out
