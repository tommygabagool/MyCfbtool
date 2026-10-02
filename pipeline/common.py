"""Shared helpers: play outcomes (success, explosives, conversions, garbage time), the metric definitions,
and the ridge-regression fit that rates every offense and defense on each metric."""
import pandas as pd, numpy as np, scipy.sparse as sp
FBSC = {'ACC','American Athletic','Big 12','Big Ten','Conference USA','FBS Independents',
        'Mid-American','Mountain West','Pac-12','SEC','Sun Belt'}
COLS = ['game_id','week','pos_team','def_pos_team','home','away','neutral_site','play_type','play_text','rush','pass','pass_attempt',
        'completion','sack','int','pass_td','rush_td','fumble_vec','yards_gained','completion_yds','rush_yds','down','distance','EPA',
        'pos_score_diff_start','period','penalty_no_play','home_team_conference','away_team_conference','home_team','away_team']

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

def fit(s, FBS, prior=None):
    """Ridge: y = mu + off_team + def_team + FCS-group terms + home. Returns effects per metric.
    prior: {metric: {'off': {team: (mean, lam)}, 'def': {...}}} shrinks toward last season's rating instead of zero."""
    M = metric_defs(s)
    teams = sorted(set(s.off) | set(s.dfn)); ix = {t: i for i, t in enumerate(teams)}; N = len(teams)
    isF = np.array([t not in FBS for t in teams])
    res = {}
    for k, (f, col) in M.items():
        d = s[f & ~s.garbage & s[col].notna()]
        y = d[col].astype(float).values; n = len(d)
        oi = d.off.map(ix).values; di = d.dfn.map(ix).values; rows = np.arange(n)
        X = sp.hstack([sp.csr_matrix((np.ones(n), (rows, oi)), shape=(n, N)), sp.csr_matrix((np.ones(n), (rows, di)), shape=(n, N)),
                       sp.csr_matrix(isF[oi].astype(float)[:, None]), sp.csr_matrix(isF[di].astype(float)[:, None]),
                       sp.csr_matrix(d.homeflag.values.astype(float)[:, None]), sp.csr_matrix(np.ones((n, 1)))]).tocsr()
        s2 = y.var()
        g = d[d.dfn.isin(FBS)].groupby('dfn')[col].agg(['mean', 'size'])
        tau2 = max(g['mean'].var() - (s2 / g['size']).mean(), 0.05 * g['mean'].var())
        lam0 = float(np.clip(s2 / tau2, 20, 5000))
        P = np.r_[np.full(2 * N, lam0), 1.0, 1.0, 1.0, 0.0]; mu = np.zeros(2 * N + 4)
        if prior is not None and k in prior:
            pr = prior[k]
            P[:2 * N] = s2 / pr['tau2_default']                      # teams with no usable history: shrink toward average (FCS toward FCS)
            for side, off in (('off', 0), ('def', N)):
                lam = s2 / pr['tau2_' + side]
                for t, m0 in pr['mu_' + side].items():
                    if t in ix and t in FBS: mu[off + ix[t]] = m0; P[off + ix[t]] = lam
        A = (X.T @ X).toarray() + np.diag(P); b = X.T @ y + P * mu
        beta = np.linalg.solve(A, b)
        oe = beta[:N] + np.where(isF, beta[2 * N], 0.0); de = beta[N:2 * N] + np.where(isF, beta[2 * N + 1], 0.0)
        oe = oe - oe[~isF].mean(); de = de - de[~isF].mean()
        res[k] = dict(off={t: float(oe[ix[t]]) for t in teams}, dfn={t: float(de[ix[t]]) for t in teams},
                      lam0=lam0, s2=float(s2), n=n, home=float(beta[2 * N + 2]), fcs_def=float(beta[2 * N + 1]))
    return res
