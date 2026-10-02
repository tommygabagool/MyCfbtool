"""Step 8. Running backs, wide receivers and tight ends: raw + opponent-adjusted stats, archetypes and game
logs, merged with the quarterbacks plus page metadata. Writes work/all_data.json."""
import paths  # noqa: F401  (sets the working directory to work/)
import pandas as pd, numpy as np, json
s = pd.read_parquet('plays_skill.parquet'); PL = pd.read_pickle('players.pkl')
T = json.load(open('teams.json')); FBS = set(T['fbs']); conf = T['conf']; E = json.load(open('def_effects.json'))
inP = lambda t: conf.get(t) in {'ACC', 'Big 12', 'Big Ten', 'SEC'} or t == 'Notre Dame'
for k in ['ry', 'rsr', 'xr', 'rtd', 'repa', 'stf', 'cmp', 'ypa', 'ptd', 'xp', 'psr', 'pepa', 'ypc']:
    s['e_' + k] = s.dfn.map(E[k]).fillna(0.0)
s['stf'] = s['stf'].astype(float); s['rz'] = s.ytg <= 20
raw = pd.read_parquet('../data/pbp.parquet', columns=['game_id', 'start_date'])
gd = pd.to_datetime(raw.dropna().drop_duplicates('game_id').set_index('game_id').start_date, utc=True).dt.tz_convert('America/New_York')
C = s[s.is_rush & ~s.kneel & s.rus_id.notna()].copy(); C['pid'] = C.rus_id.astype(int)
G = s[s.is_att & s.rec_id.notna()].copy(); G['pid'] = G.rec_id.astype(int)
nm = lambda x: x.mean() if x.notna().any() else np.nan
# ---------- per-player aggregates
ru = C.groupby('pid').agg(car=('ryds', 'size'), ry=('ryds', 'sum'), rtd=('rtd', 'sum'), xr=('xr', 'sum'), stf=('stf', 'mean'), sr=('succ', nm), epa=('epa', nm),
                          rzc=('rz', 'sum'), sos_r=('e_repa', 'mean'))
C['a_ry'] = C.ryds - C.e_ry; C['a_rtd'] = C.rtd - C.e_rtd; C['a_xr'] = C.xr - C.e_xr; C['a_stf'] = C.stf - C.e_stf
C['a_sr'] = C.succ - C.e_rsr; C['a_epa'] = C.epa - C.e_repa
ru = ru.join(C.groupby('pid').agg(j_ry=('a_ry', 'sum'), j_rtd=('a_rtd', 'sum'), j_xr=('a_xr', 'sum'), j_stf=('a_stf', 'mean'), j_sr=('a_sr', nm), j_epa=('a_epa', nm)))
rc = G.groupby('pid').agg(tgt=('cmp', 'size'), rec=('cmp', 'sum'), recy=('pyds', 'sum'), rectd=('ptd', 'sum'), x20=('xp', 'sum'), tsr=('succ', nm), tepa=('epa', nm),
                          rzt=('rz', 'sum'), t3=('late', 'sum'), sos_p=('e_pepa', 'mean'))
G['a_rec'] = G.cmp - G.e_cmp; G['a_recy'] = G.pyds - G.e_ypa; G['a_rectd'] = G.ptd - G.e_ptd; G['a_x20'] = G.xp - G.e_xp
G['a_tsr'] = G.succ - G.e_psr; G['a_tepa'] = G.epa - G.e_pepa; G['a_ypr'] = np.where(G.cmp, G.pyds - G.e_ypc, np.nan)
rc = rc.join(G.groupby('pid').agg(j_rec=('a_rec', 'sum'), j_recy=('a_recy', 'sum'), j_rectd=('a_rectd', 'sum'), j_x20=('a_x20', 'sum'), j_tsr=('a_tsr', nm),
                                  j_tepa=('a_tepa', nm), j_ypr=('a_ypr', nm)))
A = ru.join(rc, how='outer')
for c in ['car', 'ry', 'rtd', 'xr', 'rzc', 'tgt', 'rec', 'recy', 'rectd', 'x20', 'rzt', 't3', 'j_ry', 'j_rtd', 'j_xr', 'j_rec', 'j_recy', 'j_rectd', 'j_x20']:
    A[c] = A[c].fillna(0.0)
both = pd.concat([C[['pid', 'off', 'game_id']], G[['pid', 'off', 'game_id']]])
A['team'] = both.groupby('pid').off.agg(lambda v: v.value_counts().index[0])
games = both.groupby('pid').game_id.apply(set)
A['gp'] = games.map(len)
# team denominators over the games each player appeared in
def tg(mask): return s[mask].groupby(['off', 'game_id']).size()
DEN = {'ta': tg(s.is_att), 'tr': tg(s.is_rush & ~s.kneel), 'trzr': tg(s.is_rush & ~s.kneel & s.rz), 'trza': tg(s.is_att & s.rz), 't3a': tg(s.is_att & s.late)}
for k, ser in DEN.items():
    d = ser.to_dict(); A[k] = [sum(d.get((t, g), 0) for g in games[p]) for p, t in zip(A.index, A.team)]
A['pos'] = PL.pos.reindex(A.index)
amb = A.pos.isin(['ATH', '?'])
A.loc[amb, 'pos'] = np.where(A.loc[amb, 'car'] > A.loc[amb, 'tgt'], 'RB', 'WR')
A.loc[A.pos == 'FB', 'pos'] = 'RB'
A['name'] = PL.name.reindex(A.index); A['fbs'] = A.team.isin(FBS); A['p4'] = A.team.map(inP)
tgames = s.groupby('off').game_id.nunique(); A['tgm'] = A.team.map(tgames)

def shr(rate, n, k):
    rate = pd.Series(rate, index=n.index, dtype=float); m = np.nansum(rate * n) / n[rate.notna()].sum()
    r = rate.fillna(m); return (r * n + m * k) / (n + k)
def label(r):
    o = r.sort_values(ascending=False); a, b = o.index[0], o.index[1]
    if o.iloc[0] <= 0.2: return a, f'{a} (weak fit)'
    if o.iloc[1] >= o.iloc[0] - 0.25: return a, f'{a} / {b}'
    return a, a
def score(X, pop, feats, recipe):
    F = pd.DataFrame({k: f(X) for k, f in feats.items()}, index=X.index)
    mu, sd = F[pop].mean(), F[pop].std()
    Z = ((F - mu) / sd).clip(-3, 3)
    S = pd.DataFrame({name: sum(sign * Z[c] for c, sign in parts) / len(parts) for name, parts in recipe.items()})
    L = S.apply(label, axis=1, result_type='expand'); L.columns = ['a', 'lab']
    return S, L

# ---------- running backs
R = A[(A.pos == 'RB') & A.fbs].copy()
R['dp'] = R.groupby('team').car.rank(ascending=False, method='first').astype(int)
popR = R.car >= 20
featR = {'cpg': lambda X: X.car / X.gp, 'csh': lambda X: X.car / X.tr, 'tpg_all': lambda X: (X.car + X.rec) / X.gp,
         'xrt': lambda X: shr(X.j_xr / X.car, X.car, 40), 'ypc': lambda X: shr(X.j_ry / X.car, X.car, 40),
         'rsr': lambda X: shr(X.j_sr, X.car, 40), 'stf': lambda X: shr(X.j_stf, X.car, 40), 'rzs': lambda X: shr(X.rzc / X.trzr.replace(0, np.nan), X.trzr, 10),
         'tpg': lambda X: X.tgt / X.gp, 'tsh': lambda X: X.tgt / X.ta, 'rypg': lambda X: X.j_recy / X.gp}
ARCH_R = {'Workhorse': [('cpg', 1), ('csh', 1), ('tpg_all', 1)], 'Home-Run Hitter': [('xrt', 1), ('ypc', 1)],
          'Receiving Back': [('tpg', 1), ('tsh', 1), ('rypg', 1)], 'Grinder': [('rsr', 1), ('stf', -1), ('rzs', 1)]}
SR, LR = score(R, popR, featR, ARCH_R); R = R.join(SR).join(LR)
showR = R.p4 & ((R.dp == 1) | (R.car >= 6 * R.tgm))
# ---------- receivers (WR and TE share the recipe, scored within position)
def recv(pos, minpop, show_rule):
    X = A[(A.pos == pos) & A.fbs].copy()
    X['dp'] = X.groupby('team').tgt.rank(ascending=False, method='first').astype(int)
    pop = X.tgt >= minpop
    feat = {'tsh': lambda X: X.tgt / X.ta, 'tpg': lambda X: X.tgt / X.gp, 'rypg': lambda X: X.j_recy / X.gp,
            'ypr': lambda X: shr(X.j_ypr, X.rec, 10), 'x20r': lambda X: shr(X.j_x20 / X.tgt, X.tgt, 20),
            'cp': lambda X: shr(X.j_rec / X.tgt, X.tgt, 20), 'tsr': lambda X: shr(X.j_tsr, X.tgt, 20), 't3s': lambda X: shr(X.t3 / X.t3a.replace(0, np.nan), X.t3a, 10),
            'rzs': lambda X: shr(X.rzt / X.trza.replace(0, np.nan), X.trza, 8), 'rzpg': lambda X: X.rzt / X.gp, 'tdr': lambda X: shr(X.j_rectd / X.tgt, X.tgt, 30)}
    names = {'WR': ['Alpha', 'Deep Threat', 'Red-Zone Threat', 'Possession'], 'TE': ['Receiving Weapon', 'Seam Stretcher', 'Red-Zone Target', 'Safety Valve']}[pos]
    rec = {names[0]: [('tsh', 1), ('tpg', 1), ('rypg', 1)], names[1]: [('ypr', 1), ('x20r', 1)],
           names[2]: [('rzs', 1), ('rzpg', 1), ('tdr', 1)], names[3]: [('cp', 1), ('tsr', 1), ('t3s', 1)]}
    S, L = score(X, pop, feat, rec); X = X.join(S).join(L)
    return X, show_rule(X), names, int(pop.sum())
W, showW, NW, popW = recv('WR', 10, lambda X: X.p4 & ((X.dp == 1) | (X.tgt >= 3 * X.tgm)))
TE, showT, NT, popT = recv('TE', 6, lambda X: X.p4 & (((X.dp == 1) & (X.tgt >= 4)) | (X.tgt >= 2 * X.tgm)))
print('archetype pools: RB', int(popR.sum()), 'WR', popW, 'TE', popT)

# ---------- export
def rk(metric): e = pd.Series(E[metric]); return e[e.index.isin(FBS)].rank(method='min').astype(int)
RUNR, PASSR = rk('repa'), rk('pepa')
def bio(p):
    r = PL.loc[p]; parts = []
    if isinstance(r.cls, str) and r.cls: parts.append(r.cls)
    if isinstance(r.ht, str) and r.ht: parts.append(r.ht.replace("' ", '-').replace('"', ''))
    if isinstance(r.wt, str) and r.wt: parts.append(r.wt)
    return ', '.join(parts)
R3 = lambda v: None if v is None or (isinstance(v, float) and np.isnan(v)) else round(float(v), 3)
def glog(p, kind):
    c, g = C[C.pid == p], G[G.pid == p]; out = []
    for gid in sorted(set(c.game_id) | set(g.game_id), key=lambda z: gd[z]):
        cc, gg = c[c.game_id == gid], g[g.game_id == gid]; x = pd.concat([cc, gg])
        opp = x.dfn.iloc[0]; hf = int(np.sign(x.homeflag.mean())); rkr = RUNR if kind == 'rb' else PASSR
        z = dict(dt=f"{gd[gid]:%b} {gd[gid].day}", o=opp, h=hf, od=int(rkr[opp]) if opp in rkr.index else 'FCS',
                 car=len(cc), ry=int(cc.ryds.sum()), rtd=int(cc.rtd.sum()), tgt=len(gg), rec=int(gg.cmp.sum()), recy=int(gg.pyds.sum()), rectd=int(gg.ptd.sum()))
        base = cc if kind == 'rb' else gg; ea = (base.epa - (base.e_repa if kind == 'rb' else base.e_pepa))
        z['e'] = R3(base.epa.mean()) if base.epa.notna().any() else None; z['ea'] = R3(ea.mean()) if ea.notna().any() else None
        out.append(z)
    return out
def pack_rb(r, m):
    j = m == 'j'; car = max(r.car, 1)
    ry = r.j_ry if j else r.ry; rtd = r.j_rtd if j else r.rtd; xr = r.j_xr if j else r.xr
    rec = r.j_rec if j else r.rec; recy = r.j_recy if j else r.recy; rectd = r.j_rectd if j else r.rectd
    d = dict(car=r.car, ry=ry, rtd=rtd, ypc=ry / car, sr=100 * (r.j_sr if j else r.sr), epa=r.j_epa if j else r.epa, xr=xr, xrt=100 * xr / car,
             stf=100 * (r.j_stf if j else r.stf), tgt=r.tgt, rec=rec, recy=recy, rectd=rectd, scr=ry + recy, tot=rtd + rectd,
             cpg=r.car / r.gp, csh=100 * r.car / max(r.tr, 1), rzs=100 * r.rzc / max(r.trzr, 1))
    return {k: R3(v) for k, v in d.items()}
def pack_rc(r, m):
    j = m == 'j'; t = max(r.tgt, 1)
    rec = r.j_rec if j else r.rec; recy = r.j_recy if j else r.recy; rectd = r.j_rectd if j else r.rectd; x20 = r.j_x20 if j else r.x20
    ry = r.j_ry if j else r.ry; rtd = r.j_rtd if j else r.rtd
    ypr = (r.j_ypr if j else (r.recy / r.rec if r.rec else np.nan))
    d = dict(tgt=r.tgt, rec=rec, recy=recy, rectd=rectd, cp=100 * rec / t, ypt=recy / t, ypr=ypr, sr=100 * (r.j_tsr if j else r.tsr),
             epa=r.j_tepa if j else r.tepa, x20=x20, tsh=100 * r.tgt / max(r.ta, 1), rzt=r.rzt, rzs=100 * r.rzt / max(r.trza, 1),
             t3s=100 * r.t3 / max(r.t3a, 1), tpg=r.tgt / r.gp, car=r.car, ry=ry, scr=recy + ry, tot=rectd + rtd)
    return {k: R3(v) for k, v in d.items()}
def export(X, show, names, kind):
    X = X[show].copy(); sos = X.sos_r if kind == 'rb' else X.sos_p
    X['sos'] = sos; X['sosr'] = X.sos.rank(method='min').astype(int)
    out = []
    for p, r in X.iterrows():
        c = C[C.pid == p].ryds if kind == 'rb' else G[(G.pid == p) & G.cmp].pyds
        cd = ([int((c <= 0).sum()), int(((c > 0) & (c < 4)).sum()), int(((c >= 4) & (c < 10)).sum()), int(((c >= 10) & (c < 20)).sum()), int((c >= 20).sum())]
              if kind == 'rb' else [int((c < 5).sum()), int(((c >= 5) & (c < 10)).sum()), int(((c >= 10) & (c < 20)).sum()), int(((c >= 20) & (c < 30)).sum()), int((c >= 30).sum())])
        pk = pack_rb if kind == 'rb' else pack_rc
        out.append(dict(k=str(p), n=r['name'], t=r.team, c='Independent' if conf.get(r.team) == 'FBS Independents' else conf.get(r.team), gp=int(r.gp), dp=int(r.dp),
                        a=r.a, lab=r.lab, sc=[round(float(r[n]), 2) for n in names], sos=round(float(r.sos), 4), sosr=int(r.sosr), b=bio(p),
                        r=pk(r, 'r'), j=pk(r, 'j'), cd=cd, g=glog(p, kind)))
    return out
OUT = {'rb': export(R, showR, list(ARCH_R), 'rb'), 'wr': export(W, showW, NW, 'wr'), 'te': export(TE, showT, NT, 'te')}
# QBs: add a stable key and roster bio
Q = json.load(open('qb_data.json')); qteam = C.groupby('pid').off.agg(lambda v: v.value_counts().index[0])
for q in Q:
    q['k'] = q['n']; hit = [p for p in PL.index[(PL.name == q['n']) & (PL.pos == 'QB')] if qteam.get(p) == q['t']]
    q['b'] = bio(hit[0]) if hit else ''
OUT['qb'] = Q
# page metadata: data window, model pools, Power 4 + ND teams with no qualifying tight end
MON = {1: 'Jan.', 2: 'Feb.', 3: 'March', 4: 'April', 5: 'May', 6: 'June', 7: 'July', 8: 'Aug.', 9: 'Sept.', 10: 'Oct.', 11: 'Nov.', 12: 'Dec.'}
last = max(gd[g] for g in s.game_id.unique() if g in gd.index)
OUT['meta'] = dict(week=int(s.week.max()), through=f'{MON[last.month]} {last.day}', n_fbs=len(FBS),
                   pools=dict(qb=len(pd.read_pickle('qb_arch.pkl')), rb=int(popR.sum()), wr=popW, te=popT),
                   te_missing=sorted({q['t'] for q in Q} - {d['t'] for d in OUT['te']}))
json.dump(OUT, open('all_data.json', 'w'), separators=(',', ':'))
print({k: len(v) for k, v in OUT.items()}, 'size KB', round(len(json.dumps(OUT, separators=(',', ':'))) / 1024))
print('QB bios found', sum(1 for q in Q if q['b']), 'of', len(Q))
from collections import Counter
for k in ['rb', 'wr', 'te']:
    print(k, Counter(d['a'] for d in OUT[k]), 'hybrids', sum('/' in d['lab'] for d in OUT[k]), 'weak', sum('weak' in d['lab'] for d in OUT[k]))
    top = sorted(OUT[k], key=lambda d: -(d['j']['ry'] if k == 'rb' else d['j']['recy']))[:6]
    print('  top adj yds:', [(d['n'], d['t'], round(d['j']['ry'] if k == 'rb' else d['j']['recy']), d['lab']) for d in top])
    print('  no bio:', [d['n'] for d in OUT[k] if not d['b']][:8], '| teams missing:', sorted({t for t in {x['t'] for x in OUT['qb']}} - {d['t'] for d in OUT[k]}))
