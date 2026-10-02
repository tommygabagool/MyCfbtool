"""Step 6. Export the Power 4 + Notre Dame starting quarterbacks with game logs. Writes work/qb_data.json."""
import paths  # noqa: F401  (sets the working directory to work/)
import pandas as pd, numpy as np, json
s = pd.read_parquet('plays_adj.parquet'); B = pd.read_pickle('qb_arch.pkl')
T = json.load(open('teams.json')); FBS = set(T['fbs']); E = json.load(open('def_effects.json'))
raw = pd.read_parquet('../data/pbp.parquet', columns=['game_id', 'start_date'])
gd = raw.dropna().drop_duplicates('game_id').set_index('game_id').start_date
gd = pd.to_datetime(gd, utc=True).dt.tz_convert('America/New_York')
# FBS defensive ranks by opponent-adjusted EPA/play allowed (1 = stingiest)
de = pd.Series(E['epa']); fr = de[de.index.isin(FBS)].rank(method='min').astype(int)
P = B[B.conf.isin(['ACC', 'Big 12', 'Big Ten', 'SEC']) | (B.team == 'Notre Dame')].copy()
P['sosr'] = P.sos.rank(method='min').astype(int)          # 1 = toughest defenses faced
FPL = ['Field General', 'Gunslinger', 'Pure Runner', 'Pocket Passer']
def pack(r, pre):
    g = lambda k: float(r[pre + k])
    live = r.live
    d = dict(py=g('py'), ptd=g('ptd'), i=g('i'), sk=g('sk'), ry=g('ry'), rtd=g('rtd'), xp=g('xp'), xr=g('xr'),
             sr=100 * g('sr'), epa=g('epa'), ypc=g('ypc'), conv=100 * g('conv'), cp=100 * g('cmp') / r.att)
    d['x'] = d['xp'] + d['xr']; d['xrt'] = 100 * d['x'] / live; d['tot'] = d['ptd'] + d['rtd']
    return {k: round(v, 3) for k, v in d.items()}
out = []
for _, r in P.iterrows():
    d = s[(s.off == r.team) & (s.qb == r.qb)]
    c = d[d.is_att & d.cmp].pyds
    cd = [int((c < 5).sum()), int(((c >= 5) & (c < 10)).sum()), int(((c >= 10) & (c < 20)).sum()), int(((c >= 20) & (c < 30)).sum()), int((c >= 30).sum())]
    games = []
    for gid, x in d.groupby('game_id'):
        a, rr, lv = x[x.is_att], x[x.is_rush], x[x.live & x.epa.notna()]
        opp = x.dfn.iloc[0]; hf = int(np.sign(x.homeflag.mean()))
        games.append(dict(dt=f"{gd[gid]:%b} {gd[gid].day}", ts=int(gd[gid].timestamp()), o=opp, h=hf,
                          od=int(fr[opp]) if opp in fr.index else 'FCS', oe=round(E['epa'][opp], 3),
                          cmp=int(a.cmp.sum()), att=len(a), py=int(a.pyds.sum()), ptd=int(a.ptd.sum()), i=int(a.intc.sum()),
                          car=len(rr), ry=int(rr.ryds.sum()), rtd=int(rr.rtd.sum()),
                          e=round(float(lv.epa.mean()), 3), ea=round(float((lv.epa - lv.e_epa2).mean()), 3)))
    games.sort(key=lambda z: z['ts'])
    for z in games: z.pop('ts')
    out.append(dict(n=r.qb, t=r.team, c='Independent' if r.conf == 'FBS Independents' else r.conf, gp=int(r.gp),
                    a=r.a, lab=r.lab, sc=[round(float(r[f]), 2) for f in FPL],
                    att=int(r.att), car=int(r.car), sky=int(r.sky), sos=round(float(r.sos), 4), sosr=int(r.sosr),
                    r=pack(r, 'r_'), j=pack(r, 'a_'), cd=cd, g=games))
json.dump(out, open('qb_data.json', 'w'), separators=(',', ':'))
print(len(out), 'QBs;', round(len(json.dumps(out)) / 1024), 'KB')
x = next(o for o in out if o['n'] == 'Kevin Jennings'); print(json.dumps({k: x[k] for k in ['n', 't', 'lab', 'sc', 'sos', 'sosr', 'r', 'j']}, indent=None)[:900]); print(x['g'])
print('FBS def ranks sample:', fr.sort_values().head(5).to_dict())
