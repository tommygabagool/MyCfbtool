"""Step 11. Data for the Teams page: every Power 4 team and Notre Dame with model ratings, record, roster with season
totals, full schedule, and for each game both teams' players (actual box score once played, model projection from Week 2
on), plus what each projection rests on (every player's schedule-adjusted rates, every defense's ratings and ranks, and the
model's weights) for the matchup panel. Writes work/teams_data.json."""
import paths  # noqa: F401  (sets the working directory to work/)
import pandas as pd, numpy as np, json, pickle
from pathlib import Path
from common import P4C, tiers, MCOLS, TCOLS
from pmodel import FAM

P = pickle.load(open('proj.pkl', 'rb')); PL = pd.read_pickle('players.pkl')
CAL = P['CAL']                     # 2025 calibration updated with this season's games (project.py)
CAL25 = json.load(open('calib.json'))
T = json.load(open('teams.json')); conf = T['conf']; FBS = set(T['fbs'])
ALL = json.load(open('all_data.json'))
STAT, R, LG = P['STAT'], P['R'], P['LG']
P4 = {t for t, c in conf.items() if c in P4C} | {'Notre Dame'}
S = pd.read_parquet('../data/schedules_2026.parquet'); S['gid'] = S.game_id.astype(int); S = S.drop_duplicates('gid').set_index('gid')
S['ts'] = pd.to_datetime(S.start_date, utc=True)
bet = pd.read_parquet('../data/betting_2026.parquet'); bet['gid'] = bet.game_id.astype(int); bet = bet.drop_duplicates('gid').set_index('gid')
LINES = Path('../data/espn_lines_2026.json')  # ESPN's lines for games the betting file doesn't have yet (download_data.py)
pre = json.loads(LINES.read_text()) if LINES.exists() else {}
G = S[S.home_team.isin(P4) | S.away_team.isin(P4)].sort_values('ts')
done = lambda g: g in P['BOX'] and bool(S.at[g, 'completed'])

# ---------- team colors and abbreviations (ESPN rosters / schedule)
ro = pd.read_parquet('../data/game_rosters_2026.parquet', columns=['team_id', 'team_color', 'team_alternate_color', 'team_short_display_name'])
ro = ro.drop_duplicates('team_id').set_index('team_id')
tid, abbr = {}, {}
for side in ('home', 'away'):
    for n, i, a in zip(S[side + '_team'], S[side + '_id'], S[side + '_abbreviation']):
        tid[n] = int(i)
        if isinstance(a, str) and a: abbr[n] = a
def color(t, k):
    i = tid.get(t)
    v = ro[k].get(i) if i is not None and i in ro.index else None
    return '#' + v if isinstance(v, str) and len(v) == 6 else None

# ---------- ratings: FBS ranks (1 = best) for the latest fit
def rk(metric, side):
    v = pd.Series(R[metric][side]); v = v[v.index.isin(FBS)]
    return v.rank(ascending=(side == 'dfn'), method='min').astype(int).to_dict(), v.to_dict()
UNITS = {'o': ('epa', 'off'), 'd': ('epa', 'dfn'), 'po': ('pepa', 'off'), 'ro': ('repa', 'off'), 'pd': ('pepa', 'dfn'), 'rd': ('repa', 'dfn'),
         'so': ('sr', 'off'), 'sd': ('sr', 'dfn')}
RK = {u: rk(*m) for u, m in UNITS.items()}
p4share = np.mean([tiers(t, conf, FBS)[0] for t in FBS])
Ld = P['DPACE'][0]; MC = dict(zip(MCOLS, CAL['m']))
def power(t):
    """Projected margin against an average FBS team on a neutral field: the game model's margin terms with the opponent's
    ratings, pace and league tier all at the FBS average."""
    d = lambda k: R[k]['off'].get(t, 0.0) - R[k]['dfn'].get(t, 0.0)
    return (MC['E'] * LG['plays'] * d('epa') + MC['Pd'] * Ld * d('ppd') + MC['Sr'] * LG['plays'] * d('sr')
            + MC['p4'] * (tiers(t, conf, FBS)[0] - p4share))
PW = pd.Series({t: power(t) for t in FBS}); PWR = PW.rank(ascending=False, method='min').astype(int)
DEF = P['DEF']; DXK = ['cmp', 'ypa', 'ptd', 'int', 'ry', 'rtd']
DXR = {}
for k in DXK:
    v = pd.Series(DEF[k]['dfn']); v = v[v.index.isin(FBS)]
    DXR[k] = v.rank(ascending=(k != 'int'), method='min').astype(int).to_dict()
def dx(t):
    """[[effect, FBS rank or None]] per metric in DXK: what defense t allows per play above or below an average FBS defense."""
    return [[round(float(DEF[k]['dfn'].get(t, 0.0)), 4), DXR[k].get(t)] for k in DXK]
# each projected player's latest inputs: per family [plays so far, and per stat [raw rate, average defensive effect faced,
# schedule-adjusted rate, group average, the FBS rank of a defense that allows what his opponents averaged]]; volume
# averages per game played, recent shares, played one of the team's last two games
MXS = {fam: list(stats) for fam, (vol, stats) in FAM.items()}
MET = {st: m for fam, (vol, stats) in FAM.items() for st, m in stats.items()}
FBSDX = {k: np.sort(pd.Series(DEF[k]['dfn'])[lambda v: v.index.isin(FBS)].values) for k in DXK}
def eq_rank(k, e):
    """FBS rank (1 = stingiest; for interceptions, 1 = most) of a defense allowing e per play."""
    v = FBSDX[k]; return int(1 + ((v > e).sum() if k == 'int' else (v < e).sum()))
def mx(c):
    out = dict(b=[round(float(c['b_' + k]), 2) for k in ('att', 'car', 'tgt')], sh=[round(float(c['sh_' + k]), 4) for k in ('att', 'car', 'tgt')],
               av=int(c['avail']))
    for fam, stats in MXS.items():
        n = float(c['n_' + fam])
        if n > 0:
            out[fam] = [int(n)] + [[round(float(c[f'{q}_{st}']), 4) if pd.notna(c[f'{q}_{st}']) else None for q in ('raw', 'eh', 'adj', 'avg')]
                                   + [eq_rank(MET[st], float(c['eh_' + st])) if pd.notna(c['eh_' + st]) else None] for st in stats]
    return out

# ---------- players-page links: QBs are keyed by name, everyone else by athlete id
link = {}
for pos in ('rb', 'wr', 'te'):
    for d in ALL[pos]: link[int(d['k'])] = [pos, d['k']]
qlink = {(d['t'], d['n']): ['qb', d['k']] for d in ALL['qb']}

def bio(p):
    if p not in PL.index: return ''
    r = PL.loc[p]; parts = [x for x in (r.cls, r.ht.replace("' ", '-').replace('"', '') if isinstance(r.ht, str) else None, r.wt) if isinstance(x, str) and x]
    return ', '.join(parts)

# ---------- players: everyone who shows up in a box score or projection of these games
season = {}
for g, bx in P['BOX'].items():
    for p, arr in bx.items():
        a = season.setdefault(p, [0] * len(STAT) + [0]); a[:len(STAT)] = [x + y for x, y in zip(a[:len(STAT)], arr)]; a[-1] += 1
pids = set()
for g in G.index:
    pids |= set(P['BOX'].get(g, {})) | set(P['PROJ'].get(g, {}))
for t in P4:
    pids |= {p for p, tt in P['PTEAM'].items() if tt == t}
players = {}
for p in pids:
    t = P['PTEAM'].get(p)
    if p in PL.index:
        n, pos, jn = PL.at[p, 'name'], PL.at[p, 'pos'], PL.at[p, 'jersey']
    else:
        n, pos, jn = P['QNAME'].get(p), 'QB', None
    if p in P['QNAME'] and season.get(p, [0])[0] > 0 and pos in ('?', 'ATH', 'OTHER', None): pos = 'QB'
    pos = {'FB': 'RB', '?': 'ATH', 'OTHER': 'ATH'}.get(pos, pos)
    d = dict(n=n or f'#{p}', t=t, pos=pos, st=season.get(p, [0] * (len(STAT) + 1)))
    if jn is not None and str(jn) not in ('nan', 'None'): d['jn'] = str(jn)
    b = bio(p)
    if b: d['b'] = b
    lk = link.get(p) or qlink.get((t, P['QNAME'].get(p))) or (qlink.get((t, n)) if pos == 'QB' else None)
    if lk: d['pk'] = lk
    if p in P['COMPS']: d['mx'] = mx(P['COMPS'][p])
    players[str(p)] = d

# ---------- games
r1 = lambda v: round(float(v), 1)
games = {}
for g, z in G.iterrows():
    e = dict(wk=int(z.week), ts=int(z.ts.timestamp()), h=z.home_team, a=z.away_team, ns=int(bool(z.neutral_site)), cg=int(bool(z.conference_game)),
             v=z.venue if isinstance(z.venue, str) else '', done=int(done(g)), tbd=int(bool(z.start_time_tbd)))
    if pd.notna(z.home_rank): e['hr'] = int(z.home_rank)
    if pd.notna(z.away_rank): e['ar'] = int(z.away_rank)
    if bool(z.completed) and pd.notna(z.home_points): e['hs'], e['as'] = int(z.home_points), int(z.away_points)
    if g in bet.index and bool(bet.at[g, 'game_spread_available']):
        e['ln'] = dict(sp=float(bet.at[g, 'home_team_spread']), ou=float(bet.at[g, 'over_under']) if pd.notna(bet.at[g, 'over_under']) else None)
    elif str(g) in pre:
        e['ln'] = pre[str(g)]
    m = P['MODEL'].get(g)
    if m:
        e['m'] = dict(hp=r1(m['hp']), ap=r1(m['ap']), wp=round(m['wp'], 3), asof=int(m['asof']))
        tv = m.get('tv', {})
        if tv: e['m']['tv'] = [[r1(x) for x in tv[k]] if k in tv else None for k in (z.home_team, z.away_team)]   # team plays, passes, carries, targets
    if g in P['BOX'] and e['done']: e['box'] = {str(p): v for p, v in P['BOX'][g].items()}
    if g in P['PROJ']: e['pr'] = {str(p): [r1(x) for x in v] for p, v in P['PROJ'][g].items()}
    games[str(g)] = e

# ---------- teams (full profiles for Power 4 + ND; light entries for opponents)
def record(t, cg_only=False):
    w = l = 0
    for g, z in G.iterrows():
        if t not in (z.home_team, z.away_team) or not (bool(z.completed) and pd.notna(z.home_points)) or (cg_only and not z.conference_game): continue
        mine, theirs = (z.home_points, z.away_points) if t == z.home_team else (z.away_points, z.home_points)
        w += mine > theirs; l += mine < theirs
    return [int(w), int(l)]
opp = {t for z in G.itertuples() for t in (z.home_team, z.away_team)}
teams = {}
for t in sorted(opp):
    c = conf.get(t) or (S[S.home_team == t].home_conference.dropna().tolist() + S[S.away_team == t].away_conference.dropna().tolist() + [None])[0]
    d = dict(ab=abbr.get(t, t[:4].upper()), c='Independent' if c == 'FBS Independents' else (c or 'FCS'), col=color(t, 'team_color'),
             col2=color(t, 'team_alternate_color'), fbs=int(t in FBS))
    d['dx'] = dx(t)
    if t in FBS:
        d['rk'] = {u: RK[u][0].get(t) for u in UNITS}
        d['pw'], d['pwr'] = r1(PW[t]), int(PWR[t])
    if t in P4:
        sched = [str(g) for g, z in G.iterrows() if t in (z.home_team, z.away_team)]
        pts = [(z.home_points, z.away_points) if t == z.home_team else (z.away_points, z.home_points) for g, z in G.iterrows()
               if t in (z.home_team, z.away_team) and bool(z.completed) and pd.notna(z.home_points)]
        d.update(p4=1, rec=record(t), crec=record(t, True), sched=sched,
                 pf=r1(np.mean([a for a, _ in pts])) if pts else None, pa=r1(np.mean([b for _, b in pts])) if pts else None,
                 v={u: round(RK[u][1].get(t, 0.0), 3) for u in UNITS}, pace=r1(P['PACE'].get(t, LG['plays'])), dbr=round(P['DBR'].get(t, LG['db']), 3),
                 roster=sorted([k for k, v in players.items() if v['t'] == t], key=lambda k: -sum(players[k]['st'][i] for i in (0, 6, 9))))
    teams[t] = d

meta = dict(week=int(P['LAST']), through=ALL['meta']['through'], n_fbs=len(FBS), stat=STAT, acc=P['ACC'],
            cal=dict(m=dict(zip(MCOLS, np.round(CAL['m'], 3).tolist())), t=dict(zip(TCOLS, np.round(CAL['t'], 3).tolist())), sigma=round(CAL['sigma'], 2),
                     n26=CAL['n26']),
            cal25=dict(sigma=round(CAL25['sigma'], 2), n=CAL25['n'], mae=round(CAL25['mae'], 1), tmae=round(CAL25['tmae'], 1)),
            lg={k: round(float(v), 3) for k, v in LG.items()}, p4=sorted(P4), units={u: list(m) for u, m in UNITS.items()},
            pc=dict(rate={k: np.round(v, 3).tolist() for k, v in P['PC']['rate'].items()}, vol={k: np.round(v, 3).tolist() for k, v in P['PC']['vol'].items()},
                    plays=np.round(P['PC']['plays'], 3).tolist(), dbr=np.round(P['PC']['dbr'], 4).tolist(), n=P['PC']['n']),
            dxk=DXK, dhome={k: round(DEF[k]['home'], 4) for k in DXK}, nfbs_dx=len(DXR['ypa']), mxs=MXS)
OUT = dict(meta=meta, teams=teams, players=players, games=games)
json.dump(OUT, open('teams_data.json', 'w'), separators=(',', ':'))
sz = lambda o: round(len(json.dumps(o, separators=(',', ':'))) / 1024)
print(f"teams_data.json: {sz(OUT)} KB (teams {sz(teams)}, players {sz(players)}, games {sz(games)}); {len(teams)} teams ({len(P4)} full), "
      f"{len(players)} players, {len(games)} games ({sum(g['done'] for g in games.values())} played)")
print('players linked to the Players page:', sum('pk' in v for v in players.values()), 'of', sum(len(ALL[k]) for k in ('qb', 'rb', 'wr', 'te')))
x = teams['Alabama']; print('Alabama:', {k: x[k] for k in ('ab', 'c', 'col', 'rec', 'crec', 'pf', 'pa', 'pw', 'pwr', 'rk', 'pace')}, len(x['roster']), 'players')
