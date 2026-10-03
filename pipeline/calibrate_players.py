"""Step 3c. Calibrate the player projections on last season. Replays 2025 week by week (ratings, usage and rates from
earlier weeks only, the game model from calibrate.py for each game's projected margin and total) and fits, by least
squares on what actually happened:
  - team plays from pace and the projected total, and pass rate from the team's own rate and the projected margin;
  - each player's volume as a blend of his recent share of the team's volume and his average per game played, less a
    blowout term (starters sit when the projected margin is large);
  - each stat's rate per play as the league average, plus a share of how far his schedule-adjusted rate sits above or
    below it, plus a share of the opponent's defensive rating (and a small constant).
Graded: players averaging 15+ pass attempts, 8+ carries or 3+ targets per game played, in games they played.
Writes work/calib_players.json."""
import paths  # noqa: F401  (sets the working directory to work/)
import pandas as pd, numpy as np, json, pickle
from common import *
from pmodel import STAT, PK, FAM, GRADE, CLIP, Ctx, box, usage_flags, team_volume, prate, fill_ids

R = pickle.load(open('seasons.pkl', 'rb')); Yc = pd.read_csv('persistence_conf.csv'); CAL = json.load(open('calib.json'))
IDS = ['completion_player_id', 'incompletion_player_id', 'interception_thrown_player_id', 'sack_taken_player_id', 'rush_player_id',
       'reception_player_id', 'target_player_id']
p = pd.read_parquet('../data/pbp_2025.parquet', columns=COLS + IDS)
FBS = fbs_teams(p); conf = R[2025]['_conf']
s = outcomes(p)
sched = pd.read_parquet('../data/schedules_2025.parquet'); sched = sched[sched.season_type == 'regular']
s = s[s.game_id.astype(int).isin(set(sched.game_id.astype(int)))]
s['gid'] = s.game_id.astype(int)
num = lambda c: pd.to_numeric(s[c], errors='coerce')
# player ids from the participant fields, then from the play text's jersey numbers and names (most late-season 2025 touchdown
# plays came through with no ids at all)
s['qid'] = np.where(s.is_att, num('completion_player_id').fillna(num('incompletion_player_id')).fillna(num('interception_thrown_player_id')),
                    np.where(s.is_sack, num('sack_taken_player_id'), np.nan))
s['rec_id'] = np.where(s.is_att, np.where(s.cmp, num('reception_player_id').fillna(num('target_player_id')), num('target_player_id')), np.nan)
s['rus_id'] = np.where(s.is_rush & ~s.kneel, num('rush_player_id'), np.nan)
ro = pd.read_parquet('../data/game_rosters_2025.parquet', columns=['game_id', 'team_id', 'athlete_id', 'jersey', 'first_name', 'last_name', 'position_href', 'week'])
miss0 = {c: int(s[c].isna()[m].sum()) for c, m in (('qid', s.is_att | s.is_sack), ('rec_id', s.is_att & ~s.intc), ('rus_id', s.is_rush & ~s.kneel))}
s, filled = fill_ids(s, ro, R[2025]['_tid'])
print('ids missing before', miss0, '| filled from the play text', filled)
s = usage_flags(s)
dr = drive_table(s)
ro['pos'] = ro.position_href.str.extract(r'positions/(\d+)')[0].map({'1': 'WR', '7': 'TE', '8': 'QB', '9': 'RB', '10': 'RB', '0': 'ATH'}).fillna('OTHER')
POS = ro.sort_values('week').drop_duplicates('athlete_id', keep='last').assign(a=lambda d: pd.to_numeric(d.athlete_id, errors='coerce')).set_index('a').pos.to_dict()
tz = talent_z('../data/talent_2025.parquet', R[2025]['_tid'], FBS)
prior = season_prior(R, Yc, base=2024, conf=conf, tz=tz, loosen=LOOSEN)
pts = points_lookup(sched)
S = sched.assign(gid=sched.game_id.astype(int)).drop_duplicates('gid').set_index('gid')
TS = pd.to_datetime(S.start_date, utc=True).to_dict()
gw = s.groupby('gid').week.first()
games = S[S.index.isin(gw.index)]
ACT = box(s).set_index(['gid', 'off', 'pid'])
TACT = s.groupby(['gid', 'off']).agg(live=('live', 'sum'), db=('db', 'sum'))

trows, prows = [], []
for w in range(2, int(gw.max()) + 1):
    c = Ctx(s, dr, FBS, conf, prior, pts, TS, POS, w, keys=set(PK) | {'epa', 'sr'})
    L = c.lg
    for g, z in games[games.index.map(gw) == w].iterrows():
        if not (z.home_team in FBS or z.away_team in FBS): continue
        hf = 0 if z.neutral_site else 1
        fh = side_feats(z.home_team, z.away_team, hf, c.r, FBS, conf, L['plays'], L['ppp'], c.pace_o, c.pace_d, c.Ld, c.dpo, c.dpd)
        fa = side_feats(z.away_team, z.home_team, -hf, c.r, FBS, conf, L['plays'], L['ppp'], c.pace_o, c.pace_d, c.Ld, c.dpo, c.dpd)
        hp, ap, margin, total = game_points(CAL, fh, fa, hf)
        for t, o, h, pm in ((z.home_team, z.away_team, hf, margin), (z.away_team, z.home_team, -hf, -margin)):
            if t not in FBS or (g, t) not in TACT.index: continue
            ti = c.team_inputs(t, o, h)
            trows.append(dict(g=g, t=t, pm=pm, pt=total, **ti, a_live=TACT.at[(g, t), 'live'], a_db=TACT.at[(g, t), 'db']))
            C = c.comps(t, o, h)
            for pid, r in C.iterrows():
                key = (g, t, int(pid))
                act = ACT.loc[key] if key in ACT.index else None
                prows.append(dict(r.to_dict(), g=g, t=t, pid=int(pid), pm=pm, pt=total, **{k: v for k, v in ti.items()},
                                  played=act is not None, **{'a_' + k: (float(act[k]) if act is not None else 0.0) for k in STAT}))
    print('week', w, len(prows), flush=True)
T = pd.DataFrame(trows); X = pd.DataFrame(prows)

def lsq(A, y, wt=None):
    wt = np.ones(len(y)) if wt is None else np.asarray(wt, float)
    return np.linalg.solve((A * wt[:, None]).T @ A + 1e-6 * np.eye(A.shape[1]), (A * wt[:, None]).T @ y)
# team volume: plays ~ pace and projected total; pass rate ~ the team's rate and projected margin
PC = {'plays': lsq(np.c_[np.ones(len(T)), T.plays0, T.pt], T.a_live.values).tolist(),
      'dbr': lsq(np.c_[np.ones(len(T)), T.dbr0, T.pm], (T.a_db / T.a_live).values).tolist()}
TV = {i: team_volume(r, r.pm, r.pt, PC) for i, r in X.iterrows()}
for k in ('att', 'car', 'tgt'): X['tv_' + k] = [TV[i][k] for i in X.index]
# player volume and per-play rates, graded on established players in games they played
played = X.played & ((X.a_att + X.a_car + X.a_tgt) > 0)
PC['vol'], PC['rate'] = {}, {}
for fam, (vol, stats) in FAM.items():
    m = played & (X['b_' + vol] >= GRADE[vol])
    sv = X.loc[m, 'tv_' + vol] * X.loc[m, 'sh_' + vol]
    A = np.c_[sv, X.loc[m, 'b_' + vol] * X.loc[m, 'avail'], sv * X.loc[m, 'pm'].abs() / 14]
    PC['vol'][fam] = lsq(A, X.loc[m, 'a_' + vol].values).tolist()
    mv = m & (X['a_' + vol] > 0)
    for st in stats:
        A = np.c_[np.ones(mv.sum()), X.loc[mv, 'adj_' + st] - X.loc[mv, 'avg_' + st], X.loc[mv, 'eo_' + st]]
        y = (X.loc[mv, 'a_' + st] / X.loc[mv, 'a_' + vol] - X.loc[mv, 'avg_' + st]).values
        PC['rate'][st] = lsq(A, y, X.loc[mv, 'a_' + vol].values).tolist()
# in-sample check against each player's season average so far
out = []
for fam, (vol, stats) in FAM.items():
    m = played & (X['b_' + vol] >= GRADE[vol])
    a, b, c = PC['vol'][fam]; sv = X['tv_' + vol] * X['sh_' + vol]
    v = np.maximum(0, a * sv + b * X['b_' + vol] * X.avail + c * sv * X.pm.abs() / 14)
    for st in stats:
        rate = np.clip(prate(PC['rate'][st], X['adj_' + st], X['eo_' + st], X['avg_' + st]), *CLIP[st])
        e = (v * rate - X['a_' + st])[m].abs().mean()
        out.append(f'{st} {e:.2f}')
print('graded player-games:', {v: int((played & (X['b_' + v] >= GRADE[v])).sum()) for v in GRADE}, '| in-sample MAE:', ', '.join(out))
print('plays', np.round(PC['plays'], 3), 'pass rate', np.round(PC['dbr'], 4), 'volume', {k: np.round(v, 3).tolist() for k, v in PC['vol'].items()})
print('rates (constant, share of his edge over average, share of the opponent effect):', {k: np.round(v, 3).tolist() for k, v in PC['rate'].items()})
PC['n'] = {v: int((played & (X['b_' + v] >= GRADE[v])).sum()) for v in GRADE}
json.dump(PC, open('calib_players.json', 'w'))
