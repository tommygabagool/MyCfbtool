"""Optional. Check play-by-play attribution against ESPN box scores, game by game. The box-score file can lag
the play-by-play, so only games present in both are compared."""
import paths  # noqa: F401  (sets the working directory to work/)
import pandas as pd

b = pd.read_parquet('../data/player_box_2026.parquet')
for c in ['passingYards', 'rushingAttempts', 'rushingYards', 'receptions', 'receivingYards']:
    b[c] = pd.to_numeric(b[c], errors='coerce')
b['athlete_id'] = pd.to_numeric(b.athlete_id, errors='coerce'); b['game_id'] = pd.to_numeric(b.game_id, errors='coerce')
s = pd.read_parquet('plays_skill.parquet'); s['gid'] = s.game_id.astype(int); PL = pd.read_pickle('players.pkl')
share = lambda m, a, c: f'{(m[a] == m[c]).mean():.1%}'

r = (s[s.is_att & s.rec_id.notna()].groupby(['rec_id', 'gid']).agg(rec=('cmp', 'sum'), yds=('pyds', 'sum')).reset_index()
     .merge(b[b.receptions.notna()], left_on=['rec_id', 'gid'], right_on=['athlete_id', 'game_id']))
print(f'receiver-games {len(r):>5}: receptions exact {share(r, "rec", "receptions")}, yards exact {share(r, "yds", "receivingYards")}')
u = (s[s.is_rush & ~s.kneel & s.rus_id.notna()].groupby(['rus_id', 'gid']).agg(car=('ryds', 'size'), yds=('ryds', 'sum')).reset_index()
     .merge(b[b.rushingAttempts.notna()], left_on=['rus_id', 'gid'], right_on=['athlete_id', 'game_id']))
u = u[u.rus_id.astype(int).map(PL.pos).isin(['RB', 'WR', 'TE'])]
print(f'rusher-games   {len(u):>5}: carries exact {share(u, "car", "rushingAttempts")}, yards exact {share(u, "yds", "rushingYards")} (non-QBs)')
q = (s[s.is_att & s.qb.notna()].groupby(['qb', 'gid']).pyds.sum().reset_index()
     .merge(b[b.passingYards.notna()], left_on=['qb', 'gid'], right_on=['athlete_name', 'game_id']))
print(f'QB-games       {len(q):>5}: passing yards exact {share(q, "pyds", "passingYards")}')
