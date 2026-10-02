"""Step 7. Attach rusher/receiver identity (ESPN athlete ids, with a play-text fallback matched against ESPN's
game rosters), roster position and bio, and a red-zone flag to every play.
Writes work/plays_skill.parquet and work/players.pkl."""
import paths  # noqa: F401  (sets the working directory to work/)
import pandas as pd, numpy as np, re, json, unicodedata
s = pd.read_parquet('plays.parquet')
C = ['game_id', 'reception_player', 'reception_player_id', 'target_player', 'target_player_id', 'rush_player', 'rush_player_id',
     'position_target', 'position_reception', 'yards_to_goal', 'home_team', 'away_team', 'home_team_id', 'away_team_id']
p = pd.read_parquet('../data/pbp.parquet', columns=C)
x = p.loc[s.index]
tid2name = {}
for a, b in list(zip(p.home_team_id, p.home_team)) + list(zip(p.away_team_id, p.away_team)):
    if pd.notna(a): tid2name[int(a)] = b

# ---- ESPN game rosters -> per-team registry
POS = {'1': 'WR', '7': 'TE', '8': 'QB', '9': 'RB', '10': 'FB', '0': 'ATH'}
ro = pd.read_parquet('../data/game_rosters_2026.parquet')
ro['team'] = ro.team_id.astype(int).map(tid2name)
ro['pos'] = ro.position_href.str.extract(r'positions/(\d+)')[0].map(POS).fillna('OTHER')
print('roster rows without a pbp team name:', int(ro.team.isna().sum()))
ro = ro[ro.team.notna()]
def fold(z): return unicodedata.normalize('NFKD', str(z)).encode('ascii', 'ignore').decode()
def norm(z): return re.sub(r'[^a-z]', '', fold(z).lower())
SUF = re.compile(r'(?:,?\s(?:Jr\.?|Sr\.?|II|III|IV|V))$')
ath = ro.sort_values('week').drop_duplicates('athlete_id', keep='last').set_index('athlete_id')
reg = {}                                    # team -> key -> set(athlete ids)
for aid, r in ath.iterrows():
    fn, ln = str(r.first_name or ''), SUF.sub('', str(r.last_name or ''))
    keys = {norm(r.full_name), norm(SUF.sub('', str(r.full_name))), norm(r.display_name), norm(r.short_name),
            norm(fn[:1] + ln), norm(fn[:2] + ln)}
    for k in keys:
        if k: reg.setdefault(r.team, {}).setdefault(k, set()).add(int(aid))
game_roster = ro.groupby('game_id').athlete_id.apply(lambda v: set(int(a) for a in v)).to_dict()
jersey = ath.jersey.to_dict()
SKILL = {'WR', 'TE', 'RB', 'FB', 'ATH', 'QB'}

def resolve(team, game, raw, num):
    ids = reg.get(team, {}).get(norm(SUF.sub('', raw)), set())
    if len(ids) > 1 and num:  ids = {i for i in ids if str(jersey.get(i)) == num} or ids
    if len(ids) > 1: ids = {i for i in ids if i in game_roster.get(int(game), set())} or ids
    if len(ids) > 1: ids = {i for i in ids if ath.at[i, 'pos'] in SKILL} or ids
    return next(iter(ids)) if len(ids) == 1 else None
NAME = r"(?:#(\d+)\s?)?((?:[A-Z][a-z]{0,3}\.\s?(?:St\.\s?)?[^\s,.;:()]+(?:\s(?!for\b|thrown\b|caught\b|rush\b|run\b)[A-Z][^\s,.;:()]+)?)|(?:[A-Z][^\s,;:()#]*(?:\s(?!for\b|thrown\b|caught\b|rush\b|run\b)[A-Z][^\s,;:()#]*){1,3}))(?:,?\s(?:Jr\.?|Sr\.?|II|III|IV))?"
RE_REC = re.compile(r'pass(?: complete| incomplete)?[a-z ]*? to ' + NAME)
RE_RUS = re.compile(NAME + r'\s(?:rush|run|scramble)\b')
RE_RUS2 = re.compile(r'^(?:\(\d+:\d+\)\s*)?' + NAME + r'\s\d+\s[Yy]d\s(?:Run|Rush)')
RE_REC2 = re.compile(r'^(?:\(\d+:\d+\)\s*)?' + NAME + r'\s\d+\s[Yy]d\s[Pp]ass from')

# ---- ids straight from the participant fields
s['game_id'] = x.game_id
s['rec_id'] = np.where(s.cmp, x.reception_player_id.fillna(x.target_player_id), x.target_player_id)
s.loc[~s.is_att, 'rec_id'] = np.nan
s['rus_id'] = np.where(s.is_rush, x.rush_player_id, np.nan)
s['ytg'] = x.yards_to_goal
fixed = {'rec': 0, 'rus': 0}
for idx in s.index[s.is_att & ~s.intc & s.rec_id.isna()]:
    m = RE_REC.search(str(s.at[idx, 'play_text'])) or RE_REC2.search(str(s.at[idx, 'play_text']))
    if m:
        a = resolve(s.at[idx, 'off'], s.at[idx, 'game_id'], m.group(2), m.group(1))
        if a: s.at[idx, 'rec_id'] = a; fixed['rec'] += 1
for idx in s.index[s.is_rush & ~s.kneel & s.rus_id.isna()]:
    m = RE_RUS2.search(str(s.at[idx, 'play_text'])) or RE_RUS.search(str(s.at[idx, 'play_text']))
    if m:
        a = resolve(s.at[idx, 'off'], s.at[idx, 'game_id'], m.group(2), m.group(1))
        if a: s.at[idx, 'rus_id'] = a; fixed['rus'] += 1
print('resolved from text:', fixed)
print('still missing: receiver', int((s.is_att & ~s.intc & s.rec_id.isna()).sum()), 'of', int((s.is_att & ~s.intc).sum()),
      '| rusher', int((s.is_rush & ~s.kneel & s.rus_id.isna()).sum()), 'of', int((s.is_rush & ~s.kneel).sum()))

# ---- player table: name, roster position, bio; position falls back to the label ESPN puts on his targets
lab = pd.concat([x[['target_player_id', 'position_target']].set_axis(['i', 'p'], axis=1),
                 x[['reception_player_id', 'position_reception']].set_axis(['i', 'p'], axis=1)]).dropna()
lab = lab.groupby('i').p.agg(lambda v: v.value_counts().index[0])
nm = pd.concat([x[['target_player_id', 'target_player']].set_axis(['i', 'n'], axis=1), x[['reception_player_id', 'reception_player']].set_axis(['i', 'n'], axis=1),
                x[['rush_player_id', 'rush_player']].set_axis(['i', 'n'], axis=1)]).dropna().drop_duplicates('i').set_index('i').n
ids = set(s.rec_id.dropna().astype(int)) | set(s.rus_id.dropna().astype(int))
rows = []
for i in ids:
    if i in ath.index:
        r = ath.loc[i]; pos = r.pos if r.pos != 'OTHER' else lab.get(i, 'OTHER')
        rows.append(dict(id=i, name=r.full_name, pos=pos, cls=r.experience_display_value, ht=r.display_height, wt=r.display_weight, jersey=r.jersey))
    else:
        rows.append(dict(id=i, name=nm.get(i), pos=lab.get(i, '?'), cls=None, ht=None, wt=None, jersey=None))
PL = pd.DataFrame(rows).set_index('id'); PL.to_pickle('players.pkl')
print('players', len(PL), PL.pos.value_counts().head(10).to_dict())
s.to_parquet('plays_skill.parquet')
