"""Step 1. Load 2026 play-by-play and keep scrimmage plays, with three source fixes: completion yardage that
drops the minus sign on losses, fumbled runs/catches flagged as neither run nor pass, and ESPN "ghost" copies
of a play stamped (00:00). Attributes every dropback and QB run to a quarterback and flags garbage time.
Writes work/plays.parquet and work/teams.json."""
import paths  # noqa: F401  (sets the working directory to work/)
import pandas as pd, numpy as np, re
p = pd.read_parquet('../data/pbp.parquet')

FBSC = {'ACC','American Athletic','Big 12','Big Ten','Conference USA','FBS Independents',
        'Mid-American','Mountain West','Pac-12','SEC','Sun Belt'}
g = p.drop_duplicates('game_id')
m = g[g.home_team_division.notna()]
conf = {}
for _, r in m.iterrows():
    conf[r.home_team] = r.home_team_conference; conf[r.away_team] = r.away_team_conference
FBS = {t for t, c in conf.items() if c in FBSC}

# fumbled runs (and a few fumbled catches) come through with rush = pass = 0; put them back in
_t = p.play_text.fillna('')
_f = (p.play_type.str.contains('Fumble', na=False) & (p['rush'] != 1) & (p['pass'] != 1)
      & ~_t.str.contains(r'\b(?:punt|kickoff|kick|onside)\b|attempt (?:failed|good)|two-point|2pt', case=False))
p = p.copy()
p.loc[_f & _t.str.contains(r'\b(?:rush|run|scramble)\b') & ~_t.str.contains(r'\bpass\b|\bsacked\b'), 'rush'] = 1
p.loc[_f & _t.str.contains('pass complete'), ['pass', 'pass_attempt', 'completion']] = 1
# ---- scrimmage plays: runs, pass attempts, sacks (no 2-pt tries, no nullified plays)
s = p[((p['rush'] == 1) | (p['pass'] == 1)) & ~p.play_type.str.startswith('Two Point')].copy()
s = s[s.penalty_no_play.fillna(False) != True]
# ghost rows: ESPN occasionally repeats a play, stamped (00:00), just before the real one; drop the copies
def _strip(t): return re.sub(r'for (?:\d+ yards? (?:gain|loss)|no gain)', '', re.sub(r'^\(\d{1,2}:\d{2}\)\s*', '', t if isinstance(t, str) else ''))
_ix = list(s.index); _ghost = set()
for _p, _i in enumerate(_ix):
    _t = s.at[_i, 'play_text']
    if isinstance(_t, str) and _t.startswith('(00:00)'):
        for _q in range(_p + 1, min(_p + 6, len(_ix))):
            _j = _ix[_q]
            if s.at[_j, 'game_id'] != s.at[_i, 'game_id']: break
            _tj = s.at[_j, 'play_text']
            if isinstance(_tj, str) and not _tj.startswith('(00:00)') and _strip(_tj) == _strip(_t):
                _ghost.update(_ix[_p:_q]); break
print('ghost rows dropped:', sorted(_ghost))
s = s.drop(index=list(_ghost))
s['is_sack'] = (s.sack == 1)
s['is_att'] = (s.pass_attempt == 1)
s['is_rush'] = (s.rush == 1) & ~s.is_sack
s['kneel'] = s.is_rush & s.play_text.str.contains(r'kneel', case=False, na=False)

# ---- who: passer / rusher from the player-stat fields, with a text fallback
s['qb'] = np.where(s.is_att, s.completion_player.fillna(s.incompletion_player).fillna(s.interception_thrown_player),
          np.where(s.is_sack, s.sack_taken_player, s.rush_player))
s['qb'] = s['qb'].where(s['qb'].notna(), None)

# roster of known names per offense, for resolving "#15 H.Geriner"-style text
known = {}
for col in ['completion_player','incompletion_player','interception_thrown_player','sack_taken_player','rush_player']:
    for t, n in s[['pos_team', col]].dropna().itertuples(index=False):
        known.setdefault(t, set()).add(n)
SUF = {'jr','jr.','sr','sr.','ii','iii','iv','v'}
def last_tokens(full):
    toks = [x for x in full.replace('.', ' ').split() if x.lower().strip('.') not in SUF]
    return toks
def norm(x): return re.sub(r'[^a-z]', '', x.lower())
def resolve(team, text, verb_re):
    if not isinstance(text, str): return None
    names = known.get(team, set())
    # abbreviated: "#15 H.Geriner pass", "#16 J.Overton, Jr. rush", "#9 T.St. Clair pass", "Av. Johnson pass", "T. Reynolds sacked"
    for mm in re.finditer(r'(?:^|[\s#\d)-])([A-Z][a-z]?)\.\s?((?:St\.\s?)?[A-Z][A-Za-z\'\-]+(?:\s[A-Z][A-Za-z\'\-]+){0,2}?)(?:,?\s(?:Jr\.?|Sr\.?|II|III|IV))?\s+' + verb_re, text):
        ini, last = mm.group(1), norm(mm.group(2))
        c = [n for n in names if n.startswith(ini) and norm(' '.join(last_tokens(n)[1:])).endswith(last)]
        if len(c) == 1: return c[0]
    # full name at start of text: "Keelon Russell run for ..."
    mm = re.match(r'^(?:\(\d+:\d+\)\s*)?([A-Z][A-Za-z\'\-\.]+(?:\s[A-Z][A-Za-z\'\-\.]+){1,3})\s+' + verb_re, text)
    if mm and mm.group(1) in names: return mm.group(1)
    return None

miss = s.qb.isna()
fill = []
for idx, r in s[miss].iterrows():
    verb = r'(?:pass|sacked)' if (r.is_att or r.is_sack) else r'(?:rush|run|scramble)'
    fill.append((idx, resolve(r.pos_team, r.play_text, verb)))
for idx, n in fill:
    if n: s.at[idx, 'qb'] = n
# dropbacks with no passer in the text at all (e.g. "DJ McKinney 55 Yd Interception Return"): use the offense's passer on the nearest dropback in the same game
s['ord'] = s.index
db = (s.is_att | s.is_sack)
nopat = ~s.play_text.str.contains(r'\b(?:pass|sacked)\b', case=False, na=False)
cand = s[db & s.qb.notna()][['game_id','pos_team','ord','qb']]
for idx, r in s[db & s.qb.isna() & nopat].iterrows():
    c = cand[(cand.game_id == r.game_id) & (cand.pos_team == r.pos_team)]
    if len(c): s.at[idx, 'qb'] = c.loc[(c.ord - idx).abs().idxmin(), 'qb']
print('attribution missing before', int(miss.sum()), 'after', int(s.qb.isna().sum()), 'of', len(s))

# ---- outcomes
yg = s.yards_gained.fillna(0)
s['cmp'] = (s.completion == 1) & s.is_att
cy = s.completion_yds
bad = cy.isna() | ((cy == -s.yards_gained) & (s.yards_gained < 0)) | s.play_text.str.contains('nullified', case=False, na=False)
s['pyds'] = np.where(s.cmp, np.where(bad, yg, cy), 0.0)   # completion_yds drops the sign on losses and ignores spot fouls
s['ryds'] = np.where(s.is_rush, s.rush_yds.fillna(yg), 0.0)
s['sky'] = np.where(s.is_sack, -yg.clip(upper=0), 0.0)
s['ptd'] = (s.pass_td == 1) & s.is_att
s['rtd'] = (s.rush_td == 1) & s.is_rush
s['intc'] = (s['int'] == 1) & s.is_att
s['xp'] = s.cmp & (s.pyds >= 20)
s['xr'] = s.is_rush & (s.ryds >= 10)
s['stf'] = s.is_rush & (s.ryds <= 0)
gain = np.where(s.is_att, s.pyds, np.where(s.is_rush, s.ryds, yg))
td = s.ptd | s.rtd
thr = np.select([s.down == 1, s.down == 2, s.down >= 3], [0.5 * s.distance, 0.7 * s.distance, s.distance], np.nan)
to = s.intc | (s.fumble_vec == 1) & (s.play_type.str.contains('Opponent', na=False))
s['succ'] = np.where(s.down.isna() | s.distance.isna(), np.nan, ((gain >= thr) | td) & ~to).astype(float)
s.loc[s.down.isna() | s.distance.isna(), 'succ'] = np.nan
s['late'] = s.down >= 3
s['conv'] = np.where(s.late, (((gain >= s.distance) | td) & ~to).astype(float), np.nan)
s['epa'] = s.EPA
m_ = s.pos_score_diff_start.abs()
s['garbage'] = ((s.period == 1) & (m_ > 43)) | ((s.period == 2) & (m_ > 37)) | ((s.period == 3) & (m_ > 27)) | ((s.period >= 4) & (m_ > 21))
s['off'] = s.pos_team; s['dfn'] = s.def_pos_team
s['homeflag'] = np.where(s.neutral_site == True, 0, np.where(s.pos_team == s.home, 1, np.where(s.pos_team == s.away, -1, 0)))
keep = ['game_id','week','off','dfn','home','away','homeflag','period','down','distance','qb','is_att','is_sack','is_rush','kneel',
        'cmp','pyds','ryds','sky','ptd','rtd','intc','xp','xr','stf','succ','late','conv','epa','garbage','play_text']
s[keep].to_parquet('plays.parquet')
import json; json.dump({'conf': conf, 'fbs': sorted(FBS)}, open('teams.json', 'w'))
print('plays', len(s), 'garbage share', s.garbage.mean().round(3), 'EPA null', s.epa.isna().mean().round(4), 'succ null', s.succ.isna().mean().round(4))
