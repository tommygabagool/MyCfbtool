"""Step 0. Download raw data into data/. Existing files are kept; pass --force to re-download the current-season
files (play-by-play, game rosters, box scores, schedules, betting lines), which sportsdataverse updates through the season.
The betting file only picks a game up once it's final, so lines for games not yet played come from ESPN's scoreboard."""
import paths  # noqa: F401  (sets the working directory to work/)
import json
import sys
import urllib.parse
import urllib.request
from pathlib import Path

SEASON = 2026
BASE = 'https://github.com/sportsdataverse/sportsdataverse-data/releases/download'
FILES = {  # local name -> source
    'pbp.parquet': f'{BASE}/cfbfastR_cfb_pbp/play_by_play_{SEASON}.parquet',
    'pbp_2025.parquet': f'{BASE}/cfbfastR_cfb_pbp/play_by_play_2025.parquet',
    'pbp_2024.parquet': f'{BASE}/cfbfastR_cfb_pbp/play_by_play_2024.parquet',
    'game_rosters_2026.parquet': f'{BASE}/espn_cfb_game_rosters/game_rosters_{SEASON}.parquet',
    'player_box_2026.parquet': f'{BASE}/espn_cfb_player_box/player_box_{SEASON}.parquet',  # only used by validate.py
    'schedules_2026.parquet': f'{BASE}/cfb_schedules/cfb_schedules_{SEASON}.parquet',     # dates, venues, scores (project.py, teams.py)
    'betting_2026.parquet': f'{BASE}/espn_cfb_betting/betting_{SEASON}.parquet',           # Vegas lines for played games (project.py, teams.py)
    'schedules_2025.parquet': f'{BASE}/cfb_schedules/cfb_schedules_2025.parquet',          # scores for the 2025 calibration (calibrate.py)
}
CURRENT = {'pbp.parquet', 'game_rosters_2026.parquet', 'player_box_2026.parquet', 'schedules_2026.parquet', 'betting_2026.parquet'}
ESPN = 'https://site.api.espn.com/apis/site/v2/sports/football/college-football/scoreboard'
LINES = Path('../data/espn_lines_2026.json')  # Vegas lines for games not yet played (teams.py)


def espn_lines():
    """Spread (home team's, negative when the home team is favored, as in the betting file) and over/under for every
    FBS game ESPN posts a line for. ESPN drops a game's line once it's final, so lines saved by earlier runs are kept."""
    get = lambda **q: json.load(urllib.request.urlopen(f'{ESPN}?' + urllib.parse.urlencode(dict(groups=80, limit=300, dates=SEASON, **q))))
    lines = json.loads(LINES.read_text()) if LINES.exists() else {}
    n = 0
    for part in get(seasontype=2, week=1)['leagues'][0]['calendar']:
        if part['value'] not in ('2', '3'):  # regular season and postseason
            continue
        for wk in part['entries']:
            for e in get(seasontype=part['value'], week=wk['value'])['events']:
                o = (e['competitions'][0].get('odds') or [{}])[0]
                if o.get('spread') is None:
                    continue
                lines[e['id']] = dict(sp=float(o['spread']), ou=float(o['overUnder']) if o.get('overUnder') is not None else None)
                n += 1
    LINES.write_text(json.dumps(lines, sort_keys=True))
    print(f'  {n} games with a current line, {len(lines)} saved')


force = '--force' in sys.argv[1:]
for name, url in FILES.items():
    out = Path('../data') / name
    if out.exists() and not (force and name in CURRENT):
        print(f'have {name}')
        continue
    print(f'downloading {name} ...', flush=True)
    urllib.request.urlretrieve(url, out)
    print(f'  {out.stat().st_size / 1e6:.1f} MB')
if LINES.exists() and not force:
    print(f'have {LINES.name}')
else:
    print(f'fetching upcoming lines from ESPN into {LINES.name} ...', flush=True)
    try:
        espn_lines()
    except (OSError, ValueError, KeyError) as err:  # a supplementary source: keep building without it
        print(f'  skipped ({err!r}); games not yet played will show no Vegas line')
