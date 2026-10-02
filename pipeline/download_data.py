"""Step 0. Download raw data into data/. Existing files are kept; pass --force to re-download the current-season
files (play-by-play, game rosters, box scores, schedules, betting lines), which sportsdataverse updates through the season."""
import paths  # noqa: F401  (sets the working directory to work/)
import sys
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
    'betting_2026.parquet': f'{BASE}/espn_cfb_betting/betting_{SEASON}.parquet',           # Vegas lines (project.py, teams.py)
    'schedules_2025.parquet': f'{BASE}/cfb_schedules/cfb_schedules_2025.parquet',          # scores for the 2025 calibration (calibrate.py)
}
CURRENT = {'pbp.parquet', 'game_rosters_2026.parquet', 'player_box_2026.parquet', 'schedules_2026.parquet', 'betting_2026.parquet'}
force = '--force' in sys.argv[1:]
for name, url in FILES.items():
    out = Path('../data') / name
    if out.exists() and not (force and name in CURRENT):
        print(f'have {name}')
        continue
    print(f'downloading {name} ...', flush=True)
    urllib.request.urlretrieve(url, out)
    print(f'  {out.stat().st_size / 1e6:.1f} MB')
