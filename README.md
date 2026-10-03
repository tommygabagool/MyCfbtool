# CFB 2026: Power 4 teams and players

Opponent-adjusted ratings, game and player projections, and play-style archetypes for the 67 Power 4 programs and Notre Dame, built from cfbfastR play-by-play. The pipeline writes a small static site into `docs/`: a landing page and two parent pages, Teams and Players. Every page is a single self-contained HTML file with its data embedded, so the site works from GitHub Pages or straight from disk.

| Page | File | What's on it |
|---|---|---|
| Landing | `docs/index.html` | Links to Teams and Players, the data window and the model's track record. Old dashboard links (`index.html#qb`, `#rb/...`) redirect to the Players page. |
| Teams | `docs/teams/index.html` | Every Power 4 team and Notre Dame: a directory by conference with the coming week's slate; team profiles (model power rating, unit ranks, schedule with results and projections, roster with season totals and player profiles); and a game view with the projected score, win probability, Vegas line, unit matchups and each player's projected line (projected next to actual once the game is played). |
| Players | `docs/players/index.html` | The QB, RB, WR and TE dashboard: an opponent-adjusted/raw toggle, archetype and conference filters, three scatter views drawn on a football field, a scouting card (key stats against the position average, style fingerprint, yardage profile, game log with each opponent's defensive rank, a link to his team's page), leaderboards, an archetype mix by conference, and a sortable full board. |

The pages share a header that links them. Routes live in the URL hash, so any view can be linked: `teams/index.html#team/Ohio%20State`, `teams/index.html#game/<game id>`, `teams/index.html#how` (how the projections work), `players/index.html#wr` or `players/index.html#wr/<player key>` (the key is the athlete id, or the name for quarterbacks).

## Quick start

```bash
python3 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
./run_all.sh            # first run downloads about 304 MB of data, then builds the three pages in docs/
```

Open `docs/index.html`, the landing page, in a browser. Links between pages are relative paths to explicit `index.html` files, so the site works from `file://` as well as from a server. To refresh during the season, run `./run_all.sh --force`, which re-downloads this season's files and rebuilds everything; the page text (week, date, counts, track record) updates from the data.

To host it, commit `docs/` (`index.html`, `players/index.html`, `teams/index.html`) and turn on GitHub Pages under Settings > Pages > Deploy from a branch, choosing `main` and `/docs`.

## Pipeline

All scripts live in `pipeline/` and run from `work/` (handled by `paths.py`). Raw downloads go to `data/`, intermediate files to `work/`, the pages to `docs/`. `run_all.sh` runs them in this order.

| Step | Script | What it does | Writes |
|---|---|---|---|
| 0 | `download_data.py` | Fetches play-by-play (2024 to 2026), ESPN game rosters and box scores, schedules (2025 and 2026) and 2026 betting lines, plus ESPN's current lines for games not yet played | `data/*.parquet`, `data/espn_lines_2026.json` |
| 1 | `prep.py` | Scrimmage plays, outcome flags, QB attribution, garbage time, source-data fixes | `plays.parquet`, `teams.json` |
| 2 | `seasons.py` | Ratings for 2024 and 2025 and their year-to-year carryover, overall and split into conference average and standing within the conference | `seasons.pkl`, `persistence.csv`, `persistence_conf.csv` |
| 3 | `fit2026.py` | 2026 offense and defense ratings for every metric, seeded by regressed 2025 ratings | `def_effects.json`, `off_effects.json` |
| 3b | `calibrate.py` | Replays 2025 week by week to fit the game model's points scale, league-tier terms and margin spread | `calib.json` |
| 4 | `qbstats.py` | Raw and opponent-adjusted stats for every FBS starting QB | `plays_adj.parquet`, `qb_all.pkl` |
| 5 | `arch.py` | QB archetype scores | `qb_arch.pkl` |
| 6 | `dataset.py` | Power 4 + Notre Dame QBs with game logs | `qb_data.json` |
| 7 | `prep_skill.py` | Rusher and receiver identity, roster positions and bios | `plays_skill.parquet`, `players.pkl` |
| 8 | `skill.py` | RB, WR and TE stats, archetypes and game logs, merged with QBs and page metadata | `all_data.json` |
| 10 | `project.py` | Projected score and player lines for every Power 4 / Notre Dame game: true pregame projections for played games from Week 2 on, and the backtest against Vegas lines and a season-average baseline | `proj.pkl` |
| 11 | `teams.py` | Teams-page data: team ratings and ranks, records, schedules, rosters with season totals, and each game's box score and projections | `teams_data.json` |
| 12 | `build.py` | Fills the three templates (`home_template.html`, `players_template.html`, `teams_template.html`) with the data | `docs/index.html`, `docs/players/index.html`, `docs/teams/index.html` |
| opt. | `validate.py` | Game-by-game match rates against ESPN box scores | (prints) |

`common.py` holds the shared outcome definitions, the metric list, the rating model and the game-points formula used by both `calibrate.py` and `project.py`.

`paths.py` also pins BLAS to one thread (`OPENBLAS_NUM_THREADS`, `OMP_NUM_THREADS` and `MKL_NUM_THREADS`, unless already set): the rating fits solve small dense systems, and multithreaded OpenBLAS made the ridge fits about 40 times slower on many-core machines. Game dates are formatted with `f"{d:%b} {d.day}"` rather than `strftime('%b %-d')`, which Windows doesn't support, so the pipeline runs unchanged on Windows.

## Method

**Opponent adjustment.** For each metric (EPA per play, success rate, yards per attempt, completion rate, explosive plays, stuff rate, EPA per target and so on), a ridge regression rates every offense and every defense, plus home field, from all 2026 plays. Blowout plays are left out of the fit (a margin of more than 43 points in the 1st quarter, 37 in the 2nd, 27 in the 3rd, 21 in the 4th). Each FBS defense starts from its 2025 rating scaled by how much defensive ratings carried over from 2024 to 2025, so early-season ratings are pulled well back toward average. FCS teams are rated individually, starting from the FCS average. A player's adjusted number subtracts, play by play, what the defense on the other side typically allows above or below an average FBS defense, so adjusted totals estimate his output against an average FBS defense on the same plays. Running backs' carries use run-defense ratings; targets use pass-defense ratings. Schedule faced is the average defensive rating on his plays.

**Who's included.**
- QB: the passer with the most attempts in his team's latest game.
- RB: each team's lead back by carries, plus any back averaging 6+ carries per team game.
- WR: each team's most-targeted wide receiver, plus any averaging 3+ targets per team game.
- TE: each team's most-targeted tight end with 4+ targets, plus any averaging 2+ targets per team game.

**Archetypes.** Four style scores per player, in standard deviations from the FBS average at his position (pools: QBs with 15+ attempts, RBs with 20+ carries, WRs with 10+ targets, TEs with 6+ targets). Performance inputs are opponent-adjusted, usage inputs are raw, and rate stats are shrunk toward the average for low-volume players. The top score is the archetype; a second score within 0.25 makes him a hybrid; a top score at or below +0.2 is a weak fit.

| Position | Archetypes |
|---|---|
| QB | Field General, Gunslinger, Pure Runner, Pocket Passer |
| RB | Workhorse, Home-Run Hitter, Receiving Back, Grinder |
| WR | Alpha, Deep Threat, Red-Zone Threat, Possession |
| TE | Receiving Weapon, Seam Stretcher, Red-Zone Target, Safety Valve |

The inputs behind each archetype are listed in the page footer for each tab and in `arch.py` / `skill.py`.

## Projections

The Teams page projects every Power 4 and Notre Dame game: each team's points, the win probability, and every player's passing, rushing and receiving line. Its footer explains the method for readers; this is the same thing with the constants.

**Team ratings.** The same ridge regression as the Players page rates every offense and defense, plus home field, on EPA per play and the other metrics, with blowout plays left out. For projections each team starts from its 2025 rating split two ways: its conference's average, which carries over strongly from year to year, and its standing within the conference, which carries over much less. Both carryover rates are measured from 2024 to 2025 (`seasons.py`, `persistence_conf.csv`).

**Projected score.** For each team:

- plays = its offensive pace + the opponent's pace allowed − the league average (paces are scrimmage plays per game, shrunk two games toward average);
- net EPA edge = its offense rating + the opponent's defense rating + home field (zero at a neutral site);
- points = plays × (league points per play + scale × edge) + tier term.

Independents count as having an FBS-average conference, so their whole 2025 rating is treated as standing (otherwise Notre Dame would carry over nearly all of last season).

The scale converts EPA-per-play ratings to points per play (more than 1 because ridge ratings are shrunk). The tier terms correct for scrimmage EPA misjudging the gaps between leagues; each is split evenly between the two scores so the projected total doesn't change. Win probability treats the actual margin as normally distributed around the projected margin. League points per play and plays per game come from FBS-vs-FBS games only. A team's projected points are the expected value of a score that can't go below zero (team scores vary by about 12 points), so a big mismatch projects a few points for the underdog rather than a shutout.

These constants start from `calibrate.py`, which replays the 2025 regular season week by week (782 games involving an FBS team, ratings built only from earlier weeks): scale 1.20, Power 4 term 9.8 points, FCS term 6.7, margin standard deviation 16.6. `project.py` then updates them walk-forward with this season's completed games involving an FBS team, the 2025 values counting as 100 games: the projections for Week *w* use only games before Week *w*. This matters because cfbfastR's expected-points model leans on the pregame spread, and much more so in 2026 than in 2025, so 2026 EPA understates mismatches that 2025 constants don't account for. Through Week 4 the updated constants are scale 1.46, Power 4 term 12.9, FCS term −3.1, standard deviation 17.0. A team's model power on the Teams page is its projected margin against an average FBS team on a neutral field.

**Player projections.** Team volume comes first: dropbacks are plays × the team's dropback rate (shrunk 120 plays toward the league rate), sacks come from the league sack rate plus both teams' sack ratings, and the remaining plays are carries. Each player gets his share of his team's pass attempts, carries and targets, with recent games counting more (a game's weight halves every two team games back) and a game he missed counting as zero share. His efficiency is his opponent-adjusted rate (his result on each play minus what that defense typically allows), pulled toward the FBS average for his position when the sample is small (the pull is worth 25 targets for catch rate and up to 250 attempts for interception rate), plus the opponent's defensive rating and home field. Receivers' catches, yards and touchdowns are then reconciled with the quarterbacks' by meeting in the middle, so both sides of the passing game add up.

**Pregame, not hindsight.** For every played game from Week 2 on, ratings, usage shares and player rates are rebuilt from the weeks before that game only, so projected-vs-actual on the page is a real forecast. Week 1 games have no projection, because the model needs a week of 2026 games. Upcoming games use everything to date.

**Track record.** Through Week 4 of 2026, across the 135 Power 4 / Notre Dame games in Weeks 2 to 4 (all of which had a Vegas line):

| | Model | Vegas |
|---|---|---|
| Winners picked | 82.2% | 88.1% |
| Margin error (mean absolute, points) | 14.3 | 10.6 |
| Total-points error (mean absolute) | 12.7 | 11.5 |

Vegas is clearly more accurate on games; the model's value is that it projects every player. Its win probabilities have been about right: when it made the favorite 50 to 65% likely to win, favorites won 56% (34 games, 58% expected); 65 to 80%, 74% (27 games, 72% expected); 80 to 90%, 89% (19 games, 86% expected); 90% or more, 100% (55 games, 95% expected).

Player projections, compared with a naive guess (each player's season-to-date average per game played), for Power 4 and Notre Dame players who played; errors are mean absolute:

| Stat | Player-games | Model | Season average | Correlation with actual |
|---|---|---|---|---|
| QB passing yards (projected 15+ attempts) | 190 | 67.3 | 74.9 | 0.25 |
| Rushing yards (projected 8+ carries) | 254 | 33.6 | 37.0 | 0.33 |
| Receiving yards (projected 3+ targets) | 603 | 30.0 | 33.9 | 0.37 |
| Receptions (projected 3+ targets) | 603 | 1.9 | 2.0 | 0.35 |

The usage half-life, shrinkage amounts and the 100-game weight on the 2025 constants were set by hand, not tuned on 2026 results.

**Limits.** No injury reports or depth-chart news (a player who missed recent games is projected at a reduced share); no projection for Week 1; FCS opponents have little data; special teams and turnovers aren't modeled separately; projections are averages, not ceilings.

## Fixes applied to the source data

1. `completion_yds` drops the minus sign on catches behind the line and ignores spot fouls; those plays use `yards_gained` instead (about 640 plays through Week 4 of 2026).
2. Fumbled runs and catches arrive flagged as neither a run nor a pass; they're put back (about 615 plays).
3. ESPN sometimes repeats a play, stamped (00:00), just before the real one; the copies are dropped.
4. Many plays have no player IDs. Names in the play text are matched to ESPN game rosters by initial and last name, using jersey number and that game's roster to break ties.

Rushing yards exclude sacks, and kneel-downs are dropped.

## Validation

`python pipeline/validate.py` compares game totals with ESPN box scores. Through Week 4 of 2026: receptions match exactly in 98.5% of receiver-games and receiving yards in 97.9%; carries in 96.3% and rushing yards in 96.2% of non-QB rusher-games; passing yards in 92.3% of QB-games.

## Limitations

- No PFF grades, ESPN QBR or Strength of Record.
- College play-by-play has no air yards, yards after catch, drops, routes or blocking, so receiving archetypes describe production and usage only, and blocking never counts for backs or tight ends.
- Targets on interceptions aren't credited to a receiver, because the play text rarely names one.
- Early-season samples are small; adjustments are modest by design and grow as ratings firm up.

## Adapting to another season

Years are hardcoded in:

- `download_data.py`: the season, the prior-season play-by-play and schedule file names, and the local file names (including `espn_lines_2026.json`, also read by `teams.py`);
- `seasons.py`: the two prior seasons;
- `fit2026.py` and `project.py`: the 2025 seed (`season_prior`'s default base year in `common.py`);
- `project.py`: postseason games are placed after the regular season (play-by-play labels them week 1); bowls and playoff games all share that one slot;
- `calibrate.py`: the replayed season (2025, from `pbp_2025.parquet` and `schedules_2025.parquet`) and its 2024 starting ratings;
- the `game_rosters_2026` / `player_box_2026` file names in `prep_skill.py`, `validate.py` and `teams.py`, and the `schedules_2026` / `betting_2026` file names in `project.py` and `teams.py`;
- the "2026" text in the three templates (page titles, the "CFB 2026" brand in the shared header, the Players page's "2026 season" copy).

Conference membership comes from the play-by-play.

## Data sources

- Play-by-play: [cfbfastR](https://github.com/sportsdataverse/cfbfastR) data via the [sportsdataverse-data](https://github.com/sportsdataverse/sportsdataverse-data) releases.
- Game rosters and player box scores: ESPN data via the same releases.
- Schedules (dates, venues, scores) and betting lines: the same releases' `cfb_schedules` and `espn_cfb_betting` files. The betting file only adds a game once it's final, so lines for games not yet played come from ESPN's public scoreboard API (`site.api.espn.com/.../college-football/scoreboard`, the line it shows for each FBS game). ESPN removes a game's line once it's final; `download_data.py` keeps the lines it saved on earlier runs, so a just-finished game keeps a line until the betting file picks it up. The betting file's line wins wherever both have one, and the backtest uses only the betting file.

Raw data isn't committed (`data/` and `work/` are in `.gitignore`); check the sources' terms before redistributing it.

## Requirements

Python 3 with pandas, numpy, pyarrow and scipy. Tested on Python 3.12.3 with pandas 3.0.2, numpy 2.4.4, pyarrow 25.0.1 and scipy 1.17.1. Headless-browser screenshots aren't part of the pipeline.

No license file is included; add one before making the repo public if you want others to reuse the code.
