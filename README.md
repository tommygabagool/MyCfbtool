# CFB 2026: Power 4 teams and players

Opponent-adjusted ratings, game and player projections, and play-style archetypes for the 67 Power 4 programs and Notre Dame, built from cfbfastR play-by-play. The pipeline writes a small static site into `docs/`: a landing page and two parent pages, Teams and Players. Every page is a single self-contained HTML file with its data embedded, so the site works from GitHub Pages or straight from disk.

| Page | File | What's on it |
|---|---|---|
| Landing | `docs/index.html` | Links to Teams and Players, the data window and the model's track record. Old dashboard links (`index.html#qb`, `#rb/...`) redirect to the Players page. |
| Teams | `docs/teams/index.html` | Every Power 4 team and Notre Dame: a directory by conference with the coming week's slate; team profiles (model power rating, unit ranks, schedule with results and projections, roster with season totals and player profiles); and a game view with the projected score, win probability, Vegas line, unit matchups and each player's projected line (projected next to actual once the game is played). Tapping a player opens his matchup: how his projection is built against that defense (a waterfall from the FBS average), his rates against the defenses he has faced, and his game-by-game rate plotted against each opponent's defensive rank. |
| Players | `docs/players/index.html` | The QB, RB, WR and TE dashboard: an opponent-adjusted/raw toggle, archetype and conference filters, three scatter views drawn on a football field, a scouting card (key stats against the position average, style fingerprint, yardage profile, game log with each opponent's defensive rank, a link to his team's page), leaderboards, an archetype mix by conference, and a sortable full board. |

The pages share a header that links them. Routes live in the URL hash, so any view can be linked: `teams/index.html#team/Ohio%20State`, `teams/index.html#game/<game id>`, `teams/index.html#game/<game id>/<player id>` (that player's matchup), `teams/index.html#how` (how the projections work), `players/index.html#wr` or `players/index.html#wr/<player key>` (the key is the athlete id, or the name for quarterbacks).

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
| 0 | `download_data.py` | Fetches play-by-play (2024 to 2026), ESPN game rosters (2025 and 2026) and box scores, schedules (2025 and 2026), 2026 betting lines, roster talent (2025 and 2026), plus ESPN's current lines for games not yet played | `data/*.parquet`, `data/espn_lines_2026.json` |
| 1 | `prep.py` | Scrimmage plays, outcome flags, QB attribution, garbage time, source-data fixes | `plays.parquet`, `teams.json` |
| 2 | `seasons.py` | Ratings for 2024 and 2025 (every play metric and points per drive) and their year-to-year carryover, overall and split into conference average, standing within the conference and roster talent | `seasons.pkl`, `persistence.csv`, `persistence_conf.csv` |
| 3 | `fit2026.py` | 2026 offense and defense ratings for every metric, seeded by regressed 2025 ratings | `def_effects.json`, `off_effects.json` |
| 3b | `calibrate.py` | Replays 2025 week by week to fit the game model: how much the EPA, points-per-drive and success-rate edges count toward the margin and the total, league-tier and home terms, and the margin spread | `calib.json` |
| 3c | `calibrate_players.py` | Replays 2025 week by week to fit the player model: team plays and pass rate (pace, game script), how a player's volume blends his recent share with his per-game average (and shrinks in blowouts), and how much of his schedule-adjusted edge and of the opponent's defensive rating each stat's rate keeps | `calib_players.json` |
| 4 | `qbstats.py` | Raw and opponent-adjusted stats for every FBS starting QB | `plays_adj.parquet`, `qb_all.pkl` |
| 5 | `arch.py` | QB archetype scores | `qb_arch.pkl` |
| 6 | `dataset.py` | Power 4 + Notre Dame QBs with game logs | `qb_data.json` |
| 7 | `prep_skill.py` | Rusher and receiver identity, roster positions and bios | `plays_skill.parquet`, `players.pkl` |
| 8 | `skill.py` | RB, WR and TE stats, archetypes and game logs, merged with QBs and page metadata | `all_data.json` |
| 10 | `project.py` | Projected score and player lines for every Power 4 / Notre Dame game: true pregame projections for played games from Week 2 on, and the backtest against Vegas lines and a season-average baseline | `proj.pkl` |
| 11 | `teams.py` | Teams-page data: team ratings and ranks, records, schedules, rosters with season totals, each game's box score and projections, and the matchup inputs (every projected player's schedule-adjusted rates, every defense's ratings and FBS ranks, the fitted weights) | `teams_data.json` |
| 12 | `build.py` | Fills the three templates (`home_template.html`, `players_template.html`, `teams_template.html`) with the data | `docs/index.html`, `docs/players/index.html`, `docs/teams/index.html` |
| opt. | `validate.py` | Game-by-game match rates against ESPN box scores | (prints) |

`common.py` holds the shared outcome definitions, the metric list, the rating model (play metrics and points per drive), the preseason prior and the game model used by `calibrate.py`, `calibrate_players.py` and `project.py`. `pmodel.py` holds the player model (team volume, usage shares, schedule-adjusted rates and the fitted combination) shared by `calibrate_players.py` and `project.py`, so the 2025 fit and the 2026 projections run the same code.

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

The Teams page projects every Power 4 and Notre Dame game: each team's points, the win probability, and every player's passing, rushing and receiving line. Its footer explains the method for readers, and tapping a player in a game shows how his projection was built against that defense; this is the same thing with the constants.

**Team ratings.** The same ridge regression as the Players page rates every offense and defense, plus home field, on EPA per play, success rate and the other play metrics, and, from drives, on points per drive (7 for a touchdown drive, 3 for a field goal; drives that begin in garbage time or end by running out the clock are left out). For projections each team starts from its 2025 rating split three ways: its conference's average, its standing within the conference, and its 2026 roster talent (the 247Sports composite, as a z-score among FBS teams). How much each part carries over is measured from 2024 to 2025 (`seasons.py`, `persistence_conf.csv`); for defenses, talent takes over most of what the conference average used to carry. That starting point is then held four times more loosely than the year-to-year spread implies (`LOOSEN` in `common.py`), so this season's games take over quickly. Independents count as having an FBS-average conference, so their whole 2025 rating is treated as standing.

**Projected score.** Each team's edge is measured three ways, each scaled to the game:

- E = expected plays × (its offense's EPA-per-play rating + the opponent defense's rating + home field);
- Pd = expected drives × the same on points per drive;
- Sr = expected plays × the same on success rate.

Expected plays and drives come from both teams' pace (shrunk two games toward average). The margin is the plays difference × league points per play, plus weighted home-minus-away differences in E, Pd and Sr, a Power 4 term, an FCS term and a home term (the edges already include home field; the home term trims the total home edge to what games show). The total is a weighted sum of both teams' plays × league points per play, E, Pd, Sr and a constant. Each team's points are (total ± margin) / 2, as the expected value of a score that can't go below zero (team scores vary by about 12 points), and win probability treats the actual margin as normally distributed around the projection. A team's model power is its projected margin against an average FBS team on a neutral field.

The weights start from `calibrate.py`, which replays the 2025 regular season week by week (792 games involving an FBS team, ratings built only from earlier weeks): margin weights E 0.08, Pd 0.67, Sr 0.21, Power 4 +6.8, FCS −5.6, home −1.5; standard deviation 15.7. That replay's margin error is 12.4 points (the previous EPA-only model's: 13.2) and its total error 12.5 (13.2). `project.py` then updates the weights walk-forward with this season's completed games involving an FBS team, the 2025 values counting as 100 games, so the projections for Week *w* use only games before Week *w*. Through Week 4 (232 games): E 0.07, Pd 0.63, Sr 0.22, Power 4 +9.1, FCS −6.5, home −3.2; standard deviation 16.0. Points per drive carries most of the weight.

**Player projections** (`pmodel.py`, weights from `calibrate_players.py`, fit on the 2025 season replayed week by week).

- *Team volume.* Plays = 33.1 + 0.405 × pace plays + 0.12 × the projected total. Pass rate = −0.043 + 1.073 × the team's dropback rate (shrunk 120 plays toward the league rate) − 0.0014 × its projected margin, so a team projected to lead by 10 passes about 1.4 points less often. Sacks come from the league sack rate plus both teams' sack ratings; pass attempts are dropbacks minus sacks, carries are the remaining plays, and targets are attempts × the team's credited-target rate.
- *Player volume.* His recent share of the team's attempts, carries or targets (a game's weight halves every 2 team games back for attempts and carries, every 3 for targets; a game he missed counts as zero share) × team volume, blended with his average per game played, minus a blowout term: pass attempts 0.66 × share-based + 0.41 × per-game average − 0.058 × share-based × |projected margin| / 14; carries 0.81, 0.27, −0.115; targets 0.59, 0.47, −0.076. A player who missed both of his team's last two games keeps only the share-based part.
- *Rate per play,* for each stat: the FBS average for his position group + *a* × his edge over it + *b* × the opponent's defensive rating (plus home field) + a small constant. His edge uses his rate so far adjusted for the defenses he faced (each play's result minus what that defense allows above an average FBS defense), pulled toward the group average with 40 plays of average evidence. Fitted (*a*, *b*): completion rate (0.65, 0.74), yards per attempt (0.62, 0.81), TD rate (0.35, 0.84), interception rate (−0.09, 0.85), yards per carry (0.54, 0.80), rushing TD rate (0.43, 1.13), catch rate (0.72, 0.60), yards per target (0.48, 0.64), receiving TD rate (0.37, 0.85). A defense counts for less than its full rating because ratings built from a few games are noisy; a player keeps about half his edge because hot starts regress.

Passers' and receivers' lines aren't forced to agree game by game; across games they match on average (both sides of the passing game come to about 238 projected yards per team-game in the 2025 replay), and forcing them to agree made both less accurate.

**Pregame, not hindsight.** For every played game from Week 2 on, ratings, usage shares and player rates are rebuilt from the weeks before that game only, so projected-vs-actual on the page is a real forecast. Week 1 games have no projection, because the model needs a week of 2026 games. Upcoming games use everything to date.

**Track record.** Through Week 4 of 2026, across the 135 Power 4 / Notre Dame games in Weeks 2 to 4 (all of which had a Vegas line):

| | Model | Previous model | Vegas |
|---|---|---|---|
| Winners picked | 82.2% | 82.2% | 88.1% |
| Margin error (mean absolute, points) | 12.4 | 14.3 | 10.6 |
| Total-points error (mean absolute) | 11.7 | 12.7 | 11.5 |

Vegas is still more accurate on margins; the model is now close on totals, and its value is that it projects every player. Projected margins are now calibrated (regressing actual on projected margin gives a slope of 1.02; the previous model's was 1.22, meaning it understated mismatches). Win probabilities: when it made the favorite 50 to 65% likely to win, favorites won 38% (24 games, 58% expected); 65 to 80%, 79% (28 games, 73% expected); 80 to 90%, 94% (17 games, 86% expected); 90% or more, 97% (66 games, 97% expected).

Player projections, compared with a naive guess (each player's season-to-date average per game played), for Power 4 and Notre Dame players who played; errors are mean absolute:

| Stat | Player-games | Model | Season average | Correlation with actual |
|---|---|---|---|---|
| QB passing yards (projected 15+ attempts) | 188 | 66.8 | 75.3 | 0.22 |
| Rushing yards (projected 8+ carries) | 254 | 32.6 | 37.3 | 0.36 |
| Receiving yards (projected 3+ targets) | 565 | 28.7 | 34.0 | 0.37 |
| Receptions (projected 3+ targets) | 565 | 1.8 | 2.1 | 0.38 |
| QB passing TDs (projected 15+ attempts) | 188 | 1.04 | 1.36 | 0.22 |
| Rushing TDs (projected 8+ carries) | 254 | 0.67 | 0.80 | 0.22 |
| Receiving TDs (projected 3+ targets) | 565 | 0.48 | 0.53 | 0.22 |

On exactly the same player-games, the previous model's errors were 67.8 (passing yards), 33.2 (rushing yards), 29.6 (receiving yards), 1.92 (receptions), 1.03, 0.67 and 0.48 (TDs); this one's 66.8, 32.1, 28.1, 1.77, 1.05, 0.67 and 0.46.

**Limits.** No injury reports or depth-chart news (ESPN's injury feed in the same releases is too sparse to use), so a player who missed recent games is projected at a reduced share; no projection for Week 1; FCS opponents have little data; special teams and turnovers aren't modeled separately; projections are averages, not ceilings.

## Research behind the model

The game and player models were rebuilt in October 2026 from a replay study. Every candidate was scored out of sample on games it never saw, against Vegas closing lines where they exist.

**Test bed.** The 2024 and 2025 seasons and 2026 to date, each replayed week by week from earlier games only, starting from the prior season's ratings: 1,888 games involving an FBS team, about 1,700 with a Vegas line. Game models were fit on two seasons and tested on the third. Player models were replayed over all of 2025 and 2026 to date (about 37,000 player-team-games) and scored with five-fold cross-validation grouped by game.

**Score model** (mean absolute error in points across all 1,888 games; Vegas on the same games: 11.65 margin, 12.37 total):

| Variant | Margin | Total |
|---|---|---|
| Previous model: EPA per play only | 13.51 | 12.79 |
| + points per drive | 12.91 | 12.67 |
| + success rate | 12.85 | 12.66 |
| + roster talent in the preseason prior | 12.81 | 12.66 |
| + prior held four times more loosely (final) | 12.72 | 12.64 |

What didn't help: havoc rate (tackles for loss, sacks, pass breakups, takeaways; ESPN's advanced defensive box score has it), weighting recent games more (every week of the season counting equally was best), returning production, a direct talent term in the game equation, a robust (Huber) fit, and refitting on FBS-vs-FBS games only. Refitting the weights in season gained about 0.05 points, which the walk-forward update already does. Blending the model with Vegas improved on Vegas by 0.01 points, so the model holds almost no information the line doesn't already have; it doesn't beat the spread (about 50% against it across the three seasons). For 2026 Power 4 games, the rebuilt model's margin error fell from 14.6 to about 12.5.

**Player model.** Out of sample over 2025 and 2026 to date (all FBS players with an established role, mean absolute error):

| Stat | Season average | Previous model | Final model |
|---|---|---|---|
| Passing yards | 72.4 | 70.8 | 69.0 |
| Rushing yards | 34.1 | 31.0 | 30.7 |
| Receiving yards | 27.1 | 25.1 | 24.4 |
| Receptions | 1.70 | 1.66 | 1.58 |
| Passing TDs | 1.10 | 0.97 | 0.95 |
| Rushing TDs | 0.64 | 0.59 | 0.58 |
| Receiving TDs | 0.37 | 0.36 | 0.35 |

What the replay showed:

- The previous model applied a defense's rating at full strength; the fits keep 60 to 85% of it (rushing touchdowns are the exception, at about 110%), and a player keeps roughly half of his schedule-adjusted edge in yards per play (more in completion and catch rate, about a third in touchdown rate).
- Blending a player's recent share with his plain per-game average, and shrinking projected plays toward the league average (actual plays move only about 40% as much as pace numbers suggest), both cut errors.
- Game script is real but small: a team projected to lead by 10 passes about 1.4 points less often. The extra carries a favorite gets in a blowout go to backups, which is what the blowout term captures; without it, game script made rushing projections worse.
- Forcing passers' and receivers' totals to agree game by game cost accuracy on both.

## Fixes applied to the source data

1. `completion_yds` drops the minus sign on catches behind the line and ignores spot fouls; those plays use `yards_gained` instead (about 640 plays through Week 4 of 2026).
2. Fumbled runs and catches arrive flagged as neither a run nor a pass; they're put back (about 615 plays).
3. ESPN sometimes repeats a play, stamped (00:00), just before the real one; the copies are dropped.
4. Many plays have no player IDs. Names in the play text are matched to ESPN game rosters by initial and last name, using jersey number and that game's roster to break ties.
5. In the 2025 play-by-play (used for calibration), most touchdown plays from Week 10 on came through with no participant ids at all: 30% of passing and 16% of rushing touchdowns for FBS offenses had them. `calibrate_players.py` fills them from the play text's jersey number and name ("#2 B.Shapen pass complete ... to #6 D.Booth") matched to that game's ESPN roster, requiring the jersey, initial and last name to agree; on late-season plays whose ids were known, it was right every time. Coverage rose to 90% and 86%. Without it, late-season touchdowns and long plays vanished from players' lines and the fitted rates came out too low.

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

- `download_data.py`: the season, the prior-season play-by-play, schedule and roster file names, the talent file names, and the local file names (including `espn_lines_2026.json`, also read by `teams.py`);
- `seasons.py`: the two prior seasons;
- `fit2026.py` and `project.py`: the 2025 seed (`season_prior`'s default base year in `common.py`);
- `project.py`: postseason games are placed after the regular season (play-by-play labels them week 1); bowls and playoff games all share that one slot;
- `calibrate.py` and `calibrate_players.py`: the replayed season (2025, from `pbp_2025.parquet`, `schedules_2025.parquet`, `game_rosters_2025.parquet` and `talent_2025.parquet`) and its 2024 starting ratings;
- `seasons.py`: `talent_2025.parquet` for the carryover fit; `project.py`: `talent_2026.parquet` for this season's starting ratings;
- the `game_rosters_2026` / `player_box_2026` file names in `prep_skill.py`, `validate.py` and `teams.py`, and the `schedules_2026` / `betting_2026` file names in `project.py` and `teams.py`;
- the "2026" text in the three templates (page titles, the "CFB 2026" brand in the shared header, the Players page's "2026 season" copy).

Conference membership comes from the play-by-play.

## Data sources

- Play-by-play: [cfbfastR](https://github.com/sportsdataverse/cfbfastR) data via the [sportsdataverse-data](https://github.com/sportsdataverse/sportsdataverse-data) releases.
- Game rosters and player box scores: ESPN data via the same releases.
- Schedules (dates, venues, scores) and betting lines: the same releases' `cfb_schedules` and `espn_cfb_betting` files.
- Roster talent: the same releases' `cfb_team_talent` files (the 247Sports team talent composite). The betting file only adds a game once it's final, so lines for games not yet played come from ESPN's public scoreboard API (`site.api.espn.com/.../college-football/scoreboard`, the line it shows for each FBS game). ESPN removes a game's line once it's final; `download_data.py` keeps the lines it saved on earlier runs, so a just-finished game keeps a line until the betting file picks it up. The betting file's line wins wherever both have one, and the backtest uses only the betting file.

Raw data isn't committed (`data/` and `work/` are in `.gitignore`); check the sources' terms before redistributing it.

## Requirements

Python 3 with pandas, numpy, pyarrow and scipy. Tested on Python 3.12.3 with pandas 3.0.2, numpy 2.4.4, pyarrow 25.0.1 and scipy 1.17.1, and on Python 3.14.3 (Windows) with pandas 3.0.6, numpy 2.5.3, pyarrow 25.0.1 and scipy 1.18.1. A full run takes about ten minutes after the first download, most of it the two 2025 replays. Headless-browser screenshots aren't part of the pipeline.

No license file is included; add one before making the repo public if you want others to reuse the code.
