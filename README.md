# Who's under center: Power 4 skill-position dashboard

Opponent-adjusted stats and play-style archetypes for Power 4 and Notre Dame quarterbacks, running backs, wide receivers and tight ends, built from cfbfastR play-by-play. The pipeline writes one self-contained page, `docs/index.html`, with QB, RB, WR and TE tabs.

Each tab has an opponent-adjusted/raw toggle, archetype and conference filters, three scatter views drawn on a football field, a scouting card (key stats against the position average, style fingerprint, yardage profile, game log with each opponent's defensive rank), leaderboards, an archetype mix by conference, and a sortable full board.

## Quick start

```bash
python3 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
./run_all.sh            # first run downloads about 304 MB of data, then builds docs/index.html
```

Open `docs/index.html` in a browser. To refresh during the season, run `./run_all.sh --force`, which re-downloads this season's files and rebuilds everything; the page text (week, date, counts) updates from the data.

To host it, commit `docs/index.html` and turn on GitHub Pages under Settings > Pages > Deploy from a branch, choosing `main` and `/docs`.

## Pipeline

All scripts live in `pipeline/` and run from `work/` (handled by `paths.py`). Raw downloads go to `data/`, intermediate files to `work/`, the page to `docs/`.

| Step | Script | What it does | Writes |
|---|---|---|---|
| 0 | `download_data.py` | Fetches play-by-play (2024 to 2026), ESPN game rosters and box scores | `data/*.parquet` |
| 1 | `prep.py` | Scrimmage plays, outcome flags, QB attribution, garbage time, source-data fixes | `plays.parquet`, `teams.json` |
| 2 | `seasons.py` | Defense ratings for 2024 and 2025 and their year-to-year carryover | `seasons.pkl`, `persistence.csv` |
| 3 | `fit2026.py` | 2026 offense and defense ratings for every metric, seeded by regressed 2025 ratings | `def_effects.json`, `off_effects.json` |
| 4 | `qbstats.py` | Raw and opponent-adjusted stats for every FBS starting QB | `plays_adj.parquet`, `qb_all.pkl` |
| 5 | `arch.py` | QB archetype scores | `qb_arch.pkl` |
| 6 | `dataset.py` | Power 4 + Notre Dame QBs with game logs | `qb_data.json` |
| 7 | `prep_skill.py` | Rusher and receiver identity, roster positions and bios | `plays_skill.parquet`, `players.pkl` |
| 8 | `skill.py` | RB, WR and TE stats, archetypes and game logs, merged with QBs and page metadata | `all_data.json` |
| 9 | `build.py` | Fills `dashboard_template.html` with the data | `docs/index.html` |
| opt. | `validate.py` | Game-by-game match rates against ESPN box scores | (prints) |

`common.py` holds the shared outcome definitions, the metric list and the rating model.

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

Years are hardcoded in `download_data.py` (season and file names), `seasons.py` (the two prior seasons), `fit2026.py` (the 2025 seed), the `game_rosters_2026` / `player_box_2026` file names in `prep_skill.py` and `validate.py`, and the "2026 season" text in the template. Conference membership comes from the play-by-play.

## Data sources

- Play-by-play: [cfbfastR](https://github.com/sportsdataverse/cfbfastR) data via the [sportsdataverse-data](https://github.com/sportsdataverse/sportsdataverse-data) releases.
- Game rosters and player box scores: ESPN data via the same releases.

Raw data isn't committed (`data/` and `work/` are in `.gitignore`); check the sources' terms before redistributing it.

## Requirements

Python 3 with pandas, numpy, pyarrow and scipy. Tested on Python 3.12.3 with pandas 3.0.2, numpy 2.4.4, pyarrow 25.0.1 and scipy 1.17.1. Headless-browser screenshots aren't part of the pipeline.

No license file is included; add one before making the repo public if you want others to reuse the code.
