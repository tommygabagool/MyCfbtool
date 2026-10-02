"""Step 12. Fill the HTML templates with the data and page metadata. Writes three self-contained pages (fonts load from
Google Fonts): docs/index.html (landing page), docs/players/index.html (QB/RB/WR/TE dashboard, from all_data.json) and
docs/teams/index.html (Power 4 + Notre Dame teams, schedules and game projections, from teams_data.json)."""
import paths  # noqa: F401  (sets the working directory to work/)
import json
from pathlib import Path

TPL = Path(__file__).parent
DOCS = Path('../docs')
compact = lambda o: json.dumps(o, separators=(',', ':'))

def write(tpl_name, out, data, fill):
    tpl = (TPL / tpl_name).read_text(encoding='utf-8')
    for k, v in fill.items():
        tpl = tpl.replace(k, v)
    assert '{{' not in tpl, f'unfilled template placeholder in {tpl_name}'
    out = DOCS / out
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(tpl.replace('__DATA__', compact(data)), encoding='utf-8')
    print(f'wrote docs/{out.relative_to(DOCS).as_posix()} ({out.stat().st_size // 1024} KB)')

# ---------- Players page
data = json.load(open('all_data.json'))
m = data['meta']
miss = m['te_missing']
te_missing = '' if not miss else ' ' + (miss[0] if len(miss) == 1 else ', '.join(miss[:-1]) + ' and ' + miss[-1]) + \
    (' has' if len(miss) == 1 else ' have') + ' no tight end who qualifies.'
fill = {'{{WEEK}}': str(m['week']), '{{THROUGH}}': m['through'], '{{N_FBS}}': str(m['n_fbs']),
        '{{QB_POOL}}': str(m['pools']['qb']), '{{RB_POOL}}': str(m['pools']['rb']), '{{WR_POOL}}': str(m['pools']['wr']),
        '{{TE_POOL}}': str(m['pools']['te']), '{{TE_MISSING}}': te_missing}
write('players_template.html', 'players/index.html', data, fill)

# ---------- Teams page
teams = json.load(open('teams_data.json'))
tm = teams['meta']
tfill = {'{{WEEK}}': str(tm['week']), '{{THROUGH}}': tm['through']}
if (TPL / 'teams_template.html').exists():
    write('teams_template.html', 'teams/index.html', teams, tfill)
else:
    print('WARNING: pipeline/teams_template.html not found; docs/teams/index.html not built')

# ---------- landing page: a few counts and the model's track record
g = teams['games'].values()
home = dict(week=tm['week'], through=tm['through'], acc=tm['acc'], nteams=len(tm['p4']), ngames=len(teams['games']),
            nplayed=sum(x['done'] for x in g), npos={p: len(data[p]) for p in ('qb', 'rb', 'wr', 'te')})
write('home_template.html', 'index.html', home, tfill)
print(f'through Week {m["week"]}, games through {m["through"]}')
