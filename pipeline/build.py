"""Step 9. Fill the HTML template with the data and page metadata. Writes docs/index.html, a single
self-contained page (fonts load from Google Fonts)."""
import paths  # noqa: F401  (sets the working directory to work/)
import json
from pathlib import Path

tpl = (Path(__file__).parent / 'dashboard_template.html').read_text(encoding='utf-8')
data = json.load(open('all_data.json'))
m = data['meta']
miss = m['te_missing']
te_missing = '' if not miss else ' ' + (miss[0] if len(miss) == 1 else ', '.join(miss[:-1]) + ' and ' + miss[-1]) + \
    (' has' if len(miss) == 1 else ' have') + ' no tight end who qualifies.'
fill = {'{{WEEK}}': str(m['week']), '{{THROUGH}}': m['through'], '{{N_FBS}}': str(m['n_fbs']),
        '{{QB_POOL}}': str(m['pools']['qb']), '{{RB_POOL}}': str(m['pools']['rb']), '{{WR_POOL}}': str(m['pools']['wr']),
        '{{TE_POOL}}': str(m['pools']['te']), '{{TE_MISSING}}': te_missing}
for k, v in fill.items():
    tpl = tpl.replace(k, v)
assert '{{' not in tpl, 'unfilled template placeholder'
out = Path('../docs/index.html')
out.write_text(tpl.replace('__DATA__', json.dumps(data, separators=(',', ':'))), encoding='utf-8')
print(f'wrote docs/index.html ({out.stat().st_size // 1024} KB): through Week {m["week"]}, games through {m["through"]}')
