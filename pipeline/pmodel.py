"""Player projections, shared by calibrate_players.py (which fits the weights on last season, replayed week by week) and
project.py. A player's line is his projected volume (pass attempts, carries, targets) times his projected rate per play
for each stat.
  Volume: the team's projected plays, pass rate and sack rate (pace, game script), times his recency-weighted share of the
  team's attempts, carries or targets, blended with his average per game played, and trimmed in a projected blowout
  (starters sit).
  Rate: the league average for his position group, plus a share of his edge over it (his rate so far adjusted for the
  defenses he faced, pulled toward the group average when the sample is small), plus a share of this opponent's
  defensive rating and home field; calib_players.json holds the shares."""
import pandas as pd, numpy as np
from common import fit, pace_points, drive_pace, rating

STAT = ['att', 'cmp', 'py', 'ptd', 'int', 'sk', 'car', 'ry', 'rtd', 'tgt', 'rec', 'recy', 'rectd']
PK = ['cmp', 'ypa', 'ptd', 'int', 'ry', 'rtd', 'sk']   # play metrics behind the player rates
HL = dict(att=2.0, car=2.0, tgt=3.0)                   # usage weights halve every 2 (passes, carries) or 3 (targets) team games back
KN = 40.0                                              # small-sample pull toward the group average, in plays
FAM = {'pass': ('att', {'cmp': 'cmp', 'py': 'ypa', 'ptd': 'ptd', 'int': 'int'}),     # family: volume, {stat: play metric}
       'rush': ('car', {'ry': 'ry', 'rtd': 'rtd'}),
       'recv': ('tgt', {'rec': 'cmp', 'recy': 'ypa', 'rectd': 'ptd'})}
SUM = {'cmp': 'cmp', 'ypa': 'pyds', 'ptd': 'ptd', 'int': 'intc', 'ry': 'ryds', 'rtd': 'rtd'}   # play column behind each metric
GRADE = dict(att=15, car=8, tgt=3)                     # calibration grades players averaging this many per game played
CLIP = dict(cmp=(.25, .9), py=(2, 14), ptd=(0, .15), int=(0, .08), ry=(-1, 12), rtd=(0, .15), rec=(.2, .95), recy=(1, 18), rectd=(0, .2))


def fill_ids(s, ro, tid):
    """Fill missing passer, rusher and target ids from the play text's jersey number and name ("#2 B.Shapen pass complete
    ... to #6 D.Booth"), matched to that game's ESPN roster. Needed for 2025: from Week 10 on most touchdown plays came
    through with no participant ids. ro: game roster rows (game_id, team_id, athlete_id, jersey, first/last name, position_href);
    tid: {team name: ESPN team id}. Returns (s, {column: ids filled})."""
    import re
    norm = lambda z: re.sub(r'[^a-z]', '', str(z).lower())
    skill = ro.position_href.str.extract(r'positions/(\d+)')[0].isin(['1', '7', '8', '9', '10', '0'])
    reg = {}
    for g, t, j, a, fn, ln, sk in zip(ro.game_id, ro.team_id, ro.jersey, ro.athlete_id, ro.first_name, ro.last_name, skill):
        if pd.isna(j) or pd.isna(a): continue
        reg.setdefault((int(g), int(t), str(j)), []).append((int(a), str(fn or '')[:1].upper(), norm(re.sub(r'\s(?:Jr\.?|Sr\.?|II|III|IV|V)$', '', str(ln or ''))), sk))
    NAME = r"#(\d+)\s+([A-Z][a-zA-Z]{0,2})\.\s?((?:St\.\s?)?[A-Z][\w'\-]+)"
    pats = {'qid': re.compile(NAME + r"(?:[ ,][\w.]+)?\s(?:pass|sacked)\b"), 'rec_id': re.compile(r"\bpass (?:complete|incomplete)[a-z ]*? to " + NAME),
            'rus_id': re.compile(NAME + r"(?:[ ,][\w.]+)?\s(?:rush|run|scramble)\b")}
    need = {'qid': (s.is_att | s.is_sack), 'rec_id': s.is_att & ~s.intc, 'rus_id': s.is_rush & ~s.kneel}
    tnum = s.off.map(tid)
    filled = {}
    for col, pat in pats.items():
        idx = s.index[need[col] & s[col].isna() & tnum.notna()]
        vals = {}
        for i in idx:
            m = pat.search(str(s.at[i, 'play_text']))
            if not m: continue
            last = norm(m.group(3))
            c = [x for x in reg.get((int(s.at[i, 'gid']), int(tnum[i]), m.group(1)), [])     # jersey, initial and last name must agree
                 if x[1] == m.group(2)[:1].upper() and last and (x[2].startswith(last) or last.startswith(x[2][:len(last)]) and len(x[2]) >= 3)]
            if len(c) > 1: c = [x for x in c if x[3]] or c
            if len(c) == 1: vals[i] = c[0][0]
        if vals: s.loc[list(vals), col] = pd.Series(vals, dtype=float)
        filled[col] = len(vals)
    return s, filled


def usage_flags(s):
    """Volume flags on plays that carry qid (passer), rus_id (rusher) and rec_id (target) columns."""
    s['db'] = s.is_att | s.is_sack; s['car'] = s.is_rush & ~s.kneel; s['live'] = s.db | s.car
    s['tgt'] = s.is_att & s.rec_id.notna()
    return s


def box(d):
    """Actual line per (gid, off, pid) from plays d."""
    a = d[d.is_att & d.qid.notna()].groupby(['gid', 'off', 'qid']).agg(att=('is_att', 'size'), cmp=('cmp', 'sum'), py=('pyds', 'sum'),
                                                                     ptd=('ptd', 'sum'), int=('intc', 'sum'))
    k = d[d.is_sack & d.qid.notna()].groupby(['gid', 'off', 'qid']).size().rename('sk')
    c = d[d.car & d.rus_id.notna()].groupby(['gid', 'off', 'rus_id']).agg(car=('car', 'size'), ry=('ryds', 'sum'), rtd=('rtd', 'sum'))
    t = d[d.tgt].groupby(['gid', 'off', 'rec_id']).agg(tgt=('tgt', 'size'), rec=('cmp', 'sum'), recy=('pyds', 'sum'), rectd=('ptd', 'sum'))
    for z in (a, k, c, t): z.index = z.index.set_names(['gid', 'off', 'pid'])
    B = pd.concat([a, k, c, t], axis=1).fillna(0.0).reset_index()
    B['pid'] = B.pid.astype(np.int64)
    for c_ in STAT:
        if c_ not in B: B[c_] = 0.0
    return B[['gid', 'off', 'pid'] + STAT].astype({c_: float for c_ in STAT})


class Ctx:
    """Everything the projections need, built from the plays before week w only."""
    def __init__(self, s, dr, FBS, conf, prior, PTS, TS, POS, w, keys=None):
        tr = s[s.week < w].copy(); dw = dr[dr.week < w]
        self.w, self.FBS, self.conf = w, FBS, conf
        self.r = r = fit(tr, FBS, prior, keys=keys, drives=dw)
        for k, v in r.items():           # an FBS team with no plays yet starts from its preseason prior, not from average
            if k not in prior: continue
            for side, ps in (('off', 'mu_off'), ('dfn', 'mu_def')):
                for t in FBS - set(v[side]): v[side][t] = prior[k][ps].get(t, 0.0)
        for k in PK:
            tr['e_' + k] = tr.dfn.map(r[k]['dfn']).fillna(r[k]['fcs_def_c'])
        ng = tr[~tr.garbage]; fo = ng[ng.off.isin(FBS) & ng.dfn.isin(FBS)]
        # league baselines (FBS vs FBS, non-garbage): the fits are centered on these
        self.lg = dict(db=fo.db.sum() / fo.live.sum(), sk=fo.is_sack.sum() / fo.db.sum(), tgt=fo.tgt.sum() / max(1, fo.is_att.sum()),
                       cmp=fo[fo.is_att].cmp.mean(), ypa=fo[fo.is_att].pyds.mean(), ptd=fo[fo.is_att].ptd.mean(), int=fo[fo.is_att].intc.mean(),
                       ry=fo[fo.car].ryds.mean(), rtd=fo[fo.car].rtd.mean())
        self.lg['plays'], self.lg['ppp'], self.pace_o, self.pace_d = pace_points(tr, FBS, PTS)
        self.Ld, self.dpo, self.dpd = drive_pace(dw)
        # team dropback rate and credited-target rate, non-garbage, shrunk 120 plays / 80 attempts toward average
        tm = ng.groupby('off').agg(db=('db', 'sum'), live=('live', 'sum'), att=('is_att', 'sum'), tgt=('tgt', 'sum'))
        self.dbr = ((tm.db + 120 * self.lg['db']) / (tm.live + 120)).to_dict()
        self.tgr = ((tm.tgt + 80 * self.lg['tgt']) / (tm.att + 80)).to_dict()
        # usage per player per game, and team totals per game
        U = pd.concat([tr[tr.is_att & tr.qid.notna()].groupby(['gid', 'off', 'qid']).size().rename('att').rename_axis(['gid', 'off', 'pid']),
                       tr[tr.car & tr.rus_id.notna()].groupby(['gid', 'off', 'rus_id']).size().rename('car').rename_axis(['gid', 'off', 'pid']),
                       tr[tr.tgt].groupby(['gid', 'off', 'rec_id']).size().rename('tgt').rename_axis(['gid', 'off', 'pid'])], axis=1).fillna(0.0).reset_index()
        U['pid'] = U.pid.astype(np.int64); self.U = U
        self.TG = tr.groupby(['gid', 'off']).agg(att=('is_att', 'sum'), car=('car', 'sum'), tgt=('tgt', 'sum')).reset_index()
        T0 = pd.Timestamp(0, tz='UTC')
        self.tgames = {t: sorted(gs, key=lambda g: TS.get(g, T0)) for t, gs in tr.groupby('off').gid.unique().items()}
        # each player's sums so far: raw results and the defensive ratings he faced, by family; position groups
        A = tr[tr.is_att & tr.qid.notna()].assign(pid=lambda d: d.qid.astype(np.int64))
        C = tr[tr.car & tr.rus_id.notna()].assign(pid=lambda d: d.rus_id.astype(np.int64))
        G = tr[tr.tgt].assign(pid=lambda d: d.rec_id.astype(np.int64))
        passers = set(A.pid)
        pos = lambda d: d.pid.map(POS)
        C['grp'] = np.where(C.pid.isin(passers) | (pos(C) == 'QB'), 'QB', np.where(pos(C).isin(['WR', 'TE']), pos(C), 'RB'))
        G['grp'] = np.where(pos(G).isin(['WR', 'TE', 'RB']), pos(G), 'WR'); A['grp'] = 'QB'
        self.sums, self.gavg = {}, {}
        for fam, D in (('pass', A), ('rush', C), ('recv', G)):
            mets = sorted(set(FAM[fam][1].values()))
            agg = {'n': ('gid', 'size')}
            for m in mets: agg[m] = (SUM[m], 'sum'); agg['e_' + m] = ('e_' + m, 'sum')
            S_ = D.groupby('pid').agg(**agg)
            S_['grp'] = D.groupby('pid').grp.agg(lambda v: v.value_counts().index[0])
            self.sums[fam] = S_
            fb = D[D.off.isin(FBS)]
            self.gavg[fam] = fb.groupby('grp')[[SUM[m] for m in mets] + ['e_' + m for m in mets]].mean().rename(columns={SUM[m]: m for m in mets})
        # games played and averages per game played
        B = box(tr); self.base = B.groupby('pid')[STAT].mean(); self.pg = B.groupby('pid').gid.agg(set).to_dict()

    def eff(self, k, t, o, hf, both=False):
        """Rating offset for metric k: opponent o's defense rating plus home edge (plus team t's offense rating if both)."""
        r = self.r[k]; return rating(r, 'off', t, self.FBS) * both + rating(r, 'dfn', o, self.FBS) + r['home'] * hf

    def shares(self, t):
        """Recency-weighted share of the team's attempts, carries and targets for everyone who has played for team t."""
        gs = self.tgames.get(t, [])
        if not gs: return pd.DataFrame(columns=['att', 'car', 'tgt'])
        U = self.U[(self.U.off == t) & self.U.gid.isin(gs)]; TG = self.TG[(self.TG.off == t) & self.TG.gid.isin(gs)]
        out = {}
        for k in ('att', 'car', 'tgt'):
            w = pd.Series([0.5 ** ((len(gs) - 1 - i) / HL[k]) for i in range(len(gs))], index=gs)
            num = (U[k] * U.gid.map(w)).groupby(U.pid).sum(); den = float((TG[k] * TG.gid.map(w)).sum())
            out[k] = num / den if den > 0 else num * 0.0
        return pd.DataFrame(out).fillna(0.0)

    def team_inputs(self, t, o, hf):
        """Pace-based plays, the team's pass rate, sack rate and credited-target rate, before game script."""
        L = self.lg
        return dict(plays0=self.pace_o.get(t, L['plays']) + self.pace_d.get(o, L['plays']) - L['plays'],
                    dbr0=float(np.clip(self.dbr.get(t, L['db']), .3, .75)),
                    sk=float(np.clip(L['sk'] + self.eff('sk', t, o, hf, True), .01, .2)), tgr=self.tgr.get(t, L['tgt']))

    def comps(self, t, o, hf):
        """One row per player who has played for t: usage shares, averages per game played, whether he played one of the
        team's last two games, and for each stat his schedule-adjusted rate, this opponent's effect and the group average."""
        sh = self.shares(t)
        if sh.empty: return pd.DataFrame()
        gs = self.tgames.get(t, []); last2 = set(gs[-2:])
        rows = []
        for p in sh.index:
            p = int(p); r = dict(pid=p, sh_att=sh.at[p, 'att'], sh_car=sh.at[p, 'car'], sh_tgt=sh.at[p, 'tgt'])
            for k in ('att', 'car', 'tgt'): r['b_' + k] = float(self.base.at[p, k]) if p in self.base.index else 0.0
            r['avail'] = float(bool(self.pg.get(p, set()) & last2))
            r['gp'] = len(self.pg.get(p, set()))
            for fam, (vol, stats) in FAM.items():
                S_ = self.sums[fam]; ga = self.gavg[fam]
                n = float(S_.at[p, 'n']) if p in S_.index else 0.0
                grp = S_.at[p, 'grp'] if p in S_.index else ('QB' if fam == 'pass' else 'RB' if fam == 'rush' else 'WR')
                if grp not in ga.index: grp = ga.index[0]
                r['n_' + fam] = n
                for st, m in stats.items():
                    raw = float(S_.at[p, m]) if n else 0.0; eh = float(S_.at[p, 'e_' + m]) if n else 0.0
                    ar, ae = float(ga.at[grp, m]), float(ga.at[grp, 'e_' + m])
                    r['adj_' + st] = (raw - eh + KN * (ar - ae)) / (n + KN)       # rate against an average defense, shrunk
                    r['raw_' + st] = raw / n if n else np.nan                     # plain rate so far
                    r['eh_' + st] = eh / n if n else np.nan                       # average defensive effect he has faced
                    r['eo_' + st] = self.eff(m, t, o, hf)                         # this opponent (plus home field)
                    r['avg_' + st] = ar
            rows.append(r)
        return pd.DataFrame(rows).set_index('pid')


def team_volume(ti, pm, pt, PC):
    """Team plays, dropbacks, sacks, pass attempts, carries and targets from team_inputs(), the projected margin (pm, this
    team's view) and total (pt), with calib_players.json's pace and game-script terms."""
    a = PC['plays']; b = PC['dbr']
    plays = a[0] + a[1] * ti['plays0'] + a[2] * pt
    dbr = float(np.clip(b[0] + b[1] * ti['dbr0'] + b[2] * pm, .25, .8))
    db = plays * dbr; att = db * (1 - ti['sk'])
    return dict(plays=plays, dbr=dbr, db=db, sk=db * ti['sk'], att=att, car=plays - db, tgt=att * ti['tgr'])


def prate(c, adj, eo, avg):
    """Projected rate per play: group average + c[0] + c[1] x (schedule-adjusted rate - average) + c[2] x opponent effect."""
    return avg + c[0] + c[1] * (adj - avg) + c[2] * eo


def volume(c, share_vol, per_game, avail, pm):
    """A player's volume: c[0] x his share of the team's projected volume + c[1] x his average per game played (if he played
    one of the team's last two games) + c[2] x the share-based volume x the projected margin's size / 14 (negative: starters
    sit in blowouts)."""
    return max(0.0, c[0] * share_vol + c[1] * per_game * avail + c[2] * share_vol * abs(pm) / 14)


def apply(C, tv, PC, pm):
    """{pid: {stat: projection}} from comps() rows C, team volume tv, the team's projected margin pm and calib_players.json PC."""
    out = {}
    for p, r in C.iterrows():
        v = dict.fromkeys(STAT, 0.0)
        for fam, (vol, stats) in FAM.items():
            x = volume(PC['vol'][fam], tv[vol] * r['sh_' + vol], r['b_' + vol], r['avail'], pm)
            v[vol] = x
            for st in stats:
                v[st] = x * float(np.clip(prate(PC['rate'][st], r['adj_' + st], r['eo_' + st], r['avg_' + st]), *CLIP[st]))
        v['sk'] = tv['sk'] * r.sh_att
        out[int(p)] = v
    return out
