"""Step 5. Quarterback archetype style scores (Field General, Gunslinger, Pure Runner, Pocket Passer).
Writes work/qb_arch.pkl."""
import paths  # noqa: F401  (sets the working directory to work/)
import pandas as pd, numpy as np
Q = pd.read_pickle('qb_all.pkl')
B = Q[Q.att >= 15].copy()
def shr(rate, n, k):
    m = np.nansum(rate * n) / n[~np.isnan(rate)].sum()
    r = np.where(np.isnan(rate), m, rate)
    return (r * n + m * k) / (n + k)
F = pd.DataFrame(index=B.index)
F['cp'] = shr(B.a_cmp / B.att, B.att, 40); F['dsr'] = shr(B.a_dsr, B.db, 40); F['conv'] = shr(B.a_conv, B.latep, 20)
F['ir'] = shr(B.a_i / B.att, B.att, 80); F['skr'] = shr(B.a_sk / B.db, B.db, 40)
F['xpr'] = shr(B.a_xp / B.att, B.att, 40); F['ypc'] = shr(B.a_ypc, B.comp, 20)
F['rypg'] = B.a_ry / B.gp; F['rshare'] = B.carl / B.live; F['xrpg'] = B.a_xr / B.gp; F['rsr'] = shr(B.a_rsr, B.carl, 15)
F['apg'] = B.att / B.gp
Z = ((F - F.mean()) / F.std()).clip(-3, 3)
S = pd.DataFrame({
 'Field General': Z[['cp', 'dsr', 'conv']].sum(1).sub(Z.ir).sub(Z.skr) / 5,
 'Gunslinger': (Z.xpr + Z.ypc + Z.ir) / 3,
 'Pure Runner': (Z.rypg + Z.rshare + Z.xrpg + Z.rsr) / 4,
 'Pocket Passer': (Z.apg - Z.rshare - Z.skr) / 3})
def label(r):
    o = r.sort_values(ascending=False); a, b = o.index[0], o.index[1]
    if o.iloc[0] <= 0.2: return a, f'{a} (weak fit)'
    if o.iloc[1] >= o.iloc[0] - 0.25: return a, f'{a} / {b}'
    return a, a
L = S.apply(label, axis=1, result_type='expand'); L.columns = ['a', 'lab']
B = B.join(S).join(L)
B.to_pickle('qb_arch.pkl')
P = B[B.conf.isin(['ACC', 'Big 12', 'Big Ten', 'SEC']) | (B.team == 'Notre Dame')]
print(len(B), 'modelled;', len(P), 'P4+ND')
print(P.a.value_counts().to_dict()); print('hybrids', P.lab.str.contains('/').sum(), 'weak', P.lab.str.contains('weak').sum())
pd.set_option('display.width', 220)
print(P.sort_values(['a', 'lab'])[['qb', 'team', 'lab', 'Field General', 'Gunslinger', 'Pure Runner', 'Pocket Passer']].round(2).to_string(index=False))
