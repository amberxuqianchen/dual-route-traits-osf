"""
Build the max-T null and adjusted p-values from the permutation output.

Safe to run on a partial run -- it uses only permutations completed for EVERY
trait in a dataset, since max|z| must be taken over the whole family. Reports how
many permutations are usable and the shape of the null, which is the first thing
to check: a well-calibrated null max|z| over 8-10 tests should sit around 2.5-3.
The earlier version of this analysis produced a null median of 17, which would
mean the permutation scheme is not sampling the intended null.

Adjusted p uses the (1 + #exceedances) / (1 + B) convention, so the smallest
attainable value is 1/(B+1).

Writes 3_pipeline/lmm_familiarity_maximal/maxt_results.csv

Usage:
    python 2_code/supplementary/collate_maxt.py
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

import numpy as np
import pandas as pd

from config import PIPELINE_DIR

LMM_DIR = PIPELINE_DIR / 'lmm_familiarity_maximal'
PERM_DIR = LMM_DIR / 'maxt'

obs = pd.read_csv(LMM_DIR / 'diagnostics_summary.csv')
obs = obs[obs['fit'] == 'selected'].copy()
obs['ds'] = obs['label'].str.split('_').str[1]
obs['trait'] = obs['label'].str.split('_', n=2).str[2]

rows = []
for ds in ['cn', 'us']:
    traits = sorted(t for t in obs.loc[obs['ds'] == ds, 'trait']
                    if t != 'traitspace')
    perm = {}
    for t in traits:
        f = PERM_DIR / f'perm_{ds}_{t}.csv'
        if not f.exists():
            print(f'[{ds}] missing {f.name} -- skipping dataset')
            perm = None
            break
        d = pd.read_csv(f).drop_duplicates('b').set_index('b')
        perm[t] = d
    if perm is None:
        continue

    # only permutations present for every trait can contribute to a family max
    common = sorted(set.intersection(*(set(d.index) for d in perm.values())))
    if not common:
        continue
    pd_ok = {t: perm[t].loc[common, 'pdHess'].fillna(False).astype(bool)
             for t in traits}
    rate = {t: 1 - pd_ok[t].mean() for t in traits}
    # a family max is only coherent if EVERY trait's fit for that b is usable,
    # so one bad trait discards the whole permutation
    clean = [b for k, b in enumerate(common)
             if all(pd_ok[t].iloc[k] for t in traits)]
    print(f'[{ds}] {len(common)} permutations complete across all {len(traits)} '
          f'traits')
    print(f'      non-PD-Hessian rate per trait: '
          f'{ {t: f"{r:.0%}" for t, r in rate.items() if r > 0} or "none"}')
    print(f'      permutations with every trait PD: {len(clean)}/{len(common)} '
          f'({len(clean)/len(common):.0%})')

    # `all`   = every complete permutation, using z as fitted
    # `pdonly`= only permutations where all traits gave a positive-definite
    #           Hessian, so every SE entering the max is trustworthy
    for subset, bs in [('all', common), ('pdonly', clean)]:
        if not bs:
            continue
        for ch in ['sem', 'vis']:
            Z = np.vstack([perm[t].loc[bs, f'{ch}_z'].values for t in traits])
            null = np.nanmax(np.abs(Z), axis=0)      # family max per permutation
            B = len(null)
            print(f'   [{subset}] {ch}: B={B} null max|z| '
                  f'median={np.median(null):.2f} '
                  f'p95={np.percentile(null, 95):.2f} max={null.max():.2f}')
            for t in traits:
                z = obs.loc[(obs['ds'] == ds) & (obs['trait'] == t),
                            f'{ch}_z'].iloc[0]
                rows.append(dict(
                    ds=ds, trait=t, channel=ch, subset=subset, z_obs=z, B=B,
                    null_median=np.median(null),
                    null_p95=np.percentile(null, 95),
                    p_maxt=(1 + int((null >= abs(z)).sum())) / (1 + B)))

if rows:
    out = pd.DataFrame(rows)
    out.to_csv(LMM_DIR / 'maxt_results.csv', index=False)
    print(f'\nwrote {LMM_DIR / "maxt_results.csv"}')
    print(out[(out['p_maxt'] < .05)].sort_values(['subset','p_maxt']).to_string(index=False)
          or 'nothing survives max-T')
else:
    print('no usable permutations yet')
