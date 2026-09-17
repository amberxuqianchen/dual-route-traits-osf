"""
Holm-Bonferroni FWER correction and standardized betas for the
maximal-random-effects familiarity LMMs.

Permutation max-T is not feasible at this random-effects structure: the selected
models take a median of 22 min each to fit (see `secs` in
full_ladder_all_levels.csv), so a 1,000-permutation max-T over 20 outcomes is
~10^4 refits.  Holm controls the FWER under arbitrary dependence among the
tests, so it is valid here; because the tests are positively dependent it is
conservative, i.e. anything surviving Holm would also survive a step-down max-T.

Family: within moderator coding x dataset x channel, over the individual traits
(CN 10, US 8).  The trait space is a single outcome by construction and so is
left uncorrected (NaN).

Standardized betas: the models are fitted with z-scored predictors and the
outcome in raw rating units, so the fitted coefficient is semi-standardised.
Dividing by the SD of the outcome gives a fully standardized coefficient.  This
matters for comparability: the trait-space outcome (Euclidean distance over the
whole trait profile) has a much larger SD than any single trait, and the two
datasets differ in outcome SD too, so semi-standardised betas are not comparable
across outcomes or samples.  The product term is NOT itself z-scored -- for
interactions the constituent variables are standardized and the product formed
from them (Aiken & West), which is what the fit already does.  Standardization is
a pure rescaling, so z statistics and p values are unchanged.

Writes 3_pipeline/lmm_familiarity_maximal/selected_models_holm.csv

Usage:
    python 2_code/supplementary/add_holm_lmm_maximal.py --pairs-dir DIR
"""

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

import numpy as np
import pandas as pd

from config import PIPELINE_DIR

LMM_DIR = PIPELINE_DIR / 'lmm_familiarity_maximal'


def holm(p):
    """Holm-Bonferroni adjusted p-values."""
    p = np.asarray(p, float)
    m = len(p)
    order = np.argsort(p)
    adj = np.empty(m)
    running = 0.0
    for i, idx in enumerate(order):
        running = max(running, (m - i) * p[idx])
        adj[idx] = min(running, 1.0)
    return adj


ap = argparse.ArgumentParser()
ap.add_argument('--pairs-dir', required=True,
                help='output dir of export_pairs_all_traits.py')
args = ap.parse_args()
pairs_dir = Path(args.pairs_dir)

df = pd.read_csv(LMM_DIR / 'selected_models.csv')
df['sem_p_holm'] = np.nan
df['vis_p_holm'] = np.nan

# ── Standardized betas: divide the semi-standardised coefficient by SD(outcome)
prefix = {'fam': 'all', 'arm': 'bn'}
sd_out = []
for _, r in df.iterrows():
    f = pairs_dir / f"{prefix[r['mod']]}_{r['ds']}_{r['trait']}.csv"
    sd_out.append(pd.read_csv(f, usecols=['abs_diff'])['abs_diff'].std())
df['sd_outcome'] = sd_out
df['sem_std'] = df['sem'] / df['sd_outcome']
df['vis_std'] = df['vis'] / df['sd_outcome']
df['sem_se_std'] = df['sem_se'] / df['sd_outcome']
df['vis_se_std'] = df['vis_se'] / df['sd_outcome']

for (mod, ds), g in df.groupby(['mod', 'ds']):
    tr = g[g['trait'] != 'traitspace']
    for ch in ['sem', 'vis']:
        df.loc[tr.index, f'{ch}_p_holm'] = holm(tr[f'{ch}_p'].values)

out = LMM_DIR / 'selected_models_holm.csv'
df.to_csv(out, index=False)
print(f'wrote {out}')

for (mod, ds), g in df.groupby(['mod', 'ds']):
    m = (g['trait'] != 'traitspace').sum()
    surv = [(r['trait'], ch, r[f'{ch}_p_holm'])
            for _, r in g.iterrows() for ch in ['sem', 'vis']
            if r[f'{ch}_p_holm'] < .05]
    print(f'{mod}/{ds} (family of {m} per channel): '
          + (', '.join(f'{t} {c} p_holm={p:.4g}' for t, c, p in surv) or 'none survive'))
