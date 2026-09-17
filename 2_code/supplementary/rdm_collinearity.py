"""
Collinearity between the semantic and visual predictors.

H2 rests on partitioning trait-RDM variance into unique semantic and unique
visual components.  That partition is only interpretable if the two model RDMs
are not near-redundant, so this reports how correlated they actually are:

  1. semantic-visual RDM correlation per dataset,
     as Spearman rho (the metric the RSA uses) and Pearson r (the metric that
     governs the OLS variance partitioning and hence the VIF)
  2. the variance inflation factor for the two-predictor RSA regression,
     VIF = 1 / (1 - r^2) with r the Pearson correlation of the two predictors
  3. the VIF of every fixed effect in the perceiver-level LMM, computed on the
     model's own design matrix (sem, vis, FAM, SXF, VXF, FD)

Writes 3_pipeline/rdm_collinearity.csv and 3_pipeline/lmm_vif.csv

Usage: python 2_code/supplementary/rdm_collinearity.py
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

import numpy as np
import pandas as pd
from scipy.stats import spearmanr, pearsonr

from config import DATASETS, PIPELINE_DIR
from data_loader import load_rdms
from rsa_utils import flatten_rdm

LMM_DIR = PIPELINE_DIR / 'lmm_familiarity_maximal'

# ── 1-2. RDM correlation and the RSA regression VIF ──────────────────────────
rows = []
for ds, cfg in DATASETS.items():
    sem, vis = load_rdms(cfg)
    masks = {'all': np.ones(sem.shape[0], bool)}
    for split, m in masks.items():
        idx = np.where(m)[0]
        s = flatten_rdm(sem[np.ix_(idx, idx)])
        v = flatten_rdm(vis[np.ix_(idx, idx)])
        ok = ~np.isnan(s) & ~np.isnan(v)
        s, v = s[ok], v[ok]
        rho, p_rho = spearmanr(s, v)
        r, p_r = pearsonr(s, v)
        rows.append(dict(dataset=ds, split=split, n_targets=len(idx),
                         n_pairs=len(s), spearman_rho=rho, spearman_p=p_rho,
                         pearson_r=r, pearson_p=p_r,
                         shared_var_pct=100 * r ** 2,
                         vif=1 / (1 - r ** 2)))
out = pd.DataFrame(rows)
out.to_csv(PIPELINE_DIR / 'rdm_collinearity.csv', index=False)
print(out.round(4).to_string(index=False))

# ── 3. VIF of the LMM fixed effects ──────────────────────────────────────────
# Same construction as glmmtmb_diagnostics.R: predictors z-scored, products
# formed from the z-scored constituents.
vif_rows = []
for ds in DATASETS:
    f = LMM_DIR / 'pairs' / f'all_{ds}_traitspace.csv'
    d = pd.read_csv(f)
    d['FAM'] = (d.fam_mean - d.fam_mean.mean()) / d.fam_mean.std()
    d['FD'] = (d.fam_diff - d.fam_diff.mean()) / d.fam_diff.std()
    d['SXF'] = d.sem_dissim_z * d.FAM
    d['VXF'] = d.vis_dissim_z * d.FAM
    terms = ['sem_dissim_z', 'vis_dissim_z', 'FAM', 'SXF', 'VXF', 'FD']
    X = d[terms].to_numpy()
    # VIF_j = 1 / (1 - R^2_j), R^2_j from regressing column j on the others
    C = np.corrcoef(X, rowvar=False)
    Cinv = np.linalg.inv(C)
    for j, t in enumerate(terms):
        vif_rows.append(dict(dataset=ds, term=t, vif=Cinv[j, j],
                             r2_on_others=1 - 1 / Cinv[j, j]))
    vif_rows.append(dict(dataset=ds, term='(sem-vis correlation)',
                         vif=np.nan, r2_on_others=C[0, 1]))
vif = pd.DataFrame(vif_rows)
vif.to_csv(LMM_DIR / 'lmm_vif.csv', index=False)
print()
print(vif.round(4).to_string(index=False))
