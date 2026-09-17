"""
Is the recognition-split interaction model's equal-variance assumption tenable?

The two-fit and single-interaction specifications disagree in sign for several
CN traits.  They differ in exactly one substantive assumption: the interaction
model forces both recognition conditions to share the residual variance and the
three random-effect variances, while the two-fit version lets each condition
have its own.  This script tests whether that sharing is defensible.

Three complementary diagnostics per trait:

1. Per-arm variance components (lme4, one fit per condition) — residual sigma
   and the participant / target1 / target2 random-effect SDs, with the
   recognised-to-unrecognised ratio.  Descriptive but direct.

2. Brown-Forsythe test on the pooled interaction model's residuals, comparing
   absolute deviation from the group median between conditions.  A formal test
   of residual homoscedasticity for the model actually being used.

3. nlme likelihood-ratio test: lme(... random = ~1|participant) with
   weights = varIdent(form = ~1|recognized) against the homoscedastic version.
   nlme cannot fit crossed target random effects, so this drops them; it tests
   the residual-variance assumption specifically, not the full model.

Usage:
    python 2_code/supplementary/run_recognition_variance_check.py --dataset cn --variant midpoint
"""

import argparse
import sys
import warnings
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))
sys.path.insert(0, str(Path(__file__).parent))

import numpy as np
import pandas as pd
from scipy.stats import f as f_dist
from config import DATASETS, PIPELINE_DIR
from data_loader import load_ratings, load_rdms, load_familiarity_ratings
from lmm_utils import configure_r
import run_lmm_recognition_split as rsplit
from run_lmm_recognition_split import _build_pairs, CRITERIA, SUFFIX

warnings.filterwarnings('ignore')


def _variance_components(df):
    """Residual sigma and random-effect SDs from the base model on one arm."""
    from pymer4.models import Lmer
    m = Lmer('abs_diff ~ sem_dissim_z + vis_dissim_z'
             ' + (1|participant) + (1|target1) + (1|target2)', data=df)
    m.fit(summarize=False)
    rv = m.ranef_var  # index: participant, target1, target2, Residual
    out = {}
    for k in rv.index:
        out[str(k)] = float(rv.loc[k, 'Std'])
    return out


def _brown_forsythe(resid, group):
    """Brown-Forsythe (median-centred Levene) test for equal variance."""
    z = np.empty_like(resid, dtype=float)
    for g in np.unique(group):
        m = group == g
        z[m] = np.abs(resid[m] - np.median(resid[m]))
    groups = [z[group == g] for g in np.unique(group)]
    k, n = len(groups), len(z)
    zbar = z.mean()
    num = sum(len(g) * (g.mean() - zbar) ** 2 for g in groups) / (k - 1)
    den = sum(((g - g.mean()) ** 2).sum() for g in groups) / (n - k)
    F = num / den
    return float(F), float(f_dist.sf(F, k - 1, n - k))


_R_LRT = """
suppressPackageStartupMessages(library(nlme))
.var_lrt <- function(df) {
    ctl <- lmeControl(opt = "optim", maxIter = 200, msMaxIter = 200)
    f <- abs_diff ~ sem_dissim_z * recognized + vis_dissim_z * recognized
    m0 <- tryCatch(lme(f, random = ~1|participant, data = df, method = "REML",
                       control = ctl), error = function(e) NULL)
    m1 <- tryCatch(lme(f, random = ~1|participant, data = df, method = "REML",
                       weights = varIdent(form = ~1|recognized), control = ctl),
                   error = function(e) NULL)
    if (is.null(m0) || is.null(m1)) return(c(NA, NA, NA))
    a <- anova(m0, m1)
    c(a[2, "L.Ratio"], a[2, "p-value"], as.numeric(coef(m1$modelStruct$varStruct, unconstrained = FALSE))[1])
}
"""


def run(ds_name, variant, traits=None):
    cfg = DATASETS[ds_name]
    rsplit.RECOGNITION_CRITERION = CRITERIA[variant]
    ratings_dict = load_ratings(cfg)
    fam = load_familiarity_ratings(cfg)
    sem_rdm, vis_rdm = load_rdms(cfg)

    from rpy2 import robjects
    from rpy2.robjects import pandas2ri
    robjects.r(_R_LRT)

    traits = traits or cfg['traits']
    rows = []
    for i, trait in enumerate(traits, 1):
        df = _build_pairs(ratings_dict[trait], fam, sem_rdm, vis_rdm, ds_name)
        print(f"  [{i}/{len(traits)}] {trait} (n={len(df)}) ...", flush=True)

        vr = _variance_components(df[df.recognized == 1])
        vu = _variance_components(df[df.recognized == 0])

        # Brown-Forsythe on pooled interaction-model residuals
        from pymer4.models import Lmer
        pooled = Lmer('abs_diff ~ sem_dissim_z * recognized + vis_dissim_z * recognized'
                      ' + (1|participant) + (1|target1) + (1|target2)', data=df)
        pooled.fit(summarize=False)
        F, p_bf = _brown_forsythe(pooled.residuals, df.recognized.values)

        with (robjects.default_converter + pandas2ri.converter).context():
            lrt = list(robjects.r['.var_lrt'](df))

        row = {'dataset': ds_name, 'variant': variant, 'trait': trait,
               'n_rec': int(df.recognized.sum()),
               'n_unrec': int((df.recognized == 0).sum()),
               'sigma_rec': vr.get('Residual'), 'sigma_unrec': vu.get('Residual'),
               'sigma_ratio': vr.get('Residual') / vu.get('Residual'),
               'sd_ppt_rec': vr.get('participant'), 'sd_ppt_unrec': vu.get('participant'),
               'sd_t1_rec': vr.get('target1'), 'sd_t1_unrec': vu.get('target1'),
               'bf_F': F, 'bf_p': p_bf,
               'lrt_stat': lrt[0], 'lrt_p': lrt[1], 'varIdent_ratio': lrt[2]}
        rows.append(row)
        print(f"      sigma rec/unrec = {row['sigma_rec']:.3f}/{row['sigma_unrec']:.3f} "
              f"(ratio {row['sigma_ratio']:.3f}) | BF p={p_bf:.3g} | "
              f"LRT chi2={lrt[0]:.1f} p={lrt[1]:.3g}", flush=True)

        outdir = PIPELINE_DIR / ds_name
        pd.DataFrame(rows).to_csv(
            outdir / f'recognition_variance_check{SUFFIX[variant]}.csv', index=False)

    return pd.DataFrame(rows)


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--dataset', choices=['us', 'cn'], default='cn')
    p.add_argument('--variant', choices=list(CRITERIA), default='midpoint')
    p.add_argument('--traits', nargs='*', default=None)
    a = p.parse_args()
    configure_r()
    print(f"\n{'='*70}\nVariance-assumption check — {a.dataset.upper()} "
          f"[{a.variant}]\n{'='*70}")
    run(a.dataset, a.variant, a.traits)


if __name__ == '__main__':
    main()
