"""
Identity-level inference for variance-partitioning unique variance.

The ANOVA F-test used for the unique contributions treats the n(n-1)/2 RDM
cells as independent observations (df2 = n_pairs - 3).  They are not: each of
the 50 targets contributes to 49 pairs, so that df badly overstates the
effective sample size.  Both procedures here resample the 50 targets instead:

- permutation: shuffle the target labels of the trait RDM (rows and columns
  together), recompute the unique variance, and read the observed value off
  the resulting null distribution.  Upper-tail, because unique variance is
  non-negative by construction.
- bootstrap: resample the 50 targets with replacement, drop the pairs where a
  duplicated target meets itself, and take percentile confidence intervals for
  the unique variance.

The bootstrap resamples also give a paired comparison of the two routes: the
difference in unique variance (semantic minus visual) is recomputed on each
resample, giving a percentile CI and a two-sided p.

A single permutation is applied to every outcome at once, so the same draws
also give a max-T family-wise correction: the null of the maximum unique
variance across all traits, built separately per channel and per dataset
(family = traits x 1 channel x 1 dataset).  The trait space is a single outcome
by construction and is excluded from the family, matching how the per-trait RSA
correlations are corrected elsewhere in the pipeline.

Runs for the 'all' split, so the dots in the variance-partitioning figures
use the same inference.

Outputs
-------
3_pipeline/{ds}/vp_identity_inference.csv

Usage:
    python 2_code/run_vp_identity_inference.py --dataset us|cn|both
"""

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

import numpy as np
import pandas as pd
from config import DATASETS, PIPELINE_DIR, SPLITS, local_common_traits
from data_loader import load_ratings, load_rdms
from rsa_utils import build_trait_rdm, build_trait_space_rdm

N_PERM = 5000
N_BOOT = 5000
SEED = 42


def _unique_variance(y, x1, x2):
    """(unique_sem, unique_vis, total_r2) for OLS y ~ x1 + x2.

    Closed form for two predictors: with everything standardised,
    R2_full = (r_y1^2 + r_y2^2 - 2 r_y1 r_y2 r_12) / (1 - r_12^2), and the
    single-predictor R2 are just r_y1^2 and r_y2^2.
    """
    r_y1 = np.corrcoef(y, x1)[0, 1]
    r_y2 = np.corrcoef(y, x2)[0, 1]
    r_12 = np.corrcoef(x1, x2)[0, 1]
    r2_full = (r_y1**2 + r_y2**2 - 2 * r_y1 * r_y2 * r_12) / (1 - r_12**2)
    return r2_full - r_y2**2, r2_full - r_y1**2, r2_full


def _outcome_rdm(outcome, ratings_dict, traits, mask=None):
    if outcome == 'traitspace':
        return build_trait_space_rdm(ratings_dict, traits, mask)
    return build_trait_rdm(ratings_dict[outcome], mask)


def run_dataset(ds_name: str) -> Path:
    cfg = DATASETS[ds_name]
    rng = np.random.default_rng(SEED)
    ratings_dict = load_ratings(cfg)
    sem_rdm_full, vis_rdm_full = load_rdms(cfg)
    masks = {'all': np.ones(cfg['n_targets'], dtype=bool)}

    rows = []
    for split in SPLITS:
        rows.extend(_run_split(ds_name, cfg, ratings_dict, sem_rdm_full, vis_rdm_full,
                               masks[split], split, rng))

    out = PIPELINE_DIR / ds_name / 'vp_identity_inference.csv'
    pd.DataFrame(rows).to_csv(out, index=False)
    print(f"Saved {out}")
    return out


def _run_split(ds_name, cfg, ratings_dict, sem_full, vis_full, mask, split, rng):
    idx = np.where(mask)[0]
    sem_rdm = sem_full[np.ix_(idx, idx)]
    vis_rdm = vis_full[np.ix_(idx, idx)]
    n = len(idx)
    ii, jj = np.triu_indices(n, k=1)
    print(f"[{split}] n={n}")

    outcomes = ['traitspace'] + list(cfg['traits'])
    rdms = {o: _outcome_rdm(o, ratings_dict, local_common_traits(ds_name), mask)
            for o in outcomes}
    sv, vv = sem_rdm[ii, jj], vis_rdm[ii, jj]

    obs = {o: _unique_variance(rdms[o][ii, jj], sv, vv) for o in outcomes}

    # ── permutation over target labels, shared across outcomes ────────────
    # One permutation per iteration is applied to every trait RDM, so the same
    # draws serve both the per-outcome null and the max-T null.
    null_sem = np.empty((N_PERM, len(outcomes)))
    null_vis = np.empty((N_PERM, len(outcomes)))
    for b in range(N_PERM):
        pm = rng.permutation(n)
        for k, o in enumerate(outcomes):
            yp = rdms[o][np.ix_(pm, pm)][ii, jj]
            null_sem[b, k], null_vis[b, k], _ = _unique_variance(yp, sv, vv)

    # max-T family: the traits, per channel, within this dataset
    fam = [k for k, o in enumerate(outcomes) if o != 'traitspace']
    maxnull_sem = null_sem[:, fam].max(axis=1)
    maxnull_vis = null_vis[:, fam].max(axis=1)

    out_rows = []
    for k, outcome in enumerate(outcomes):
        obs_sem, obs_vis, r2_full = obs[outcome]
        p_sem = (np.sum(null_sem[:, k] >= obs_sem) + 1) / (N_PERM + 1)
        p_vis = (np.sum(null_vis[:, k] >= obs_vis) + 1) / (N_PERM + 1)
        if outcome == 'traitspace':      # single outcome, no family
            pm_sem = pm_vis = np.nan
        else:
            pm_sem = (np.sum(maxnull_sem >= obs_sem) + 1) / (N_PERM + 1)
            pm_vis = (np.sum(maxnull_vis >= obs_vis) + 1) / (N_PERM + 1)

        # ── bootstrap over targets ────────────────────────────────────────
        trait_rdm = rdms[outcome]
        boot_sem = np.empty(N_BOOT)
        boot_vis = np.empty(N_BOOT)
        for b in range(N_BOOT):
            idx = rng.integers(0, n, n)
            a, c = idx[ii], idx[jj]
            keep = a != c
            a, c = a[keep], c[keep]
            boot_sem[b], boot_vis[b], _ = _unique_variance(
                trait_rdm[a, c], sem_rdm[a, c], vis_rdm[a, c])
        sem_lo, sem_hi = np.percentile(boot_sem, [2.5, 97.5])
        vis_lo, vis_hi = np.percentile(boot_vis, [2.5, 97.5])

        # semantic vs visual unique variance, same resamples so the two are paired
        bdiff = boot_sem - boot_vis
        d_lo, d_hi = np.percentile(bdiff, [2.5, 97.5])
        p_diff = min(max(2 * min((bdiff <= 0).mean(), (bdiff >= 0).mean()),
                         1 / N_BOOT), 1.0)

        out_rows.append(dict(
            dataset=ds_name, split=split, outcome=outcome, n_targets=n, n_pairs=len(ii),
            total_r2=r2_full, unique_sem=obs_sem, unique_vis=obs_vis,
            unique_sem_p_perm=p_sem, unique_vis_p_perm=p_vis,
            unique_sem_p_maxt=pm_sem, unique_vis_p_maxt=pm_vis,
            unique_sem_ci_lower=sem_lo, unique_sem_ci_upper=sem_hi,
            unique_vis_ci_lower=vis_lo, unique_vis_ci_upper=vis_hi,
            diff_sem_minus_vis=obs_sem - obs_vis,
            diff_ci_lower=d_lo, diff_ci_upper=d_hi, diff_p_boot=p_diff,
            n_perm=N_PERM, n_boot=N_BOOT))
        print(f"  {outcome:12s} sem {obs_sem:.4f} p={p_sem:.4f} pmaxt={pm_sem:.4f} "
              f"[{sem_lo:.4f},{sem_hi:.4f}]  vis {obs_vis:.4f} p={p_vis:.4f} "
              f"pmaxt={pm_vis:.4f} [{vis_lo:.4f},{vis_hi:.4f}]  "
              f"diff {obs_sem - obs_vis:+.4f} p={p_diff:.4f}", flush=True)

    return out_rows


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--dataset', choices=['us', 'cn', 'both'], default='both')
    args = ap.parse_args()
    for ds in (['us', 'cn'] if args.dataset == 'both' else [args.dataset]):
        print(f"\n{'='*60}\nVP identity-level inference — {ds.upper()}\n{'='*60}")
        run_dataset(ds)


if __name__ == '__main__':
    main()
