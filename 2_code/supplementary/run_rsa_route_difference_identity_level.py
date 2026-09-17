"""Identity-level test of the semantic-vs-visual RSA difference.

The Fisher's z comparison reported in the Results treats the n*(n-1)/2 target pairs
as independent observations. They are not: each of the 50 identities contributes 49
pairs, and the two correlations being compared are computed on the same trait RDM.
This script re-tests rho_sem - rho_vis with the identity as the resampling unit:

  - bootstrap over identities (pairs formed from the same original identity are
    dropped, because they are zero-distance in every RDM)
  - leave-one-identity-out jackknife

Covers the trait-space outcome as well as every individual trait, so the
semantic-vs-visual comparison is on the same footing everywhere it is reported.

Writes 3_pipeline/{ds}/rsa_route_difference_identity_level.csv
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import norm, rankdata

sys.path.insert(0, str(Path(__file__).parent.parent))
ROOT = Path(__file__).resolve().parent.parent.parent

from config import DATASETS, PIPELINE_DIR
from data_loader import load_ratings, load_rdms
from config import local_common_traits
from rsa_utils import (build_trait_rdm, build_trait_space_rdm, flatten_rdm,
                       rsa_spearman)

N_BOOT = 5000
SEED = 42


def _rho(a, b):
    return float(np.corrcoef(rankdata(a), rankdata(b))[0, 1])


def run_dataset(ds_name: str) -> pd.DataFrame:
    cfg = DATASETS[ds_name]
    ratings = load_ratings(cfg)
    sem_rdm, vis_rdm = load_rdms(cfg)
    n = sem_rdm.shape[0]
    n_pairs = n * (n - 1) // 2
    ii, jj = np.triu_indices(n, k=1)
    rng = np.random.default_rng(SEED)

    rows = []
    # 'traitspace' is the primary outcome and carries no multiple-comparison
    # family; the per-trait outcomes follow.
    for trait in ['traitspace'] + list(cfg['traits']):
        trait_rdm = (build_trait_space_rdm(ratings, local_common_traits(ds_name))
                     if trait == 'traitspace' else build_trait_rdm(ratings[trait]))
        t, s, v = (flatten_rdm(x) for x in (trait_rdm, sem_rdm, vis_rdm))
        r_sem, r_vis = rsa_spearman(t, s), rsa_spearman(t, v)
        obs = r_sem - r_vis

        diffs = np.empty(N_BOOT)
        boot_sem = np.empty(N_BOOT)
        boot_vis = np.empty(N_BOOT)
        for b in range(N_BOOT):
            idx = rng.integers(0, n, n)
            keep = idx[ii] != idx[jj]
            a, c = idx[ii][keep], idx[jj][keep]
            boot_sem[b] = _rho(trait_rdm[a, c], sem_rdm[a, c])
            boot_vis[b] = _rho(trait_rdm[a, c], vis_rdm[a, c])
            diffs[b] = boot_sem[b] - boot_vis[b]
        lo, hi = np.percentile(diffs, [2.5, 97.5])
        sem_lo, sem_hi = np.percentile(boot_sem, [2.5, 97.5])
        vis_lo, vis_hi = np.percentile(boot_vis, [2.5, 97.5])
        p_boot = min(max(2 * min((diffs <= 0).mean(), (diffs >= 0).mean()), 1 / N_BOOT), 1.0)

        jk = []
        for k in range(n):
            m = np.setdiff1d(np.arange(n), k)
            tb, sb, vb = (flatten_rdm(x[np.ix_(m, m)]) for x in (trait_rdm, sem_rdm, vis_rdm))
            jk.append(rsa_spearman(tb, sb) - rsa_spearman(tb, vb))
        jk = np.array(jk)
        se_jk = np.sqrt((n - 1) / n * ((jk - jk.mean()) ** 2).sum())
        p_jack = float(2 * norm.sf(abs(obs / se_jk)))

        z1, z2 = np.arctanh(np.clip([r_sem, r_vis], -0.9999, 0.9999))
        z_fisher = float((z1 - z2) / np.sqrt(2 / (n_pairs - 3)))

        rows.append(dict(dataset=ds_name, trait=trait, n_celebrities=n,
                         rsa_sem_r=r_sem, rsa_vis_r=r_vis, diff_sem_minus_vis=obs,
                         sem_boot_ci_lower=sem_lo, sem_boot_ci_upper=sem_hi,
                         vis_boot_ci_lower=vis_lo, vis_boot_ci_upper=vis_hi,
                         boot_ci_lower=lo, boot_ci_upper=hi, p_boot=p_boot,
                         se_jackknife=se_jk, p_jackknife=p_jack,
                         z_fisher_pairwise=z_fisher,
                         p_fisher_pairwise=float(2 * norm.sf(abs(z_fisher))),
                         n_boot=N_BOOT))

    df = pd.DataFrame(rows)
    out = PIPELINE_DIR / ds_name / 'rsa_route_difference_identity_level.csv'
    out.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(out, index=False)
    print(f"Saved {out}")
    return df


if __name__ == '__main__':
    for ds in ['us', 'cn']:
        d = run_dataset(ds)
        flip = d[(d.p_fisher_pairwise < .05) & (d.p_boot >= .05)]
        print(f"[{ds}] Fisher-significant but not identity-level significant: "
              f"{', '.join(flip.trait) if len(flip) else 'none'}\n")
