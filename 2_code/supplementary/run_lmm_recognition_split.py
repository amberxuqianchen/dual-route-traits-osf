"""
Trial-level recognition split, LMM version (+ Fisher's z on the RSA correlations).

Companion to run_rsa_recognition_trial_split.py.  Where that script compares
unique variance between two target-mean RDMs, this one keeps every perceiver x
pair observation and fits the base model separately within each condition:

    abs_diff ~ sem_dissim_z + vis_dissim_z
               + (1|participant) + (1|target1) + (1|target2)

fitted once on recognised pairs and once on unrecognised pairs.  A pair counts
as recognised when the perceiver recognised BOTH targets and unrecognised when
they recognised NEITHER; mixed pairs are dropped so the contrast matches the RSA
split.  US familiarity is a binary recognition judgement (1/7), so a graded
familiarity moderator is not available there — hence separate fits rather than
an interaction term.

The recognised-minus-unrecognised difference in each beta is tested with a Wald
z on the two independent-subset estimates.

This differs from run_lmm.py --model fam, whose moderator is the continuous mean
familiarity of the two targets.

Fisher's z additionally compares the recognised vs unrecognised RSA correlations
per route, in two forms:
  - naive : treats the n*(n-1)/2 target pairs as independent (repo convention)
  - identity-level bootstrap : resamples the 50 identities, which is the honest
    unit of analysis (see run_rsa_route_difference_identity_level.py)

Usage:
    python 2_code/supplementary/run_lmm_recognition_split.py --dataset both
"""

import argparse
import sys
import warnings
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

import numpy as np
import pandas as pd
from scipy.stats import norm, rankdata
from config import DATASETS, PIPELINE_DIR
from data_loader import load_ratings, load_rdms, load_familiarity_ratings
from lmm_utils import configure_r

warnings.filterwarnings('ignore')

N_BOOT = 5000
SEED = 42

# Recognition criteria, as (is_recognised, is_unrecognised) predicates.
#   midpoint : CN split at the scale midpoint (>=4 vs <=3); US unchanged.
#   any      : CN treats any rating above the floor as recognition (>=2 vs ==1),
#              making the CN contrast structurally parallel to the binary US item.
CRITERIA = {
    'midpoint': {
        'us': (lambda f: f == 7, lambda f: f == 1),
        'cn': (lambda f: f >= 4, lambda f: f <= 3),
    },
    'any': {
        'us': (lambda f: f == 7, lambda f: f == 1),
        'cn': (lambda f: f >= 2, lambda f: f == 1),
    },
    'strict': {
        'us': (lambda f: f == 7, lambda f: f == 1),
        'cn': (lambda f: f >= 5, lambda f: f <= 4),
    },
}
SUFFIX = {'midpoint': '', 'any': '_any', 'strict': '_strict'}

# Set by main(); workers inherit it. Default preserves the original analysis.
RECOGNITION_CRITERION = CRITERIA['midpoint']


def _rho(a, b):
    return float(np.corrcoef(rankdata(a), rankdata(b))[0, 1])


def _fisher_z(r1, n1, r2, n2):
    """Two-tailed Fisher's z comparing two correlations; returns (z, p)."""
    z1, z2 = np.arctanh(np.clip([r1, r2], -0.9999, 0.9999))
    se = np.sqrt(1 / (n1 - 3) + 1 / (n2 - 3))
    z = (z1 - z2) / se
    return float(z), float(2 * norm.sf(abs(z)))


def _build_pairs(ratings, fam, sem_rdm, vis_rdm, ds_name):
    """Perceiver x pair rows for both-recognised and neither-recognised pairs."""
    is_rec, is_unrec = RECOGNITION_CRITERION[ds_name]
    n = sem_rdm.shape[0]
    ii, jj = np.triu_indices(n, k=1)

    sem = sem_rdm[ii, jj]
    vis = vis_rdm[ii, jj]
    sem_z = (sem - sem.mean()) / sem.std()
    vis_z = (vis - vis.mean()) / vis.std()

    frames = []
    for p in range(ratings.shape[0]):
        r, f = ratings[p], fam[p]
        rec, unrec = is_rec(f), is_unrec(f)
        d = np.abs(r[ii] - r[jj])

        both = rec[ii] & rec[jj]
        neither = unrec[ii] & unrec[jj]
        keep = (both | neither) & ~np.isnan(d)
        if not keep.any():
            continue

        frames.append(pd.DataFrame({
            'participant':  p,
            'target1':      ii[keep],
            'target2':      jj[keep],
            'abs_diff':     d[keep],
            'sem_dissim_z': sem_z[keep],
            'vis_dissim_z': vis_z[keep],
            'recognized':   both[keep].astype(float),
        }))

    df = pd.concat(frames, ignore_index=True)
    for c in ('participant', 'target1', 'target2'):
        df[c] = df[c].astype(str)
    return df


def _fit(df, suffix):
    """Base LMM on one condition's pairs; keys are tagged with `suffix`."""
    from pymer4.models import Lmer
    m = Lmer('abs_diff ~ sem_dissim_z + vis_dissim_z'
             ' + (1|participant) + (1|target1) + (1|target2)', data=df)
    m.fit(summarize=False)
    fe = m.coefs
    out = {f'n_obs_{suffix}': len(df), f'converged_{suffix}': not m.warnings}
    for term in ['sem_dissim_z', 'vis_dissim_z']:
        row = fe.loc[term]
        out[f'{term}_beta_{suffix}'] = row['Estimate']
        out[f'{term}_se_{suffix}'] = row['SE']
        out[f'{term}_p_{suffix}'] = row['P-val']
        out[f'{term}_ci_lower_{suffix}'] = row['2.5_ci']
        out[f'{term}_ci_upper_{suffix}'] = row['97.5_ci']
    return out


def _wald(b1, se1, b2, se2):
    """Wald z and two-tailed p for b1 - b2 from two disjoint subsets."""
    se = np.sqrt(se1 ** 2 + se2 ** 2)
    z = (b1 - b2) / se
    return float(z), float(2 * norm.sf(abs(z)))


def _fisher_block(ratings, fam, sem_rdm, vis_rdm, ds_name, rng):
    """Recognised vs unrecognised RSA correlations + naive and identity-level tests."""
    is_rec, is_unrec = RECOGNITION_CRITERION[ds_name]
    n = sem_rdm.shape[0]
    ii, jj = np.triu_indices(n, k=1)
    n_pairs = len(ii)

    def means(pred):
        m = pred(fam) & ~np.isnan(ratings)
        cnt = m.sum(axis=0)
        s = np.where(m, np.nan_to_num(ratings), 0.0).sum(axis=0)
        return np.where(cnt > 0, s / np.maximum(cnt, 1), np.nan)

    mr, mu = means(is_rec), means(is_unrec)
    tr = np.abs(mr[ii] - mr[jj])
    tu = np.abs(mu[ii] - mu[jj])
    sem, vis = sem_rdm[ii, jj], vis_rdm[ii, jj]

    out = {}
    obs = {}
    for lab, mdl in (('sem', sem), ('vis', vis)):
        r_rec, r_unrec = _rho(tr, mdl), _rho(tu, mdl)
        z, p = _fisher_z(r_rec, n_pairs, r_unrec, n_pairs)
        obs[lab] = r_rec - r_unrec
        out.update({f'rsa_{lab}_r_rec': r_rec, f'rsa_{lab}_r_unrec': r_unrec,
                    f'fisher_z_{lab}': z, f'fisher_p_{lab}_naive': p})

    # Identity-level bootstrap: resample the 50 identities, drop self-pairs
    boot = {'sem': np.empty(N_BOOT), 'vis': np.empty(N_BOOT)}
    for b in range(N_BOOT):
        idx = rng.integers(0, n, n)
        keep = idx[ii] != idx[jj]
        a, c = idx[ii][keep], idx[jj][keep]
        dr = np.abs(mr[a] - mr[c])
        du = np.abs(mu[a] - mu[c])
        for lab, rdm in (('sem', sem_rdm), ('vis', vis_rdm)):
            mv = rdm[a, c]
            boot[lab][b] = _rho(dr, mv) - _rho(du, mv)

    for lab in ('sem', 'vis'):
        d = boot[lab][np.isfinite(boot[lab])]
        out[f'boot_delta_{lab}_lo'] = float(np.percentile(d, 2.5))
        out[f'boot_delta_{lab}_hi'] = float(np.percentile(d, 97.5))
        # two-tailed bootstrap p for delta != 0
        frac = float(np.mean(d <= 0)) if obs[lab] > 0 else float(np.mean(d >= 0))
        out[f'fisher_p_{lab}_identity'] = min(1.0, 2 * max(frac, 1.0 / len(d)))
    return out


def run_dataset(ds_name: str, variant: str = 'midpoint'):
    cfg = DATASETS[ds_name]
    print(f"\n{'='*70}\nRecognition-split LMM — {ds_name.upper()} "
          f"[{variant}]\n{'='*70}")

    ratings_dict = load_ratings(cfg)
    fam = load_familiarity_ratings(cfg)
    sem_rdm, vis_rdm = load_rdms(cfg)
    rng = np.random.default_rng(SEED)

    outdir = PIPELINE_DIR / ds_name
    outdir.mkdir(parents=True, exist_ok=True)
    outpath = outdir / f'lmm_recognition_split{SUFFIX[variant]}.csv'

    results = []
    for i, trait in enumerate(cfg['traits'], 1):
        ratings = ratings_dict[trait]
        df = _build_pairs(ratings, fam, sem_rdm, vis_rdm, ds_name)
        n_both = int(df.recognized.sum())
        print(f"  [{i}/{len(cfg['traits'])}] {trait}: {len(df)} pairs "
              f"({n_both} both-recognised, {len(df)-n_both} neither) ...",
              flush=True)

        row = {'dataset': ds_name, 'trait': trait,
               'sem_model': cfg['sem_model'], 'vis_model': cfg['vis_model'],
               'n_pairs_rec': n_both, 'n_pairs_unrec': len(df) - n_both}
        row.update(_fit(df[df.recognized == 1], 'rec'))
        row.update(_fit(df[df.recognized == 0], 'unrec'))

        for term in ('sem_dissim_z', 'vis_dissim_z'):
            z, p = _wald(row[f'{term}_beta_rec'], row[f'{term}_se_rec'],
                         row[f'{term}_beta_unrec'], row[f'{term}_se_unrec'])
            row[f'{term}_delta'] = row[f'{term}_beta_rec'] - row[f'{term}_beta_unrec']
            row[f'{term}_delta_z'] = z
            row[f'{term}_delta_p'] = p

        row.update(_fisher_block(ratings, fam, sem_rdm, vis_rdm, ds_name, rng))
        results.append(row)

        print(f"      sem β rec={row['sem_dissim_z_beta_rec']:+.4f} "
              f"unrec={row['sem_dissim_z_beta_unrec']:+.4f} "
              f"Δ={row['sem_dissim_z_delta']:+.4f} p={row['sem_dissim_z_delta_p']:.4g}")
        print(f"      vis β rec={row['vis_dissim_z_beta_rec']:+.4f} "
              f"unrec={row['vis_dissim_z_beta_unrec']:+.4f} "
              f"Δ={row['vis_dissim_z_delta']:+.4f} p={row['vis_dissim_z_delta_p']:.4g}")
        print(f"      Fisher sem z={row['fisher_z_sem']:+.2f} "
              f"p_naive={row['fisher_p_sem_naive']:.4g} "
              f"p_identity={row['fisher_p_sem_identity']:.4g} | "
              f"vis z={row['fisher_z_vis']:+.2f} "
              f"p_naive={row['fisher_p_vis_naive']:.4g} "
              f"p_identity={row['fisher_p_vis_identity']:.4g}", flush=True)

        pd.DataFrame(results).to_csv(outpath, index=False)

    print(f"\nSaved {len(results)} rows → {outpath}")
    return pd.DataFrame(results)


def main():
    global RECOGNITION_CRITERION
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', choices=['us', 'cn', 'both'], default='both')
    parser.add_argument('--variant', choices=list(CRITERIA), default='midpoint')
    args = parser.parse_args()
    RECOGNITION_CRITERION = CRITERIA[args.variant]
    configure_r()
    for ds in (['us', 'cn'] if args.dataset == 'both' else [args.dataset]):
        run_dataset(ds, args.variant)


if __name__ == '__main__':
    main()
