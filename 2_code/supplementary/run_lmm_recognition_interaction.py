"""
Recognition split as a single interaction LMM (companion to the two-fit version).

Rather than fitting each recognition condition separately and differencing the
coefficients, this fits one model over all retained pairs with recognition as a
0/1 factor:

    abs_diff ~ (sem_dissim_z + vis_dissim_z) * recognized
               + (1|participant) + (1|target1) + (1|target2)

    recognized = 1  perceiver recognised BOTH targets of the pair
    recognized = 0  perceiver recognised NEITHER

The sem_dissim_z:recognized and vis_dissim_z:recognized terms are the estimates
of interest; they answer the same question as Delta-beta in the two-fit version.

Why this is preferable under unbalanced arms: the two-fit approach estimates
separate random-effect and residual variances within each condition, so the
smaller arm contributes noisy variance components, and the difference then has
to be tested with a Wald z that assumes the two estimates are independent (they
are not — they share participants and target identities).  The single model
pools all observations, estimates one set of variance components, and returns
the interaction standard error directly.

Variants are shared with run_lmm_recognition_split.py (midpoint / any / strict).

Usage:
    python 2_code/supplementary/run_lmm_recognition_interaction.py --dataset both --variant midpoint
"""

import argparse
import sys
import warnings
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))
sys.path.insert(0, str(Path(__file__).parent))

import numpy as np
import pandas as pd
from config import DATASETS, PIPELINE_DIR
from data_loader import load_ratings, load_rdms, load_familiarity_ratings
from lmm_utils import configure_r
import run_lmm_recognition_split as rsplit
from run_lmm_recognition_split import _build_pairs, CRITERIA, SUFFIX

warnings.filterwarnings('ignore')

TERMS = ['sem_dissim_z', 'vis_dissim_z', 'recognized',
         'sem_dissim_z:recognized', 'vis_dissim_z:recognized']


def _fit(df):
    from pymer4.models import Lmer
    m = Lmer('abs_diff ~ sem_dissim_z * recognized + vis_dissim_z * recognized'
             ' + (1|participant) + (1|target1) + (1|target2)', data=df)
    m.fit(summarize=False)
    fe = m.coefs
    out = {'n_obs': len(df), 'converged': not m.warnings, 'AIC': m.AIC}
    for term in TERMS:
        key = term.replace(':', '_X_')
        # R orders interaction factors by first appearance, so the visual
        # interaction comes back as 'recognized:vis_dissim_z'
        name = term if term in fe.index else ':'.join(reversed(term.split(':')))
        if name not in fe.index:
            continue
        r = fe.loc[name]
        out[f'{key}_beta'] = r['Estimate']
        out[f'{key}_se'] = r['SE']
        out[f'{key}_t'] = r['T-stat']
        out[f'{key}_p'] = r['P-val']
        out[f'{key}_ci_lower'] = r['2.5_ci']
        out[f'{key}_ci_upper'] = r['97.5_ci']
    return out


def run_dataset(ds_name, variant):
    cfg = DATASETS[ds_name]
    print(f"\n{'='*70}\nRecognition interaction LMM — {ds_name.upper()} "
          f"[{variant}]\n{'='*70}", flush=True)

    ratings_dict = load_ratings(cfg)
    fam = load_familiarity_ratings(cfg)
    sem_rdm, vis_rdm = load_rdms(cfg)

    outdir = PIPELINE_DIR / ds_name
    outdir.mkdir(parents=True, exist_ok=True)
    outpath = outdir / f'lmm_recognition_interaction{SUFFIX[variant]}.csv'

    results = []
    for i, trait in enumerate(cfg['traits'], 1):
        df = _build_pairs(ratings_dict[trait], fam, sem_rdm, vis_rdm, ds_name)
        n_rec = int(df.recognized.sum())
        print(f"  [{i}/{len(cfg['traits'])}] {trait}: {len(df)} pairs "
              f"({n_rec} rec / {len(df) - n_rec} unrec) ...", flush=True)

        row = {'dataset': ds_name, 'trait': trait, 'variant': variant,
               'n_pairs_rec': n_rec, 'n_pairs_unrec': len(df) - n_rec}
        row.update(_fit(df))
        results.append(row)
        print(f"      sem×rec β={row['sem_dissim_z_X_recognized_beta']:+.4f} "
              f"(SE {row['sem_dissim_z_X_recognized_se']:.4f}) "
              f"p={row['sem_dissim_z_X_recognized_p']:.4g} | "
              f"vis×rec β={row['vis_dissim_z_X_recognized_beta']:+.4f} "
              f"(SE {row['vis_dissim_z_X_recognized_se']:.4f}) "
              f"p={row['vis_dissim_z_X_recognized_p']:.4g}", flush=True)
        pd.DataFrame(results).to_csv(outpath, index=False)

    print(f"\nSaved {len(results)} rows → {outpath}")
    return pd.DataFrame(results)


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--dataset', choices=['us', 'cn', 'both'], default='both')
    p.add_argument('--variant', choices=list(CRITERIA), default='midpoint')
    a = p.parse_args()
    rsplit.RECOGNITION_CRITERION = CRITERIA[a.variant]
    configure_r()
    for ds in (['us', 'cn'] if a.dataset == 'both' else [a.dataset]):
        run_dataset(ds, a.variant)


if __name__ == '__main__':
    main()
