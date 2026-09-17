"""
RSA + Partial RSA + Variance Partitioning + Fam Permutation using the
trait-space RDM (pairwise Euclidean distance across all trait means).

Outputs
-------
3_pipeline/{ds}/trait_space_rsa_results.csv

Usage:
    python 2_code/run_rsa_trait_space.py --dataset us
    python 2_code/run_rsa_trait_space.py --dataset cn
    python 2_code/run_rsa_trait_space.py --dataset both
"""

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

import numpy as np
import pandas as pd
from config import DATASETS, SPLITS, PIPELINE_DIR, local_common_traits
from data_loader import load_ratings, load_rdms
from rsa_utils import (flatten_rdm, build_trait_space_rdm, permutation_test,
                       partial_rsa, variance_partitioning,
                                              maxt_permutation_rsa)

N_PERM      = 5000
N_PERM_FWER = 1000
np.random.seed(42)


def run_dataset(ds_name: str):
    cfg = DATASETS[ds_name]
    traits = local_common_traits(ds_name)
    print(f"\n{'='*60}")
    print(f"Trait-space RSA — {ds_name.upper()} ({cfg['sem_model']} vs {cfg['vis_model']})")
    print(f"Common traits ({len(traits)}): {traits}")
    print(f"{'='*60}")

    ratings_dict         = load_ratings(cfg)
    sem_rdm, vis_rdm     = load_rdms(cfg)
    split_masks = {'all': np.ones(cfg['n_targets'], dtype=bool)}

    results = []
    for split in SPLITS:
        mask  = split_masks[split]
        n_cel = int(mask.sum())
        print(f"[{split}] n={n_cel}", flush=True)

        trait_rdm = build_trait_space_rdm(ratings_dict, traits, mask)
        idx       = np.where(mask)[0]
        sem_sub   = sem_rdm[np.ix_(idx, idx)]
        vis_sub   = vis_rdm[np.ix_(idx, idx)]

        trait_vec = flatten_rdm(trait_rdm)
        sem_vec   = flatten_rdm(sem_sub)
        vis_vec   = flatten_rdm(vis_sub)

        rsa_sem_r, rsa_sem_p         = permutation_test(trait_rdm, sem_sub, N_PERM)
        rsa_vis_r, rsa_vis_p         = permutation_test(trait_rdm, vis_sub, N_PERM)
        partial_sem_r, partial_sem_p = partial_rsa(trait_rdm, sem_sub, vis_sub, N_PERM)
        partial_vis_r, partial_vis_p = partial_rsa(trait_rdm, vis_sub, sem_sub, N_PERM)
        vp = variance_partitioning(trait_vec, sem_vec, vis_vec)

        # FWER-corrected p (p_fwer): trait-space is a single outcome (family size 1,
        # not a family of per-trait tests), so this reuses the same joint-permutation
        # max-T machinery as the per-trait analysis with a one-item trait dict — the
        # "correction" here is just the finite-sample (+1)/(n+1) adjustment on its own
        # null, giving a value that tracks rsa_*_p closely (different draws/n_perm).
        maxt_out = maxt_permutation_rsa({'trait_space': trait_rdm}, sem_sub, vis_sub,
                                        n_perm=N_PERM_FWER, seed=42)
        rsa_sem_p_fwer = maxt_out['corrected_p']['trait_space']['rsa_sem_p_maxt']
        rsa_vis_p_fwer = maxt_out['corrected_p']['trait_space']['rsa_vis_p_maxt']

        print(f"  sem r={rsa_sem_r:.3f} p={rsa_sem_p:.4f} p_fwer={rsa_sem_p_fwer:.4f} | "
              f"vis r={rsa_vis_r:.3f} p={rsa_vis_p:.4f} p_fwer={rsa_vis_p_fwer:.4f}")

        results.append({
            'dataset':       ds_name,
            'split':         split,
            'n_celebrities': n_cel,
            'n_traits':      len(traits),
            'sem_model':     cfg['sem_model'],
            'vis_model':     cfg['vis_model'],
            'rsa_sem_r':     rsa_sem_r,
            'rsa_sem_p':     rsa_sem_p,
            'rsa_sem_p_fwer': rsa_sem_p_fwer,
            'rsa_vis_r':     rsa_vis_r,
            'rsa_vis_p':     rsa_vis_p,
            'rsa_vis_p_fwer': rsa_vis_p_fwer,
            'partial_sem_r': partial_sem_r,
            'partial_sem_p': partial_sem_p,
            'partial_vis_r': partial_vis_r,
            'partial_vis_p': partial_vis_p,
            **vp,
        })

    outdir = PIPELINE_DIR / ds_name
    outdir.mkdir(parents=True, exist_ok=True)

    rsa_path = outdir / 'trait_space_rsa_results.csv'
    pd.DataFrame(results).to_csv(rsa_path, index=False)
    print(f"\nSaved {len(results)} rows → {rsa_path}")

    # Familiarity unique-variance permutation
    return pd.DataFrame(results)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', choices=['us', 'cn', 'both'], default='both')
    args = parser.parse_args()
    datasets = ['us', 'cn'] if args.dataset == 'both' else [args.dataset]
    for ds in datasets:
        run_dataset(ds)


if __name__ == '__main__':
    main()
