"""
RSA + Partial RSA + Variance Partitioning for one or both datasets.

Usage:
    python 2_code/run_rsa.py --dataset us
    python 2_code/run_rsa.py --dataset cn
    python 2_code/run_rsa.py --dataset both
"""

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

import numpy as np
import pandas as pd
from config import DATASETS, SPLITS, PIPELINE_DIR
from data_loader import load_ratings, load_rdms
from rsa_utils import (flatten_rdm, build_trait_rdm, permutation_test,
                       partial_rsa, variance_partitioning)

N_PERM = 5000
np.random.seed(42)


def run_dataset(ds_name: str):
    cfg = DATASETS[ds_name]
    print(f"\n{'='*60}")
    print(f"RSA — {ds_name.upper()} ({cfg['sem_model']} vs {cfg['vis_model']})")
    print(f"{'='*60}")

    ratings_dict     = load_ratings(cfg)
    sem_rdm, vis_rdm = load_rdms(cfg)
    split_masks = {'all': np.ones(cfg['n_targets'], dtype=bool)}

    results = []
    total = len(cfg['traits']) * len(SPLITS)
    done  = 0

    for trait in cfg['traits']:
        ratings = ratings_dict[trait]

        for split in SPLITS:
            done += 1
            mask = split_masks[split]
            n_cel = int(mask.sum())
            print(f"[{done}/{total}] {trait} | {split} (n={n_cel})", flush=True)

            trait_rdm = build_trait_rdm(ratings, mask)
            sem_sub   = sem_rdm[np.ix_(np.where(mask)[0], np.where(mask)[0])]
            vis_sub   = vis_rdm[np.ix_(np.where(mask)[0], np.where(mask)[0])]

            trait_vec = flatten_rdm(trait_rdm)
            sem_vec   = flatten_rdm(sem_sub)
            vis_vec   = flatten_rdm(vis_sub)

            rsa_sem_r, rsa_sem_p         = permutation_test(trait_rdm, sem_sub, N_PERM)
            rsa_vis_r, rsa_vis_p         = permutation_test(trait_rdm, vis_sub, N_PERM)
            partial_sem_r, partial_sem_p = partial_rsa(trait_rdm, sem_sub, vis_sub, N_PERM)
            partial_vis_r, partial_vis_p = partial_rsa(trait_rdm, vis_sub, sem_sub, N_PERM)
            vp = variance_partitioning(trait_vec, sem_vec, vis_vec)

            print(f"  sem r={rsa_sem_r:.3f} p={rsa_sem_p:.4f} | "
                  f"vis r={rsa_vis_r:.3f} p={rsa_vis_p:.4f}")

            results.append({
                'dataset':       ds_name,
                'trait':         trait,
                'split':         split,
                'n_celebrities': n_cel,
                'sem_model':     cfg['sem_model'],
                'vis_model':     cfg['vis_model'],
                'rsa_sem_r':     rsa_sem_r,
                'rsa_sem_p':     rsa_sem_p,
                'rsa_vis_r':     rsa_vis_r,
                'rsa_vis_p':     rsa_vis_p,
                'partial_sem_r': partial_sem_r,
                'partial_sem_p': partial_sem_p,
                'partial_vis_r': partial_vis_r,
                'partial_vis_p': partial_vis_p,
                **vp,
            })

        # Intermediate save after each trait
        outdir = PIPELINE_DIR / ds_name
        outdir.mkdir(parents=True, exist_ok=True)
        pd.DataFrame(results).to_csv(
            outdir / 'dual_rsa_results_partial.csv', index=False)

    out = pd.DataFrame(results)
    outpath = PIPELINE_DIR / ds_name / 'dual_rsa_results.csv'
    out.to_csv(outpath, index=False)
    print(f"\nSaved {len(out)} rows → {outpath}")
    return out


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', choices=['us', 'cn', 'both'], default='both')
    args = parser.parse_args()

    datasets = ['us', 'cn'] if args.dataset == 'both' else [args.dataset]
    for ds in datasets:
        run_dataset(ds)


if __name__ == '__main__':
    main()
