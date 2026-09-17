"""
RSA with max-T FWER correction (primary analysis).

Family: all traits per channel (sem and vis corrected separately),
        'all' split only.
Permutation: 1000 iterations, same target-label permutation applied to
             both model RDMs simultaneously across all traits.

Usage:
    python 2_code/run_rsa_maxt.py --dataset us
    python 2_code/run_rsa_maxt.py --dataset cn
    python 2_code/run_rsa_maxt.py --dataset both
"""

import argparse
import sys
import warnings
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

import numpy as np
import pandas as pd
from config import DATASETS, SPLITS, PIPELINE_DIR
from data_loader import load_ratings, load_rdms
from rsa_utils import (flatten_rdm, build_trait_rdm, rsa_spearman,
                       _partial_r_vecs, maxt_permutation_rsa,
                       variance_partitioning)

N_PERM = 1000
np.random.seed(42)


def run_dataset(ds_name: str):
    cfg = DATASETS[ds_name]
    print(f"\n{'='*60}")
    print(f"RSA max-T — {ds_name.upper()} ({cfg['sem_model']} vs {cfg['vis_model']})")
    print(f"Family: 'all' split, {len(cfg['traits'])} traits × {{sem,vis}} × {{simple,partial}}")
    print(f"Permutations: {N_PERM}")
    print(f"{'='*60}")

    ratings_dict     = load_ratings(cfg)
    sem_rdm, vis_rdm = load_rdms(cfg)
    split_masks = {'all': np.ones(cfg['n_targets'], dtype=bool)}

    # ── Step 1: compute observed statistics for all splits ────────────────────
    observed_rows = []
    for trait in cfg['traits']:
        ratings = ratings_dict[trait]
        for split in SPLITS:
            mask  = split_masks[split]
            n_cel = int(mask.sum())
            idx   = np.where(mask)[0]

            trait_rdm = build_trait_rdm(ratings, mask)
            sem_sub   = sem_rdm[np.ix_(idx, idx)]
            vis_sub   = vis_rdm[np.ix_(idx, idx)]

            tv  = flatten_rdm(trait_rdm)
            sv  = flatten_rdm(sem_sub)
            vv  = flatten_rdm(vis_sub)

            vp = variance_partitioning(tv, sv, vv)
            row = {
                'dataset':       ds_name,
                'trait':         trait,
                'split':         split,
                'n_celebrities': n_cel,
                'sem_model':     cfg['sem_model'],
                'vis_model':     cfg['vis_model'],
                'rsa_sem_r':     rsa_spearman(tv, sv),
                'rsa_vis_r':     rsa_spearman(tv, vv),
                'partial_sem_r': _partial_r_vecs(tv, sv, vv),
                'partial_vis_r': _partial_r_vecs(tv, vv, sv),
                **vp,
                # max-T corrected p-values filled below for 'all' split only
                'rsa_sem_p_maxt':     float('nan'),
                'rsa_vis_p_maxt':     float('nan'),
                'partial_sem_p_maxt': float('nan'),
                'partial_vis_p_maxt': float('nan'),
                'n_perm_maxt':        float('nan'),
            }
            observed_rows.append(row)

    # ── Step 2: max-T permutation for each split ─────────────────────────────
    outdir = PIPELINE_DIR / ds_name
    outdir.mkdir(parents=True, exist_ok=True)

    maxt_by_split = {}
    for split in SPLITS:
        mask = split_masks[split]
        idx  = np.where(mask)[0]
        sem_sub = sem_rdm[np.ix_(idx, idx)]
        vis_sub = vis_rdm[np.ix_(idx, idx)]

        print(f"\nBuilding max-T null distribution ({N_PERM} perms, '{split}' split, "
              f"n={mask.sum()} celebrities)...")
        trait_rdms_split = {
            trait: build_trait_rdm(ratings_dict[trait], mask)
            for trait in cfg['traits']
        }
        maxt_out = maxt_permutation_rsa(trait_rdms_split, sem_sub, vis_sub,
                                         n_perm=N_PERM, seed=42)
        maxt_by_split[split] = maxt_out

        q95_ss = np.percentile(maxt_out['null_sem'],         95)
        q95_sv = np.percentile(maxt_out['null_vis'],         95)
        q95_ps = np.percentile(maxt_out['null_partial_sem'], 95)
        q95_pv = np.percentile(maxt_out['null_partial_vis'], 95)
        print(f"  Tmax null 95th pct — sem: {q95_ss:.4f}, vis: {q95_sv:.4f}, "
              f"partial_sem: {q95_ps:.4f}, partial_vis: {q95_pv:.4f}")

        np.save(outdir / 'rsa_maxt_null_sem.npy',         maxt_out['null_sem'])
        np.save(outdir / 'rsa_maxt_null_vis.npy',         maxt_out['null_vis'])
        np.save(outdir / 'rsa_maxt_null_partial_sem.npy', maxt_out['null_partial_sem'])
        np.save(outdir / 'rsa_maxt_null_partial_vis.npy', maxt_out['null_partial_vis'])

    # ── Step 3: fill max-T p-values for all splits ───────────────────────────
    for row in observed_rows:
        split = row['split']
        trait = row['trait']
        cp    = maxt_by_split[split]['corrected_p'][trait]
        row['rsa_sem_p_maxt']     = cp['rsa_sem_p_maxt']
        row['rsa_vis_p_maxt']     = cp['rsa_vis_p_maxt']
        row['partial_sem_p_maxt'] = cp['partial_sem_p_maxt']
        row['partial_vis_p_maxt'] = cp['partial_vis_p_maxt']
        row['n_perm_maxt']        = N_PERM
        if split == 'all':
            print(f"  {trait:12s}  sem r={row['rsa_sem_r']:.3f} p_maxt={cp['rsa_sem_p_maxt']:.4f}"
                  f"  |  vis r={row['rsa_vis_r']:.3f} p_maxt={cp['rsa_vis_p_maxt']:.4f}"
                  f"  |  partial sem p_maxt={cp['partial_sem_p_maxt']:.4f}"
                  f"  |  partial vis p_maxt={cp['partial_vis_p_maxt']:.4f}")

    out     = pd.DataFrame(observed_rows)
    outpath = outdir / 'dual_rsa_maxt_results.csv'
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
