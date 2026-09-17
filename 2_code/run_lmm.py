"""
Linear Mixed Models for one or both datasets.

Usage:
    python 2_code/run_lmm.py --dataset us
    python 2_code/run_lmm.py --dataset cn
    python 2_code/run_lmm.py --dataset both
    python 2_code/run_lmm.py --dataset both --model fam_interaction
    python 2_code/run_lmm.py --dataset both --model both
"""

import argparse
import sys
import warnings
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

import numpy as np
import pandas as pd
from config import DATASETS, SPLITS, PIPELINE_DIR
from data_loader import load_ratings, load_rdms, load_familiarity_ratings
from lmm_utils import configure_r, build_pair_data, fit_lmm

warnings.filterwarnings('ignore')
np.random.seed(42)


def run_dataset(ds_name: str):
    cfg = DATASETS[ds_name]
    print(f"\n{'='*60}")
    print(f"LMM — {ds_name.upper()} ({cfg['sem_model']} vs {cfg['vis_model']})")
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
            print(f"\n[{done}/{total}] {trait} | {split} (n={n_cel})")

            try:
                df_pairs = build_pair_data(ratings, mask, sem_rdm, vis_rdm)
                print(f"  Pairs: {len(df_pairs)}")
                lmm_out = fit_lmm(df_pairs)

                row = {
                    'dataset':       ds_name,
                    'trait':         trait,
                    'split':         split,
                    'n_celebrities': n_cel,
                    'sem_model':     cfg['sem_model'],
                    'vis_model':     cfg['vis_model'],
                }
                row.update(lmm_out)
                results.append(row)
                print(f"  converged={lmm_out['converged']}  AIC={lmm_out['AIC']:.1f}  "
                      f"LR_p={lmm_out['lr_p']:.4g}")
                print(f"  sem β={lmm_out['sem_dissim_z_beta']:.4f} "
                      f"p={lmm_out['sem_dissim_z_p']:.4f}  |  "
                      f"vis β={lmm_out['vis_dissim_z_beta']:.4f} "
                      f"p={lmm_out['vis_dissim_z_p']:.4f}")
            except Exception as e:
                print(f"  ERROR: {e}")
                results.append({'dataset': ds_name, 'trait': trait,
                                'split': split, 'error': str(e)})

        # Intermediate save after each trait
        outdir = PIPELINE_DIR / ds_name
        outdir.mkdir(parents=True, exist_ok=True)
        pd.DataFrame(results).to_csv(
            outdir / 'dual_lmm_results_partial.csv', index=False)

    out = pd.DataFrame(results)
    outpath = PIPELINE_DIR / ds_name / 'dual_lmm_results.csv'
    out.to_csv(outpath, index=False)
    print(f"\nSaved {len(out)} rows → {outpath}")
    return out


def run_fam_interaction_dataset(ds_name: str):
    cfg = DATASETS[ds_name]
    print(f"\n{'='*60}")
    print(f"LMM (fam_interaction) — {ds_name.upper()} "
          f"({cfg['sem_model']} vs {cfg['vis_model']})")
    print(f"{'='*60}")

    ratings_dict     = load_ratings(cfg)
    fam_ratings      = load_familiarity_ratings(cfg)
    sem_rdm, vis_rdm = load_rdms(cfg)
    split_masks = {'all': np.ones(cfg['n_targets'], dtype=bool)}

    results = []
    total = len(cfg['traits']) * len(SPLITS)
    done  = 0

    for trait in cfg['traits']:
        ratings = ratings_dict[trait]

        for split in SPLITS:
            done += 1
            mask  = split_masks[split]
            n_cel = int(mask.sum())
            print(f"\n[{done}/{total}] {trait} | {split} (n={n_cel})")

            try:
                df_pairs = build_pair_data(ratings, mask, sem_rdm, vis_rdm,
                                           fam_ratings=fam_ratings)
                print(f"  Pairs: {len(df_pairs)}")
                lmm_out = fit_lmm(df_pairs, fam_interaction=True)

                row = {
                    'dataset':       ds_name,
                    'trait':         trait,
                    'split':         split,
                    'n_celebrities': n_cel,
                    'sem_model':     cfg['sem_model'],
                    'vis_model':     cfg['vis_model'],
                }
                row.update(lmm_out)
                results.append(row)
                print(f"  converged={lmm_out['converged']}  AIC={lmm_out['AIC']:.1f}  "
                      f"LR_p={lmm_out['lr_p']:.4g}")
                print(f"  sem β={lmm_out['sem_dissim_z_beta']:.4f} "
                      f"p={lmm_out['sem_dissim_z_p']:.4f}  |  "
                      f"vis β={lmm_out['vis_dissim_z_beta']:.4f} "
                      f"p={lmm_out['vis_dissim_z_p']:.4f}  |  "
                      f"sem×fam_mean β={lmm_out.get('sem_dissim_z_X_fam_mean_z_beta', float('nan')):.4f} "
                      f"p={lmm_out.get('sem_dissim_z_X_fam_mean_z_p', float('nan')):.4f}  |  "
                      f"vis×fam_mean β={lmm_out.get('vis_dissim_z_X_fam_mean_z_beta', float('nan')):.4f} "
                      f"p={lmm_out.get('vis_dissim_z_X_fam_mean_z_p', float('nan')):.4f}")
            except Exception as e:
                print(f"  ERROR: {e}")
                results.append({'dataset': ds_name, 'trait': trait,
                                'split': split, 'error': str(e)})

        outdir = PIPELINE_DIR / ds_name
        outdir.mkdir(parents=True, exist_ok=True)
        pd.DataFrame(results).to_csv(
            outdir / 'dual_lmm_results_fam_interaction_partial.csv', index=False)

    out = pd.DataFrame(results)
    outpath = PIPELINE_DIR / ds_name / 'dual_lmm_results_fam_interaction.csv'
    out.to_csv(outpath, index=False)
    print(f"\nSaved {len(out)} rows → {outpath}")
    return out


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', choices=['us', 'cn', 'both'], default='both')
    parser.add_argument('--model', choices=['base', 'fam_interaction', 'both'],
                        default='base',
                        help='base = median-split model; fam_interaction = perceiver-level '
                             'familiarity interaction model')
    args = parser.parse_args()

    configure_r()
    from pymer4.models import Lmer  # noqa: F401 — ensure import after R setup

    datasets = ['us', 'cn'] if args.dataset == 'both' else [args.dataset]
    models   = ['base', 'fam_interaction'] if args.model == 'both' else [args.model]

    for ds in datasets:
        if 'base' in models:
            run_dataset(ds)
        if 'fam_interaction' in models:
            run_fam_interaction_dataset(ds)


if __name__ == '__main__':
    main()
