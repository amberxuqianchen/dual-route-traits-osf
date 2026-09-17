"""
Pooled LMM combining US and CN datasets on the 8 common traits.

Adds `dataset` (0=us, 1=cn) as a fixed-effect covariate. Participant and
target IDs are prefixed with the dataset label to avoid collisions.

Usage:
    python 2_code/run_merged_lmm.py
"""

import sys
import warnings
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

import numpy as np
import pandas as pd
from config import DATASETS, COMMON_TRAITS, SPLITS, PIPELINE_DIR
from data_loader import load_ratings, load_rdms
from lmm_utils import configure_r, build_pair_data, fit_lmm

warnings.filterwarnings('ignore')
np.random.seed(42)

# Canonical trait name for each dataset  (handles youthful → young in US)
def _canonical(ds_name, trait):
    return DATASETS[ds_name]['canonical_trait_map'].get(trait, trait)

# Reverse map: canonical → dataset-local name
def _local_trait(ds_name, canonical):
    inv = {v: k for k, v in DATASETS[ds_name]['canonical_trait_map'].items()}
    return inv.get(canonical, canonical)


def build_merged_pairs(canonical_trait: str, mask_key: str) -> pd.DataFrame:
    dfs = []
    for ds_name, cfg in DATASETS.items():
        local = _local_trait(ds_name, canonical_trait)
        if local not in cfg['traits']:
            print(f"  Skipping {ds_name}: trait '{local}' not found")
            continue

        ratings_dict     = load_ratings(cfg)
        sem_rdm, vis_rdm = load_rdms(cfg)
        mask_map = {'all': np.ones(cfg['n_targets'], dtype=bool)}
        mask = mask_map[mask_key]
        df   = build_pair_data(ratings_dict[local], mask, sem_rdm, vis_rdm,
                               dataset_label=ds_name)
        dfs.append(df)

    if not dfs:
        return pd.DataFrame()

    merged = pd.concat(dfs, ignore_index=True)
    # Re-z-score dissimilarities across the pooled pairs
    for col in ['sem_dissim_z', 'vis_dissim_z']:
        merged[col] = (merged[col] - merged[col].mean()) / merged[col].std()
    # Encode dataset as numeric for LMM
    merged['dataset_num'] = (merged['dataset'] == 'cn').astype(float)
    return merged


def main():
    configure_r()
    from pymer4.models import Lmer  # noqa: F401

    outdir = PIPELINE_DIR / 'merged'
    outdir.mkdir(parents=True, exist_ok=True)

    results = []
    total = len(COMMON_TRAITS) * len(SPLITS)
    done  = 0

    for trait in COMMON_TRAITS:
        for split in SPLITS:
            done += 1
            print(f"\n[{done}/{total}] {trait} | {split} (merged)")

            try:
                df_pairs = build_merged_pairs(trait, split)
                if df_pairs.empty:
                    print("  No data — skipped")
                    continue
                print(f"  Pairs: {len(df_pairs)}")
                lmm_out = fit_lmm(df_pairs, extra_fixed=['dataset_num'])

                row = {
                    'trait': trait,
                    'split': split,
                    'n_us':  (df_pairs['dataset'] == 'us').sum(),
                    'n_cn':  (df_pairs['dataset'] == 'cn').sum(),
                }
                row.update(lmm_out)
                results.append(row)
                print(f"  converged={lmm_out['converged']}  "
                      f"sem β={lmm_out['sem_dissim_z_beta']:.4f} "
                      f"p={lmm_out['sem_dissim_z_p']:.4f}  |  "
                      f"vis β={lmm_out['vis_dissim_z_beta']:.4f} "
                      f"p={lmm_out['vis_dissim_z_p']:.4f}")
            except Exception as e:
                print(f"  ERROR: {e}")
                results.append({'trait': trait, 'split': split, 'error': str(e)})

        # Intermediate save after each trait
        pd.DataFrame(results).to_csv(
            outdir / 'dual_lmm_results_merged_partial.csv', index=False)

    out = pd.DataFrame(results)
    outpath = outdir / 'dual_lmm_results_merged.csv'
    out.to_csv(outpath, index=False)
    print(f"\nSaved {len(out)} rows → {outpath}")


if __name__ == '__main__':
    main()
