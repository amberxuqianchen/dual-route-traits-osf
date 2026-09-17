"""
Export pair-level data for the Bayesian recognition models (brms_recognition.R).

Two files per dataset x trait:

  pairs_bn_{ds}_{trait}.csv   both/neither pairs only, with `recognized` 0/1.
                              Used by the `twofit` and `joint` variants.

  pairs_all_{ds}_{trait}.csv  every pair with familiarity on both targets,
                              including mixed ones, plus `fam_mean` and
                              `fam_min`.  Used by `fammean` (full sample) and
                              `famhigh` (fam_min > 5).  Mixed pairs are kept
                              here because a continuous familiarity moderator
                              has no reason to discard them — that filter only
                              exists to make the binary contrast clean.

Usage:
    python 2_code/supplementary/export_pairs_for_brms.py --outdir DIR \
        --traits cn:feminine cn:competent us:feminine us:strong
"""

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))
sys.path.insert(0, str(Path(__file__).parent))

import numpy as np
import pandas as pd
from config import DATASETS
from data_loader import load_ratings, load_rdms, load_familiarity_ratings
import run_lmm_recognition_split as rsplit
from run_lmm_recognition_split import _build_pairs, CRITERIA


def export_all_pairs(ratings, fam, sem_rdm, vis_rdm, ds_name):
    """Every pair with familiarity on both targets; carries fam_mean / fam_min."""
    n = sem_rdm.shape[0]
    ii, jj = np.triu_indices(n, k=1)
    sem, vis = sem_rdm[ii, jj], vis_rdm[ii, jj]
    sem_z = (sem - sem.mean()) / sem.std()
    vis_z = (vis - vis.mean()) / vis.std()

    frames = []
    for p in range(ratings.shape[0]):
        r, f = ratings[p], fam[p]
        d = np.abs(r[ii] - r[jj])
        fi, fj = f[ii], f[jj]
        keep = ~np.isnan(d) & ~np.isnan(fi) & ~np.isnan(fj)
        if not keep.any():
            continue
        frames.append(pd.DataFrame({
            'participant': p,
            'target1': ii[keep], 'target2': jj[keep],
            'abs_diff': d[keep],
            'sem_dissim_z': sem_z[keep], 'vis_dissim_z': vis_z[keep],
            'fam_mean': (fi[keep] + fj[keep]) / 2.0,
            'fam_min': np.minimum(fi[keep], fj[keep]),
        }))
    df = pd.concat(frames, ignore_index=True)
    for c in ('participant', 'target1', 'target2'):
        df[c] = df[c].astype(str)
    return df


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--outdir', required=True)
    ap.add_argument('--variant', default='midpoint', choices=list(CRITERIA))
    ap.add_argument('--traits', nargs='+', required=True,
                    help='entries like cn:feminine us:strong')
    a = ap.parse_args()
    out = Path(a.outdir)
    out.mkdir(parents=True, exist_ok=True)
    rsplit.RECOGNITION_CRITERION = CRITERIA[a.variant]

    cache = {}
    for spec in a.traits:
        ds, trait = spec.split(':')
        if ds not in cache:
            cfg = DATASETS[ds]
            cache[ds] = (cfg, load_ratings(cfg), load_familiarity_ratings(cfg),
                         *load_rdms(cfg))
        cfg, R, fam, sem, vis = cache[ds]

        bn = _build_pairs(R[trait], fam, sem, vis, ds)
        p1 = out / f'pairs_bn_{ds}_{trait}.csv'
        bn.to_csv(p1, index=False)

        al = export_all_pairs(R[trait], fam, sem, vis, ds)
        p2 = out / f'pairs_all_{ds}_{trait}.csv'
        al.to_csv(p2, index=False)

        hi = al[al.fam_min > 5]
        print(f'{ds}/{trait}: both-neither {len(bn):>7} '
              f'(rec {int(bn.recognized.sum())}) | all {len(al):>7} | '
              f'fam_min>5 {len(hi):>7} '
              f'(fam_mean sd {hi.fam_mean.std():.3f})', flush=True)


if __name__ == '__main__':
    main()
