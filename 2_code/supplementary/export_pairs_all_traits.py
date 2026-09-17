"""
Export pair-level data for every trait and for the trait space, in both designs.

Two files per dataset x outcome:

  bn_{ds}_{name}.csv   both/neither pairs only, with `recognized` 0/1.
                       Familiarity is binarised (recognised vs not) and only
                       pairs where BOTH targets fall on the same side are kept.
                       Used by the arm-specific joint model.

  all_{ds}_{name}.csv  every pair with familiarity on both targets, carrying
                       `fam_mean` (the moderator) and `fam_diff` (|f_i - f_j|,
                       entered as a control so the moderator reflects the pair's
                       overall familiarity level rather than its imbalance).

`name` is a trait, or `traitspace` for the multivariate outcome.

The outcome column is called `abs_diff` in every file.  For a single trait it is
|mean rating difference| for that trait; for the trait space it is the Euclidean
distance between the two targets' full trait profiles for that perceiver.  The
shared name lets one model script serve both.

Usage:
    python 2_code/supplementary/export_pairs_all_traits.py --outdir DIR [--variant midpoint]
"""

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))
sys.path.insert(0, str(Path(__file__).parent))

import numpy as np
import pandas as pd
from config import DATASETS, local_common_traits
from data_loader import load_ratings, load_rdms, load_familiarity_ratings
from run_lmm_recognition_split import CRITERIA


def trait_space_traits(ds_name):
    """Traits defining the omnibus space for every model family."""
    return local_common_traits(ds_name)


def _model_vecs(sem_rdm, vis_rdm):
    n = sem_rdm.shape[0]
    ii, jj = np.triu_indices(n, k=1)
    sem, vis = sem_rdm[ii, jj], vis_rdm[ii, jj]
    return ii, jj, (sem - sem.mean()) / sem.std(), (vis - vis.mean()) / vis.std()


def _outcome_single(ratings, p, ii, jj):
    r = ratings[p]
    return np.abs(r[ii] - r[jj])


def _outcome_space(coords, p, ii, jj):
    """Euclidean distance between trait profiles; NaN if all traits missing."""
    c = coords[p]                       # (n_targets, n_traits)
    d = c[ii] - c[jj]
    dist = np.sqrt(np.nansum(d ** 2, axis=1))
    dist[np.all(np.isnan(d), axis=1)] = np.nan
    return dist


def build(outcome_fn, fam, ii, jj, sem_z, vis_z, n_part, is_rec, is_unrec):
    """Returns (both/neither frame, all-pairs frame)."""
    bn, al = [], []
    for p in range(n_part):
        y = outcome_fn(p)
        f = fam[p]
        fi, fj = f[ii], f[jj]
        ok = ~np.isnan(y) & ~np.isnan(fi) & ~np.isnan(fj)
        if not ok.any():
            continue
        base = dict(participant=p, target1=ii[ok], target2=jj[ok],
                    abs_diff=y[ok], sem_dissim_z=sem_z[ok], vis_dissim_z=vis_z[ok])
        al.append(pd.DataFrame({**base,
                                'fam_mean': (fi[ok] + fj[ok]) / 2.0,
                                'fam_diff': np.abs(fi[ok] - fj[ok])}))
        rec, unrec = is_rec(f), is_unrec(f)
        both, neither = rec[ii] & rec[jj], unrec[ii] & unrec[jj]
        k = (both | neither) & ok
        if k.any():
            # fam_mean/fam_diff are carried here too: variant 3 uses the SAME
            # same-arm subset as variant 2 but keeps the original graded
            # familiarity as the moderator instead of the binary arm.
            bn.append(pd.DataFrame(dict(
                participant=p, target1=ii[k], target2=jj[k], abs_diff=y[k],
                sem_dissim_z=sem_z[k], vis_dissim_z=vis_z[k],
                recognized=both[k].astype(float),
                fam_mean=(fi[k] + fj[k]) / 2.0,
                fam_diff=np.abs(fi[k] - fj[k]))))
    return pd.concat(bn, ignore_index=True), pd.concat(al, ignore_index=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--outdir', required=True)
    ap.add_argument('--variant', default='midpoint', choices=list(CRITERIA))
    a = ap.parse_args()
    out = Path(a.outdir)
    out.mkdir(parents=True, exist_ok=True)

    for ds, cfg in DATASETS.items():
        R = load_ratings(cfg)
        fam = load_familiarity_ratings(cfg)
        sem_rdm, vis_rdm = load_rdms(cfg)
        ii, jj, sem_z, vis_z = _model_vecs(sem_rdm, vis_rdm)
        is_rec, is_unrec = CRITERIA[a.variant][ds]
        n_part = R[cfg['traits'][0]].shape[0]

        jobs = [(t, lambda p, t=t: _outcome_single(R[t], p, ii, jj))
                for t in cfg['traits']]
        # Trait space: the eight dimensions shared across datasets.  This must
        # match the group-level RSA/VP definition and the manuscript; the two
        # additional CN-only traits are analyzed individually, not folded into
        # the omnibus cross-dataset outcome.
        traits = trait_space_traits(ds)
        coords = np.stack([np.column_stack([R[t][p] for t in traits])
                           for p in range(n_part)])
        jobs.append(('traitspace', lambda p: _outcome_space(coords, p, ii, jj)))

        for name, fn in jobs:
            bn, al = build(fn, fam, ii, jj, sem_z, vis_z, n_part,
                           is_rec, is_unrec)
            for c in ('participant', 'target1', 'target2'):
                bn[c] = bn[c].astype(str)
                al[c] = al[c].astype(str)
            bn.to_csv(out / f'bn_{ds}_{name}.csv', index=False)
            al.to_csv(out / f'all_{ds}_{name}.csv', index=False)
            print(f'{ds}/{name}: bn {len(bn):>7} (rec {int(bn.recognized.sum()):>6}) '
                  f'| all {len(al):>7}', flush=True)


if __name__ == '__main__':
    main()
