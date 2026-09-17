"""
Leave-one-out influence diagnostics for the recognition interaction terms.

Which perceivers and which target identities are driving the sem x recognized
and vis x recognized coefficients?

Full LMM refits are infeasible at this scale (330 participants x 50 targets x
18 traits).  However the saturated OLS interaction coefficient is algebraically
the difference in per-arm slopes, and empirically tracks the LMM interaction
closely (see run_recognition_variance_check.py), so OLS is used for the
exhaustive sweep and the top-ranked units are then confirmed with real LMM
refits.

For each unit u:
    delta(u) = beta_full - beta_without_u          (a DFBETA)
    ratio(u) = delta(u) / SD of delta across units (a standardised influence)

A target is dropped by removing every pair that contains it; a participant by
removing all of their rows.

Writes 3_pipeline/{ds}/recognition_influence_{targets,participants}.csv

Usage:
    python 2_code/supplementary/run_recognition_influence.py --dataset both --variant midpoint
    python 2_code/supplementary/run_recognition_influence.py --dataset cn --confirm feminine
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
import run_lmm_recognition_split as rsplit
from run_lmm_recognition_split import _build_pairs, CRITERIA, SUFFIX

warnings.filterwarnings('ignore')


def _design(df):
    S = df.sem_dissim_z.values
    V = df.vis_dissim_z.values
    G = df.recognized.values
    X = np.column_stack([np.ones(len(df)), S, V, G, S * G, V * G])
    return X, df.abs_diff.values


def _coefs(X, y):
    """OLS via normal equations; returns (sem_X_rec, vis_X_rec) = columns 4, 5."""
    b = np.linalg.solve(X.T @ X, X.T @ y)
    return b[4], b[5]


def _sweep(df, unit_col_masks, full):
    """DFBETAs for each unit given precomputed boolean keep-masks."""
    X, y = _design(df)
    out = []
    for name, keep in unit_col_masks:
        if keep.sum() < 100:
            out.append((name, np.nan, np.nan, int((~keep).sum())))
            continue
        try:
            s, v = _coefs(X[keep], y[keep])
        except np.linalg.LinAlgError:
            out.append((name, np.nan, np.nan, int((~keep).sum())))
            continue
        out.append((name, full[0] - s, full[1] - v, int((~keep).sum())))
    return out


def run(ds_name, variant):
    cfg = DATASETS[ds_name]
    rsplit.RECOGNITION_CRITERION = CRITERIA[variant]
    ratings_dict = load_ratings(cfg)
    fam = load_familiarity_ratings(cfg)
    sem_rdm, vis_rdm = load_rdms(cfg)

    tgt_rows, ppt_rows = [], []
    for trait in cfg['traits']:
        df = _build_pairs(ratings_dict[trait], fam, sem_rdm, vis_rdm, ds_name)
        X, y = _design(df)
        full = _coefs(X, y)

        t1 = df.target1.values
        t2 = df.target2.values
        targets = sorted(set(t1) | set(t2), key=int)
        masks = [(t, ~((t1 == t) | (t2 == t))) for t in targets]
        for name, ds_, dv_, n in _sweep(df, masks, full):
            tgt_rows.append({'dataset': ds_name, 'variant': variant, 'trait': trait,
                             'target': name, 'n_pairs_dropped': n,
                             'dfbeta_sem': ds_, 'dfbeta_vis': dv_,
                             'beta_sem_full': full[0], 'beta_vis_full': full[1]})

        pv = df.participant.values
        ppts = sorted(set(pv), key=int)
        masks = [(p, pv != p) for p in ppts]
        for name, ds_, dv_, n in _sweep(df, masks, full):
            ppt_rows.append({'dataset': ds_name, 'variant': variant, 'trait': trait,
                             'participant': name, 'n_pairs_dropped': n,
                             'dfbeta_sem': ds_, 'dfbeta_vis': dv_,
                             'beta_sem_full': full[0], 'beta_vis_full': full[1]})
        print(f"  {trait}: {len(targets)} targets, {len(ppts)} participants swept", flush=True)

    outdir = PIPELINE_DIR / ds_name
    T = pd.DataFrame(tgt_rows)
    P = pd.DataFrame(ppt_rows)
    for d, nm in ((T, 'targets'), (P, 'participants')):
        for ch in ('sem', 'vis'):
            d[f'z_{ch}'] = d.groupby('trait')[f'dfbeta_{ch}'].transform(
                lambda x: (x - x.mean()) / x.std())
        d.to_csv(outdir / f'recognition_influence_{nm}{SUFFIX[variant]}.csv', index=False)
    return T, P


def confirm(ds_name, variant, trait, units, kind):
    """Refit the real LMM dropping each named unit, to verify the OLS ranking."""
    from lmm_utils import configure_r
    from pymer4.models import Lmer
    configure_r()
    cfg = DATASETS[ds_name]
    rsplit.RECOGNITION_CRITERION = CRITERIA[variant]
    df = _build_pairs(load_ratings(cfg)[trait], load_familiarity_ratings(cfg),
                      *load_rdms(cfg), ds_name)

    def fit(d):
        m = Lmer('abs_diff ~ sem_dissim_z * recognized + vis_dissim_z * recognized'
                 ' + (1|participant) + (1|target1) + (1|target2)', data=d)
        m.fit(summarize=False)
        fe = m.coefs
        g = lambda n: (fe.loc[n, 'Estimate'] if n in fe.index
                       else fe.loc[':'.join(reversed(n.split(':'))), 'Estimate'])
        return g('sem_dissim_z:recognized'), g('vis_dissim_z:recognized')

    base = fit(df)
    print(f"\n{ds_name.upper()} {trait} [{variant}] full: "
          f"semXrec={base[0]:+.4f} visXrec={base[1]:+.4f}")
    for u in units:
        keep = (~((df.target1 == u) | (df.target2 == u)) if kind == 'target'
                else df.participant != u)
        s, v = fit(df[keep])
        print(f"  drop {kind} {u:>4}: semXrec={s:+.4f} ({s-base[0]:+.4f})  "
              f"visXrec={v:+.4f} ({v-base[1]:+.4f})", flush=True)


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--dataset', choices=['us', 'cn', 'both'], default='both')
    p.add_argument('--variant', choices=list(CRITERIA), default='midpoint')
    p.add_argument('--confirm', default=None, help='trait name to confirm with real LMM refits')
    p.add_argument('--units', nargs='*', default=None)
    p.add_argument('--kind', choices=['target', 'participant'], default='target')
    a = p.parse_args()

    if a.confirm:
        confirm(a.dataset, a.variant, a.confirm, a.units or [], a.kind)
        return
    for ds in (['us', 'cn'] if a.dataset == 'both' else [a.dataset]):
        print(f"\n{'='*70}\nInfluence sweep — {ds.upper()} [{a.variant}]\n{'='*70}")
        run(ds, a.variant)


if __name__ == '__main__':
    main()
