"""
Max-T FWER correction for the recognition-split LMM deltas.

Observed statistic (from run_lmm_recognition_split.py) is the Wald z for
beta_recognised - beta_unrecognised, per trait per channel.

Null: the familiarity matrix is permuted across participants WITHIN each target
column.  This preserves every target's recognition rate exactly — so target
popularity is controlled — while breaking the link between a perceiver's
recognition of a target and their trait ratings of it.  Pairs are then rebuilt
and both condition models refit, so the null inherits the same both/neither pair
counts and n-imbalance as the real data.

This deviates from run_lmm_maxt.py, which permutes the model-RDM target labels.
That null destroys the sem/vis-to-trait relationship entirely and is not
appropriate for a between-condition difference: it would drive both betas to
zero rather than asking whether recognition specifically matters.

One permutation is shared across all traits, so max |z| across traits gives
FWER control per channel per dataset.

Family: all traits x {sem, vis} per dataset.
  US: 8 traits, CN: 10 traits

Checkpointing: null arrays saved after every batch; re-run to resume.

Usage:
    python 2_code/supplementary/run_lmm_recognition_maxt.py --dataset both
    python 2_code/supplementary/run_lmm_recognition_maxt.py --dataset us --n-perm 200 --n-workers 24
"""

import argparse
import sys
import warnings
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))
sys.path.insert(0, str(Path(__file__).parent))

import numpy as np
import pandas as pd
from config import DATASETS, PIPELINE_DIR
from data_loader import load_ratings, load_rdms, load_familiarity_ratings
import run_lmm_recognition_split as rsplit
from run_lmm_recognition_split import _build_pairs, _wald, CRITERIA, SUFFIX

warnings.filterwarnings('ignore')

N_PERM_DEFAULT = 200
N_WORKERS_DEFAULT = 24
BATCH = 24
SEED = 42

_G = {}


def _init_worker(ds_name, variant):
    """Configure R once per worker and cache this dataset's arrays."""
    warnings.filterwarnings('ignore')
    from lmm_utils import configure_r
    configure_r()
    rsplit.RECOGNITION_CRITERION = CRITERIA[variant]
    cfg = DATASETS[ds_name]
    sem, vis = load_rdms(cfg)
    _G['ds'] = ds_name
    _G['cfg'] = cfg
    _G['ratings'] = load_ratings(cfg)
    _G['fam'] = load_familiarity_ratings(cfg)
    _G['sem'] = sem
    _G['vis'] = vis


def _betas(df):
    """(beta, se) for sem and vis from the base LMM on one condition."""
    from pymer4.models import Lmer
    m = Lmer('abs_diff ~ sem_dissim_z + vis_dissim_z'
             ' + (1|participant) + (1|target1) + (1|target2)', data=df)
    m.fit(summarize=False)
    fe = m.coefs
    return {t: (fe.loc[t, 'Estimate'], fe.loc[t, 'SE'])
            for t in ('sem_dissim_z', 'vis_dissim_z')}


def _one_perm(k):
    """Return (max|z_sem|, max|z_vis|) across traits for permutation k."""
    cfg, ds = _G['cfg'], _G['ds']
    fam, sem, vis = _G['fam'], _G['sem'], _G['vis']
    rng = np.random.default_rng(SEED + k)

    # Permute familiarity across participants within each target column
    pf = fam.copy()
    for t in range(pf.shape[1]):
        pf[:, t] = fam[rng.permutation(fam.shape[0]), t]

    max_sem = max_vis = 0.0
    for trait in cfg['traits']:
        df = _build_pairs(_G['ratings'][trait], pf, sem, vis, ds)
        if df.recognized.nunique() < 2:
            continue
        try:
            br = _betas(df[df.recognized == 1])
            bu = _betas(df[df.recognized == 0])
        except Exception:
            continue
        for term, store in (('sem_dissim_z', 'sem'), ('vis_dissim_z', 'vis')):
            z, _ = _wald(*br[term], *bu[term])
            if store == 'sem':
                max_sem = max(max_sem, abs(z))
            else:
                max_vis = max(max_vis, abs(z))
    return max_sem, max_vis


def run_dataset(ds_name, n_perm, n_workers, variant='midpoint'):
    cfg = DATASETS[ds_name]
    outdir = PIPELINE_DIR / ds_name
    sfx = SUFFIX[variant]
    obs = pd.read_csv(outdir / f'lmm_recognition_split{sfx}.csv')

    ck_sem = outdir / f'lmm_recognition_maxt_null_sem{sfx}.npy'
    ck_vis = outdir / f'lmm_recognition_maxt_null_vis{sfx}.npy'
    null_sem = list(np.load(ck_sem)) if ck_sem.exists() else []
    null_vis = list(np.load(ck_vis)) if ck_vis.exists() else []
    done = len(null_sem)
    print(f"\n{'='*70}\nRecognition-LMM max-T — {ds_name.upper()} [{variant}]  "
          f"({done}/{n_perm} done)\n{'='*70}", flush=True)

    todo = list(range(done, n_perm))
    with ProcessPoolExecutor(max_workers=n_workers,
                             initializer=_init_worker,
                             initargs=(ds_name, variant)) as ex:
        for i in range(0, len(todo), BATCH):
            chunk = todo[i:i + BATCH]
            for s, v in ex.map(_one_perm, chunk):
                null_sem.append(s)
                null_vis.append(v)
            np.save(ck_sem, np.array(null_sem))
            np.save(ck_vis, np.array(null_vis))
            print(f"  {len(null_sem)}/{n_perm} perms  "
                  f"max|z| sem p95={np.percentile(null_sem, 95):.2f} "
                  f"vis p95={np.percentile(null_vis, 95):.2f}", flush=True)

    ns, nv = np.array(null_sem), np.array(null_vis)
    rows = []
    for _, r in obs.iterrows():
        rows.append({
            'dataset': ds_name, 'trait': r['trait'],
            'sem_delta': r['sem_dissim_z_delta'],
            'sem_delta_z': r['sem_dissim_z_delta_z'],
            'sem_delta_p': r['sem_dissim_z_delta_p'],
            'sem_delta_p_maxt': (np.sum(ns >= abs(r['sem_dissim_z_delta_z'])) + 1) / (len(ns) + 1),
            'vis_delta': r['vis_dissim_z_delta'],
            'vis_delta_z': r['vis_dissim_z_delta_z'],
            'vis_delta_p': r['vis_dissim_z_delta_p'],
            'vis_delta_p_maxt': (np.sum(nv >= abs(r['vis_dissim_z_delta_z'])) + 1) / (len(nv) + 1),
            'n_perm': len(ns),
        })
    out = pd.DataFrame(rows)
    outpath = outdir / f'lmm_recognition_maxt_results{sfx}.csv'
    out.to_csv(outpath, index=False)
    print(out[['trait', 'sem_delta', 'sem_delta_p', 'sem_delta_p_maxt',
               'vis_delta', 'vis_delta_p', 'vis_delta_p_maxt']].round(4).to_string(index=False))
    print(f"\nSaved → {outpath}")
    return out


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--dataset', choices=['us', 'cn', 'both'], default='both')
    p.add_argument('--n-perm', type=int, default=N_PERM_DEFAULT)
    p.add_argument('--n-workers', type=int, default=N_WORKERS_DEFAULT)
    p.add_argument('--variant', choices=list(CRITERIA), default='midpoint')
    a = p.parse_args()
    rsplit.RECOGNITION_CRITERION = CRITERIA[a.variant]
    for ds in (['us', 'cn'] if a.dataset == 'both' else [a.dataset]):
        run_dataset(ds, a.n_perm, a.n_workers, a.variant)


if __name__ == '__main__':
    main()
