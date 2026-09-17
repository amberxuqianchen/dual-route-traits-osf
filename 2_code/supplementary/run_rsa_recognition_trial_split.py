"""
Trial-level recognition split: is the conceptual (semantic) contribution larger
for targets a perceiver actually recognises?

Unlike a target-level median split on mean familiarity, the recognition
label here is a perceiver x target property.  For each trait two trait RDMs are
built over the SAME full set of 50 targets:

    recognized   trait RDM : target means over recognised trials only
    unrecognized trait RDM : target means over unrecognised trials only

Both are compared against the identical semantic/visual model RDMs, so any
difference in unique variance cannot come from the two halves containing
different celebrities.

Recognition criterion
---------------------
us : familiarity is binary (1 = did not recognise, 7 = recognised)
cn : 7-point scale, recognised >= 4, unrecognised <= 3 (scale midpoint counts as recognised)

Null
----
Recognition labels are shuffled within each target column, preserving the number
of recognised trials per target.  The null therefore carries the same per-target
sample-size imbalance (and hence the same measurement-noise asymmetry) as the
observed data, so the p-values are not driven by recognised means being based on
fewer trials.

Usage:
    python 2_code/supplementary/run_rsa_recognition_trial_split.py --dataset both
"""

import argparse
import sys
import warnings
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

import numpy as np
import pandas as pd
from scipy.spatial.distance import squareform
from scipy.stats import spearmanr
from config import DATASETS, PIPELINE_DIR
from data_loader import load_ratings, load_rdms, load_familiarity_ratings

warnings.filterwarnings('ignore')

N_PERM = 5000
N_REL_SPLITS = 100
SEED = 42

# (recognized predicate, unrecognized predicate) on the familiarity rating
RECOGNITION_CRITERION = {
    'us': (lambda f: f == 7, lambda f: f == 1),
    'cn': (lambda f: f >= 4, lambda f: f <= 3),
}


def _unique(mean_r, sem_vec, vis_vec):
    """(unique_sem, unique_vis, shared, total_r2) for a target-mean vector."""
    tv = squareform(np.abs(mean_r[:, None] - mean_r[None, :]), checks=False)
    m = ~np.isnan(tv) & ~np.isnan(sem_vec) & ~np.isnan(vis_vec)
    y, s, v = tv[m], sem_vec[m], vis_vec[m]

    def _r2(*xs):
        X = np.column_stack([np.ones(len(y)), *xs])
        b = np.linalg.lstsq(X, y, rcond=None)[0]
        return 1.0 - np.sum((y - X @ b) ** 2) / np.sum((y - y.mean()) ** 2)

    r2_full, r2_sem, r2_vis = _r2(s, v), _r2(s), _r2(v)
    return r2_full - r2_vis, r2_full - r2_sem, r2_sem + r2_vis - r2_full, r2_full


def _vp(mean_r, sem_vec, vis_vec):
    """Full variance partitioning plus marginal RSA correlations (reporting only)."""
    u_sem, u_vis, shared, r2_full = _unique(mean_r, sem_vec, vis_vec)
    tv = squareform(np.abs(mean_r[:, None] - mean_r[None, :]), checks=False)
    m = ~np.isnan(tv) & ~np.isnan(sem_vec) & ~np.isnan(vis_vec)
    return {
        'unique_sem': u_sem,
        'unique_vis': u_vis,
        'shared':     shared,
        'total_r2':   r2_full,
        'rsa_sem_r':  spearmanr(tv[m], sem_vec[m])[0],
        'rsa_vis_r':  spearmanr(tv[m], vis_vec[m])[0],
    }


def _labels(fam, ds_name):
    """(n_participants, n_targets) recognition labels: 1 = recognised, 0 = not, -1 = excluded."""
    is_rec, is_unrec = RECOGNITION_CRITERION[ds_name]
    lab = np.full(fam.shape, -1, dtype=np.int8)
    ok = ~np.isnan(fam)
    lab[ok & is_rec(fam)] = 1
    lab[ok & is_unrec(fam)] = 0
    return lab


def _shuffle_labels(lab, rng):
    """Permute recognition labels within each target column (preserves per-target counts)."""
    out = lab.copy()
    for t in range(lab.shape[1]):
        idx = np.where(lab[:, t] >= 0)[0]
        out[idx, t] = lab[rng.permutation(idx), t]
    return out


def _cond_means(ratings, lab):
    """Target-mean vectors under a recognition-label matrix -> (recognised, unrecognised)."""
    valid = ~np.isnan(ratings)
    out = []
    for want in (1, 0):
        m = valid & (lab == want)
        n = m.sum(axis=0)
        s = np.where(m, np.nan_to_num(ratings), 0.0).sum(axis=0)
        out.append(np.where(n > 0, s / np.maximum(n, 1), np.nan))
    return out[0], out[1]


def _trial_pools(ratings, fam, cfg, ds_name):
    """Per-target arrays of valid trial ratings, ordered recognised-first.

    Returns (pools, n_rec) where pools[t] is a 1-D array of that target's ratings
    with the first n_rec[t] entries coming from recognised trials.
    """
    is_rec, is_unrec = RECOGNITION_CRITERION[ds_name]
    pools, n_rec = [], []
    for t in range(cfg['n_targets']):
        r, f = ratings[:, t], fam[:, t]
        ok = ~np.isnan(r) & ~np.isnan(f)
        rec = ok & is_rec(f)
        unrec = ok & is_unrec(f)
        pools.append(np.concatenate([r[rec], r[unrec]]))
        n_rec.append(int(rec.sum()))
    return pools, np.array(n_rec)


def _means(pools, n_rec):
    """Target-mean vectors for the recognised and unrecognised halves."""
    rec = np.array([p[:k].mean() if k > 0 else np.nan
                    for p, k in zip(pools, n_rec)])
    unrec = np.array([p[k:].mean() if len(p) - k > 0 else np.nan
                      for p, k in zip(pools, n_rec)])
    return rec, unrec


def _split_half_reliability(pools, lo, hi, rng):
    """Spearman-Brown corrected split-half reliability of a trait RDM.

    `lo:hi` selects the condition's slice of each target's trial pool.
    Reliability is the correlation between trait RDMs built from two random
    halves of that condition's trials.
    """
    rs = []
    for _ in range(N_REL_SPLITS):
        a, b = [], []
        for p, l, h in zip(pools, lo, hi):
            x = p[l:h]
            if len(x) < 2:
                a.append(np.nan)
                b.append(np.nan)
                continue
            perm = rng.permutation(len(x))
            half = len(x) // 2
            a.append(x[perm[:half]].mean())
            b.append(x[perm[half:]].mean())
        va = squareform(np.abs(np.subtract.outer(np.array(a), np.array(a))), checks=False)
        vb = squareform(np.abs(np.subtract.outer(np.array(b), np.array(b))), checks=False)
        m = ~np.isnan(va) & ~np.isnan(vb)
        r = spearmanr(va[m], vb[m])[0]
        rs.append(2 * r / (1 + r))
    return float(np.nanmean(rs))


def run_dataset(ds_name: str):
    cfg = DATASETS[ds_name]
    print(f"\n{'='*70}")
    print(f"Trial-level recognition split RSA — {ds_name.upper()}")
    print(f"{'='*70}")

    ratings_dict = load_ratings(cfg)
    fam = load_familiarity_ratings(cfg)
    sem_rdm, vis_rdm = load_rdms(cfg)
    sem_vec = squareform(sem_rdm, checks=False)
    vis_vec = squareform(vis_rdm, checks=False)

    rng = np.random.default_rng(SEED)
    traits = cfg['traits']
    lab = _labels(fam, ds_name)

    # Observed statistics
    obs, rel, counts = {}, {}, {}
    for trait in traits:
        ratings = ratings_dict[trait]
        mr, mu = _cond_means(ratings, lab)
        vp_rec = _vp(mr, sem_vec, vis_vec)
        vp_unrec = _vp(mu, sem_vec, vis_vec)
        obs[trait] = (vp_rec, vp_unrec,
                      vp_rec['unique_sem'] - vp_unrec['unique_sem'],
                      vp_rec['unique_vis'] - vp_unrec['unique_vis'])

        pools, n_rec = _trial_pools(ratings, fam, cfg, ds_name)
        n_tot = np.array([len(p) for p in pools])
        counts[trait] = (n_rec, n_tot - n_rec)
        rel[trait] = (_split_half_reliability(pools, np.zeros_like(n_rec), n_rec, rng),
                      _split_half_reliability(pools, n_rec, n_tot, rng))

    # Permutation: one shared reshuffle of recognition labels applied to every trait,
    # preserving the cross-trait dependence needed for max-T
    cnt = {t: [0, 0] for t in traits}
    null_max_sem = np.zeros(N_PERM)
    null_max_vis = np.zeros(N_PERM)
    for k in range(N_PERM):
        if k % 500 == 0:
            print(f"  perm {k}/{N_PERM}", flush=True)
        plab = _shuffle_labels(lab, rng)
        m_sem = m_vis = 0.0
        for trait in traits:
            pr, pu = _cond_means(ratings_dict[trait], plab)
            rs, rv, _, _ = _unique(pr, sem_vec, vis_vec)
            us, uv, _, _ = _unique(pu, sem_vec, vis_vec)
            d_s, d_v = abs(rs - us), abs(rv - uv)
            if d_s >= abs(obs[trait][2]):
                cnt[trait][0] += 1
            if d_v >= abs(obs[trait][3]):
                cnt[trait][1] += 1
            m_sem, m_vis = max(m_sem, d_s), max(m_vis, d_v)
        null_max_sem[k], null_max_vis[k] = m_sem, m_vis

    results = []
    for trait in traits:
        vp_rec, vp_unrec, d_sem, d_vis = obs[trait]
        n_rec, n_unrec = counts[trait]
        rel_rec, rel_unrec = rel[trait]
        p_sem = (cnt[trait][0] + 1) / (N_PERM + 1)
        p_vis = (cnt[trait][1] + 1) / (N_PERM + 1)
        p_sem_maxt = (np.sum(null_max_sem >= abs(d_sem)) + 1) / (N_PERM + 1)
        p_vis_maxt = (np.sum(null_max_vis >= abs(d_vis)) + 1) / (N_PERM + 1)

        results.append({
            'dataset':    ds_name,
            'trait':      trait,
            'sem_model':  cfg['sem_model'],
            'vis_model':  cfg['vis_model'],
            'n_trials_rec':   int(n_rec.sum()),
            'n_trials_unrec': int(n_unrec.sum()),
            'min_n_rec_per_target':   int(n_rec.min()),
            'min_n_unrec_per_target': int(n_unrec.min()),
            'unique_sem_rec':   vp_rec['unique_sem'],
            'unique_sem_unrec': vp_unrec['unique_sem'],
            'delta_unique_sem': d_sem,
            'p_unique_sem':     p_sem,
            'p_unique_sem_maxt': p_sem_maxt,
            'unique_vis_rec':   vp_rec['unique_vis'],
            'unique_vis_unrec': vp_unrec['unique_vis'],
            'delta_unique_vis': d_vis,
            'p_unique_vis':     p_vis,
            'p_unique_vis_maxt': p_vis_maxt,
            'shared_rec':       vp_rec['shared'],
            'shared_unrec':     vp_unrec['shared'],
            'total_r2_rec':     vp_rec['total_r2'],
            'total_r2_unrec':   vp_unrec['total_r2'],
            'rsa_sem_r_rec':    vp_rec['rsa_sem_r'],
            'rsa_sem_r_unrec':  vp_unrec['rsa_sem_r'],
            'rsa_vis_r_rec':    vp_rec['rsa_vis_r'],
            'rsa_vis_r_unrec':  vp_unrec['rsa_vis_r'],
            'reliability_rec':   rel_rec,
            'reliability_unrec': rel_unrec,
            'n_perm':           N_PERM,
        })
        print(f"  {trait:<12} Δsem={d_sem:+.4f} p={p_sem:.4f} maxT={p_sem_maxt:.4f} | "
              f"Δvis={d_vis:+.4f} p={p_vis:.4f} maxT={p_vis_maxt:.4f} | "
              f"rel {rel_rec:.2f}/{rel_unrec:.2f}")

    out = pd.DataFrame(results)
    outdir = PIPELINE_DIR / ds_name
    outdir.mkdir(parents=True, exist_ok=True)
    outpath = outdir / 'rsa_recognition_trial_split.csv'
    out.to_csv(outpath, index=False)
    print(f"\nSaved {len(out)} rows → {outpath}")
    return out


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', choices=['us', 'cn', 'both'], default='both')
    args = parser.parse_args()
    for ds in (['us', 'cn'] if args.dataset == 'both' else [args.dataset]):
        run_dataset(ds)


if __name__ == '__main__':
    main()
