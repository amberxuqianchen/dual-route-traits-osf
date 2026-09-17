"""
Max-T FWER correction for LMM familiarity-interaction t-statistics.

For each of 1,000 permutations:
  1. Permute target labels in model RDMs (same pi applied to sem + vis).
  2. Recompute sem_dissim_z / vis_dissim_z for all pairs (fam columns stay fixed).
  3. Refit interaction LMM for every trait.
  4. Track max |t| across traits x {sem:fam, vis:fam} -> Tmax null distribution.

Family: 'all' split, all traits x {sem_dissim_z:fam_mean_z, vis_dissim_z:fam_mean_z}.
  US: 8 traits x 2 = 16 tests
  CN: 10 traits x 2 = 20 tests

Checkpointing: null array saved after every chunk; re-run to resume.

Usage:
    python 2_code/run_lmm_fam_interaction_maxt.py --dataset us
    python 2_code/run_lmm_fam_interaction_maxt.py --dataset cn
    python 2_code/run_lmm_fam_interaction_maxt.py --dataset both
    python 2_code/run_lmm_fam_interaction_maxt.py --dataset us --n-workers 16
"""

import argparse
import sys
import warnings
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

import numpy as np
import pandas as pd

from config import DATASETS, PIPELINE_DIR
from data_loader import load_ratings, load_rdms, load_familiarity_ratings
from lmm_utils import configure_r, build_pair_data
from rsa_utils import maxt_corrected_p

warnings.filterwarnings('ignore')

N_PERM_DEFAULT    = 1000
N_WORKERS_DEFAULT = 8
CHUNK_SIZE        = 50
SEED              = 42


# ── R permutation function ────────────────────────────────────────────────────

_R_FUNC = """
library(lme4)
library(parallel)

.lmm_fam_perm_chunk <- function(dfs, pair_ks, sem_mat, vis_mat, n_cores=1L) {
    # dfs      : R list of data.frames, one per trait.
    #            Each has: participant, target1, target2, abs_diff,
    #            sem_dissim_z, vis_dissim_z, fam_dissim_z, fam_mean_z
    #            (fam columns are fixed; sem/vis are overwritten each iter).
    # pair_ks  : R list of 1-based integer vectors indexing rows of sem_mat / vis_mat.
    # sem_mat  : R matrix (n_pairs x n_chunk), permuted z-scored sem dissimilarities.
    # vis_mat  : R matrix (n_pairs x n_chunk), permuted z-scored vis dissimilarities.
    # n_cores  : number of cores for mclapply.
    # Returns  : named list with $sem and $vis, each numeric(n_chunk) --
    #            max |t_sem:fam| and max |t_vis:fam| across traits per permutation.

    n_traits <- length(dfs)

    run_one <- function(k) {
        max_t_sem_fam <- 0.0
        max_t_vis_fam <- 0.0
        for (t in seq_len(n_traits)) {
            df <- dfs[[t]]
            pk <- pair_ks[[t]]
            df[["sem_dissim_z"]] <- sem_mat[pk, k]
            df[["vis_dissim_z"]] <- vis_mat[pk, k]
            mod <- tryCatch(
                suppressMessages(suppressWarnings(
                    lme4::lmer(
                        abs_diff ~ sem_dissim_z + vis_dissim_z +
                            fam_dissim_z + fam_mean_z +
                            sem_dissim_z:fam_mean_z + vis_dissim_z:fam_mean_z +
                            (1 | participant) + (1 | target1) + (1 | target2),
                        data    = df,
                        REML    = FALSE,
                        control = lmerControl(
                            optimizer = "bobyqa",
                            optCtrl   = list(maxfun = 1e5)
                        )
                    )
                )),
                error = function(e) NULL
            )
            if (!is.null(mod)) {
                cs <- coef(summary(mod))
                rnames <- rownames(cs)
                sem_fam_row <- rnames[grepl("sem_dissim_z:fam_mean_z", rnames, fixed=TRUE) |
                                      grepl("fam_mean_z:sem_dissim_z", rnames, fixed=TRUE)]
                vis_fam_row <- rnames[grepl("vis_dissim_z:fam_mean_z", rnames, fixed=TRUE) |
                                      grepl("fam_mean_z:vis_dissim_z", rnames, fixed=TRUE)]
                if (length(sem_fam_row) > 0) {
                    t_sf <- abs(cs[sem_fam_row[1L], "t value"])
                    if (t_sf > max_t_sem_fam) max_t_sem_fam <- t_sf
                }
                if (length(vis_fam_row) > 0) {
                    t_vf <- abs(cs[vis_fam_row[1L], "t value"])
                    if (t_vf > max_t_vis_fam) max_t_vis_fam <- t_vf
                }
            }
        }
        c(max_t_sem_fam, max_t_vis_fam)
    }

    n_chunk <- ncol(sem_mat)
    res <- if (n_cores > 1L) {
        parallel::mclapply(seq_len(n_chunk), run_one, mc.cores = n_cores)
    } else {
        lapply(seq_len(n_chunk), run_one)
    }
    mat <- do.call(cbind, res)   # 2 x n_chunk
    list(sem = mat[1L, ], vis = mat[2L, ])
}
"""


# ── rpy2 helpers ──────────────────────────────────────────────────────────────

def _setup_r():
    configure_r()
    import rpy2.robjects as ro
    ro.r(_R_FUNC)
    return ro


def _df_to_r(ro, df):
    import rpy2.robjects.pandas2ri as pandas2ri
    with ro.conversion.localconverter(ro.default_converter + pandas2ri.converter):
        return ro.conversion.py2rpy(df)


def _prepare_r_objects(ro, trait_dfs, pairs_i, pairs_j, target_idx):
    from rpy2.robjects import ListVector, IntVector

    pair_to_k = {
        (int(target_idx[li]), int(target_idx[lj])): k
        for k, (li, lj) in enumerate(zip(pairs_i, pairs_j))
    }

    r_dfs_items     = []
    r_pair_ks_items = []

    cols = ['participant', 'target1', 'target2', 'abs_diff',
            'sem_dissim_z', 'vis_dissim_z', 'fam_dissim_z', 'fam_mean_z']

    for t_idx, (trait, df) in enumerate(trait_dfs.items()):
        t1 = df['target1'].astype(int).values
        t2 = df['target2'].astype(int).values
        pk_0 = np.array([pair_to_k[(a, b)] for a, b in zip(t1, t2)], dtype=np.int32)

        r_df = _df_to_r(ro, df[cols])
        r_dfs_items.append((str(t_idx), r_df))
        r_pair_ks_items.append((str(t_idx), IntVector((pk_0 + 1).tolist())))
        print(f"    {trait}: {len(df):,} rows transferred to R")

    return ListVector(r_dfs_items), ListVector(r_pair_ks_items)


def _run_chunk(ro, r_dfs, r_pair_ks, sem_chunk, vis_chunk, n_workers):
    from rpy2.robjects import FloatVector, IntVector

    n_pairs, n_iter = sem_chunk.shape
    sem_r = ro.r.matrix(
        FloatVector(sem_chunk.ravel(order='F').tolist()),
        nrow=n_pairs, ncol=n_iter
    )
    vis_r = ro.r.matrix(
        FloatVector(vis_chunk.ravel(order='F').tolist()),
        nrow=n_pairs, ncol=n_iter
    )
    result = ro.r['.lmm_fam_perm_chunk'](r_dfs, r_pair_ks, sem_r, vis_r,
                                          IntVector([n_workers]))
    return np.array(result.rx2('sem')), np.array(result.rx2('vis'))


# ── Per-dataset runner ────────────────────────────────────────────────────────

def run_dataset(ds_name: str, ro, n_perm: int, n_workers: int):
    cfg      = DATASETS[ds_name]
    n_traits = len(cfg['traits'])
    print(f"\n{'='*60}")
    print(f"LMM fam-interaction max-T — {ds_name.upper()}")
    print(f"Family: 'all' split, {n_traits} traits — sem:fam and vis:fam corrected separately")
    print(f"Permutations: {n_perm}  |  chunk size: {CHUNK_SIZE}  |  workers: {n_workers}")
    print(f"{'='*60}")

    outdir = PIPELINE_DIR / ds_name
    outdir.mkdir(parents=True, exist_ok=True)

    lmm_path = outdir / 'dual_lmm_results_fam_interaction.csv'
    if not lmm_path.exists():
        raise FileNotFoundError(
            f"{lmm_path} not found. "
            f"Run 'python 2_code/run_lmm.py --dataset {ds_name} --model fam_interaction' first."
        )
    lmm_df = pd.read_csv(lmm_path)

    ratings_dict     = load_ratings(cfg)
    fam_ratings      = load_familiarity_ratings(cfg)
    sem_rdm, vis_rdm = load_rdms(cfg)

    n_targets  = cfg['n_targets']
    target_idx = np.arange(n_targets)
    all_mask   = np.ones(n_targets, dtype=bool)

    pairs_i, pairs_j = np.triu_indices(n_targets, k=1)

    sem_sub = sem_rdm[np.ix_(target_idx, target_idx)]
    vis_sub = vis_rdm[np.ix_(target_idx, target_idx)]

    print(f"\nBuilding pair DataFrames (with fam columns) and transferring to R...")
    trait_dfs = {
        trait: build_pair_data(ratings_dict[trait], all_mask, sem_rdm, vis_rdm,
                               fam_ratings=fam_ratings)
        for trait in cfg['traits']
    }
    r_dfs, r_pair_ks = _prepare_r_objects(
        ro, trait_dfs, pairs_i, pairs_j, target_idx
    )

    rng       = np.random.default_rng(SEED)
    all_perms = [rng.permutation(n_targets) for _ in range(n_perm)]

    checkpoint_path = outdir / 'lmm_fam_interaction_maxt_null_dist_checkpoint.npy'
    null_dist = np.full((2, n_perm), np.nan)
    start_k   = 0

    if checkpoint_path.exists():
        saved = np.load(checkpoint_path)
        if saved.shape == (2, n_perm):
            null_dist = saved
            start_k   = int(np.sum(~np.isnan(null_dist[0])))
            print(f"\nCheckpoint found: resuming from permutation {start_k}/{n_perm}")
        else:
            print(f"\nCheckpoint shape {saved.shape} incompatible; starting fresh.")

    print(f"\nRunning null distribution ({n_perm - start_k} permutations remaining)...")
    t_wall = time.time()

    def _zscore(v):
        s = v.std()
        return (v - v.mean()) / (s if s > 0 else 1.0)

    for chunk_start in range(start_k, n_perm, CHUNK_SIZE):
        chunk_end   = min(chunk_start + CHUNK_SIZE, n_perm)
        chunk_perms = all_perms[chunk_start:chunk_end]
        n_chunk     = len(chunk_perms)

        sem_chunk = np.column_stack([
            _zscore(sem_sub[p[pairs_i], p[pairs_j]]) for p in chunk_perms
        ])
        vis_chunk = np.column_stack([
            _zscore(vis_sub[p[pairs_i], p[pairs_j]]) for p in chunk_perms
        ])

        t0 = time.time()
        chunk_sem_tmax, chunk_vis_tmax = _run_chunk(
            ro, r_dfs, r_pair_ks, sem_chunk, vis_chunk, n_workers)
        chunk_time = time.time() - t0

        null_dist[0, chunk_start:chunk_end] = chunk_sem_tmax
        null_dist[1, chunk_start:chunk_end] = chunk_vis_tmax
        np.save(checkpoint_path, null_dist)

        done      = chunk_end
        elapsed   = time.time() - t_wall
        rate      = (done - start_k) / elapsed if elapsed > 0 else 1
        remaining = (n_perm - done) / rate if rate > 0 else 0
        print(
            f"  [{done:4d}/{n_perm}]  "
            f"chunk Tmax_sem:fam={chunk_sem_tmax.max():.3f}  "
            f"chunk Tmax_vis:fam={chunk_vis_tmax.max():.3f}  "
            f"chunk time={chunk_time:.0f}s  "
            f"ETA={remaining/3600:.1f}h",
            flush=True
        )

    n_done = int(np.sum(~np.isnan(null_dist[0])))
    assert n_done == n_perm, f"Only {n_done}/{n_perm} permutations completed."
    null_sem = null_dist[0]
    null_vis = null_dist[1]
    q95_sem = np.percentile(null_sem, 95)
    q95_vis = np.percentile(null_vis, 95)
    print(f"\nNull distribution complete.  Tmax 95th pct: sem:fam={q95_sem:.4f}, vis:fam={q95_vis:.4f}")

    # ── Corrected p-values ────────────────────────────────────────────────────
    out_df = lmm_df.copy()
    out_df['sem_X_fam_p_maxt'] = float('nan')
    out_df['vis_X_fam_p_maxt'] = float('nan')
    out_df['n_perm_maxt']      = float('nan')

    all_rows = out_df['split'] == 'all'
    print(f"\nCorrected p-values (Tmax 95th pct: sem:fam={q95_sem:.3f}, vis:fam={q95_vis:.3f}):")
    for idx, row in out_df[all_rows].iterrows():
        t_sf = row['sem_dissim_z_X_fam_mean_z_t']
        t_vf = row['vis_dissim_z_X_fam_mean_z_t']
        p_sf = maxt_corrected_p(null_sem, t_sf)
        p_vf = maxt_corrected_p(null_vis, t_vf)
        out_df.at[idx, 'sem_X_fam_p_maxt'] = p_sf
        out_df.at[idx, 'vis_X_fam_p_maxt'] = p_vf
        out_df.at[idx, 'n_perm_maxt']       = n_perm
        print(
            f"  {row['trait']:12s}  "
            f"t_sem:fam={t_sf:7.2f} p={row['sem_dissim_z_X_fam_mean_z_p']:.4g} → p_maxt={p_sf:.4f}  |  "
            f"t_vis:fam={t_vf:7.2f} p={row['vis_dissim_z_X_fam_mean_z_p']:.4g} → p_maxt={p_vf:.4f}"
        )

    np.save(outdir / 'lmm_fam_interaction_maxt_null_sem.npy', null_sem)
    np.save(outdir / 'lmm_fam_interaction_maxt_null_vis.npy', null_vis)
    out_df.to_csv(outdir / 'dual_lmm_fam_interaction_maxt_results.csv', index=False)
    print(f"\nSaved → {outdir / 'dual_lmm_fam_interaction_maxt_results.csv'}")
    return out_df


# ── Entry point ───────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', choices=['us', 'cn', 'both'], default='both')
    parser.add_argument('--n-perm', type=int, default=N_PERM_DEFAULT)
    parser.add_argument('--n-workers', type=int, default=N_WORKERS_DEFAULT)
    args = parser.parse_args()

    ro       = _setup_r()
    datasets = ['us', 'cn'] if args.dataset == 'both' else [args.dataset]
    for ds in datasets:
        run_dataset(ds, ro, args.n_perm, args.n_workers)


if __name__ == '__main__':
    main()
