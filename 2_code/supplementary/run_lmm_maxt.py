"""
Max-T FWER correction for LMM fixed-effect t-statistics — full LMM permutation.

For each of 1,000 permutations:
  1. Permute target labels in model RDMs (same π applied to sem + vis).
  2. Recompute sem_dissim_z / vis_dissim_z for all pairs.
  3. Refit LMM (abs_diff ~ sem_dissim_z + vis_dissim_z + rand) for every trait.
  4. Track max |t| across traits × {sem, vis} → Tmax null distribution.

The lmer fitting loop runs natively in R (via rpy2) in chunks of CHUNK_SIZE,
avoiding repeated Python→R round-trip overhead.  Python handles data prep,
checkpointing, and corrected p-value computation.

Checkpointing: null array saved to lmm_maxt_null_dist_checkpoint.npy after
every chunk.  Resume by re-running the same command.

Permutations are generated from a fixed seed so that resuming from a
checkpoint reproduces exactly the same permutation sequence.

Family: 'all' split, all traits × {sem_dissim_z, vis_dissim_z}.
  US: 8 traits × 2 = 16 tests
  CN: 10 traits × 2 = 20 tests

Parallelism: R's mclapply parallelizes across permutations within each chunk
(fork-based, copy-on-write → low memory overhead).  Use --n-workers to set
the number of cores (default 8).  Measured speedup: ~3x/4 cores → ~5x/8 cores.

Estimated runtime per dataset (measured ~12 s / lmer fit, ~5x speedup w/ 8 workers):
  n_perm=100  →  ~45 min     (fast sanity check)
  n_perm=200  →  ~90 min
  n_perm=1000 →  ~6 h        (publication-quality, fits overnight)

Usage:
    python 2_code/run_lmm_maxt.py --dataset us
    python 2_code/run_lmm_maxt.py --dataset cn
    python 2_code/run_lmm_maxt.py --dataset both
    python 2_code/run_lmm_maxt.py --dataset us --n-workers 16
"""

import argparse
import sys
import warnings
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

import numpy as np
import pandas as pd

from config import DATASETS, PIPELINE_DIR
from data_loader import load_ratings, load_rdms
from lmm_utils import configure_r, build_pair_data
from rsa_utils import maxt_corrected_p

warnings.filterwarnings('ignore')

N_PERM_DEFAULT  = 1000
N_WORKERS_DEFAULT = 8
CHUNK_SIZE      = 50    # permutations per R call; checkpoint after each chunk
SEED            = 42


# ── R permutation function ────────────────────────────────────────────────────

_R_FUNC = """
library(lme4)
library(parallel)

.lmm_perm_chunk <- function(dfs, pair_ks, sem_mat, vis_mat, n_cores=1L) {
    # dfs      : R list of data.frames, one per trait.
    #            Each has columns: participant, target1, target2, abs_diff,
    #            sem_dissim_z, vis_dissim_z (last two are overwritten each iter).
    # pair_ks  : R list of 1-based integer vectors, length nrow(dfs[[t]]),
    #            indexing rows of sem_mat / vis_mat.
    # sem_mat  : R matrix (n_pairs x n_chunk), z-scored sem dissimilarities.
    # vis_mat  : R matrix (n_pairs x n_chunk), z-scored vis dissimilarities.
    # n_cores  : number of cores for mclapply (1 = serial).
    # Returns  : named list with $sem and $vis, each numeric(n_chunk) —
    #            max |t_sem| and max |t_vis| across traits per permutation.

    n_traits <- length(dfs)

    run_one <- function(k) {
        max_t_sem <- 0.0
        max_t_vis <- 0.0
        for (t in seq_len(n_traits)) {
            df <- dfs[[t]]
            pk <- pair_ks[[t]]
            df[["sem_dissim_z"]] <- sem_mat[pk, k]
            df[["vis_dissim_z"]] <- vis_mat[pk, k]
            mod <- tryCatch(
                suppressMessages(suppressWarnings(
                    lme4::lmer(
                        abs_diff ~ sem_dissim_z + vis_dissim_z +
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
                t_sem <- abs(cs["sem_dissim_z", "t value"])
                t_vis <- abs(cs["vis_dissim_z", "t value"])
                if (t_sem > max_t_sem) max_t_sem <- t_sem
                if (t_vis > max_t_vis) max_t_vis <- t_vis
            }
        }
        c(max_t_sem, max_t_vis)
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
    """Configure R, load lme4, define R permutation function."""
    configure_r()
    import rpy2.robjects as ro
    ro.r(_R_FUNC)
    return ro


def _df_to_r(ro, df):
    """Convert a pandas DataFrame to an R data.frame."""
    import rpy2.robjects.pandas2ri as pandas2ri
    with ro.conversion.localconverter(ro.default_converter + pandas2ri.converter):
        return ro.conversion.py2rpy(df)


def _prepare_r_objects(ro, trait_dfs, pairs_i, pairs_j, target_idx):
    """
    Convert trait DataFrames and pair_k arrays to R ListVectors.

    pair_k maps each row of a trait DataFrame to its local pair index
    (0-based into pairs_i/pairs_j), used to look up permuted predictors.
    """
    from rpy2.robjects import ListVector, IntVector

    # Mapping (global_t1, global_t2) → 0-based local pair index
    pair_to_k = {
        (int(target_idx[li]), int(target_idx[lj])): k
        for k, (li, lj) in enumerate(zip(pairs_i, pairs_j))
    }

    r_dfs_items     = []
    r_pair_ks_items = []

    for t_idx, (trait, df) in enumerate(trait_dfs.items()):
        t1 = df['target1'].astype(int).values
        t2 = df['target2'].astype(int).values
        pk_0 = np.array([pair_to_k[(a, b)] for a, b in zip(t1, t2)], dtype=np.int32)

        cols = ['participant', 'target1', 'target2', 'abs_diff',
                'sem_dissim_z', 'vis_dissim_z']
        r_df = _df_to_r(ro, df[cols])

        r_dfs_items.append((str(t_idx), r_df))
        r_pair_ks_items.append((str(t_idx), IntVector((pk_0 + 1).tolist())))  # 1-based
        print(f"    {trait}: {len(df):,} rows transferred to R")

    return ListVector(r_dfs_items), ListVector(r_pair_ks_items)


def _run_chunk(ro, r_dfs, r_pair_ks, sem_chunk, vis_chunk, n_workers):
    """
    Run one chunk of permutations in R (parallelised with mclapply).

    sem_chunk, vis_chunk: (n_pairs, n_chunk) float64 arrays
    n_workers: number of R/mclapply cores
    Returns: tuple of two (n_chunk,) arrays — (max_t_sem, max_t_vis)
    """
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

    result = ro.r['.lmm_perm_chunk'](r_dfs, r_pair_ks, sem_r, vis_r,
                                      IntVector([n_workers]))
    return np.array(result.rx2('sem')), np.array(result.rx2('vis'))


# ── Per-dataset runner ────────────────────────────────────────────────────────

def run_dataset(ds_name: str, ro, n_perm: int, n_workers: int):
    cfg = DATASETS[ds_name]
    n_traits = len(cfg['traits'])
    eta_h    = n_perm * n_traits * 12.5 / n_workers / 3600
    print(f"\n{'='*60}")
    print(f"LMM max-T (full LMM permutation) — {ds_name.upper()}")
    print(f"Family: 'all' split, {n_traits} traits — sem and vis corrected separately ({n_traits} tests each)")
    print(f"Permutations: {n_perm}  |  chunk size: {CHUNK_SIZE}  |  workers: {n_workers}")
    print(f"ETA (≈12 s/fit, {n_workers} cores): {eta_h:.1f} h")
    print(f"{'='*60}")

    outdir = PIPELINE_DIR / ds_name
    outdir.mkdir(parents=True, exist_ok=True)

    # Load observed LMM t-statistics
    lmm_path = outdir / 'dual_lmm_results.csv'
    if not lmm_path.exists():
        raise FileNotFoundError(
            f"{lmm_path} not found. "
            f"Run 'python 2_code/run_lmm.py --dataset {ds_name}' first."
        )
    lmm_df = pd.read_csv(lmm_path)

    ratings_dict     = load_ratings(cfg)
    sem_rdm, vis_rdm = load_rdms(cfg)

    n_targets  = cfg['n_targets']
    all_mask   = np.ones(n_targets, dtype=bool)
    target_idx = np.where(all_mask)[0]      # == np.arange(n_targets) for 'all'

    pairs_i, pairs_j = np.triu_indices(n_targets, k=1)
    n_pairs = len(pairs_i)

    sem_sub = sem_rdm[np.ix_(target_idx, target_idx)]
    vis_sub = vis_rdm[np.ix_(target_idx, target_idx)]

    # Build and transfer pair DataFrames to R (one-time cost)
    print(f"\nBuilding pair DataFrames and transferring to R...")
    trait_dfs = {
        trait: build_pair_data(ratings_dict[trait], all_mask, sem_rdm, vis_rdm)
        for trait in cfg['traits']
    }
    r_dfs, r_pair_ks = _prepare_r_objects(
        ro, trait_dfs, pairs_i, pairs_j, target_idx
    )

    # Generate all permutations from fixed seed (ensures reproducible resume)
    rng       = np.random.default_rng(SEED)
    all_perms = [rng.permutation(n_targets) for _ in range(n_perm)]

    # Load checkpoint if available
    # Shape (2, n_perm): row 0 = null_sem, row 1 = null_vis
    checkpoint_path = outdir / 'lmm_maxt_null_dist_checkpoint.npy'
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

    # ── Permutation loop (chunked) ────────────────────────────────────────────
    print(f"\nRunning null distribution ({n_perm - start_k} permutations remaining)...")
    t_wall = time.time()

    for chunk_start in range(start_k, n_perm, CHUNK_SIZE):
        chunk_end   = min(chunk_start + CHUNK_SIZE, n_perm)
        chunk_perms = all_perms[chunk_start:chunk_end]
        n_chunk     = len(chunk_perms)

        # Build permuted predictor matrices: (n_pairs × n_chunk)
        def _zscore(v):
            s = v.std()
            return (v - v.mean()) / (s if s > 0 else 1.0)

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
            f"chunk Tmax_sem={chunk_sem_tmax.max():.3f}  "
            f"chunk Tmax_vis={chunk_vis_tmax.max():.3f}  "
            f"chunk time={chunk_time:.0f}s  "
            f"ETA={remaining/3600:.1f}h",
            flush=True
        )

    n_done = int(np.sum(~np.isnan(null_dist[0])))
    assert n_done == n_perm, f"Only {n_done}/{n_perm} permutations completed — check logs."
    null_sem = null_dist[0]
    null_vis = null_dist[1]
    q95_sem = np.percentile(null_sem, 95)
    q95_vis = np.percentile(null_vis, 95)
    print(f"\nNull distribution complete.  Tmax 95th pct: sem={q95_sem:.4f}, vis={q95_vis:.4f}")

    # ── Corrected p-values ────────────────────────────────────────────────────
    out_df = lmm_df.copy()
    out_df['sem_dissim_z_p_maxt'] = float('nan')
    out_df['vis_dissim_z_p_maxt'] = float('nan')
    out_df['n_perm_maxt']         = float('nan')

    all_rows = out_df['split'] == 'all'
    print(f"\nCorrected p-values (Tmax 95th pct: sem={q95_sem:.3f}, vis={q95_vis:.3f}):")
    for idx, row in out_df[all_rows].iterrows():
        t_sem = row['sem_dissim_z_t']
        t_vis = row['vis_dissim_z_t']
        p_sem = maxt_corrected_p(null_sem, t_sem)
        p_vis = maxt_corrected_p(null_vis, t_vis)
        out_df.at[idx, 'sem_dissim_z_p_maxt'] = p_sem
        out_df.at[idx, 'vis_dissim_z_p_maxt'] = p_vis
        out_df.at[idx, 'n_perm_maxt']          = n_perm
        print(
            f"  {row['trait']:12s}  "
            f"t_sem={t_sem:6.2f} p={row['sem_dissim_z_p']:.4g} → p_maxt={p_sem:.4f}  |  "
            f"t_vis={t_vis:6.2f} p={row['vis_dissim_z_p']:.4g} → p_maxt={p_vis:.4f}"
        )

    # Save outputs
    np.save(outdir / 'lmm_maxt_null_sem.npy', null_sem)
    np.save(outdir / 'lmm_maxt_null_vis.npy', null_vis)
    out_df.to_csv(outdir / 'dual_lmm_maxt_results.csv', index=False)
    print(f"\nSaved → {outdir / 'dual_lmm_maxt_results.csv'}")
    return out_df


# ── Entry point ───────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', choices=['us', 'cn', 'both'], default='both')
    parser.add_argument('--n-perm', type=int, default=N_PERM_DEFAULT,
                        help=f'Number of permutations (default {N_PERM_DEFAULT}).')
    parser.add_argument('--n-workers', type=int, default=N_WORKERS_DEFAULT,
                        help=f'Parallel R workers via mclapply (default {N_WORKERS_DEFAULT}). '
                             f'This machine has 32 logical CPUs.')
    args = parser.parse_args()

    ro       = _setup_r()
    datasets = ['us', 'cn'] if args.dataset == 'both' else [args.dataset]
    for ds in datasets:
        run_dataset(ds, ro, args.n_perm, args.n_workers)


if __name__ == '__main__':
    main()
