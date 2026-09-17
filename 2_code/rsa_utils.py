"""
Pure-math RSA utilities (no I/O).

Functions
---------
flatten_rdm(rdm)
build_trait_rdm(ratings, target_mask=None)
build_trait_space_rdm(ratings_dict, traits, target_mask=None)
rsa_spearman(vec_a, vec_b)
permutation_test(trait_rdm, target_rdm, n_perm=5000)
partial_rsa(trait_rdm, target_rdm, confound_rdm, n_perm=5000)
variance_partitioning(trait_vec, sem_vec, vis_vec)
"""

import numpy as np
from scipy.spatial.distance import cdist, squareform
from scipy.stats import spearmanr
import statsmodels.api as sm

np.random.seed(42)


def flatten_rdm(rdm: np.ndarray) -> np.ndarray:
    """Upper triangle (excluding diagonal) → 1-D vector."""
    return squareform(rdm, checks=False)


def build_trait_rdm(ratings: np.ndarray, target_mask: np.ndarray = None) -> np.ndarray:
    """Pairwise |mean_rating(i) - mean_rating(j)| across targets.

    Parameters
    ----------
    ratings : (n_participants, n_targets)
    target_mask : boolean (n_targets,) — subset targets; None = use all
    """
    if target_mask is not None:
        ratings = ratings[:, target_mask]
    mean_r = np.nanmean(ratings, axis=0)
    n = len(mean_r)
    rdm = np.zeros((n, n))
    for i in range(n):
        for j in range(i + 1, n):
            d = abs(mean_r[i] - mean_r[j])
            rdm[i, j] = rdm[j, i] = d
    return rdm


def build_trait_space_rdm(ratings_dict: dict, traits: list,
                          target_mask: np.ndarray = None) -> np.ndarray:
    """Pairwise Euclidean distance in trait space across targets.

    Each target is a point in R^n_traits (mean ratings). Raw (unstandardised)
    Euclidean distance is used because all traits share the same rating scale.

    Parameters
    ----------
    ratings_dict : dict[trait -> (n_participants, n_targets)]
    traits : ordered list of trait names to include
    target_mask : boolean (n_targets,) — subset targets; None = use all
    """
    means = np.column_stack([np.nanmean(ratings_dict[t], axis=0) for t in traits])
    if target_mask is not None:
        means = means[target_mask]
    return cdist(means, means, metric='euclidean')


def rsa_spearman(vec_a: np.ndarray, vec_b: np.ndarray) -> float:
    """Spearman r between two flat RDM vectors, ignoring NaN pairs."""
    m = ~np.isnan(vec_a) & ~np.isnan(vec_b)
    return spearmanr(vec_a[m], vec_b[m])[0]


def permutation_test(trait_rdm: np.ndarray, target_rdm: np.ndarray,
                     n_perm: int = 5000) -> tuple:
    """Standard RSA + permutation p-value.

    Permutes rows and columns of trait_rdm simultaneously (preserving
    symmetry), returns (r_observed, p_permutation).
    """
    trait_vec  = flatten_rdm(trait_rdm)
    target_vec = flatten_rdm(target_rdm)
    r_obs = rsa_spearman(trait_vec, target_vec)
    n = trait_rdm.shape[0]
    count = 0
    for _ in range(n_perm):
        perm = np.random.permutation(n)
        perm_vec = flatten_rdm(trait_rdm[np.ix_(perm, perm)])
        if abs(rsa_spearman(perm_vec, target_vec)) >= abs(r_obs):
            count += 1
    return r_obs, (count + 1) / (n_perm + 1)


def partial_rsa(trait_rdm: np.ndarray, target_rdm: np.ndarray,
                confound_rdm: np.ndarray, n_perm: int = 5000) -> tuple:
    """Partial Spearman r (target ~ trait controlling for confound).

    Both trait and target vectors are residualised on confound before
    computing Spearman r. Returns (r_partial, p_permutation).
    """
    target_vec   = flatten_rdm(target_rdm)
    confound_vec = flatten_rdm(confound_rdm)
    trait_vec    = flatten_rdm(trait_rdm)

    def _partial_r(tv, tgt, conf):
        m = ~np.isnan(tv) & ~np.isnan(tgt) & ~np.isnan(conf)
        tv, tgt, conf = tv[m], tgt[m], conf[m]
        X = sm.add_constant(conf)
        res_trait  = tv  - sm.OLS(tv,  X).fit().fittedvalues
        res_target = tgt - sm.OLS(tgt, X).fit().fittedvalues
        return spearmanr(res_trait, res_target)[0]

    r_obs = _partial_r(trait_vec, target_vec, confound_vec)
    n = trait_rdm.shape[0]
    count = 0
    for _ in range(n_perm):
        perm = np.random.permutation(n)
        perm_vec = flatten_rdm(trait_rdm[np.ix_(perm, perm)])
        if abs(_partial_r(perm_vec, target_vec, confound_vec)) >= abs(r_obs):
            count += 1
    return r_obs, (count + 1) / (n_perm + 1)


def _partial_r_vecs(trait_vec: np.ndarray, target_vec: np.ndarray,
                    confound_vec: np.ndarray) -> float:
    """Partial Spearman r between target_vec ~ trait_vec controlling for confound_vec."""
    m = ~np.isnan(trait_vec) & ~np.isnan(target_vec) & ~np.isnan(confound_vec)
    tv, tgt, conf = trait_vec[m], target_vec[m], confound_vec[m]
    X = sm.add_constant(conf)
    res_trait  = tv  - sm.OLS(tv,  X).fit().fittedvalues
    res_target = tgt - sm.OLS(tgt, X).fit().fittedvalues
    return spearmanr(res_trait, res_target)[0]


def maxt_corrected_p(null_dist: np.ndarray, observed_stat: float) -> float:
    """Max-T corrected p-value with +1 finite-sample correction."""
    null_dist = np.asarray(null_dist, dtype=float)
    return (np.sum(null_dist >= abs(observed_stat)) + 1) / (len(null_dist) + 1)

def maxt_permutation_rsa(trait_rdms: dict, sem_rdm: np.ndarray, vis_rdm: np.ndarray,
                          n_perm: int = 1000, seed: int = 42) -> dict:
    """Max-T FWER-corrected RSA across all traits for the primary 'all' split.

    One permutation of the model-RDM target labels is applied simultaneously to
    all trait RDMs, preserving the joint dependence structure.  Four separate null
    distributions are built — one per channel — so sem and vis are corrected
    independently across traits:
      - null_sem         : max |r_sem|         across all traits
      - null_vis         : max |r_vis|         across all traits
      - null_partial_sem : max |r_partial_sem| across all traits
      - null_partial_vis : max |r_partial_vis| across all traits

    Parameters
    ----------
    trait_rdms : dict[trait_name -> (n, n) RDM]  — one RDM per trait
    sem_rdm    : (n, n) semantic model RDM
    vis_rdm    : (n, n) visual model RDM
    n_perm     : number of permutation iterations
    seed       : RNG seed for reproducibility

    Returns
    -------
    dict with keys:
      null_sem         : (n_perm,) max |r| null distribution, semantic channel
      null_vis         : (n_perm,) max |r| null distribution, visual channel
      null_partial_sem : (n_perm,) max |partial r| null, semantic channel
      null_partial_vis : (n_perm,) max |partial r| null, visual channel
      obs          : dict[trait -> {rsa_sem_r, rsa_vis_r, partial_sem_r, partial_vis_r}]
      corrected_p  : dict[trait -> {rsa_sem_p_maxt, rsa_vis_p_maxt,
                                    partial_sem_p_maxt, partial_vis_p_maxt}]
    """
    rng    = np.random.default_rng(seed)
    traits = list(trait_rdms.keys())
    n      = sem_rdm.shape[0]

    sem_vec = flatten_rdm(sem_rdm)
    vis_vec = flatten_rdm(vis_rdm)

    # Observed statistics (using original model RDMs)
    obs = {}
    for trait, rdm in trait_rdms.items():
        tv = flatten_rdm(rdm)
        obs[trait] = {
            'rsa_sem_r':     rsa_spearman(tv, sem_vec),
            'rsa_vis_r':     rsa_spearman(tv, vis_vec),
            'partial_sem_r': _partial_r_vecs(tv, sem_vec, vis_vec),
            'partial_vis_r': _partial_r_vecs(tv, vis_vec, sem_vec),
        }

    # Pre-flatten trait RDMs once (they do not change across permutations)
    trait_vecs = {t: flatten_rdm(rdm) for t, rdm in trait_rdms.items()}

    null_sem         = np.zeros(n_perm)
    null_vis         = np.zeros(n_perm)
    null_partial_sem = np.zeros(n_perm)
    null_partial_vis = np.zeros(n_perm)

    for k in range(n_perm):
        # Same permutation applied to both model RDMs
        perm     = rng.permutation(n)
        sem_perm = flatten_rdm(sem_rdm[np.ix_(perm, perm)])
        vis_perm = flatten_rdm(vis_rdm[np.ix_(perm, perm)])

        max_sem = max_vis = max_ps = max_pv = 0.0
        for tv in trait_vecs.values():
            max_sem = max(max_sem, abs(rsa_spearman(tv, sem_perm)))
            max_vis = max(max_vis, abs(rsa_spearman(tv, vis_perm)))
            max_ps  = max(max_ps,  abs(_partial_r_vecs(tv, sem_perm, vis_perm)))
            max_pv  = max(max_pv,  abs(_partial_r_vecs(tv, vis_perm, sem_perm)))
        null_sem[k]         = max_sem
        null_vis[k]         = max_vis
        null_partial_sem[k] = max_ps
        null_partial_vis[k] = max_pv

    # Corrected p-values: proportion of null >= observed |r|
    corrected_p = {}
    for trait in traits:
        o = obs[trait]
        corrected_p[trait] = {
            'rsa_sem_p_maxt':     maxt_corrected_p(null_sem, o['rsa_sem_r']),
            'rsa_vis_p_maxt':     maxt_corrected_p(null_vis, o['rsa_vis_r']),
            'partial_sem_p_maxt': maxt_corrected_p(null_partial_sem, o['partial_sem_r']),
            'partial_vis_p_maxt': maxt_corrected_p(null_partial_vis, o['partial_vis_r']),
        }

    return {
        'null_sem':         null_sem,
        'null_vis':         null_vis,
        'null_partial_sem': null_partial_sem,
        'null_partial_vis': null_partial_vis,
        'obs':              obs,
        'corrected_p':      corrected_p,
    }


def variance_partitioning(trait_vec: np.ndarray, sem_vec: np.ndarray,
                          vis_vec: np.ndarray) -> dict:
    """OLS R² decomposition into unique and shared variance.

    Returns dict with keys: unique_sem, unique_vis, shared, total_r2.
    """
    m = ~np.isnan(trait_vec) & ~np.isnan(sem_vec) & ~np.isnan(vis_vec)
    y, x1, x2 = trait_vec[m], sem_vec[m], vis_vec[m]
    r2_full = sm.OLS(y, sm.add_constant(np.column_stack([x1, x2]))).fit().rsquared
    r2_sem  = sm.OLS(y, sm.add_constant(x1)).fit().rsquared
    r2_vis  = sm.OLS(y, sm.add_constant(x2)).fit().rsquared
    unique_sem = r2_full - r2_vis
    unique_vis = r2_full - r2_sem
    shared     = r2_full - unique_sem - unique_vis
    return {
        'unique_sem': unique_sem,
        'unique_vis': unique_vis,
        'shared':     shared,
        'total_r2':   r2_full,
    }
