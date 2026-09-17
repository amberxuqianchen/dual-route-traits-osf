"""
LMM utilities: R setup and pair-level data building.

Functions
---------
configure_r()
build_pair_data(ratings, target_mask, sem_rdm, vis_rdm, dataset_label=None)
fit_lmm(df_pairs, extra_fixed=None)
"""

import os
import subprocess
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import chi2


# ── R / pymer4 setup ──────────────────────────────────────────────────────────

def _candidate_r_configs():
    home = Path.home()
    env_roots = []
    explicit = os.environ.get('PYMER4_R_HOME') or os.environ.get('R_HOME')
    if explicit:
        r_home = Path(explicit).expanduser()
        if r_home.name == 'R' and r_home.parent.name == 'lib':
            env_roots.append(r_home.parent.parent)
    env_roots.extend([
        home / 'anaconda3' / 'envs' / 'R',
        home / 'anaconda3' / 'envs' / 'r_efa',
        home / 'anaconda3' / 'envs' / 'jupyter_env' / 'envs' / 'facebert',
    ])
    seen = set()
    for root in env_roots:
        root = root.resolve()
        if root in seen:
            continue
        seen.add(root)
        r_home  = root / 'lib' / 'R'
        r_lib   = r_home / 'library'
        rscript = root / 'bin' / 'Rscript'
        if r_home.exists() and r_lib.exists() and rscript.exists():
            yield {'env_root': root, 'r_home': r_home,
                   'r_lib': r_lib, 'rscript': rscript}


def _probe_r_config(config):
    lib  = str(config['r_lib']).replace('\\', '/')
    expr = (f'.libPaths(c("{lib}")); '
            'suppressPackageStartupMessages({'
            'library(lme4); library(lmerTest); library(rlang)'
            '}); cat("OK\\n")')
    result = subprocess.run([str(config['rscript']), '--vanilla', '-e', expr],
                            capture_output=True, text=True)
    return result.returncode == 0 and 'OK' in result.stdout


def configure_r():
    """Detect and configure a working R runtime for pymer4.

    Sets R_HOME, R_LIBS* env vars and imports pymer4.
    Raises RuntimeError if no suitable R installation is found.
    """
    for config in _candidate_r_configs():
        if _probe_r_config(config):
            os.environ['R_HOME']      = str(config['r_home'])
            os.environ['R_LIBS']      = str(config['r_lib'])
            os.environ['R_LIBS_USER'] = str(config['r_lib'])
            os.environ['R_LIBS_SITE'] = str(config['r_lib'])
            r_lib_path = str(config['r_lib']).replace('\\', '/')
            from rpy2 import robjects
            robjects.r(f'.libPaths(c("{r_lib_path}", .libPaths()))')
            print(f"R_HOME={config['r_home']}")
            return config
    raise RuntimeError('Could not find a working R runtime for pymer4. '
                       'Needs lme4, lmerTest, rlang.')


# ── Pair-level data ───────────────────────────────────────────────────────────

def build_pair_data(ratings: np.ndarray, target_mask: np.ndarray,
                    sem_rdm: np.ndarray, vis_rdm: np.ndarray,
                    dataset_label: str = None,
                    fam_ratings: np.ndarray = None) -> pd.DataFrame:
    """Build long-format DataFrame for one trait × split.

    Parameters
    ----------
    ratings     : (n_participants, n_targets) array for one trait
    target_mask : boolean (n_targets,)
    sem_rdm     : (n_targets, n_targets) full semantic RDM
    vis_rdm     : (n_targets, n_targets) full visual RDM
    dataset_label : optional string prefix for participant/target IDs (for merging)
    fam_ratings : optional (n_participants, n_targets) per-perceiver familiarity ratings;
                  when provided adds fam_dissim_z column (perceiver-specific pairwise |fam_i - fam_j|)

    Columns returned: participant, target1, target2, abs_diff,
                      sem_dissim_z, vis_dissim_z [, fam_dissim_z] [, dataset]
    """
    target_indices = np.where(target_mask)[0]
    n = len(target_indices)

    sem_sub = sem_rdm[np.ix_(target_indices, target_indices)]
    vis_sub = vis_rdm[np.ix_(target_indices, target_indices)]

    pairs_i, pairs_j = np.triu_indices(n, k=1)
    sem_vals = np.array([sem_sub[i, j] for i, j in zip(pairs_i, pairs_j)])
    vis_vals = np.array([vis_sub[i, j] for i, j in zip(pairs_i, pairs_j)])

    sem_z = (sem_vals - np.nanmean(sem_vals)) / np.nanstd(sem_vals)
    vis_z = (vis_vals - np.nanmean(vis_vals)) / np.nanstd(vis_vals)

    n_participants = ratings.shape[0]
    participant_ids = [str(i) for i in range(n_participants)]

    rows = []
    for p_idx, p_id in enumerate(participant_ids):
        r_p       = ratings[p_idx][target_mask]
        abs_diffs = np.abs(r_p[pairs_i] - r_p[pairs_j])
        if np.all(np.isnan(abs_diffs)):
            continue

        fam_diffs = None
        fam_means = None
        if fam_ratings is not None:
            f_p = fam_ratings[p_idx][target_mask]
            fam_diffs = np.abs(f_p[pairs_i] - f_p[pairs_j])
            fam_means = (f_p[pairs_i] + f_p[pairs_j]) / 2.0

        pid = f"{dataset_label}_{p_id}" if dataset_label else p_id
        for k in range(len(pairs_i)):
            if np.isnan(abs_diffs[k]):
                continue
            t1 = f"{dataset_label}_{target_indices[pairs_i[k]]}" if dataset_label else str(target_indices[pairs_i[k]])
            t2 = f"{dataset_label}_{target_indices[pairs_j[k]]}" if dataset_label else str(target_indices[pairs_j[k]])
            row = {
                'participant':  pid,
                'target1':      t1,
                'target2':      t2,
                'abs_diff':     abs_diffs[k],
                'sem_dissim_z': sem_z[k],
                'vis_dissim_z': vis_z[k],
            }
            if fam_diffs is not None:
                row['fam_dissim'] = float(fam_diffs[k])
                row['fam_mean']   = float(fam_means[k])
            if dataset_label is not None:
                row['dataset'] = dataset_label
            rows.append(row)

    df = pd.DataFrame(rows)
    if 'fam_dissim' in df.columns:
        df['fam_dissim_z'] = (df['fam_dissim'] - df['fam_dissim'].mean()) / df['fam_dissim'].std()
        df['fam_mean_z']   = (df['fam_mean']   - df['fam_mean'].mean())   / df['fam_mean'].std()
        df.drop(columns=['fam_dissim', 'fam_mean'], inplace=True)
    return df


# ── LMM fitting ───────────────────────────────────────────────────────────────

def build_trait_space_pair_data(ratings_dict: dict, traits: list,
                                target_mask: np.ndarray,
                                sem_rdm: np.ndarray, vis_rdm: np.ndarray,
                                dataset_label: str = None,
                                fam_ratings: np.ndarray = None) -> pd.DataFrame:
    """Build long-format DataFrame for trait-space LMM.

    Each row = one participant × celebrity-pair. The outcome column
    ``euclidean_dist`` is the pairwise Euclidean distance in trait space
    (using that participant's own ratings across all traits).

    Parameters
    ----------
    ratings_dict  : dict[trait -> (n_participants, n_targets)]
    traits        : ordered list of traits defining the space
    target_mask   : boolean (n_targets,)
    sem_rdm       : (n_targets, n_targets) full semantic RDM
    vis_rdm       : (n_targets, n_targets) full visual RDM
    dataset_label : optional string prefix for participant/target IDs
    fam_ratings   : optional (n_participants, n_targets) per-perceiver familiarity

    Columns returned: participant, target1, target2, euclidean_dist,
                      sem_dissim_z, vis_dissim_z [, fam_dissim_z, fam_mean_z]
    """
    target_indices = np.where(target_mask)[0]
    n = len(target_indices)

    sem_sub = sem_rdm[np.ix_(target_indices, target_indices)]
    vis_sub = vis_rdm[np.ix_(target_indices, target_indices)]

    pairs_i, pairs_j = np.triu_indices(n, k=1)
    sem_vals = np.array([sem_sub[i, j] for i, j in zip(pairs_i, pairs_j)])
    vis_vals = np.array([vis_sub[i, j] for i, j in zip(pairs_i, pairs_j)])
    sem_z = (sem_vals - np.nanmean(sem_vals)) / np.nanstd(sem_vals)
    vis_z = (vis_vals - np.nanmean(vis_vals)) / np.nanstd(vis_vals)

    n_participants = ratings_dict[traits[0]].shape[0]

    rows = []
    for p_idx in range(n_participants):
        # (n_sub_targets, n_traits) — each row is one celebrity's trait profile
        p_coords = np.column_stack(
            [ratings_dict[t][p_idx][target_mask] for t in traits]
        )
        diffs = p_coords[pairs_i] - p_coords[pairs_j]  # (n_pairs, n_traits)
        dists = np.sqrt(np.nansum(diffs ** 2, axis=1))
        all_nan = np.all(np.isnan(diffs), axis=1)
        dists[all_nan] = np.nan

        if np.all(np.isnan(dists)):
            continue

        fam_diffs = None
        fam_means = None
        if fam_ratings is not None:
            f_p = fam_ratings[p_idx][target_mask]
            fam_diffs = np.abs(f_p[pairs_i] - f_p[pairs_j])
            fam_means = (f_p[pairs_i] + f_p[pairs_j]) / 2.0

        pid = f"{dataset_label}_{p_idx}" if dataset_label else str(p_idx)
        for k in range(len(pairs_i)):
            if np.isnan(dists[k]):
                continue
            t1 = (f"{dataset_label}_{target_indices[pairs_i[k]]}"
                  if dataset_label else str(target_indices[pairs_i[k]]))
            t2 = (f"{dataset_label}_{target_indices[pairs_j[k]]}"
                  if dataset_label else str(target_indices[pairs_j[k]]))
            row = {
                'participant':    pid,
                'target1':        t1,
                'target2':        t2,
                'euclidean_dist': dists[k],
                'sem_dissim_z':   sem_z[k],
                'vis_dissim_z':   vis_z[k],
            }
            if fam_diffs is not None:
                row['fam_dissim'] = float(fam_diffs[k])
                row['fam_mean']   = float(fam_means[k])
            if dataset_label is not None:
                row['dataset'] = dataset_label
            rows.append(row)

    df = pd.DataFrame(rows)
    if 'fam_dissim' in df.columns:
        df['fam_dissim_z'] = (df['fam_dissim'] - df['fam_dissim'].mean()) / df['fam_dissim'].std()
        df['fam_mean_z']   = (df['fam_mean']   - df['fam_mean'].mean())   / df['fam_mean'].std()
        df.drop(columns=['fam_dissim', 'fam_mean'], inplace=True)
    return df


def fit_lmm(df_pairs: pd.DataFrame, extra_fixed: list = None,
            fam_interaction: bool = False,
            outcome: str = 'abs_diff') -> dict:
    """Fit full and null LMMs; return coefficients and model-fit stats.

    Parameters
    ----------
    df_pairs        : output of build_pair_data or build_trait_space_pair_data
    extra_fixed     : additional fixed-effect terms, e.g. ['dataset_num']
    fam_interaction : if True, fit sem*fam + vis*fam interaction model
                      (requires fam_dissim_z column in df_pairs)
    outcome         : name of the outcome column (default 'abs_diff'; use
                      'euclidean_dist' for trait-space LMM)

    Returns dict with sem/vis betas, CIs, p-values, AIC, LR test.
    """
    from pymer4.models import Lmer
    warnings.filterwarnings('ignore')

    formula_full, formula_null, lr_df = build_lmm_formulas(
        extra_fixed=extra_fixed,
        fam_interaction=fam_interaction,
        outcome=outcome,
    )

    # Select complete cases once, before either fit.  If R is allowed to drop
    # rows formula-by-formula, a familiarity full model can use fewer rows than
    # its null model (whose formula has no familiarity terms), invalidating the
    # likelihood-ratio comparison and overstating the recorded n_obs.
    numeric_required = [outcome, 'sem_dissim_z', 'vis_dissim_z']
    if fam_interaction:
        numeric_required.extend(['fam_dissim_z', 'fam_mean_z'])
    elif extra_fixed:
        numeric_required.extend(extra_fixed)
    required = ['participant', 'target1', 'target2', *numeric_required]
    missing = sorted(set(required) - set(df_pairs.columns))
    if missing:
        raise ValueError(f'LMM input is missing required columns: {missing}')

    df_model = df_pairs.dropna(subset=required).copy()
    finite = np.isfinite(
        df_model[numeric_required].to_numpy(dtype=float)
    ).all(axis=1)
    df_model = df_model.loc[finite].copy()
    if df_model.empty:
        raise ValueError('LMM input has no complete finite observations')

    model      = Lmer(formula_full, data=df_model)
    model_null = Lmer(formula_null, data=df_model)
    model.fit(summarize=False)
    model_null.fit(summarize=False)

    fe  = model.coefs
    out = {}
    for predictor in ['sem_dissim_z', 'vis_dissim_z']:
        row = fe.loc[predictor]
        out[f'{predictor}_beta']     = row['Estimate']
        out[f'{predictor}_se']       = row['SE']
        out[f'{predictor}_t']        = row['T-stat']
        out[f'{predictor}_p']        = row['P-val']
        out[f'{predictor}_ci_lower'] = row['2.5_ci']
        out[f'{predictor}_ci_upper'] = row['97.5_ci']

    if fam_interaction:
        for term in ['fam_dissim_z', 'fam_mean_z',
                     'sem_dissim_z:fam_mean_z', 'vis_dissim_z:fam_mean_z']:
            key = term.replace(':', '_X_')
            if term in fe.index:
                row = fe.loc[term]
                out[f'{key}_beta']     = row['Estimate']
                out[f'{key}_se']       = row['SE']
                out[f'{key}_t']        = row['T-stat']
                out[f'{key}_p']        = row['P-val']
                out[f'{key}_ci_lower'] = row['2.5_ci']
                out[f'{key}_ci_upper'] = row['97.5_ci']
    elif extra_fixed:
        for term in extra_fixed:
            if term in fe.index:
                row = fe.loc[term]
                out[f'{term}_beta']     = row['Estimate']
                out[f'{term}_p']        = row['P-val']
                out[f'{term}_ci_lower'] = row['2.5_ci']
                out[f'{term}_ci_upper'] = row['97.5_ci']

    out['AIC']       = model.AIC
    out['BIC']       = model.BIC
    out['AIC_null']  = model_null.AIC
    out['BIC_null']  = model_null.BIC
    lr_stat = -2 * (model_null.logLike - model.logLike)
    out['lr_stat']   = lr_stat
    out['lr_df']     = lr_df
    out['lr_p']      = chi2.sf(lr_stat, lr_df)
    out['n_obs']     = len(df_model)
    out['converged'] = not model.warnings
    return out


def build_lmm_formulas(extra_fixed: list = None,
                       fam_interaction: bool = False,
                       outcome: str = 'abs_diff') -> tuple:
    """Return full/null mixed-model formulas and LR-test df.

    Kept separate from ``fit_lmm`` so formula contracts can be tested without
    importing pymer4/R.
    """
    rand = ' + (1|participant) + (1|target1) + (1|target2)'

    if fam_interaction:
        formula_full = (
            f'{outcome} ~ sem_dissim_z + vis_dissim_z'
            ' + fam_dissim_z + fam_mean_z'
            ' + sem_dissim_z:fam_mean_z + vis_dissim_z:fam_mean_z'
            + rand
        )
        formula_null = f'{outcome} ~ 1' + rand
        return formula_full, formula_null, 6

    extras = ' + '.join(extra_fixed) if extra_fixed else ''
    suffix = f' + {extras}' if extras else ''
    formula_full = f'{outcome} ~ sem_dissim_z + vis_dissim_z{suffix}' + rand
    formula_null = f'{outcome} ~ 1' + rand
    return formula_full, formula_null, 2
