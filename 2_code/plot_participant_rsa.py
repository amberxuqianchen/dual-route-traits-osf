"""
Participant-level variance partitioning plots.

For each participant, builds their individual trait-space RDM (pairwise Euclidean
distances between celebrities in trait space using only that participant's ratings),
then computes OLS variance partitioning against the group semantic and visual RDMs.

Plot 1 — Bar chart: participant-level unique_sem / shared / unique_vis,
         every participant sorted by total R², for US and CN.

Plot 2 — Scatter: participant mean familiarity (x) vs unique_sem or unique_vis (y)
         with OLS regression line, for US and CN.

Usage:
    python 2_code/plot_participant_rsa.py
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent))

import numpy as np
import pandas as pd
from scipy.spatial.distance import squareform
from scipy.stats import linregress
import matplotlib
matplotlib.rcParams.update({
    'font.family': 'Arial',
    'font.size':   12,
    'figure.dpi':  300,
    'savefig.dpi': 300,
    'savefig.bbox': 'tight',
})
import matplotlib.pyplot as plt
from matplotlib.patches import Patch
from matplotlib.lines import Line2D

from config import DATASETS, PIPELINE_DIR, OUTPUT_DIR
from config import SEM_COLOR, VIS_COLOR
from data_loader import load_rdms, load_familiarity_ratings

# Use 8 common traits (same as COMMON_TRAITS in config)
TRAITS_US = ['warm', 'critical', 'competent', 'practical',
             'feminine', 'strong', 'youthful', 'charismatic']
TRAITS_CN = ['warm', 'critical', 'competent', 'practical',
             'feminine', 'strong', 'young', 'charismatic']


# ── Variance partitioning (no I/O, fast numpy OLS) ───────────────────────────

def _vp(trait_vec, sem_vec, vis_vec):
    """OLS R² decomposition. Returns (unique_sem, unique_vis, shared, total_r2).
    Returns all-NaN tuple if too few valid observations."""
    m = ~np.isnan(trait_vec) & ~np.isnan(sem_vec) & ~np.isnan(vis_vec)
    if m.sum() < 10:
        return np.nan, np.nan, np.nan, np.nan
    y, s, v = trait_vec[m], sem_vec[m], vis_vec[m]

    def _r2(*xs):
        X = np.column_stack([np.ones(len(y)), *xs])
        b = np.linalg.lstsq(X, y, rcond=None)[0]
        ss_res = np.sum((y - X @ b) ** 2)
        ss_tot = np.sum((y - y.mean()) ** 2)
        return 1.0 - ss_res / ss_tot if ss_tot > 0 else 0.0

    r2_full = _r2(s, v)
    unique_sem = r2_full - _r2(v)
    unique_vis = r2_full - _r2(s)
    return unique_sem, unique_vis, r2_full - unique_sem - unique_vis, r2_full


# ── Participant RDM builder ───────────────────────────────────────────────────

def _participant_rdm(ratings_p):
    """Pairwise Euclidean distance between celebrities in trait space.

    ratings_p : (n_traits, n_celebrities) — one participant's ratings.
    Returns (n_celebrities, n_celebrities) distance matrix (NaN if both
    cells lack a common valid trait).
    """
    n_t, n_c = ratings_p.shape
    rdm = np.zeros((n_c, n_c))
    for i in range(n_c):
        for j in range(i + 1, n_c):
            mask = ~np.isnan(ratings_p[:, i]) & ~np.isnan(ratings_p[:, j])
            d = np.sqrt(np.sum((ratings_p[mask, i] - ratings_p[mask, j]) ** 2)) \
                if mask.any() else np.nan
            rdm[i, j] = rdm[j, i] = d
    return rdm


# ── Per-dataset computation ───────────────────────────────────────────────────

def compute_us(cfg):
    arr = np.load(cfg['ratings_file'])          # (415, 9, 50)
    trait_idx = [cfg['traits'].index(t) for t in TRAITS_US]
    sem_rdm, vis_rdm = load_rdms(cfg)
    sem_vec = squareform(sem_rdm, checks=False)
    vis_vec = squareform(vis_rdm, checks=False)

    rows = []
    n = arr.shape[0]
    for p in range(n):
        if p % 50 == 0:
            print(f"  US participant {p}/{n}", flush=True)
        ratings_p = arr[p][trait_idx]           # (8, 50)
        rdm_p = _participant_rdm(ratings_p)
        us, uv, sh, tot = _vp(squareform(rdm_p, checks=False), sem_vec, vis_vec)
        rows.append({'participant': p, 'unique_sem': us, 'unique_vis': uv,
                     'shared': sh, 'total_r2': tot})
    return pd.DataFrame(rows)


def compute_cn(cfg):
    df_raw = pd.read_csv(cfg['ratings_file'])
    sem_rdm, vis_rdm = load_rdms(cfg)
    sem_vec = squareform(sem_rdm, checks=False)
    vis_vec = squareform(vis_rdm, checks=False)

    all_ppts = sorted(df_raw[cfg['csv_participant_col']].unique())
    n_tgt    = cfg['n_targets']

    # Build (n_participants, n_targets) matrix for each common trait
    trait_mats = {}
    for trait in TRAITS_CN:
        sub  = df_raw[df_raw[cfg['csv_trait_col']] == trait]
        wide = (sub.pivot_table(index=cfg['csv_participant_col'],
                                columns=cfg['csv_target_col'],
                                values=cfg['csv_rating_col'], aggfunc='mean')
                   .reindex(index=all_ppts, columns=range(1, n_tgt + 1)))
        trait_mats[trait] = wide.values   # (n_ppts, n_targets)

    rows = []
    n = len(all_ppts)
    for p_idx, p in enumerate(all_ppts):
        if p_idx % 50 == 0:
            print(f"  CN participant {p_idx}/{n}", flush=True)
        ratings_p = np.row_stack([trait_mats[t][p_idx] for t in TRAITS_CN])  # (8, 50)
        rdm_p = _participant_rdm(ratings_p)
        us, uv, sh, tot = _vp(squareform(rdm_p, checks=False), sem_vec, vis_vec)
        rows.append({'participant': p, 'unique_sem': us, 'unique_vis': uv,
                     'shared': sh, 'total_r2': tot})
    return pd.DataFrame(rows)


def get_mean_familiarity(cfg):
    """Per-participant mean familiarity across all celebrities."""
    fam = load_familiarity_ratings(cfg)          # (n_participants, n_targets)
    return np.nanmean(fam, axis=1)               # (n_participants,)


# ── Plot 1: bar chart ────────────────────────────────────────────────────────

def _pub_style(ax):
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)


def plot_participant_bars(vp_us, vp_cn):
    fig, axes = plt.subplots(2, 1, figsize=(18, 10))

    components = [
        ('unique_sem', 'Unique Semantic', SEM_COLOR),
        ('shared',     'Shared',          '#888888'),
        ('unique_vis', 'Unique Visual',   VIS_COLOR),
    ]

    # Shared y-axis: use the global max total_r2 across both datasets
    y_max = max(vp_us['total_r2'].max(), vp_cn['total_r2'].max()) * 1.05

    for ax, vp_df, ds_label in [(axes[0], vp_us, 'US'), (axes[1], vp_cn, 'CN')]:
        vp = vp_df.sort_values('total_r2').reset_index(drop=True)
        x  = np.arange(len(vp))
        bottoms = np.zeros(len(vp))

        for col, lbl, color in components:
            vals = vp[col].fillna(0).values
            ax.bar(x, vals, width=1.0, bottom=bottoms, color=color,
                   linewidth=0, label=lbl)
            bottoms += vals

        ax.set_ylabel('Variance Explained (R²)', fontsize=13)
        ax.set_xlabel('Participant (sorted by total R²)', fontsize=13)
        ax.set_title(
            f'{ds_label} — Participant-Level Variance Partitioning (n={len(vp)})',
            fontsize=15)
        ax.set_xlim(-0.5, len(vp) - 0.5)
        ax.set_ylim(0, y_max)
        ax.tick_params(bottom=False, labelbottom=False)
        _pub_style(ax)

    handles = [Patch(color=c, label=l) for _, l, c in components]
    fig.legend(handles=handles, loc='upper center', ncol=3, fontsize=11,
               bbox_to_anchor=(0.5, 1.02))
    plt.tight_layout()

    outdir = OUTPUT_DIR / 'comparison'
    outdir.mkdir(parents=True, exist_ok=True)
    path = outdir / 'participant_variance_partitioning.png'
    fig.savefig(path)
    plt.close(fig)
    print(f"Saved {path}")


# ── Plot 2: familiarity scatter ───────────────────────────────────────────────

def plot_familiarity_scatter(vp_us, fam_us, vp_cn, fam_cn):
    fig, axes = plt.subplots(2, 2, figsize=(14, 10), sharey='col')

    datasets = [
        ('US', vp_us, fam_us, axes[0]),
        ('CN', vp_cn, fam_cn, axes[1]),
    ]

    for ds_label, vp_df, fam_arr, ax_row in datasets:
        for ax, col, color, modality in [
            (ax_row[0], 'unique_sem', SEM_COLOR, 'Semantic'),
            (ax_row[1], 'unique_vis', VIS_COLOR, 'Visual'),
        ]:
            y_all = vp_df[col].values
            valid = ~np.isnan(fam_arr) & ~np.isnan(y_all)
            fam = fam_arr[valid]
            y   = y_all[valid]

            ax.scatter(fam, y, color=color, alpha=0.35, s=18, linewidths=0)

            slope, intercept, r, p, _ = linregress(fam, y)
            x_line = np.linspace(fam.min(), fam.max(), 200)
            stars = '***' if p < 0.001 else '**' if p < 0.01 else '*' if p < 0.05 else 'n.s.'
            ax.plot(x_line, slope * x_line + intercept, color='black', lw=2.0,
                    label=f'r = {r:.2f}{stars}')

            ax.set_xlabel('Mean Familiarity Rating', fontsize=12)
            ax.set_ylabel(f'Unique {modality} R²', fontsize=12)
            ax.set_title(f'{ds_label} — Familiarity vs Unique {modality}', fontsize=13)
            ax.legend(fontsize=11)
            _pub_style(ax)

    fig.text(0.5, 0.01,
             'Note: not for familiarity group but probably for personality trait paper',
             ha='center', va='bottom', fontsize=11, style='italic', color='gray')

    plt.tight_layout(rect=[0, 0.03, 1, 1])

    outdir = OUTPUT_DIR / 'comparison'
    outdir.mkdir(parents=True, exist_ok=True)
    path = outdir / 'familiarity_vs_unique_variance.png'
    fig.savefig(path)
    plt.close(fig)
    print(f"Saved {path}")


# ── Main ─────────────────────────────────────────────────────────────────────

def main():
    print("Computing US participant-level VP...")
    vp_us = compute_us(DATASETS['us'])
    out_us = PIPELINE_DIR / 'us' / 'participant_rsa_results.csv'
    vp_us.to_csv(out_us, index=False)
    print(f"  US done: mean total_r2 = {vp_us['total_r2'].mean():.4f}  →  {out_us}")

    print("Computing CN participant-level VP...")
    vp_cn = compute_cn(DATASETS['cn'])
    out_cn = PIPELINE_DIR / 'cn' / 'participant_rsa_results.csv'
    vp_cn.to_csv(out_cn, index=False)
    print(f"  CN done: mean total_r2 = {vp_cn['total_r2'].mean():.4f}  →  {out_cn}")

    print("Loading familiarity...")
    fam_us = get_mean_familiarity(DATASETS['us'])
    fam_cn = get_mean_familiarity(DATASETS['cn'])

    print("Plotting bar chart...")
    plot_participant_bars(vp_us, vp_cn)

    print("Plotting familiarity scatter...")
    plot_familiarity_scatter(vp_us, fam_us, vp_cn, fam_cn)

    print("Done.")


if __name__ == '__main__':
    main()
