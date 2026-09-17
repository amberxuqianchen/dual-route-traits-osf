"""
Figures 11 and 12: familiarity moderation from the maximal-random-effects LMMs.

Produces
--------
4_output/publication/fig11_trait_space_lmm_moderation.[pdf|svg|png|tiff]
4_output/publication/fig12_lmm_moderation_betas.[pdf|svg|png|tiff]

Usage:
    python 2_code/supplementary/plot_lmm_maximal_moderation.py
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

import numpy as np
import pandas as pd
import matplotlib as mpl
mpl.rcParams.update({
    'font.family': 'sans-serif',
    'font.sans-serif': ['Arial', 'Helvetica', 'DejaVu Sans', 'sans-serif'],
    'font.size': 7,
    'axes.labelsize': 7.5,
    'axes.titlesize': 8,
    'axes.linewidth': 0.75,
    'xtick.labelsize': 6.5,
    'ytick.labelsize': 6.5,
    'legend.fontsize': 6.5,
    'legend.frameon': False,
    'pdf.fonttype': 42,
    'svg.fonttype': 'none',
})
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

from config import DATASETS, PIPELINE_DIR, ROOT, SEM_COLOR, VIS_COLOR
DISPLAY_LABELS = {
    'warm': 'Warm', 'critical': 'Critical', 'competent': 'Competent',
    'practical': 'Practical', 'feminine': 'Feminine', 'strong': 'Strong',
    'youthful': 'Youthful', 'young': 'Youthful', 'charismatic': 'Charismatic',
    'trustworthy': 'Trustworthy', 'attractive': 'Attractive',
}

def _fmt_traits(traits):
    return [DISPLAY_LABELS.get(t, t.capitalize()) for t in traits]

PUB = ROOT / '4_output' / 'publication'
FULL_WIDTH = 7.086614  # 180 mm
INK = '#272727'
MUTED = '#6F6F6F'


def _style(ax):
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.tick_params(colors=INK)


def _save_pub(fig, name):
    PUB.mkdir(parents=True, exist_ok=True)
    stem = PUB / name
    fig.savefig(stem.with_suffix('.pdf'), bbox_inches='tight', facecolor='white')
    fig.savefig(stem.with_suffix('.svg'), bbox_inches='tight', facecolor='white')
    fig.savefig(stem.with_suffix('.png'), dpi=600, bbox_inches='tight', facecolor='white')
    fig.savefig(stem.with_suffix('.tiff'), dpi=600, bbox_inches='tight', facecolor='white',
                pil_kwargs={'compression': 'tiff_lzw'})
    plt.close(fig)
    print(f'Saved {stem}.[pdf|svg|png|tiff]')

sel = pd.read_csv(PIPELINE_DIR / 'lmm_familiarity_maximal' / 'selected_models_holm.csv')
sel = sel[sel['mod'] == 'fam']


def _row(ds, trait):
    return sel[(sel['ds'] == ds) & (sel['trait'] == trait)].iloc[0]


def _marker(p_unc, p_holm, color):
    """Diamond = Holm; solid circle = uncorrected only; small faint = n.s."""
    if not np.isnan(p_holm) and p_holm < .05:
        return dict(marker='D', markersize=6.2, markerfacecolor=color,
                    markeredgecolor=INK, markeredgewidth=0.8, alpha=1.0)
    if p_unc < .05:
        return dict(marker='o', markersize=6.5, markerfacecolor=color,
                    markeredgecolor=INK, markeredgewidth=0.8, alpha=1.0)
    return dict(marker='o', markersize=4.4, markerfacecolor=color,
                markeredgecolor='none', markeredgewidth=0, alpha=0.35)



# ── Figure 11: trait space ────────────────────────────────────────────────────

def _panel(ax, letter, title):
    if letter:
        ax.text(-0.10, 1.07, letter, transform=ax.transAxes, fontsize=8,
                fontweight='bold', ha='left', va='bottom')
    ax.text(0, 1.07, title, transform=ax.transAxes, fontsize=8,
            fontweight='bold', ha='left', va='bottom')


def fig_trait_space():
    fig, ax = plt.subplots(figsize=(FULL_WIDTH, 2.75), constrained_layout=True)
    y = np.arange(2)
    for off, (ch, col, lab) in enumerate(
            [('sem', SEM_COLOR, 'Semantic × familiarity'),
             ('vis', VIS_COLOR, 'Visual × familiarity')]):
        b, se, pu = [], [], []
        for ds in ['us', 'cn']:
            r = _row(ds, 'traitspace')
            b.append(r[f'{ch}_std'])
            se.append(r[f'{ch}_se_std'])
            pu.append(r[f'{ch}_p'])
        b, se = np.asarray(b), np.asarray(se)
        yy = y + (0.11 if off == 0 else -0.11)
        ax.errorbar(b, yy, xerr=1.96 * se, fmt='none', ecolor=col,
                    elinewidth=0.9, capsize=2, capthick=0.8, alpha=0.8)
        for k in range(2):
            ax.plot(b[k], yy[k], linestyle='none',
                    **_marker(pu[k], np.nan, col))

    ax.axvline(0, color=MUTED, lw=0.75, ls='--')
    ax.set_yticks(y)
    ax.set_yticklabels([DATASETS['us']['label'], DATASETS['cn']['label']])
    ax.set_ylim(1.45, -0.45)
    ax.set_xlabel('Standardized familiarity-moderation coefficient, β (95% CI)')
    _panel(ax, '', 'Trait-space familiarity moderation')
    _style(ax)

    handles = [
        Line2D([0], [0], marker='o', linestyle='none', markerfacecolor=SEM_COLOR,
               markeredgecolor='none', markersize=5.2, label='Semantic route'),
        Line2D([0], [0], marker='o', linestyle='none', markerfacecolor=VIS_COLOR,
               markeredgecolor='none', markersize=5.2, label='Visual route'),
        Line2D([0], [0], marker='o', linestyle='none', markerfacecolor=MUTED,
               markeredgecolor=INK, markeredgewidth=0.8, markersize=5.5,
               label='p < .05 (single outcome)'),
    ]
    fig.legend(handles=handles, loc='lower center', bbox_to_anchor=(0.5, 1.0),
               ncol=3, columnspacing=1.6, handletextpad=0.5)
    name = 'fig11_trait_space_lmm_moderation'
    source = PUB / 'source_data'
    source.mkdir(parents=True, exist_ok=True)
    sel[sel['trait'] == 'traitspace'].to_csv(source / f'{name}_source_data.csv', index=False)
    _save_pub(fig, name)


# ── Figure 12: per trait ──────────────────────────────────────────────────────

def fig_per_trait():
    intervals = []
    for ds in ['us', 'cn']:
        for trait in DATASETS[ds]['traits']:
            r = _row(ds, trait)
            for ch in ['sem', 'vis']:
                beta = r[f'{ch}_std']
                half = 1.96 * r[f'{ch}_se_std']
                intervals.extend([beta - half, beta + half])
    span = max(intervals) - min(intervals)
    xlim = (min(intervals) - 0.06 * span, max(intervals) + 0.06 * span)

    fig, axes = plt.subplots(1, 2, figsize=(FULL_WIDTH, 3.75), sharex=True,
                             constrained_layout=True)
    for panel, (ax, ds) in enumerate(zip(axes, ['us', 'cn'])):
        traits = DATASETS[ds]['traits']
        y = np.arange(len(traits))
        for off, (ch, col) in enumerate([('sem', SEM_COLOR), ('vis', VIS_COLOR)]):
            rows = [_row(ds, t) for t in traits]
            b = np.array([r[f'{ch}_std'] for r in rows])
            se = np.array([r[f'{ch}_se_std'] for r in rows])
            pu = np.array([r[f'{ch}_p'] for r in rows])
            ph = np.array([r[f'{ch}_p_holm'] for r in rows])
            yy = y + (0.13 if off == 0 else -0.13)
            ax.errorbar(b, yy, xerr=1.96 * se, fmt='none', ecolor=col,
                        elinewidth=0.85, capsize=2, capthick=0.75, alpha=0.8)
            for k in range(len(traits)):
                ax.plot(b[k], yy[k], linestyle='none',
                        **_marker(pu[k], ph[k], col))

        ax.axvline(0, color=MUTED, lw=0.75, ls='--')
        ax.set_yticks(y)
        ax.set_yticklabels(_fmt_traits(traits))
        ax.invert_yaxis()
        ax.set_xlim(xlim)
        ax.set_xlabel('Standardized moderation coefficient, β (95% CI)')
        dataset_label = DATASETS[ds]['label_adj']
        _panel(ax, chr(ord('a') + panel), f'{dataset_label} sample')
        _style(ax)

    handles = [
        Line2D([0], [0], marker='o', linestyle='none', markerfacecolor=SEM_COLOR,
               markeredgecolor='none', markersize=5.0, label='Semantic route'),
        Line2D([0], [0], marker='o', linestyle='none', markerfacecolor=VIS_COLOR,
               markeredgecolor='none', markersize=5.0, label='Visual route'),
        Line2D([0], [0], marker='o', linestyle='none', markerfacecolor=MUTED,
               markeredgecolor=INK, markeredgewidth=0.8, markersize=5.5,
               label='p < .05 uncorrected'),
        Line2D([0], [0], marker='D', linestyle='none', markerfacecolor=MUTED,
               markeredgecolor=INK, markeredgewidth=0.8, markersize=5.0,
               label='Survives Holm correction'),
    ]
    fig.legend(handles=handles, loc='lower center', bbox_to_anchor=(0.5, 1.0),
               ncol=4, columnspacing=1.2, handletextpad=0.45)
    name = 'fig12_lmm_moderation_betas'
    source = PUB / 'source_data'
    source.mkdir(parents=True, exist_ok=True)
    sel[sel['trait'] != 'traitspace'].to_csv(source / f'{name}_source_data.csv', index=False)
    _save_pub(fig, name)


if __name__ == '__main__':
    fig_trait_space()
    fig_per_trait()
