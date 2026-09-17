"""
Figures for the trial-level recognition split (LMM + RSA).

Produces
--------
4_output/publication/supplementary/figR1_recognition_lmm_deltas   — Δβ (recognised − unrecognised)
4_output/publication/supplementary/figR2_recognition_lmm_betas    — β per condition, both routes
4_output/publication/supplementary/figR3_recognition_rsa_unique   — RSA unique variance per condition

Usage:
    python 2_code/supplementary/plot_recognition_split.py
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

import numpy as np
import pandas as pd
import matplotlib
matplotlib.rcParams.update({
    'font.family': 'sans-serif',
    'font.sans-serif': ['Arial', 'Helvetica', 'DejaVu Sans', 'sans-serif'],
    'font.size': 7,
    'axes.labelsize': 7,
    'axes.titlesize': 8,
    'axes.linewidth': 0.6,
    'xtick.labelsize': 6,
    'ytick.labelsize': 6,
    'legend.fontsize': 6.5,
    'legend.frameon': False,
    'figure.dpi': 300,
    'savefig.dpi': 600,
    'savefig.bbox': 'tight',
    'pdf.fonttype': 42,
    'svg.fonttype': 'none',
})
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

from config import DATASETS, PIPELINE_DIR, ROOT
from config import SEM_COLOR, VIS_COLOR, SEM_LIGHT, VIS_LIGHT
DISPLAY_LABELS = {
    'warm': 'Warm', 'critical': 'Critical', 'competent': 'Competent',
    'practical': 'Practical', 'feminine': 'Feminine', 'strong': 'Strong',
    'youthful': 'Youthful', 'young': 'Youthful', 'charismatic': 'Charismatic',
    'trustworthy': 'Trustworthy', 'attractive': 'Attractive',
}

OUT = ROOT / '4_output' / 'publication' / 'supplementary'
FULL_WIDTH = 7.086614  # 180 mm
INK = '#272727'


def _fmt_traits(traits):
    return [DISPLAY_LABELS.get(t, t.capitalize()) for t in traits]


def _save_fig(fig, name):
    OUT.mkdir(parents=True, exist_ok=True)
    stem = OUT / name
    fig.savefig(stem.with_suffix('.pdf'), bbox_inches='tight', facecolor='white')
    fig.savefig(stem.with_suffix('.svg'), bbox_inches='tight', facecolor='white')
    fig.savefig(stem.with_suffix('.png'), dpi=600, bbox_inches='tight', facecolor='white')
    fig.savefig(stem.with_suffix('.tiff'), dpi=600, bbox_inches='tight', facecolor='white',
                pil_kwargs={'compression': 'tiff_lzw'})
    print(f'Saved {stem}.[pdf|svg|png|tiff]')


def _pub_style(ax):
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.spines['left'].set_color(INK)
    ax.spines['bottom'].set_color(INK)
    ax.tick_params(colors=INK)


def _load(ds):
    lmm = pd.read_csv(PIPELINE_DIR / ds / 'lmm_recognition_split.csv')
    rsa = pd.read_csv(PIPELINE_DIR / ds / 'rsa_recognition_trial_split.csv')
    return lmm, rsa


def _marker(p_unc):
    """Ringed = uncorrected p < .05; hollow = n.s."""
    if p_unc < .05:
        return dict(markeredgecolor='0.35', markeredgewidth=0.6, alpha=1.0)
    return dict(markeredgecolor='none', markeredgewidth=0, alpha=0.45)


# ── Figure R1: Δβ ─────────────────────────────────────────────────────────────

def fig_deltas():
    fig, axes = plt.subplots(1, 2, figsize=(FULL_WIDTH, 3.1))
    for ax, ds in zip(axes, ['us', 'cn']):
        lmm, _ = _load(ds)
        traits = lmm.trait.tolist()
        y = np.arange(len(traits))

        for off, (term, col, lab) in enumerate(
                [('sem_dissim_z', SEM_COLOR, 'Semantic'),
                 ('vis_dissim_z', VIS_COLOR, 'Visual')]):
            d = lmm[f'{term}_delta'].values
            se = np.sqrt(lmm[f'{term}_se_rec'] ** 2 + lmm[f'{term}_se_unrec'] ** 2).values
            pu = lmm[f'{term}_delta_p'].values
            yy = y + (0.18 if off == 0 else -0.18)
            ax.errorbar(d, yy, xerr=1.96 * se, fmt='none',
                        ecolor=col, elinewidth=0.85, capsize=2, capthick=0.75, alpha=0.8)
            for k in range(len(traits)):
                ax.plot(d[k], yy[k], 'o', color=col, markersize=4.5,
                        **_marker(pu[k]))

        ax.axvline(0, color='0.3', lw=0.75, ls='--')
        ax.set_yticks(y)
        ax.set_yticklabels(_fmt_traits(traits))
        ax.invert_yaxis()
        ax.set_xlabel('Δβ  (recognised − unrecognised)')
        ax.set_title(f"{DATASETS[ds]['label_adj']} sample", loc='left')
        _pub_style(ax)

    handles = [Line2D([0], [0], marker='o', color='w', markerfacecolor=SEM_COLOR,
                      markersize=5, label='Semantic'),
               Line2D([0], [0], marker='o', color='w', markerfacecolor=VIS_COLOR,
                      markersize=5, label='Visual'),
               Line2D([0], [0], marker='o', color='w', markerfacecolor='0.5',
                      markeredgecolor='0.35', markeredgewidth=0.6, markersize=5,
                      label='p < .05 uncorrected')]
    fig.legend(handles=handles, loc='lower center', ncol=3, frameon=False,
               bbox_to_anchor=(0.5, -0.06))
    fig.tight_layout()
    _save_fig(fig, 'figR1_recognition_lmm_deltas')
    plt.close(fig)


# ── Figure R2: β per condition ────────────────────────────────────────────────

def fig_betas():
    fig, axes = plt.subplots(1, 2, figsize=(FULL_WIDTH, 3.1))
    for ax, ds in zip(axes, ['us', 'cn']):
        lmm, _ = _load(ds)
        traits = lmm.trait.tolist()
        y = np.arange(len(traits))

        series = [('sem_dissim_z', 'rec', SEM_COLOR, 'Semantic — recognised', 0.27),
                  ('sem_dissim_z', 'unrec', SEM_LIGHT, 'Semantic — unrecognised', 0.09),
                  ('vis_dissim_z', 'rec', VIS_COLOR, 'Visual — recognised', -0.09),
                  ('vis_dissim_z', 'unrec', VIS_LIGHT, 'Visual — unrecognised', -0.27)]
        for term, cond, col, lab, off in series:
            b = lmm[f'{term}_beta_{cond}'].values
            lo = lmm[f'{term}_ci_lower_{cond}'].values
            hi = lmm[f'{term}_ci_upper_{cond}'].values
            yy = y + off
            ax.errorbar(b, yy, xerr=[b - lo, hi - b], fmt='o', color=col,
                        markersize=3.5, elinewidth=0.85, capsize=2, capthick=0.75, label=lab)

        ax.axvline(0, color='0.3', lw=0.75, ls='--')
        ax.set_yticks(y)
        ax.set_yticklabels(_fmt_traits(traits))
        ax.invert_yaxis()
        ax.set_xscale('symlog', linthresh=0.05)
        ax.set_xlabel('β  (semi-standardised)')
        ax.set_title(f"{DATASETS[ds]['label_adj']} sample", loc='left')
        _pub_style(ax)

    h, l = axes[0].get_legend_handles_labels()
    fig.legend(h, l, loc='lower center', ncol=4, frameon=False,
               bbox_to_anchor=(0.5, -0.06))
    fig.tight_layout()
    _save_fig(fig, 'figR2_recognition_lmm_betas')
    plt.close(fig)


# ── Figure R3: RSA unique variance ────────────────────────────────────────────

def fig_rsa():
    fig, axes = plt.subplots(1, 2, figsize=(FULL_WIDTH, 3.1))
    for ax, ds in zip(axes, ['us', 'cn']):
        _, rsa = _load(ds)
        traits = rsa.trait.tolist()
        x = np.arange(len(traits))
        w = 0.2
        for off, (col_name, col, lab) in enumerate([
                ('unique_sem_rec', SEM_COLOR, 'Semantic — recognised'),
                ('unique_sem_unrec', SEM_LIGHT, 'Semantic — unrecognised'),
                ('unique_vis_rec', VIS_COLOR, 'Visual — recognised'),
                ('unique_vis_unrec', VIS_LIGHT, 'Visual — unrecognised')]):
            ax.bar(x + (off - 1.5) * w, rsa[col_name].values, w,
                   color=col, label=lab)
        ax.set_xticks(x)
        ax.set_xticklabels(_fmt_traits(traits), rotation=45, ha='right')
        ax.set_ylabel('Unique variance explained (ΔR²)')
        ax.set_title(f"{DATASETS[ds]['label_adj']} sample", loc='left')
        _pub_style(ax)

    h, l = axes[0].get_legend_handles_labels()
    fig.legend(h, l, loc='lower center', ncol=4, frameon=False,
               bbox_to_anchor=(0.5, -0.14))
    fig.tight_layout()
    _save_fig(fig, 'figR3_recognition_rsa_unique')
    plt.close(fig)


if __name__ == '__main__':
    fig_deltas()
    fig_betas()
    fig_rsa()
