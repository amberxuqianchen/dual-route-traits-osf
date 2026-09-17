"""Render journal-ready versions of the main inferential figures (Figs. 5–8).

This module
assembles the US and Chinese results directly from the pipeline CSVs at a fixed
two-column journal width and exports editable vector files, 600 dpi raster
files, and the exact source-data rows used in each figure.

Usage
-----
python 2_code/plot_publication_figures.py
"""

from pathlib import Path
import shutil
import sys

sys.path.insert(0, str(Path(__file__).parent))

import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.patches import Patch
import numpy as np
import pandas as pd

from config import (DATASETS, PIPELINE_DIR, OUTPUT_DIR, ROOT, SEM_COLOR,
                    VIS_COLOR, ds_label, ds_label_adj)


# Two-column publication contract: 180 mm wide, 7–9 pt final text.
FULL_WIDTH = 7.086614  # 180 mm

SEM = SEM_COLOR
VIS = VIS_COLOR
SHARED = "#969696"
INK = "#272727"

DISPLAY_LABELS = {
    "warm": "Warm",
    "critical": "Critical",
    "competent": "Competent",
    "practical": "Practical",
    "feminine": "Feminine",
    "strong": "Strong",
    "youthful": "Youthful",
    "young": "Youthful",
    "charismatic": "Charismatic",
    "trustworthy": "Trustworthy",
    "attractive": "Attractive",
}

mpl.rcParams.update({
    "font.family": "sans-serif",
    "font.sans-serif": ["Arial", "Helvetica", "DejaVu Sans", "sans-serif"],
    "font.size": 7,
    "axes.labelsize": 7.5,
    "axes.titlesize": 8,
    "axes.linewidth": 0.75,
    "xtick.labelsize": 6.5,
    "ytick.labelsize": 6.5,
    "xtick.major.width": 0.7,
    "ytick.major.width": 0.7,
    "xtick.major.size": 2.5,
    "ytick.major.size": 2.5,
    "legend.fontsize": 6.5,
    "legend.frameon": False,
    "pdf.fonttype": 42,
    "ps.fonttype": 42,
    "svg.fonttype": "none",
    "figure.facecolor": "white",
    "axes.facecolor": "white",
})

OUT = OUTPUT_DIR / "publication"
SOURCE_OUT = OUT / "source_data"


def _traits(ds):
    return DATASETS[ds]["traits"]


def _labels(ds):
    return [DISPLAY_LABELS[t] for t in _traits(ds)]


def _ordered(df, ds, split):
    return (df[df["split"] == split]
            .set_index("trait").loc[_traits(ds)].reset_index())


def _stars(p):
    if p < 0.001:
        return "***"
    if p < 0.01:
        return "**"
    if p < 0.05:
        return "*"
    return ""


def _style(ax):
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines["left"].set_color(INK)
    ax.spines["bottom"].set_color(INK)
    ax.tick_params(colors=INK)


def _panel(ax, letter, title):
    if letter:
        ax.text(-0.09, 1.14, letter, transform=ax.transAxes, ha="left", va="bottom",
                fontsize=8, fontweight="bold", color=INK, clip_on=False)
    ax.text(0.0, 1.14, title, transform=ax.transAxes, ha="left", va="bottom",
            fontsize=8, fontweight="bold", color=INK, clip_on=False)


def _bracket(ax, x1, x2, y, label, color=INK, tick=0.009):
    ax.plot([x1, x1, x2, x2], [y - tick, y, y, y - tick], color=color,
            linewidth=0.8, clip_on=False)
    ax.text((x1 + x2) / 2, y + tick * 0.22, label, ha="center", va="bottom",
            fontsize=6.5, fontweight="bold", color=color, clip_on=False)


def _vertical_bracket(ax, x, y1, y2, label):
    tick = 0.055
    ax.plot([x - tick, x, x, x - tick], [y1, y1, y2, y2], color=INK,
            linewidth=0.75, clip_on=False)
    ax.text(x + 0.025, (y1 + y2) / 2, label, rotation=90, ha="left", va="center",
            fontsize=6.2, fontweight="bold", color=INK, clip_on=False)


def _save(fig, name):
    OUT.mkdir(parents=True, exist_ok=True)
    stem = OUT / name
    fig.savefig(stem.with_suffix(".pdf"), bbox_inches="tight", facecolor="white")
    fig.savefig(stem.with_suffix(".svg"), bbox_inches="tight", facecolor="white")
    fig.savefig(stem.with_suffix(".png"), dpi=600, bbox_inches="tight",
                facecolor="white")
    fig.savefig(stem.with_suffix(".tiff"), dpi=600, bbox_inches="tight",
                facecolor="white", pil_kwargs={"compression": "tiff_lzw"})
    plt.close(fig)
    print(f"Saved {stem}.[pdf|svg|png|tiff]")


def _export_source(name, frames):
    SOURCE_OUT.mkdir(parents=True, exist_ok=True)
    pd.concat(frames, ignore_index=True, sort=False).to_csv(
        SOURCE_OUT / f"{name}_source_data.csv", index=False)


def figure05_trait_space_rsa():
    fig, axes = plt.subplots(1, 2, figsize=(FULL_WIDTH, 2.72), sharey=True,
                             constrained_layout=True)
    ymin, ymax = -0.12, 0.93
    for i, (ax, ds) in enumerate(zip(axes, ("us", "cn"))):
        rsa = pd.read_csv(PIPELINE_DIR / ds / "trait_space_rsa_results.csv")
        row = rsa[rsa["split"] == "all"].iloc[0]
        boot = pd.read_csv(PIPELINE_DIR / ds / "rsa_route_difference_identity_level.csv")
        b = boot[boot["trait"] == "traitspace"].iloc[0]
        vals = np.array([row["rsa_sem_r"], row["rsa_vis_r"]], dtype=float)
        lo = np.array([b["sem_boot_ci_lower"], b["vis_boot_ci_lower"]], dtype=float)
        hi = np.array([b["sem_boot_ci_upper"], b["vis_boot_ci_upper"]], dtype=float)
        x = np.array([0, 1])
        ax.bar(x, vals, width=0.62, color=[SEM, VIS], edgecolor="white", linewidth=0.5)
        ax.errorbar(x, vals, yerr=[vals - lo, hi - vals], fmt="none", ecolor=INK,
                    elinewidth=0.8, capsize=2.5, capthick=0.8, zorder=3)
        for xi, p, color in zip(
                x, [row["rsa_sem_p"], row["rsa_vis_p"]], [SEM, VIS]):
            sig = _stars(float(p))
            if sig:
                ax.text(xi, -0.075, sig, ha="center", va="bottom",
                        color=color, fontweight="bold", fontsize=7)
        route_sig = _stars(float(b["p_boot"]))
        if route_sig:
            _bracket(ax, 0, 1, max(hi) + 0.055, route_sig)
        ax.set_xticks(x, ["Semantic", "Visual"])
        ax.set_ylim(ymin, ymax)
        _panel(ax, chr(ord("a") + i), f"{ds_label_adj(ds)} sample")
        _style(ax)
    axes[0].set_ylabel("Spearman ρ")
    handles = [
        Patch(facecolor=SEM, label="Semantic"),
        Patch(facecolor=VIS, label="Visual"),
        Line2D([], [], color=INK, linewidth=0.8, marker="|", markersize=5,
               label="95% bootstrap CI"),
        Line2D([], [], color=INK, marker="*", linewidth=0, markersize=6,
               label="uncorrected p < .05"),
        Line2D([], [], color=INK, linewidth=0.8, label="bootstrap route difference"),
    ]
    fig.legend(handles=handles, loc="lower center", bbox_to_anchor=(0.5, 1.06), ncol=5,
               columnspacing=1.1, handletextpad=0.45)
    fig.text(0.5, 1.20, "Trait-space RSA", ha="center", va="bottom", fontsize=8,
             fontweight="bold", color=INK)
    _save(fig, "fig05_trait_space_rsa_all")
    frames = []
    for ds in ("us", "cn"):
        rsa = pd.read_csv(PIPELINE_DIR / ds / "trait_space_rsa_results.csv")
        frames.append(rsa[(rsa["split"] == "all")].assign(panel=ds_label(ds)))
    _export_source("fig05_trait_space_rsa_all", frames)


def _rsa_panel(ax, ds, shared_ylim):
    rsa = pd.read_csv(PIPELINE_DIR / ds / "dual_rsa_results.csv")
    df = _ordered(rsa, ds, "all")
    maxt = pd.read_csv(PIPELINE_DIR / ds / "dual_rsa_maxt_results.csv")
    maxt = _ordered(maxt, ds, "all")
    boot = pd.read_csv(PIPELINE_DIR / ds / "rsa_route_difference_identity_level.csv")
    boot = boot.set_index("trait").loc[_traits(ds)].reset_index()

    x = np.arange(len(df))
    w = 0.34
    sem = df["rsa_sem_r"].to_numpy()
    vis = df["rsa_vis_r"].to_numpy()
    ax.bar(x - w / 2, sem, w, color=SEM, label="Semantic")
    ax.bar(x + w / 2, vis, w, color=VIS, label="Visual")
    for off, vals, lo_col, hi_col in (
        (-w / 2, sem, "sem_boot_ci_lower", "sem_boot_ci_upper"),
        (w / 2, vis, "vis_boot_ci_lower", "vis_boot_ci_upper"),
    ):
        lo = boot[lo_col].to_numpy()
        hi = boot[hi_col].to_numpy()
        ax.errorbar(x + off, vals, yerr=[vals - lo, hi - vals], fmt="none",
                    ecolor=INK, elinewidth=0.65, capsize=1.8, capthick=0.65, zorder=4)

    top = np.maximum(boot["sem_boot_ci_upper"].to_numpy(),
                     boot["vis_boot_ci_upper"].to_numpy())
    for j in range(len(df)):
        sig = _stars(float(boot.iloc[j]["p_boot"]))
        if sig:
            _bracket(ax, x[j] - w / 2, x[j] + w / 2, max(top[j], 0) + 0.035,
                     sig, tick=0.007)

    y_sig = shared_ylim[0] + 0.055
    y_corr = shared_ylim[0] + 0.030
    for j in range(len(df)):
        for off, raw_p, corr_p, color in (
            (-w / 2, df.iloc[j]["rsa_sem_p"], maxt.iloc[j]["rsa_sem_p_maxt"], SEM),
            (w / 2, df.iloc[j]["rsa_vis_p"], maxt.iloc[j]["rsa_vis_p_maxt"], VIS),
        ):
            sig = _stars(float(raw_p))
            if sig:
                ax.text(x[j] + off, y_sig, sig, ha="center", va="bottom",
                        color=color, fontsize=5.2, fontweight="bold")
            if float(corr_p) < 0.05:
                ax.plot(x[j] + off, y_corr, "o", color=color, markersize=2.6)
    ax.set_xticks(x, _labels(ds), rotation=34, ha="right", rotation_mode="anchor")
    ax.set_ylim(*shared_ylim)
    _style(ax)
    return df.assign(panel=ds_label(ds))


def figure06_trait_rsa():
    all_boot = []
    for ds in ("us", "cn"):
        b = pd.read_csv(PIPELINE_DIR / ds / "rsa_route_difference_identity_level.csv")
        all_boot.append(b[b["trait"].isin(_traits(ds))])
    b = pd.concat(all_boot)
    lo = min(float(b["sem_boot_ci_lower"].min()), float(b["vis_boot_ci_lower"].min()), 0)
    hi = max(float(b["sem_boot_ci_upper"].max()), float(b["vis_boot_ci_upper"].max()), 0)
    span = hi - lo
    shared_ylim = (lo - 0.24 * span, hi + 0.20 * span)

    fig, axes = plt.subplots(2, 1, figsize=(FULL_WIDTH, 5.05), sharey=True,
                             constrained_layout=True)
    frames = []
    for i, (ax, ds) in enumerate(zip(axes, ("us", "cn"))):
        frames.append(_rsa_panel(ax, ds, shared_ylim))
        _panel(ax, chr(ord("a") + i), f"{ds_label_adj(ds)} sample")
    for ax in axes:
        ax.set_ylabel("Spearman ρ")
    handles = [
        Patch(facecolor=SEM, label="Semantic"),
        Patch(facecolor=VIS, label="Visual"),
        Line2D([], [], color=INK, linewidth=0.7, marker="|", markersize=5,
               label="95% bootstrap CI"),
        Line2D([], [], color=INK, marker="*", linewidth=0, markersize=6,
               label="uncorrected p < .05"),
        Line2D([], [], color=INK, marker="o", linewidth=0, markersize=3,
               label="survives max-T"),
        Line2D([], [], color=INK, linewidth=0.8, label="bootstrap route difference"),
    ]
    fig.legend(handles=handles, loc="lower center", bbox_to_anchor=(0.5, 1.03), ncol=6,
               columnspacing=1.1, handletextpad=0.45)
    fig.text(0.5, 1.10, "Per-trait RSA", ha="center", va="bottom", fontsize=8,
             fontweight="bold", color=INK)
    _save(fig, "fig06_rsa_sem_vis_bars")
    _export_source("fig06_rsa_sem_vis_bars", frames)


def _stacked_bar(ax, x, row, width, colors, hatch=None, edgecolor="white"):
    bottom = 0.0
    for col, color in zip(("unique_sem", "shared", "unique_vis"), colors):
        value = float(row[col])
        ax.bar(x, value, width, bottom=bottom, color=color, edgecolor=edgecolor,
               linewidth=0.45, hatch=hatch)
        bottom += value
    return bottom


def _vp_significance(ax, x, row, y_star, y_dot, sep=0.07):
    """Route-specific permutation star and max-T dot, both below the bar."""
    for dx, p_col, max_col, color in (
        (-sep, "unique_sem_p_perm", "unique_sem_p_maxt", SEM),
        (sep, "unique_vis_p_perm", "unique_vis_p_maxt", VIS),
    ):
        sig = _stars(float(row[p_col]))
        if sig:
            ax.text(x + dx, y_star, sig, ha="center", va="bottom", color=color,
                    fontsize=5.2, fontweight="bold", clip_on=False)
        if pd.notna(row[max_col]) and float(row[max_col]) < 0.05:
            ax.plot(x + dx, y_dot, "o", color=color, markersize=2.6, clip_on=False)


def figure07_trait_space_vp():
    rows = []
    vpi_rows = []
    for ds in ("us", "cn"):
        rsa = pd.read_csv(PIPELINE_DIR / ds / "trait_space_rsa_results.csv")
        rows.append(rsa[rsa["split"] == "all"].iloc[0])
        vpi = pd.read_csv(PIPELINE_DIR / ds / "vp_identity_inference.csv")
        vpi_rows.append(vpi[(vpi["split"] == "all") &
                            (vpi["outcome"] == "traitspace")].iloc[0])
    ymax = max(float(r["total_r2"]) for r in rows)
    fig, ax = plt.subplots(figsize=(FULL_WIDTH * 0.64, 3.05), constrained_layout=True)
    x = np.arange(2)
    for i, (row, vpi) in enumerate(zip(rows, vpi_rows)):
        _stacked_bar(ax, x[i], row, 0.48, (SEM, SHARED, VIS))
        _vp_significance(ax, x[i], vpi, -0.043, -0.060, sep=0.10)
        sig = _stars(float(vpi["diff_p_boot"]))
        if sig:
            y1 = float(row["unique_sem"]) / 2
            y2 = float(row["unique_sem"] + row["shared"] + row["unique_vis"] / 2)
            _vertical_bracket(ax, x[i] + 0.31, y1, y2, sig)
    ax.set_xticks(x, [f"US\n(n = {int(rows[0]['n_celebrities'])})",
                      f"China\n(n = {int(rows[1]['n_celebrities'])})"])
    ax.set_ylabel("Variance explained (R²)")
    ax.set_ylim(-0.080, ymax * 1.18)
    _panel(ax, "", "Trait-space variance partitioning")
    _style(ax)
    ax.legend(handles=[Patch(facecolor=SEM, label="Unique semantic"),
                       Patch(facecolor=SHARED, label="Shared"),
                       Patch(facecolor=VIS, label="Unique visual"),
                       Line2D([], [], marker="*", color=INK, linewidth=0,
                              markersize=6, label="route pperm < .05"),
                       Line2D([], [], color=INK, linewidth=0.8,
                              label="bootstrap route difference")],
              loc="upper left", bbox_to_anchor=(0, 1.0), ncol=2)
    _save(fig, "fig07_trait_space_variance_partitioning_all")
    _export_source("fig07_trait_space_variance_partitioning_all",
                   [pd.DataFrame(rows).assign(panel=["US", "China"])])


def _vp_trait_panel(ax, ds, ylim):
    rsa = pd.read_csv(PIPELINE_DIR / ds / "dual_rsa_results.csv")
    df = _ordered(rsa, ds, "all")
    vpi = pd.read_csv(PIPELINE_DIR / ds / "vp_identity_inference.csv")
    vpi = vpi[vpi["split"] == "all"].set_index("outcome")
    x = np.arange(len(df))
    for j, row in df.iterrows():
        _stacked_bar(ax, x[j], row, 0.56, (SEM, SHARED, VIS))
        pr = vpi.loc[row["trait"]]
        _vp_significance(ax, x[j], pr, ylim[0] + 0.045, ylim[0] + 0.022, sep=0.14)
        sig = _stars(float(pr["diff_p_boot"]))
        if sig:
            y1 = float(row["unique_sem"]) / 2
            y2 = float(row["unique_sem"] + row["shared"] + row["unique_vis"] / 2)
            _vertical_bracket(ax, x[j] + 0.40, y1, y2, sig)
    ax.set_xticks(x, _labels(ds), rotation=34, ha="right", rotation_mode="anchor")
    ax.set_ylim(*ylim)
    ax.set_title(f"{ds_label_adj(ds)} sample", loc="left", pad=4)
    ax.set_ylabel("Variance explained (R²)")
    _style(ax)
    return df.assign(panel=ds_label(ds))


def figure08_trait_vp():
    totals = []
    for ds in ("us", "cn"):
        df = _ordered(pd.read_csv(PIPELINE_DIR / ds / "dual_rsa_results.csv"), ds, "all")
        totals.extend(df["total_r2"].tolist())
    ylim = (-0.075, max(totals) * 1.24)
    fig, axes = plt.subplots(2, 1, figsize=(FULL_WIDTH, 4.95), sharey=True,
                             constrained_layout=True)
    frames = []
    for i, (ax, ds) in enumerate(zip(axes, ("us", "cn"))):
        frames.append(_vp_trait_panel(ax, ds, ylim))
        _panel(ax, chr(ord("a") + i), "Per-trait variance partitioning")
    handles = [Patch(facecolor=SEM, label="Unique semantic"),
               Patch(facecolor=SHARED, label="Shared"),
               Patch(facecolor=VIS, label="Unique visual"),
               Line2D([], [], marker="*", color=INK, linewidth=0, markersize=6,
                      label="route pperm < .05"),
               Line2D([], [], marker="o", color=INK, linewidth=0, markersize=3,
                      label="survives max-T"),
               Line2D([], [], color=INK, linewidth=0.8,
                      label="bootstrap route difference")]
    fig.legend(handles=handles, loc="lower center", bbox_to_anchor=(0.5, 1.0), ncol=6,
               columnspacing=1.1, handletextpad=0.45)
    _save(fig, "fig08_variance_partitioning")
    _export_source("fig08_variance_partitioning", frames)


def export_inference_inputs():
    SOURCE_OUT.mkdir(parents=True, exist_ok=True)
    files = (
        "dual_rsa_results.csv",
        "dual_rsa_maxt_results.csv",
        "rsa_route_difference_identity_level.csv",
        "vp_identity_inference.csv",
        "trait_space_rsa_results.csv",
    )
    for ds in ("us", "cn"):
        for filename in files:
            source = PIPELINE_DIR / ds / filename
            if source.exists():
                shutil.copy2(source, SOURCE_OUT / f"{ds}_{filename}")


def main():
    figure05_trait_space_rsa()
    figure06_trait_rsa()
    figure07_trait_space_vp()
    figure08_trait_vp()
    export_inference_inputs()
    print("Publication figure set complete.")


if __name__ == "__main__":
    main()
