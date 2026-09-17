"""
Descriptive embedding, trait-rating, and RDM visualizations.

Figures produced
----------------
Main:
4_output/publication/fig04_tsne_semantic_{dataset}.png
4_output/publication/fig03_tsne_visual_{dataset}.png
4_output/publication/fig01|02_trait_top_bottom_ratings_{dataset}.png
Supplementary:
4_output/publication/supplementary/tsne_visual_occupation_{dataset}.png
4_output/publication/supplementary/tsne_trait_space_{dataset}.png
4_output/publication/supplementary/trait_correlation_matrix_{dataset}.png

Usage
-----
python 2_code/descriptive_visualization.py
python 2_code/descriptive_visualization.py --dataset cn
python 2_code/descriptive_visualization.py --dataset us --no-photos

The script uses sibling source-repo defaults when they exist:
- ../chinese-celeb/1_data/ChineseStimuli/RateImage_Chinese/image_module10
- ../chinese-celeb/3_pipeline/embeddings/embeddings_VGGFace2.npy
- ../face-bert/wikiRDM_GPT3large_18May2026
- ../face-bert/face-bert-clean/pipeline/vggface2_RDM/features_vggface2.npy
- ../face-bert/face-bert-clean/replication/data/face_identity_demographics.csv

If a raw embedding file is unavailable, t-SNE falls back to the corresponding
50 x 50 RDM with metric="precomputed".
"""

from __future__ import annotations

import argparse
import math
import pickle
import re
import sys
import warnings
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

import matplotlib

matplotlib.use("Agg")
matplotlib.rcParams.update({
    "font.family": "sans-serif",
    "font.sans-serif": ["Arial", "Helvetica", "DejaVu Sans", "sans-serif"],
    "font.size": 7,
    "axes.labelsize": 7,
    "xtick.labelsize": 6,
    "ytick.labelsize": 6,
    "legend.fontsize": 6.5,
    "axes.linewidth": 0.6,
    "xtick.major.width": 0.6,
    "ytick.major.width": 0.6,
    "figure.dpi": 300,
    "savefig.dpi": 600,
    "savefig.bbox": "tight",
    "svg.fonttype": "none",
    "pdf.fonttype": 42,
    "axes.unicode_minus": False,
})

import matplotlib.pyplot as plt
from matplotlib import font_manager
from matplotlib.offsetbox import AnnotationBbox, OffsetImage
from matplotlib.patches import Patch
import numpy as np
import pandas as pd
from scipy.spatial.distance import squareform
from scipy.stats import mannwhitneyu, pearsonr

try:
    from PIL import Image, ImageOps
except ImportError as exc:  # pragma: no cover - exercised only in missing envs
    raise SystemExit("Pillow is required for photo thumbnails. Run `pip install pillow`.") from exc

try:
    from sklearn.manifold import TSNE
    from sklearn.preprocessing import StandardScaler
except ImportError as exc:  # pragma: no cover - exercised only in missing envs
    raise SystemExit("scikit-learn is required for t-SNE. Run `pip install scikit-learn`.") from exc

from config import ds_label, DATASETS, OUTPUT_DIR, ROOT, SEM_COLOR, VIS_COLOR
from data_loader import load_familiarity_ratings, load_ratings, load_rdms
TSNE_SEED        = 42
TRAIT_SPACE_COLOR = '#5A9E6F'
OUTDIR = OUTPUT_DIR / "publication"
INK = "#272727"

# Two-column publication contract: 180 mm wide, 7-9 pt final text.
FULL_WIDTH = 7.086614  # 180 mm

# Keep the embedding axes identical across datasets and representations. The
# canvas, rather than the t-SNE panel, grows to accommodate multi-row legends.
TSNE_AX_BOUNDS_IN = (0.60, 0.47, 6.26, 5.46)  # left, bottom, width, height
TSNE_FIG_WIDTH_IN = FULL_WIDTH
TSNE_FIG_HEIGHT_IN = 6.33
TSNE_OCC_LEGEND_EXTRA_IN = 0.98
TSNE_GENDER_LEGEND_EXTRA_IN = 0.44

GENDER_PALETTE: dict[str, str] = {
    "Female": "#C02635",  # dark red
    "Male":   "#00609D",  # dark blue
}

OCCUPATION_PALETTE: dict[str, str] = {
    "Scholar/Writer":      "#4472C4",  # blue (user-specified)
    "Politician/Official": "#E15759",  # red
    "Athlete":             "#59A14F",  # green
    "Musician":            "#B07AA1",  # purple
    "Host/Media":          "#76B7B2",  # teal
    "Entrepreneur":        "#EDC948",  # gold
    "Actor/Actress":       "#F28E2B",  # orange
    "Actor/Actress/Model": "#FF9DA7",  # pink
    "Director":            "#9C755F",  # brown
}


def _configure_fonts():
    """Use an installed CJK-capable font when available."""
    preferred = [
        "Noto Sans CJK SC",
        "Noto Sans CJK JP",
        "Source Han Sans SC",
        "Microsoft YaHei",
        "SimHei",
        "Arial Unicode MS",
        "DejaVu Sans",
    ]
    available = {f.name for f in font_manager.fontManager.ttflist}
    for name in preferred:
        if name in available:
            matplotlib.rcParams["font.family"] = name
            return


def _pub_style(ax):
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines["left"].set_color(INK)
    ax.spines["bottom"].set_color(INK)
    ax.tick_params(colors=INK, labelsize=6)


def _save_fig(fig, path: Path) -> None:
    fig.savefig(str(path), dpi=600)
    pdf_path = path.with_suffix(".pdf")
    fig.savefig(str(pdf_path))
    svg_path = path.with_suffix(".svg")
    fig.savefig(str(svg_path))
    tiff_path = path.with_suffix(".tiff")
    fig.savefig(str(tiff_path), dpi=600, pil_kwargs={"compression": "tiff_lzw"})
    print(f"Saved {path} + {pdf_path.name} + {svg_path.name} + {tiff_path.name}")


def _dataset_names(ds_name: str) -> list[str]:
    """Return display names in the RDM/ratings order."""
    cfg = DATASETS[ds_name]
    if ds_name == "cn":
        metadata = ROOT / "1_data/cn/semantic_first_substantive/text/cn_first_substantive_text_metadata.csv"
        if metadata.exists():
            df = pd.read_csv(metadata).sort_values("celeb_id")
            return df["name"].astype(str).tolist()

        stats = ROOT.parent / "chinese-celeb/3_pipeline/descriptive_behavioral/descriptive_stats_per_identity.csv"
        if stats.exists():
            df = pd.read_csv(stats).sort_values("identity_idx")
            return df["identity"].astype(str).tolist()

    order = _us_identity_order_path()
    if order.exists():
        return [_clean_us_identity(line.strip()) for line in order.read_text().splitlines() if line.strip()]

    return [f"{ds_label(ds_name)} {i:02d}" for i in range(1, cfg["n_targets"] + 1)]


def _us_identity_order_path() -> Path:
    """US 50-target order; the in-repo copy takes precedence over the source repo."""
    local = ROOT / "1_data/us/identity_order.txt"
    if local.exists():
        return local
    return ROOT.parent / "face-bert/face-bert-clean/pipeline/vggface2_RDM/identity_order.txt"


def _clean_us_identity(identity: str) -> str:
    identity = re.sub(r"^ID_\d+_", "", identity).strip()
    identity = identity.replace("_", " ")
    identity = re.sub(r"(?<=[a-z])(?=[A-Z])", " ", identity)
    return re.sub(r"\s+", " ", identity)


def _identity_key(name: str) -> str:
    name = re.sub(r"^ID_\d+_", "", str(name)).strip()
    name = name.replace("_", " ")
    name = re.sub(r"(?<=[a-z])(?=[A-Z])", " ", name)
    return re.sub(r"[^a-z0-9]+", "", name.lower())


def _load_us_pickle_embeddings(path: Path, n_targets: int) -> np.ndarray | None:
    order_path = _us_identity_order_path()
    if not order_path.exists():
        warnings.warn(f"Skipping {path}: US identity_order.txt not found")
        return None

    pkl_by_key = {_identity_key(p.stem): p for p in path.glob("*.pkl")}
    rows = []
    missing = []
    for identity in [line.strip() for line in order_path.read_text().splitlines() if line.strip()]:
        pkl_path = pkl_by_key.get(_identity_key(identity))
        if pkl_path is None:
            missing.append(identity)
            continue
        with pkl_path.open("rb") as f:
            arr = np.asarray(pickle.load(f), dtype=float)
        if arr.ndim != 1:
            warnings.warn(f"Skipping {pkl_path}: expected 1-D vector, got shape {arr.shape}")
            return None
        rows.append(arr)

    if missing or len(rows) != n_targets:
        warnings.warn(f"Skipping {path}: missing {len(missing)} embeddings; expected {n_targets}, got {len(rows)}")
        return None
    return np.vstack(rows)


def _load_matrix(path: Path | None, n_targets: int) -> np.ndarray | None:
    if path is None or not path.exists():
        return None
    if path.is_dir():
        return _load_us_pickle_embeddings(path, n_targets)

    arr = np.load(path, allow_pickle=False)
    if arr.ndim == 3 and arr.shape[0] == 1:
        arr = arr[0]
    if arr.ndim != 2:
        warnings.warn(f"Skipping {path}: expected 2-D array, got shape {arr.shape}")
        return None
    if arr.shape[0] != n_targets:
        warnings.warn(f"Skipping {path}: expected {n_targets} rows, got {arr.shape[0]}")
        return None
    return np.asarray(arr, dtype=float)


def _semantic_embedding_path(ds_name: str) -> Path | None:
    if ds_name == "cn":
        candidates = [
            ROOT / "1_data/cn/semantic_first_substantive/embeddings/embeddings_first_substantive_text_embedding_3_large.npy",
            ROOT.parent / "chinese-celeb/3_pipeline/cn_first_substantive_embeddings/embeddings_first_substantive_text_embedding_3_large.npy",
            ROOT.parent / "chinese-celeb/3_pipeline/embeddings/embeddings_SentAvg_large.npy",
        ]
    else:
        candidates = [
            ROOT / "1_data/us/semantic_gpt_wiki_18May2026/embeddings",
            ROOT.parent / "face-bert/wikiRDM_GPT3large_18May2026",
            ROOT.parent / "face-bert/face-bert-clean/pipeline/wikiRDM_GPT3large_18May2026/embeddings.npy",
            ROOT.parent / "face-bert/face-bert-clean/pipeline/wikiRDM_GPT3large/embeddings.npy",
        ]
    return next((path for path in candidates if path.exists()), None)


def _visual_embedding_path(ds_name: str) -> Path | None:
    if ds_name == "cn":
        candidates = [
            ROOT / "1_data/cn/embeddings_vggface2.npy",
            ROOT.parent / "chinese-celeb/3_pipeline/embeddings/embeddings_VGGFace2.npy",
        ]
    else:
        candidates = [
            ROOT / "1_data/us/embeddings_vggface2.npy",
            ROOT.parent / "face-bert/face-bert-clean/pipeline/vggface2_RDM/features_vggface2.npy",
        ]
    return next((path for path in candidates if path.exists()), None)


def _cn_image_paths(image_dir: Path | None, n_targets: int) -> list[Path | None]:
    if image_dir is None:
        image_dir = ROOT / "1_data/cn/stimuli"
    return [image_dir / f"{i}.jpg" if (image_dir / f"{i}.jpg").exists() else None
            for i in range(1, n_targets + 1)]


def _us_image_paths(image_dir: Path | None, n_targets: int) -> list[Path | None]:
    if image_dir is None:
        image_dir = ROOT / "1_data/us/stimuli"

    order_path = _us_identity_order_path()
    meta_path = ROOT / "1_data/us/face_identity_demographics.csv"
    if not order_path.exists() or not meta_path.exists():
        return [None] * n_targets

    order = [line.strip() for line in order_path.read_text().splitlines() if line.strip()]
    meta = pd.read_csv(meta_path)
    meta = meta.copy()
    meta["_identity_key"] = meta["IDnames"].map(_identity_key)
    image_paths: list[Path | None] = []
    for identity in order:
        rows = meta[meta["_identity_key"] == _identity_key(identity)]
        path = None
        if not rows.empty:
            name = str(rows.sort_values("FaceIndex").iloc[0]["ImageName"])
            candidate = image_dir / name
            if candidate.exists():
                path = candidate
        image_paths.append(path)
    if len(image_paths) != n_targets:
        return [None] * n_targets
    return image_paths


def _image_paths(ds_name: str, image_dir: Path | None) -> list[Path | None]:
    n_targets = DATASETS[ds_name]["n_targets"]
    if ds_name == "cn":
        return _cn_image_paths(image_dir, n_targets)
    return _us_image_paths(image_dir, n_targets)


def _load_thumb(path: Path, size: int = 42) -> np.ndarray | None:
    try:
        img = Image.open(path).convert("RGB")
        resampling = getattr(Image, "Resampling", Image)
        img = ImageOps.fit(img, (size, size), method=resampling.LANCZOS, centering=(0.5, 0.45))
        return np.asarray(img)
    except Exception as exc:
        warnings.warn(f"Could not load image {path}: {exc}")
        return None


def _load_occupations(ds_name: str) -> dict[int, str]:
    """Return {celeb_id: occupation} from 1_data/{ds}/celebrity_occupations.csv."""
    path = ROOT / f"1_data/{ds_name}/celebrity_occupations.csv"
    if not path.exists():
        return {}
    df = pd.read_csv(path)
    return dict(zip(df["celeb_id"].astype(int), df["occupation"].astype(str)))


def _load_genders(ds_name: str) -> dict[int, str]:
    """Return {celeb_id: gender} from 1_data/{ds}/celebrity_genders.csv."""
    path = ROOT / f"1_data/{ds_name}/celebrity_genders.csv"
    if not path.exists():
        return {}
    df = pd.read_csv(path)
    return dict(zip(df["celeb_id"].astype(int), df["gender"].astype(str)))


def _source_label(path: Path) -> str:
    try:
        return str(path.relative_to(ROOT))
    except ValueError:
        return str(path)


def _tsne_from_data(data: np.ndarray, is_rdm: bool) -> np.ndarray:
    if is_rdm:
        matrix = np.asarray(data, dtype=float).copy()
        matrix[~np.isfinite(matrix)] = np.nanmedian(matrix[np.isfinite(matrix)])
        np.fill_diagonal(matrix, 0)
        return TSNE(
            n_components=2,
            metric="precomputed",
            init="random",
            method="exact",
            perplexity=12,
            learning_rate="auto",
            random_state=TSNE_SEED,
        ).fit_transform(matrix)

    matrix = np.asarray(data, dtype=float)
    matrix = np.nan_to_num(matrix, nan=np.nanmedian(matrix[np.isfinite(matrix)]))
    matrix = StandardScaler().fit_transform(matrix)
    return TSNE(
        n_components=2,
        perplexity=12,
        learning_rate="auto",
        # # to prevent from regenerating different plots
        # method="exact",
        init="pca",
        random_state=TSNE_SEED,
    ).fit_transform(matrix)


def _plot_tsne_panel(
    ds_name: str,
    representation: str,
    coords: np.ndarray,
    names: list[str],
    image_paths: list[Path | None],
    use_photos: bool,
    occupations: dict[int, str] | None = None,
    palette: dict[str, str] | None = None,
) -> Path:
    # semantic and visual t-SNE panels are main figures; the occupation-colored
    # visual and trait-space variants are supplementary
    outdir = OUTDIR
    if representation in ("visual_occupation", "trait_space"):
        outdir = outdir / "supplementary"
    outdir.mkdir(parents=True, exist_ok=True)
    tsne_prefix = {"visual": "fig03_", "semantic": "fig04_"}.get(representation, "")
    path = outdir / f"{tsne_prefix}tsne_{representation}_{ds_name}.png"

    if representation == "semantic":
        color = SEM_COLOR
    elif representation in ("visual", "visual_occupation"):
        color = VIS_COLOR
    else:
        color = TRAIT_SPACE_COLOR
    fig, ax = plt.subplots(figsize=(TSNE_FIG_WIDTH_IN, TSNE_FIG_HEIGHT_IN))
    ax.scatter(coords[:, 0], coords[:, 1], s=14, color=color, alpha=0.85, zorder=2)

    border_palette = palette or OCCUPATION_PALETTE
    has_photo = np.zeros(len(names), dtype=bool)
    occ_seen: dict[str, str] = {}  # border category -> hex color
    if use_photos:
        for i, (x, y, img_path) in enumerate(zip(coords[:, 0], coords[:, 1], image_paths)):
            if img_path is None:
                continue
            thumb = _load_thumb(img_path)
            if thumb is None:
                continue
            occ = (occupations or {}).get(i + 1)
            edge_color = border_palette.get(occ) if occ else "white"
            edge_lw = 3.8 if occ and edge_color != "white" else 0.6
            if occ and edge_color != "white":
                occ_seen[occ] = edge_color
            ab = AnnotationBbox(
                OffsetImage(thumb, zoom=0.52),
                (x, y),
                frameon=True,
                bboxprops={"edgecolor": edge_color, "linewidth": edge_lw},
                pad=0.02,
                zorder=3,
            )
            ax.add_artist(ab)
            has_photo[i] = True

    if not use_photos:
        label_indices = range(len(names))
    else:
        label_indices = np.where(~has_photo)[0]
    for i in label_indices:
        x, y = coords[i]
        ax.text(x, y, str(i + 1), color="black", fontsize=5.5, fontweight="bold", ha="center", va="center", zorder=4)

    title_word = {"semantic": "Semantic Embedding", "visual": "Visual Embedding",
                  "visual_occupation": "Visual Embedding", "trait_space": "Trait Space"}.get(representation, representation.capitalize())
    ax.set_title(f"{ds_label(ds_name)} {title_word} t-SNE",
                 fontsize=8, fontweight="bold", pad=6)
    ax.set_xlabel("t-SNE Dimension 1", fontsize=7, fontweight="bold")
    ax.set_ylabel("t-SNE Dimension 2", fontsize=7, fontweight="bold")
    _pub_style(ax)

    if occ_seen:
        sorted_occs = sorted(occ_seen.keys())
        handles = [Patch(color=occ_seen[occ], label=occ) for occ in sorted_occs]
        is_occupation_legend = palette is None or representation == "visual_occupation"
        n_cols = min(3 if is_occupation_legend else 2, len(handles))
        legend_extra = (TSNE_OCC_LEGEND_EXTRA_IN if is_occupation_legend
                        else TSNE_GENDER_LEGEND_EXTRA_IN)
        fig_height = TSNE_FIG_HEIGHT_IN + legend_extra
        fig.set_size_inches(TSNE_FIG_WIDTH_IN, fig_height, forward=True)
        left, bottom, width, height = TSNE_AX_BOUNDS_IN
        ax.set_position((left / TSNE_FIG_WIDTH_IN,
                         (bottom + legend_extra) / fig_height,
                         width / TSNE_FIG_WIDTH_IN,
                         height / fig_height))
        fig.legend(
            handles=handles,
            loc="lower center",
            bbox_to_anchor=(0.5, -0.006),
            ncol=n_cols,
            frameon=False,
            fontsize=6.5,
            handlelength=1.1,
            handleheight=1.1,
            borderpad=0.4,
            columnspacing=1.1,
        )
    else:
        left, bottom, width, height = TSNE_AX_BOUNDS_IN
        ax.set_position((left / TSNE_FIG_WIDTH_IN, bottom / TSNE_FIG_HEIGHT_IN,
                         width / TSNE_FIG_WIDTH_IN, height / TSNE_FIG_HEIGHT_IN))

    _save_fig(fig, path)
    plt.close(fig)
    return path


def plot_tsne(ds_name: str, image_dir: Path | None, use_photos: bool) -> list[Path]:
    cfg = DATASETS[ds_name]
    names = _dataset_names(ds_name)
    images = _image_paths(ds_name, image_dir)
    occupations = _load_occupations(ds_name)
    sem_rdm, vis_rdm = load_rdms(cfg)

    outputs: list[Path] = []
    specs = [
        ("semantic", _semantic_embedding_path(ds_name), sem_rdm),
        ("visual", _visual_embedding_path(ds_name), vis_rdm),
    ]
    for representation, emb_path, fallback_rdm in specs:
        matrix = _load_matrix(emb_path, cfg["n_targets"])
        if matrix is not None:
            coords = _tsne_from_data(matrix, is_rdm=False)
        else:
            coords = _tsne_from_data(fallback_rdm, is_rdm=True)
        if representation == "semantic":
            outputs.append(_plot_tsne_panel(ds_name, representation, coords, names, images, use_photos, occupations))
        else:  # visual: gender-colored thumbnail borders
            genders = _load_genders(ds_name)
            outputs.append(_plot_tsne_panel(ds_name, representation, coords, names, images, use_photos,
                                            genders, palette=GENDER_PALETTE))
            outputs.append(_plot_tsne_panel(ds_name, "visual_occupation", coords, names, images, use_photos, occupations))
    return outputs


def combine_main_tsne_figures() -> list[Path]:
    """Join US/CN panels pixel-for-pixel; never rescale either t-SNE axes."""
    OUTDIR.mkdir(parents=True, exist_ok=True)
    combined: list[Path] = []
    specs = [
        ("fig03_tsne_visual_{ds}.png", "fig03_tsne_visual_us_cn.png"),
        ("fig04_tsne_semantic_{ds}.png", "fig04_tsne_semantic_us_cn.png"),
    ]
    for panel_name, combined_name in specs:
        panels = []
        for ds_name in ("us", "cn"):
            with Image.open(OUTDIR / panel_name.format(ds=ds_name)) as image:
                panels.append(image.convert("RGB").copy())
        gap_px = 80
        canvas = Image.new(
            "RGB",
            (sum(panel.width for panel in panels) + gap_px,
             max(panel.height for panel in panels)),
            "white",
        )
        x = 0
        for panel in panels:
            y = (canvas.height - panel.height) // 2
            canvas.paste(panel, (x, y))
            x += panel.width + gap_px
        path = OUTDIR / combined_name
        canvas.save(path, dpi=(600, 600), optimize=True)
        print(f"Saved {path} (panels joined without rescaling)")
        combined.append(path)
    return combined


def plot_tsne_trait_space(ds_name: str, image_dir: Path | None, use_photos: bool) -> Path:
    """t-SNE of celebrities in the full trait-rating space (n_targets × n_traits)."""
    cfg     = DATASETS[ds_name]
    ratings = load_ratings(cfg)
    names   = _dataset_names(ds_name)
    images  = _image_paths(ds_name, image_dir)
    means   = np.column_stack([np.nanmean(ratings[t], axis=0) for t in cfg["traits"]])
    coords  = _tsne_from_data(means, is_rdm=False)
    return _plot_tsne_panel(ds_name, "trait_space", coords, names, images, use_photos)


def _trait_mean_matrix(ds_name: str) -> pd.DataFrame:
    cfg = DATASETS[ds_name]
    ratings = load_ratings(cfg)
    names = _dataset_names(ds_name)
    data = {trait: np.nanmean(ratings[trait], axis=0) for trait in cfg["traits"]}
    fam = load_familiarity_ratings(cfg)
    data["familiarity"] = np.nanmean(fam, axis=0)
    return pd.DataFrame(data, index=names)


def _corr_pvals(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    traits = list(df.columns)
    corr = pd.DataFrame(np.eye(len(traits)), index=traits, columns=traits)
    pvals = pd.DataFrame(np.zeros((len(traits), len(traits))), index=traits, columns=traits)
    for i, t1 in enumerate(traits):
        for j, t2 in enumerate(traits):
            if i >= j:
                continue
            valid = df[t1].notna() & df[t2].notna()
            r, p = pearsonr(df.loc[valid, t1], df.loc[valid, t2])
            corr.loc[t1, t2] = corr.loc[t2, t1] = r
            pvals.loc[t1, t2] = pvals.loc[t2, t1] = p
    return corr, pvals


def _stars(p: float) -> str:
    if p < 0.001:
        return "***"
    if p < 0.01:
        return "**"
    if p < 0.05:
        return "*"
    return ""


def plot_trait_correlation_matrix(ds_name: str) -> Path:
    import seaborn as sns

    means = _trait_mean_matrix(ds_name).rename(columns={"young": "youthful"})
    corr, pvals = _corr_pvals(means)

    fig, ax = plt.subplots(figsize=(FULL_WIDTH, FULL_WIDTH * 0.905))
    heatmap = sns.heatmap(
        corr,
        ax=ax,
        cmap="vlag",
        vmin=-1,
        vmax=1,
        center=0,
        square=True,
        linewidths=0.4,
        linecolor="white",
        annot=False,
        cbar_kws={"label": "Pearson correlation (r)", "shrink": 0.82},
    )
    for i, row in enumerate(corr.index):
        for j, col in enumerate(corr.columns):
            value = corr.loc[row, col]
            text_color = "white" if abs(value) >= 0.55 else "#111827"
            label = f"{value:.2f}" if i == j else f"{value:.2f}{_stars(pvals.loc[row, col])}"
            ax.text(j + 0.5, i + 0.5, label,
                    ha="center", va="center", fontsize=5.5, fontweight="bold", color=text_color)

    cbar = heatmap.collections[0].colorbar
    cbar.ax.tick_params(labelsize=12)
    cbar.ax.yaxis.label.set_size(13)
    cbar.ax.yaxis.label.set_weight("bold")
    ax.set_title(f"{ds_label(ds_name)} Trait Correlation Matrix", fontsize=8, fontweight="bold", pad=6)
    ax.set_xlabel("")
    ax.set_ylabel("")
    ax.set_xticklabels(ax.get_xticklabels(), rotation=45, ha="right", fontsize=6.5, fontweight="bold")
    ax.set_yticklabels(ax.get_yticklabels(), rotation=0, fontsize=6.5, fontweight="bold")
    fig.text(0.5, 0.02, "Cell values are Pearson correlations across target-level mean ratings; * p < .05, ** p < .01, *** p < .001",
             ha="center", va="center", fontsize=6, color="#374151")
    fig.tight_layout(rect=[0, 0.04, 1, 1])

    outdir = OUTDIR / "supplementary"
    outdir.mkdir(parents=True, exist_ok=True)
    path = outdir / f"trait_correlation_matrix_{ds_name}.png"
    _save_fig(fig, path)
    plt.close(fig)
    return path


def _draw_stimulus_cell(ax, img_path: Path | None):
    ax.set_xticks([])
    ax.set_yticks([])
    for spine in ax.spines.values():
        spine.set_visible(False)
    if img_path is not None:
        thumb = _load_thumb(img_path, size=160)
    else:
        thumb = None
    if thumb is None:
        ax.set_facecolor("#ECEFF4")
        ax.text(0.5, 0.5, "Image\nnot found", ha="center", va="center",
                fontsize=5, color="#4C566A", transform=ax.transAxes)
    else:
        ax.imshow(thumb)


def _display_trait_name(trait: str) -> str:
    return {"young": "youthful"}.get(trait, trait).replace("_", " ").title()


def plot_trait_top_bottom_ratings(ds_name: str, image_dir: Path | None) -> Path:
    cfg = DATASETS[ds_name]
    means = _trait_mean_matrix(ds_name)
    images = _image_paths(ds_name, image_dir)
    traits = cfg["traits"]

    n_cols = 10
    n_rows = len(traits)
    fig, axes = plt.subplots(n_rows, n_cols,
                             figsize=(FULL_WIDTH, max(3.78, n_rows * 0.666)), squeeze=False)

    for row, trait in enumerate(traits):
        vals = means[trait].to_numpy(dtype=float)
        ordered = np.argsort(vals)
        selected = list(ordered[:5]) + list(ordered[-5:])
        for col, idx in enumerate(selected):
            _draw_stimulus_cell(axes[row, col], images[idx])
        axes[row, 0].set_ylabel(_display_trait_name(trait), rotation=0, ha="right", va="center",
                                labelpad=22, fontsize=7, fontstyle="italic", fontweight="bold")

    fig.subplots_adjust(left=0.16, right=0.985, top=0.87, bottom=0.11, wspace=0.08, hspace=0.12)
    fig.suptitle(f"{ds_label(ds_name)} Trait Rating Extremes", fontsize=9, fontweight="bold", y=0.965)
    fig.text(0.36, 0.905, "Lowest-rated targets", ha="center", va="center",
             fontsize=7.5, fontweight="bold")
    fig.text(0.75, 0.905, "Highest-rated targets", ha="center", va="center",
             fontsize=7.5, fontweight="bold")

    # Span the arrow across the stimulus grid itself rather than fixed figure
    # fractions, so it stays aligned if the subplot margins change.
    grid_left = axes[0, 0].get_position().x0
    grid_right = axes[0, -1].get_position().x1
    overlay = fig.add_axes([0, 0, 1, 1], frameon=False)
    overlay.set_axis_off()
    overlay.annotate("", xy=(grid_right, 0.065), xytext=(grid_left, 0.065),
                     xycoords="figure fraction",
                     arrowprops={"arrowstyle": "-|>", "linewidth": 1.0,
                                 "color": "#3B4252", "mutation_scale": 12})
    fig.text(grid_left, 0.03, "Lowest", ha="left", va="center", fontsize=7.5, fontweight="bold")
    fig.text(grid_right, 0.03, "Highest", ha="right", va="center", fontsize=7.5, fontweight="bold")

    OUTDIR.mkdir(parents=True, exist_ok=True)
    prefix = "fig01" if ds_name == "us" else "fig02"
    path = OUTDIR / f"{prefix}_trait_top_bottom_ratings_{ds_name}.png"
    _save_fig(fig, path)
    plt.close(fig)
    return path


def run_dataset(ds_name: str, image_dir: Path | None, use_photos: bool) -> list[Path]:
    paths: list[Path] = []
    paths.extend(plot_tsne(ds_name, image_dir, use_photos))
    paths.append(plot_tsne_trait_space(ds_name, image_dir, use_photos))
    paths.append(plot_trait_correlation_matrix(ds_name))
    paths.append(plot_trait_top_bottom_ratings(ds_name, image_dir))
    return paths


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Create descriptive visualizations from dual-route data.")
    parser.add_argument(
        "--dataset",
        choices=["us", "cn", "both"],
        default="both",
        help="Dataset to plot.",
    )
    parser.add_argument(
        "--cn-image-dir",
        type=Path,
        default=None,
        help="Directory containing CN 1.jpg ... 50.jpg stimuli.",
    )
    parser.add_argument(
        "--us-image-dir",
        type=Path,
        default=None,
        help="Directory containing US CelebA image files listed in source metadata.",
    )
    parser.add_argument(
        "--no-photos",
        action="store_true",
        help="Use numeric labels instead of stimulus thumbnails in t-SNE plots.",
    )
    parser.add_argument(
        "--tsne-only",
        action="store_true",
        help="Render only the semantic and visual t-SNE figures.",
    )
    return parser.parse_args()


def main():
    _configure_fonts()
    args = parse_args()
    datasets = ["us", "cn"] if args.dataset == "both" else [args.dataset]
    produced: list[Path] = []
    for ds_name in datasets:
        image_dir = args.cn_image_dir if ds_name == "cn" else args.us_image_dir
        if args.tsne_only:
            produced.extend(plot_tsne(ds_name, image_dir, use_photos=not args.no_photos))
        else:
            produced.extend(run_dataset(ds_name, image_dir, use_photos=not args.no_photos))

    if datasets == ["us", "cn"]:
        produced.extend(combine_main_tsne_figures())

    for path in produced:
        print(f"Saved {path}")
    print("All descriptive visualizations complete.")


if __name__ == "__main__":
    main()
