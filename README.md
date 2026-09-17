# dual-route-traits

Unified publication pipeline for two dual-route social trait perception datasets.

## Datasets

| Dataset | Celebrities | Participants | Traits | Semantic model | Visual model |
|---------|-------------|--------------|--------|----------------|--------------|
| `us` (Western) | 50 | 415 | 8 | GPT cosine | VGGFace2 cosine |
| `cn` (Chinese) | 50 | 330 | 10 | GPT cosine | VGGFace2 cosine |

The main CN semantic RDM is
`1_data/cn/rdm_semantic_first_substantive_large_cosine.npy`. Selected Chinese
text, full-text audit files, and GPT embeddings are under `1_data/cn/semantic_first_substantive/`.

### Inputs bundled in `1_data/`

Everything the analyses and figures need is in this repository; no external
checkout is required.

| Path | Contents |
|------|----------|
| `1_data/{us,cn}/ratings_clean.{npy,csv}` | Trial-level trait ratings (pseudonymous integer participant IDs) |
| `1_data/{us,cn}/rdm_semantic_*.npy`, `rdm_visual_vggface2_cosine.npy` | Model RDMs used by `config.py` |
| `1_data/us/stimuli/`, `1_data/cn/stimuli/` | The 50 face stimuli per sample, one image per target identity |
| `1_data/us/wiki_text/` | The 50 English Wikipedia articles behind the US semantic embeddings |
| `1_data/cn/semantic_first_substantive/text/` | The 50 Chinese biographies plus the selected passages and audit metadata |
| `1_data/us/semantic_gpt_wiki_18May2026/embeddings/`, `1_data/cn/semantic_first_substantive/embeddings/` | Per-target semantic embedding vectors (3,072-d) |
| `1_data/{us,cn}/embeddings_vggface2.npy` | Per-target VGGFace2 visual features, used for the t-SNE panels |
| `1_data/us/identity_order.txt`, `1_data/us/face_identity_demographics.csv` | US target order and the identity-to-image mapping |

See **Data and stimulus reuse** below for where each of these came from and on
what terms it may be reused.

## Data and stimulus reuse

This repository bundles material of several different origins. They are listed
separately because a single blanket statement would misdescribe most of them.

**Analysis code** (`2_code/`) was written by the authors for this project and
may be freely inspected, run and adapted for research purposes.

**Behavioural ratings** (`1_data/{us,cn}/ratings_clean.*`) and every table
derived from them in `3_pipeline/` were collected by the authors. Participants
are identified only by arbitrary integer codes: the files contain no names,
contact details, dates, free text or demographic fields, and no participant can
be identified from them.

**Face stimuli** (`1_data/us/stimuli/`, `1_data/cn/stimuli/`) are third-party
images, included only so the descriptive figures can be regenerated. They are
not the authors' to license.

- US: 50 images from the **CelebA** dataset (Liu, Luo, Wang & Tang, ICCV 2015),
  which is released for **non-commercial research use only** and whose terms do
  not grant redistribution rights. Only the one image per target identity used
  in the figures is included.
- CN: 50 photographs of Chinese public figures collected from public web
  sources for the rating task. Copyright remains with the original rights
  holders.

Anyone reusing this material should obtain the images from their original
sources rather than relying on the copies here. Deleting both `stimuli/`
folders costs only the stimulus thumbnails in Figures 1-4; every analysis and
every inferential figure (Figures 5-8, 11-12, R1-R3) still reproduces without
them.

**Source texts** are likewise third-party. `1_data/us/wiki_text/` holds English
Wikipedia article text, licensed CC BY-SA 4.0, so reuse carries the same
attribution and share-alike conditions.
`1_data/cn/semantic_first_substantive/text/` holds Chinese-language biographies
retrieved from Baidu Baike; copyright remains with the original publisher and
the passages are included so the embedding step can be audited.

**Target metadata** (`face_identity_demographics.csv`, `celebrity_*.csv`,
`3_pipeline/target_demographics.csv`) records gender, occupation and race
codings for the public figures used as stimuli. These are researcher
assignments describing the stimulus set, not claims endorsed by the individuals
named.

**Derived model representations** (`*embeddings*`, `rdm_*.npy`) were produced
with OpenAI `text-embedding-3-large` and VGGFace2. Use of the OpenAI outputs is
subject to OpenAI's terms of use as of the generation date (18 May 2026).

## Setup

`1_data/` is already populated, so no setup step is required to run the
analyses. To confirm the inputs load:

```bash
python 2_code/data_loader.py                         # sanity-check data loading
```

## Running the analyses

The split into main vs. supplementary follows the manuscript Results. Main
analyses live in `2_code/`; everything not reported in the Results lives in
`2_code/supplementary/` and writes into `supplementary/` output folders.

### Main analyses

```bash
# Inter-rater reliability: per-trait + pooled trait-space ICCs
python 2_code/run_icc.py                        # → 3_pipeline/{us,cn}/icc_results.csv, icc_trait_space_results.csv

# RSA + Partial RSA + Variance partitioning
python 2_code/run_rsa.py --dataset us|cn|both   # → 3_pipeline/{us,cn}/dual_rsa_results.csv
python 2_code/run_rsa_maxt.py                   # max-T FWER correction → dual_rsa_maxt_results.csv

# Trait-space (multivariate) RSA + variance partitioning
python 2_code/run_rsa_trait_space.py            # → 3_pipeline/{us,cn}/trait_space_rsa_results.csv

# Perceiver-level LMMs (familiarity moderation is the analysis reported in the Results)
python 2_code/run_lmm.py --dataset us|cn|both --model fam
python 2_code/run_lmm_trait_space.py            # trait-space LMMs

# Publication figures — everything lands in 4_output/publication/
python 2_code/descriptive_visualization.py      # figs 01-04 (trait extremes, t-SNE) + supplementary/
python 2_code/plot_publication_figures.py       # figs 05-08 (RSA, variance partitioning) + source_data/
```

### Supplementary analyses (`2_code/supplementary/`, run from repo root)

```bash
python 2_code/supplementary/run_merged_lmm.py                     # pooled US+CN LMM (common traits)
```

The base LMM (`python 2_code/run_lmm.py --model base`, → `dual_lmm_results.csv`)
is likewise supplementary.

LMM multiplicity is handled with Holm correction in the selected maximal-model
results; max-T is not used for the LMM analyses.

## Outputs

### Pipeline CSVs (`3_pipeline/{us,cn}/`)

| File | Rows | Contents |
|------|------|----------|
| `dual_rsa_results.csv` | 8 (US) / 10 (CN) | RSA r, partial r, variance partitioning per trait |
| `dual_lmm_results.csv` | 8 / 10 | Base LMM β (sem + vis) per trait |
| `dual_lmm_results_fam_interaction.csv` | 8 / 10 | Moderation LMM β including fam main effect and interactions |

### Publication figures (`4_output/publication/`)

Every figure is written as `.pdf`, `.svg`, `.png` and `.tiff` at the 180 mm
two-column contract; `source_data/` holds the exact rows behind each panel.

| File | Contents |
|------|----------|
| `fig01/fig02_trait_top_bottom_ratings_{ds}.png` | Stimulus images for highest / lowest rated targets per trait |
| `fig03_tsne_visual_{ds}.png` | t-SNE of visual embeddings with stimulus thumbnails |
| `fig04_tsne_semantic_{ds}.png` | t-SNE of semantic embeddings with stimulus thumbnails |
| `fig05_trait_space_rsa_all.png` | Trait-space RSA, semantic vs. visual |
| `fig06_rsa_sem_vis_bars.png` | Per-trait RSA ρ with bootstrap route-difference brackets |
| `fig07_trait_space_variance_partitioning_all.png` | Trait-space unique / shared variance, US vs. China |
| `fig08_variance_partitioning.png` | Per-trait unique semantic / shared / unique visual R² |
| `fig11_trait_space_lmm_moderation.png` | Trait-space LMM familiarity-moderation β |
| `fig12_lmm_moderation_betas.png` | Per-trait LMM familiarity-moderation β |

Auxiliary figures written to `4_output/publication/supplementary/` are generated
locally and excluded from version control.

## Folder structure

```
dual-route-traits/
├── 1_data/us/                 ratings, RDMs, embeddings, stimuli/, wiki_text/
├── 1_data/cn/                 data + CN first-substantive semantic artifacts
│   └── stimuli/               the 50 CN face stimuli
├── 2_code/                    main analysis scripts
│   └── supplementary/         analyses not reported in the manuscript Results
├── 3_pipeline/{us,cn}/        intermediate CSVs (flat; shared by main + supplementary)
│   └── lmm_familiarity_maximal/  selected/maximal random-effects LMM tables
├── 4_output/publication/      all manuscript figures (pdf/svg/png/tiff) + source_data/
└── requirements.txt           Python dependencies
```
