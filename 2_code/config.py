"""Central configuration for both datasets."""

from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent

DATASETS = {
    'us': {
        'label':          'US',
        'label_adj':      'US',
        'ratings_file':   ROOT / '1_data/us/ratings_clean.npy',
        'ratings_format': 'npy',
        'npy_shape':      (415, 9, 50),   # (participants, items, targets)
        'npy_trait_axis': 1,
        'npy_familiarity_idx': 8,
        'traits':         ['warm', 'critical', 'competent', 'practical',
                           'feminine', 'strong', 'youthful', 'charismatic'],
        'n_targets': 50,
        'canonical_trait_map': {'youthful': 'young'},
        'sem_model':  'GPT_wiki_cosine',
        'sem_file':   ROOT / '1_data/us/rdm_semantic_gpt_wiki_cosine.npy',
        'vis_model':  'VGGFace2_cosine',
        'vis_file':   ROOT / '1_data/us/rdm_visual_vggface2_cosine.npy',
    },
    'cn': {
        'label':          'China',
        'label_adj':      'Chinese',
        'ratings_file':   ROOT / '1_data/cn/ratings_clean.csv',
        'ratings_format': 'csv',
        # Long-format CSV column names (actual columns verified from source)
        'csv_trait_col':       'session_trait',
        'csv_participant_col': 'participant',
        'csv_target_col':      'celeb_idx',
        'csv_rating_col':      'rating_clean',
        'csv_familiarity_val': 'familiarity',   # value in trait column for fam ratings
        'traits':         ['warm', 'critical', 'competent', 'practical',
                           'feminine', 'strong', 'young', 'charismatic', 'trustworthy', 'attractive'],
        'n_targets': 50,
        'canonical_trait_map': {},
        'sem_model':  'GPT_first_substantive_large_cosine',
        'sem_file':   ROOT / '1_data/cn/rdm_semantic_first_substantive_large_cosine.npy',
        'vis_model':  'VGGFace2_cosine',
        'vis_file':   ROOT / '1_data/cn/rdm_visual_vggface2_cosine.npy',
    },
}

# Traits shared across both datasets (youthful in US mapped to young)
COMMON_TRAITS = ['warm', 'critical', 'competent', 'practical',
                 'feminine', 'strong', 'young', 'charismatic']


def ds_label(ds_name):
    """Display name for figure titles and axis labels ('US', 'China')."""
    return DATASETS[ds_name]['label']


def ds_label_adj(ds_name):
    """Adjectival display name, for phrases like 'Chinese sample'."""
    return DATASETS[ds_name]['label_adj']


def local_common_traits(ds_name):
    """COMMON_TRAITS mapped to this dataset's local trait names (youthful/young)."""
    inv = {v: k for k, v in DATASETS[ds_name]['canonical_trait_map'].items()}
    return [inv.get(t, t) for t in COMMON_TRAITS]


SPLITS = ['all']

PIPELINE_DIR = ROOT / '3_pipeline'
OUTPUT_DIR   = ROOT / '4_output'

# Plot colours (semantic = blue, visual = red).  Dark/light pairs follow the
# manuscript palette so route identity stays stable across all figures.
SEM_COLOR  = '#00609D'
VIS_COLOR  = '#C02635'
SEM_LIGHT  = '#7EABCE'
VIS_LIGHT  = '#F58966'
