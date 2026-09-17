"""
Unified data loading for both datasets.

Public API
----------
load_ratings(cfg)         -> dict[trait_name -> (n_participants, 50) ndarray]
load_rdms(cfg)            -> (sem_rdm, vis_rdm) each (50, 50)
"""

import numpy as np
import pandas as pd


def load_ratings(cfg: dict) -> dict:
    """Return {trait: (n_participants, n_targets) array} with NaN for excluded trials."""
    if cfg['ratings_format'] == 'npy':
        return _load_ratings_npy(cfg)
    return _load_ratings_csv(cfg)


def _load_ratings_npy(cfg: dict) -> dict:
    arr = np.load(cfg['ratings_file'])   # (415, 9, 50)
    out = {}
    for i, trait in enumerate(cfg['traits']):
        out[trait] = arr[:, i, :]        # (n_participants, 50)
    return out


def _load_ratings_csv(cfg: dict) -> dict:
    df = pd.read_csv(cfg['ratings_file'])
    t_col = cfg['csv_trait_col']
    p_col = cfg['csv_participant_col']
    tgt_col = cfg['csv_target_col']
    r_col = cfg['csv_rating_col']
    n_targets = cfg['n_targets']

    # Consistent participant ordering across all traits (including familiarity)
    all_participants = sorted(df[p_col].unique())

    out = {}
    for trait in cfg['traits']:
        sub = df[df[t_col] == trait]
        wide = (sub.pivot_table(index=p_col, columns=tgt_col,
                                values=r_col, aggfunc='mean')
                   .reindex(index=all_participants,
                            columns=range(1, n_targets + 1)))
        out[trait] = wide.values   # (n_all_participants, 50)
    return out


def load_rdms(cfg: dict) -> tuple:
    """Return (sem_rdm, vis_rdm) each (50, 50)."""
    sem = np.load(cfg['sem_file'])
    vis = np.load(cfg['vis_file'])
    return sem, vis


def load_familiarity_ratings(cfg: dict) -> np.ndarray:
    """Return (n_participants, n_targets) per-perceiver familiarity ratings.
    For CSV format, participant ordering matches load_ratings (full sorted participant list)."""
    if cfg['ratings_format'] == 'npy':
        arr = np.load(cfg['ratings_file'])
        return arr[:, cfg['npy_familiarity_idx'], :]
    df = pd.read_csv(cfg['ratings_file'])
    all_participants = sorted(df[cfg['csv_participant_col']].unique())
    sub = df[df[cfg['csv_trait_col']] == cfg['csv_familiarity_val']]
    wide = (sub.pivot_table(index=cfg['csv_participant_col'],
                            columns=cfg['csv_target_col'],
                            values=cfg['csv_rating_col'], aggfunc='mean')
               .reindex(index=all_participants,
                        columns=range(1, cfg['n_targets'] + 1)))
    return wide.values


if __name__ == '__main__':
    import sys
    sys.path.insert(0, str(__import__('pathlib').Path(__file__).parent))
    from config import DATASETS

    for ds_name, cfg in DATASETS.items():
        print(f"\n=== {ds_name} ===")
        ratings = load_ratings(cfg)
        for trait, arr in ratings.items():
            print(f"  {trait}: {arr.shape}, NaN={np.isnan(arr).sum()}")
        sem, vis = load_rdms(cfg)
        print(f"  sem RDM: {sem.shape}  vis RDM: {vis.shape}")
