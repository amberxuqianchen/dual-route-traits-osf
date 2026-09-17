"""
Target (celebrity) stimulus characteristics: gender, race, occupation.

Gender and occupation already live in 1_data/{ds}/celebrity_{genders,occupations}.csv.
Race is coded only for the U.S. set, in the source repo's CelebA identity
demographics; this copies it into 1_data/us/celebrity_race.csv so the analysis
repo is self-contained, matching the convention of the other two files.  The
Chinese set is all Chinese celebrities, so race does not vary there and is not
coded.

Writes 1_data/us/celebrity_race.csv and 3_pipeline/target_demographics.csv
(one row per target, both samples).

Usage: python 2_code/supplementary/target_demographics.py
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

import pandas as pd

from config import ROOT, PIPELINE_DIR

US_SRC = ROOT.parent / 'face-bert/data/CelebA_ID_Demographics.csv'

# 1 = Caucasian, 2 = Black, 3 = Biracial.  The source encodes the same
# partition twice, as `Caucasian_Africa_Biracial` and `White_Black_Mix`; the two
# columns are identical row for row, so either serves.
RACE = {1: 'White', 2: 'Black', 3: 'Biracial'}

# the source spells two names differently from 1_data/us/celebrity_genders.csv
NAME_FIX = {'Conan O_Brien': "Conan O'Brien", 'John Krasinki': 'John Krasinski'}

# ── U.S. race, copied into 1_data ────────────────────────────────────────────
src = pd.read_csv(US_SRC)
src['name'] = (src['IDnames'].str.replace(r'^ID_\d+_', '', regex=True)
                             .replace(NAME_FIX))
src['race'] = src['Caucasian_Africa_Biracial'].map(RACE)

us_g = pd.read_csv(ROOT / '1_data/us/celebrity_genders.csv')
race = us_g[['celeb_id', 'name']].merge(src[['name', 'race']], on='name', how='left')
assert race['race'].notna().all(), race.loc[race['race'].isna(), 'name'].tolist()
race.to_csv(ROOT / '1_data/us/celebrity_race.csv', index=False)
print(f"wrote 1_data/us/celebrity_race.csv ({len(race)} rows)")

# ── Combined per-target table ────────────────────────────────────────────────
rows = []
for ds in ('us', 'cn'):
    g = pd.read_csv(ROOT / f'1_data/{ds}/celebrity_genders.csv')
    o = pd.read_csv(ROOT / f'1_data/{ds}/celebrity_occupations.csv')
    d = g.merge(o[['celeb_id', 'occupation']], on='celeb_id')
    d['race'] = (d['celeb_id'].map(dict(zip(race['celeb_id'], race['race'])))
                 if ds == 'us' else 'Chinese')
    d.insert(0, 'dataset', ds)
    rows.append(d)
out = pd.concat(rows, ignore_index=True)
out.to_csv(PIPELINE_DIR / 'target_demographics.csv', index=False)
print(f"wrote {PIPELINE_DIR / 'target_demographics.csv'} ({len(out)} rows)")

for ds, d in out.groupby('dataset', sort=False):
    print(f"\n=== {ds.upper()} (n = {len(d)}) ===")
    for col in ('gender', 'race', 'occupation'):
        print(f"  {col}: {d[col].value_counts().to_dict()}")
