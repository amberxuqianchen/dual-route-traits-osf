"""
Driver for glmmtmb_diagnostics.R over all 20 graded-moderator outcomes.

For each outcome it refits the ladder-selected model (recording explicit
convergence / Hessian diagnostics and the full variance-covariance structure),
refits it without the fixed interaction terms (2-df LRT), and — where the ladder
reduced the structure — refits L0 to test whether the reduction was genuinely
required.

Pair-level inputs come from export_pairs_all_traits.py (`all_{ds}_{trait}.csv`).

Usage:
    python 2_code/supplementary/run_lmm_maximal_diagnostics.py \
        --pairs-dir DIR [--workers 12]
"""

import argparse
import subprocess
import sys
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

import pandas as pd

from config import PIPELINE_DIR, ROOT

LMM_DIR = PIPELINE_DIR / 'lmm_familiarity_maximal'
R_SCRIPT = ROOT / '2_code/supplementary/glmmtmb_diagnostics.R'

ap = argparse.ArgumentParser()
ap.add_argument('--pairs-dir', required=True)
ap.add_argument('--workers', type=int, default=12)
args = ap.parse_args()

pairs_dir = Path(args.pairs_dir)
outdir = LMM_DIR / 'diagnostics'
outdir.mkdir(parents=True, exist_ok=True)

sel = pd.read_csv(LMM_DIR / 'selected_models.csv')
sel = sel[sel['mod'] == 'fam']

lad = pd.read_csv(LMM_DIR / 'full_ladder_all_levels.csv')
lad = lad[lad['mod'] == 'fam'].set_index(['label', 'level'])['secs']


def est_secs(ds, trait, lvl):
    """selected + nointer, plus an L0 refit when the ladder reduced."""
    lab = f'fam_{ds}_{trait}'
    s = 2 * lad[(lab, lvl)]
    return s + (lad[(lab, 'L0')] if lvl != 'L0' else 0)


jobs = [(r['ds'], r['trait'], r['lvl']) for _, r in sel.iterrows()]
# longest first, so wall time is bounded by the longest job rather than by scheduling
jobs.sort(key=lambda j: -est_secs(*j))


def run(job):
    ds, trait, lvl = job
    label = f'fam_{ds}_{trait}'
    pairs = pairs_dir / f'all_{ds}_{trait}.csv'
    prefix = outdir / label
    log = outdir / f'{label}.log'
    with open(log, 'w') as fh:
        p = subprocess.run(['/usr/bin/Rscript', str(R_SCRIPT), str(pairs),
                            str(prefix), label, lvl],
                           stdout=fh, stderr=subprocess.STDOUT)
    print(f'{label:24s} rc={p.returncode}', flush=True)
    return label, p.returncode


with ThreadPoolExecutor(max_workers=args.workers) as ex:
    results = list(ex.map(run, jobs))

bad = [l for l, rc in results if rc != 0]
print(f'\n{len(results) - len(bad)}/{len(results)} outcomes completed')
if bad:
    print('non-zero exit:', ', '.join(bad))

# ── Collate ──────────────────────────────────────────────────────────────────
summ = sorted(outdir.glob('*_summary.csv'))
if summ:
    pd.concat([pd.read_csv(f) for f in summ]).to_csv(
        LMM_DIR / 'diagnostics_summary.csv', index=False)
    print(f'wrote {LMM_DIR / "diagnostics_summary.csv"}')
vc = sorted(outdir.glob('*_varcorr.csv'))
if vc:
    pd.concat([pd.read_csv(f) for f in vc]).to_csv(
        LMM_DIR / 'diagnostics_varcorr.csv', index=False)
    print(f'wrote {LMM_DIR / "diagnostics_varcorr.csv"}')
