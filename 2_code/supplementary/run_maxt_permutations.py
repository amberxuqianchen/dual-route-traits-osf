"""
Permutation max-T for the maximal-random-effects familiarity LMMs.

Family: within dataset and channel, over the individual traits (CN 10, US 8).
The trait space is a single outcome by construction and is excluded.

Each permuted fit uses the random-effects level selected for the observed fit, so
the null is generated under the same specification as the observed statistic.
Cold start throughout.

NOT RUN.  Superseded by Holm correction (see 5_doc/HANDOFF.md).  Kept for the
record and in case a reviewer asks for a permutation FWER.  If you resume it,
read the two paragraphs below first -- both were measured, not estimated.

Cost.  The original "~12 days on 20 workers" estimate was wrong because 20-way
parallelism is not achievable here.  A single L0 fit is ~22 min, so B=1000 over
18 outcomes is ~6,600 core-h: >14 days even at a sustained 20 wide, and longer
at the concurrency this machine can actually hold.

Memory.  Each L0 fit spikes to ~18 GB for ~1 min during setup, then settles to
~4.3 GB for the remainder.  Twenty workers started at once spike together --
~360 GB against 251 GB of RAM with no swap -- so the kernel SIGKILLs them and
they write nothing, which is why an earlier 14 h run produced zero output.  The
driver records the return code but nothing else, and a SIGKILLed R process never
flushes its log, so the failure is silent.  Fixes, in preference order: stagger
worker starts by ~4 min so the spikes do not coincide (steady state is only
~86 GB at 20 wide), or cap --workers at ceil(RAM / 18 GB).

The run is resumable at permutation granularity -- re-invoking with the same
arguments skips completed permutations -- and `collate_maxt.py` can be run at any
time on partial results.

Usage:
    python 2_code/supplementary/run_maxt_permutations.py \
        --pairs-dir DIR [-B 1000] [--workers 18] [--chunk 50]

Keep --workers <= the number of outcomes (18), otherwise the extra workers pick
up a second chunk of an outcome already in flight and two processes append to the
same output file.
"""

import argparse
import subprocess
import sys
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

import numpy as np
import pandas as pd

from config import DATASETS, PIPELINE_DIR, ROOT
from data_loader import load_familiarity_ratings

LMM_DIR = PIPELINE_DIR / 'lmm_familiarity_maximal'
R_SCRIPT = ROOT / '2_code/supplementary/glmmtmb_permfit.R'
SEED = 20260812          # pi depends on (dataset seed, b) only, never on trait

ap = argparse.ArgumentParser()
ap.add_argument('--pairs-dir', required=True)
ap.add_argument('-B', type=int, default=1000)
ap.add_argument('--workers', type=int, default=20)
ap.add_argument('--chunk', type=int, default=50)
args = ap.parse_args()

pairs_dir = Path(args.pairs_dir)
outdir = LMM_DIR / 'maxt'
outdir.mkdir(parents=True, exist_ok=True)

# ── Familiarity matrices: participants x targets, the source of the permutation
fam_files = {}
for ds, cfg in DATASETS.items():
    f = outdir / f'fam_{ds}.csv'
    if not f.exists():
        fam = load_familiarity_ratings(cfg)
        np.savetxt(f, fam, delimiter=',')
        print(f'wrote {f}  shape={fam.shape}')
    fam_files[ds] = f

# ── Tasks: (dataset, trait, permutation chunk), trait space excluded
sel = pd.read_csv(LMM_DIR / 'selected_models.csv')
sel = sel[(sel['mod'] == 'fam') & (sel['trait'] != 'traitspace')]

# chunk-major: the first len(sel) tasks are one per outcome, so concurrent
# workers hit different outcomes -- different output files, and the permutation
# index advances across the whole family at once, which is what collate_maxt.py
# needs to form a family max on partial results
tasks = []
for b0 in range(1, args.B + 1, args.chunk):
    for _, r in sel.iterrows():
        tasks.append((r['ds'], r['trait'], r['lvl'], b0,
                      min(b0 + args.chunk - 1, args.B)))
print(f'{len(tasks)} chunks ({len(sel)} outcomes x {args.B} permutations, '
      f'chunk={args.chunk}), {args.workers} workers')


def run(task):
    ds, trait, lvl, b0, b1 = task
    out = outdir / f'perm_{ds}_{trait}.csv'
    log = outdir / f'perm_{ds}_{trait}.log'
    # one seed per dataset, so every trait shares pi within a permutation
    seed = SEED + (0 if ds == 'cn' else 500000)
    with open(log, 'a') as fh:
        p = subprocess.run(
            ['/usr/bin/Rscript', str(R_SCRIPT),
             str(pairs_dir / f'all_{ds}_{trait}.csv'), str(fam_files[ds]),
             lvl, str(out), str(b0), str(b1), str(seed)],
            stdout=fh, stderr=subprocess.STDOUT)
    print(f'{ds}_{trait} b={b0}-{b1} rc={p.returncode}', flush=True)
    return p.returncode


with ThreadPoolExecutor(max_workers=args.workers) as ex:
    rcs = list(ex.map(run, tasks))

print(f'\n{sum(1 for r in rcs if r == 0)}/{len(rcs)} chunks exited 0')
print(f'results in {outdir}/perm_*.csv -- run collate_maxt.py to build the null')
