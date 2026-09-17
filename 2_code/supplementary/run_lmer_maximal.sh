#!/usr/bin/env bash
# Parallel driver for the lme4/lmerTest maximal models.
#
# lmer's optimiser is single-threaded and the sparse Cholesky underneath does
# not thread usefully, so within-fit parallelism buys nothing -- BLAS is pinned
# to one thread deliberately.  All the parallelism is ACROSS outcomes, one core
# per fit, which is embarrassingly parallel.
#
# Each fit writes its own CSV, so the driver is idempotent: outcomes that are
# already finished, or that have a fit currently in flight, are skipped.  A run
# that dies partway can simply be relaunched.
#
# Budget roughly 1 GB and 20-90 min per fit (glmmTMB does the same structure in
# ~15 min; lme4 is markedly slower on 63 covariance parameters).
#
# Usage: bash 2_code/supplementary/run_lmer_maximal.sh <workdir> [JOBS]
#   workdir must contain the all_{ds}_{outcome}.csv exports.

set -uo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
W="${1:?usage: run_lmer_maximal.sh <workdir> [jobs]}"
JOBS="${2:-18}"
OUT="$W/lmer"
mkdir -p "$OUT"

# lmer does not benefit from threaded BLAS; keep one core per fit.
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1

JL="$OUT/jobs.txt"; : > "$JL"
skipped_done=0; skipped_running=0
for f in "$W"/all_*.csv; do
    b=$(basename "$f" .csv); b=${b#all_}
    if [[ -f "$OUT/${b}.csv" ]]; then
        skipped_done=$((skipped_done + 1)); continue
    fi
    # avoid double-launching an outcome that another invocation is already fitting
    if pgrep -f "lmer_maximal.R .*/${b}\.csv" > /dev/null 2>&1; then
        skipped_running=$((skipped_running + 1)); continue
    fi
    printf '%s\t%s\t%s\t%s\n' "$f" "$OUT/${b}.csv" "$b" "$OUT/${b}.log" >> "$JL"
done

n=$(wc -l < "$JL")
echo "queued $n | already finished $skipped_done | in flight $skipped_running | $JOBS concurrent"
[[ "$n" -eq 0 ]] && { echo "nothing to do"; exit 0; }

xargs -a "$JL" -d '\n' -P "$JOBS" -I{} bash -c '
    IFS=$'"'"'\t'"'"' read -r inp outp lab log <<< "{}"
    start=$(date +%s)
    if /usr/bin/Rscript '"$ROOT"'/2_code/supplementary/lmer_maximal.R "$inp" "$outp" "$lab" > "$log" 2>&1; then
        ddf=$(grep -oE "\[(Satterthwaite|Wald)\]" "$log" | tail -1)
        sing=$(grep -oE "singular=(TRUE|FALSE)" "$log" | tail -1)
        echo "OK   $lab  $(( $(date +%s) - start ))s  $ddf  $sing"
    else
        echo "FAIL $lab"; tail -3 "$log" | sed "s/^/       /"
    fi
'

echo
echo "finished: $(ls -1 "$OUT"/*.csv 2>/dev/null | grep -v jobs | wc -l) outcome files in $OUT"
