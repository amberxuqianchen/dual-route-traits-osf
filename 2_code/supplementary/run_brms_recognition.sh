#!/usr/bin/env bash
# Parallel driver for the Bayesian recognition models.
#
# 18 independent model fits (each 4 chains).  Jobs are run concurrently with
# xargs -P; each job's 4 chains use 4 cores, so JOBS=6 keeps the machine at ~24
# of 32 cores and leaves headroom.  Every fit writes its own .rds/_summary.csv,
# so a job that fails or is killed can be re-run alone without redoing the rest.
#
# Usage: bash 2_code/supplementary/run_brms_recognition.sh <workdir> [JOBS] [CHAINS] [ITER] [THREADS]

set -uo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
WORK="${1:?usage: run_brms_recognition.sh <workdir> [jobs] [chains] [iter]}"
JOBS="${2:-4}"
CHAINS="${3:-4}"
ITER="${4:-1000}"
THREADS="${5:-2}"

# Override PY/RS if your interpreters live elsewhere.
PY="${PY:-$(command -v python)}"
RS=/usr/bin/Rscript

mkdir -p "$WORK/fits" "$WORK/logs"

# Keep BLAS single-threaded; parallelism comes from chains and concurrent jobs.
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1

echo "[1/3] exporting pair data"
"$PY" "$ROOT/2_code/supplementary/export_pairs_for_brms.py" \
    --outdir "$WORK" \
    --traits cn:feminine cn:competent us:feminine us:strong \
    || { echo "export failed"; exit 1; }

echo "[2/3] building job list"
JOBLIST="$WORK/jobs.txt"
: > "$JOBLIST"
for spec in cn:feminine cn:competent us:feminine us:strong; do
    ds="${spec%%:*}"; tr="${spec##*:}"
    for v in twofit_rec twofit_unrec joint fammean famhigh; do
        # famhigh is not estimable for US: familiarity is binary, so both>5
        # implies both==7 and the moderator has zero variance.
        [[ "$v" == "famhigh" && "$ds" == "us" ]] && continue
        case "$v" in
            twofit_rec|twofit_unrec|joint) inp="$WORK/pairs_bn_${ds}_${tr}.csv" ;;
            *)                             inp="$WORK/pairs_all_${ds}_${tr}.csv" ;;
        esac
        echo -e "${inp}\t${v}\t${WORK}/fits/${ds}_${tr}_${v}\t${WORK}/logs/${ds}_${tr}_${v}.log" >> "$JOBLIST"
    done
done
echo "  $(wc -l < "$JOBLIST") jobs, $JOBS at a time ($CHAINS chains x $THREADS threads = $((JOBS*CHAINS*THREADS)) cores)"

echo "[3/3] fitting"
# shellcheck disable=SC2016
xargs -a "$JOBLIST" -d '\n' -P "$JOBS" -I{} bash -c '
    IFS=$'"'"'\t'"'"' read -r inp variant outpre log <<< "{}"
    if [[ -f "${outpre}_summary.csv" ]]; then
        echo "SKIP  $(basename "$outpre") (already fitted)"; exit 0
    fi
    start=$(date +%s)
    if '"$RS"' '"$ROOT"'/2_code/supplementary/brms_recognition.R \
            "$inp" "$variant" "$outpre" '"$CHAINS"' '"$ITER"' '"$THREADS"' > "$log" 2>&1; then
        echo "OK    $(basename "$outpre")  $(( $(date +%s) - start ))s"
    else
        echo "FAIL  $(basename "$outpre")  (see $log)"
        tail -3 "$log" | sed "s/^/        /"
    fi
'

echo
echo "fitted:"
ls -1 "$WORK/fits"/*_summary.csv 2>/dev/null | wc -l
echo "failures:"
grep -L "done ->" "$WORK/logs"/*.log 2>/dev/null || echo "  none"
