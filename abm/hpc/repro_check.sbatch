#!/bin/bash
# Same-seed reproducibility check: rerun an existing run directory into
# <root>/repro_check/<run_id>_rep<task id>, then compare it with <root>/output/<run_id>.
#
#   sbatch --array=1-2 --time=00:30:00 repro_check.sbatch <experiment root> <run_id> [runs dir] [tag]
#
# [runs dir] (default runs) holds the model files to rerun, e.g. a regenerated copy
# made with a newer make_variant.py; [tag] (default rep) names the output folder.
#SBATCH --job-name=abm_repro
#SBATCH --partition=rome
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --output=%x_%A_%a.out

set -euo pipefail
ROOT=$(realpath "$1")
RUN_ID=$2
RUNS=${3:-runs}
TAG=${4:-rep}
OUT="$ROOT/repro_check/${RUN_ID}_${TAG}${SLURM_ARRAY_TASK_ID}"
mkdir -p "$ROOT/repro_check"
echo "[$(date -Is)] rerun $RUNS/$RUN_ID -> $OUT on $(hostname)"

"$HOME/miniforge3/envs/cc3d_460/bin/python" "$ROOT/hpc/run_abm.py" \
    --run-dir "$ROOT/$RUNS/$RUN_ID" --out "$OUT"

# Byte-for-byte comparison of every lattice frame and data file with the production run
cd "$ROOT/output/$RUN_ID"
find LatticeData datafiles -type f | sort | while read -r f; do
    cmp -s "$f" "$OUT/$f" && echo "same $f" || echo "DIFF $f"
done > "$OUT/compare_with_production.txt"
echo "identical files: $(grep -c '^same' "$OUT/compare_with_production.txt")," \
     "different: $(grep -c '^DIFF' "$OUT/compare_with_production.txt" || true)"
