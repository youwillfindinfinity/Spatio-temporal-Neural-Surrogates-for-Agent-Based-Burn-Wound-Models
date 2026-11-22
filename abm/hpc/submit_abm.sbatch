#!/bin/bash
# SLURM array: one CompuCell3D run per task, rows taken from manifest.csv.
#
#   sbatch --array=1-16 --time=02:00:00 submit_abm.sbatch <experiment root> <grid> [manifest file]
#
# Task i runs the i-th manifest row with the requested grid size.
#SBATCH --job-name=abm
#SBATCH --partition=rome
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --output=%x_%A_%a.out

set -euo pipefail
ROOT=$(realpath "$1")
GRID=$2
MANIFEST=${3:-manifest.csv}
ROW=$(awk -F, -v g="$GRID" 'NR > 1 && $3 == g' "$ROOT/$MANIFEST" | sed -n "${SLURM_ARRAY_TASK_ID}p")
[ -n "$ROW" ] || { echo "no manifest row $SLURM_ARRAY_TASK_ID for grid $GRID"; exit 2; }
RUN_ID=$(echo "$ROW" | cut -d, -f1)
echo "[$(date -Is)] task $SLURM_ARRAY_TASK_ID -> $RUN_ID on $(hostname)"

"$HOME/miniforge3/envs/cc3d_460/bin/python" "$ROOT/hpc/run_abm.py" \
    --run-dir "$ROOT/runs/$RUN_ID" --out "$ROOT/output/$RUN_ID"
