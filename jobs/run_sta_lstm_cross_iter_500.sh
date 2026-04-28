#!/bin/bash
#SBATCH --job-name=sta_lstm_cross_iter
#SBATCH --partition=rome
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=12:00:00
#SBATCH --array=0-5
#SBATCH --output=jobs/logs/sta_lstm_cross_iter_%a.out
#SBATCH --error=jobs/logs/sta_lstm_cross_iter_%a.err

CYTOKINES=(il8 il8 il8 il10 il10 il10)
SEEDS=(1 42 100 1 42 100)
CYT=${CYTOKINES[$SLURM_ARRAY_TASK_ID]}
SEED=${SEEDS[$SLURM_ARRAY_TASK_ID]}

BASE="/gpfs/home3/rbumbuc1/codes/students/surrogates/data/ICCS2026_v2"
ITER0_DIR="${BASE}/scan_iteration_0/preprocessed/500x500"
ITER1_DIR="${BASE}/scan_iteration_1/preprocessed/500x500"

echo "Job:      $SLURM_JOB_ID (array $SLURM_ARRAY_TASK_ID)"
echo "Model:    sta_lstm cross-iteration 500x500"
echo "Cytokine: $CYT  |  Seed: $SEED"
echo "iter_0:   $ITER0_DIR"
echo "iter_1:   $ITER1_DIR"
echo "Split:    139 train / 20 val / 39 test (all from iter_1 tail)"
echo "Node:     $(hostname)"

module purge
module load 2023
module load Python/3.11.3-GCCcore-12.3.0
module load CUDA/12.1.1
source /gpfs/home3/rbumbuc1/codes/students/surrogates/.venv/bin/activate

cd /gpfs/home3/rbumbuc1/codes/students/surrogates
mkdir -p models/sta_lstm jobs/logs

for DIR in "$ITER0_DIR" "$ITER1_DIR"; do
    if [ ! -f "$DIR/metadata.json" ]; then
        echo "ERROR: preprocessed data not found at $DIR"
        exit 1
    fi
done

echo "Starting at $(date)"
/gpfs/home3/rbumbuc1/codes/students/surrogates/.venv/bin/python scripts/sta_lstm/cross_iter_500.py \
    --cytokine  $CYT  \
    --seed      $SEED \
    --iter0-dir "$ITER0_DIR" \
    --iter1-dir "$ITER1_DIR"

EXIT_CODE=$?
echo "Finished at $(date) | exit code: $EXIT_CODE"
OUT="models/sta_lstm/res_cross_iter_${CYT}_500_${SEED}.json"
if [ $EXIT_CODE -eq 0 ]; then
    echo "Result saved: $OUT"
    cat "$OUT"
else
    echo "FAILED with exit code $EXIT_CODE"
    sacct -j $SLURM_JOB_ID --format=JobID,MaxRSS,State,ExitCode 2>/dev/null || true
fi
exit $EXIT_CODE
