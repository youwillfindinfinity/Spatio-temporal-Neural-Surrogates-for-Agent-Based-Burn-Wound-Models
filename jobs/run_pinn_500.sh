#!/bin/bash
#SBATCH --job-name=pinn_500
#SBATCH --partition=gpu_a100
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --gpus=1
#SBATCH --mem=64G
#SBATCH --time=24:00:00
#SBATCH --array=0-5
#SBATCH --output=jobs/logs/pinn_500_%a.out
#SBATCH --error=jobs/logs/pinn_500_%a.err

CYTOKINES=(il8 il8 il8 il10 il10 il10)
SEEDS=(1 42 100 1 42 100)
CYT=${CYTOKINES[$SLURM_ARRAY_TASK_ID]}
SEED=${SEEDS[$SLURM_ARRAY_TASK_ID]}

SCAN_ITER=${SCAN_ITER:-0}
DATA_DIR="/gpfs/home3/rbumbuc1/codes/students/surrogates/data/ICCS2026_v2/scan_iteration_${SCAN_ITER}/preprocessed/500x500"

echo "Job:      $SLURM_JOB_ID (array $SLURM_ARRAY_TASK_ID)"
echo "Model:    pinn 500x500"
echo "Cytokine: $CYT  |  Seed: $SEED"
echo "Scan:     iteration $SCAN_ITER"
echo "Data:     $DATA_DIR"
echo "Node:     $(hostname)"

module purge
module load 2023
module load Python/3.11.3-GCCcore-12.3.0
module load CUDA/12.1.1
source /gpfs/home3/rbumbuc1/codes/students/surrogates/.venv/bin/activate

export TF_USE_LEGACY_KERAS=1

cd /gpfs/home3/rbumbuc1/codes/students/surrogates
mkdir -p models/pinn jobs/logs

if [ ! -f "$DATA_DIR/metadata.json" ]; then
    echo "ERROR: preprocessed data not found at $DATA_DIR"
    echo "       Run: python3 scripts/run_preprocessing.py data/ICCS2026_v2/scan_iteration_${SCAN_ITER}"
    exit 1
fi

RESULT_FILE="models/pinn/res_${CYT}_500_${SEED}.json"
if [ -f "$RESULT_FILE" ]; then
    echo "Result already exists at $RESULT_FILE — skipping."
    cat "$RESULT_FILE"
    exit 0
fi

echo "Starting training at $(date)"
/gpfs/home3/rbumbuc1/codes/students/surrogates/.venv/bin/python scripts/pinn/all_500.py --cytokine $CYT --seed $SEED --data-dir "$DATA_DIR"

EXIT_CODE=$?
echo "Finished at $(date) | exit code: $EXIT_CODE"
if [ $EXIT_CODE -eq 0 ]; then
    echo "Result saved: models/pinn/res_${CYT}_500_${SEED}.json"
    cat models/pinn/res_${CYT}_500_${SEED}.json
else
    echo "FAILED with exit code $EXIT_CODE"
    sacct -j $SLURM_JOB_ID --format=JobID,MaxRSS,State,ExitCode 2>/dev/null || true
fi
exit $EXIT_CODE
