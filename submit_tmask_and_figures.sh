#!/bin/bash
#   tmask_*   models that also receive the cell masks at the target time (4 architectures, both forms)
#   stalstm   the one configuration that ran out of GPU memory in submit_main.sh (skipped if done)
#   post      tables and figures; starts when all other jobs have ended
REPO=$HOME/surrogates_2D
ACCOUNT=your_account
cd $REPO || { echo "missing $REPO"; exit 1; }
[ -d "$REPO/preprocessed_abm" ] || { echo "missing preprocessed_abm"; exit 1; }
mkdir -p logs

submit () {   # name, time limit, dependency option ("" or full --dependency=...), command
  sbatch --parsable --job-name="$1" --partition=gpu_a100 --gpus=1 --cpus-per-task=18 --time="$2" $3 \
         --account=$ACCOUNT --output="logs/abm_$1_%j.out" <<EOT
#!/bin/bash
module load 2023
module load CUDA/12.1.1 cuDNN/8.9.2.26-CUDA-12.1.1
cd $REPO
export TF_CPP_MIN_LOG_LEVEL=2 REV_DATA_ROOT=$REPO/preprocessed_abm REV_OUT=rev_abm
$4
EOT
}

IDS=""
for G in 100 250; do
  if [ $G = 100 ]; then T=08:00:00; else T=20:00:00; fi
  for M in unet convlstm fno pointwise; do
    J=$(submit tmask_${M}_$G $T "" "python scripts/rev/run_main.py --model $M --grid $G --target-masks")
    echo "tmask_${M}_$G  $J"; IDS="$IDS:$J"
  done
done
J=$(submit stalstm_fix 01:30:00 "" "python scripts/rev/run_main.py --model stalstm --grid 250 --forms direct --cytokines il8")
echo "stalstm_fix  $J"; IDS="$IDS:$J"

J=$(submit post 03:00:00 "--dependency=afterany$IDS" "python scripts/rev/aggregate.py > /dev/null
python scripts/rev/aggregate_transfer.py > /dev/null
python scripts/rev/post_runs.py")
echo "post  $J"
echo
squeue -u $USER
