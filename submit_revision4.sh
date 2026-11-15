#!/bin/bash
# Fourth batch: everything re-run on the revision ABM runs (IL-10 from M2).
# Training on E3 seed 1; frozen-weight tests on E3 seeds 2-5, E2a and E2b.
REPO=$HOME/surrogates_2D
RUNS=$HOME/abm_revision/revision_cmpb/output
ACCOUNT=gisr115048
cd $REPO || { echo "missing $REPO"; exit 1; }
[ -d "$RUNS" ] || { echo "missing $RUNS"; exit 1; }
mkdir -p logs

submit () {   # name, time limit, dependency ("" or job id), command
  local dep=""; [ -n "$3" ] && dep="--dependency=afterok:$3"
  sbatch --parsable --job-name="$1" --partition=gpu_a100 --gpus=1 --cpus-per-task=18 --time="$2" $dep \
         --account=$ACCOUNT --output="logs/abm_$1_%j.out" <<EOT
#!/bin/bash
module load 2023
module load CUDA/12.1.1 cuDNN/8.9.2.26-CUDA-12.1.1
cd $REPO
export TF_CPP_MIN_LOG_LEVEL=2 REV_DATA_ROOT=$REPO/preprocessed_abm REV_OUT=rev_abm
$4
EOT
}

PREP=$(submit prep 03:00:00 "" "python scripts/rev/preprocess_runs.py --runs-root $RUNS --out $REPO/preprocessed_abm
python scripts/rev/abm_variability.py")
echo "prep            $PREP"

for G in 100 250; do
  if [ $G = 100 ]; then TM=08:00:00; TD=06:00:00; else TM=20:00:00; TD=20:00:00; fi
  for M in unet convlstm fno stalstm pointwise; do
    J=$(submit main_${M}_$G $TM $PREP "python scripts/rev/run_main.py --model $M --grid $G")
    echo "main_${M}_$G  $J"
    [ $M = unet ] && UNET=$J
  done
  # auxiliary U-Net experiments use the hyperparameters found by main_unet, so they wait for it
  J=$(submit aux_$G $TM $UNET "python scripts/rev/run_multitraj.py --grid $G
python scripts/rev/run_grid_models.py --model unet --grid $G --variants weighted ntrain35 ntrain52
python scripts/rev/run_lookback.py --grid $G")
  echo "aux_$G  $J"
  for V in base no_branch no_gate no_fields; do
    J=$(submit don_${V}_$G $TD $PREP "python scripts/rev/run_deeponet.py --grid $G --variants $V --seeds 42 1 100")
    echo "don_${V}_$G  $J"
  done
done
echo
squeue -u $USER
