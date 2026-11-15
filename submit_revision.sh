#!/bin/bash
# Submits all revision experiments as separate GPU jobs. Finished runs are skipped on re-submission.
REPO=/gpfs/scratch1/shared/jkowalczuk/surrogates_2D
ACCOUNT=gisr115048
cd $REPO || exit 1
mkdir -p logs

submit () {   # name, time limit, command
  sbatch --job-name="$1" --partition=gpu_a100 --gpus=1 --cpus-per-task=18 --time="$2" \
         --account=$ACCOUNT --output="logs/rev_$1_%j.out" <<EOT
#!/bin/bash
module load 2023
module load CUDA/12.1.1 cuDNN/8.9.2.26-CUDA-12.1.1
cd $REPO
export TF_CPP_MIN_LOG_LEVEL=2
$3
EOT
}

for G in 100 250; do
  if [ $G = 100 ]; then T1=02:00:00; T2=02:00:00; else T1=05:00:00; T2=06:00:00; fi
  submit unet_$G  $T1 "python scripts/rev/run_grid_models.py --model unet --grid $G"
  submit small_$G $T1 "python scripts/rev/run_grid_models.py --model stalstm --grid $G
python scripts/rev/run_grid_models.py --model convlstm --grid $G
python scripts/rev/run_grid_models.py --model fno --grid $G"
  for V in base no_branch no_gate no_fields; do
    submit don_${V}_$G $T2 "python scripts/rev/run_deeponet.py --grid $G --variants $V"
  done
done
echo
squeue -u $USER
