#!/bin/bash
# Third batch: main experiments. Five architectures x {increment, direct}, each with its own
# 20-trial Optuna search (seed 42), then seeds 42, 1, 100. One job per architecture and grid.
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
  if [ $G = 100 ]; then T=05:00:00; else T=12:00:00; fi
  for M in unet convlstm fno stalstm pointwise; do
    submit main_${M}_$G $T "python scripts/rev/run_main.py --model $M --grid $G"
  done
done
echo
squeue -u $USER
