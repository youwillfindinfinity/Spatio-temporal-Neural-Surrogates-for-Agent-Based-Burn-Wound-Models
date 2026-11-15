#!/bin/bash
#SBATCH --job-name=repro_unet
#SBATCH --partition=gpu_a100
#SBATCH --gpus=1
#SBATCH --cpus-per-task=18
#SBATCH --time=01:00:00
#SBATCH --account=gisr115048
#SBATCH --output=logs/repro_unet_%j.out

module load 2023
module load CUDA/12.1.1 cuDNN/8.9.2.26-CUDA-12.1.1
cd /gpfs/scratch1/shared/jkowalczuk/surrogates_2D
export TF_CPP_MIN_LOG_LEVEL=2
python -c "import tensorflow as tf; print('GPU:', tf.config.list_physical_devices('GPU'))"
for G in 100 250; do
  for C in il10 il8; do
    python scripts/repro_unet.py --grid $G --cytokine $C --seed 42
  done
done
