#!/bin/bash
# Second batch: remaining DeepONet seeds, look-back sweep, PINN with strict vs softened residual.
REPO=/gpfs/scratch1/shared/jkowalczuk/surrogates_2D
ACCOUNT=gisr115048
cd $REPO || exit 1
mkdir -p logs

echo "Checking DeepXDE (needed for the PINN jobs)..."
python -c "import deepxde, tensorflow_probability" 2>/dev/null || pip install --user -q deepxde "tensorflow-probability==0.23.0"
python -c "import deepxde, tensorflow_probability; print('DeepXDE', deepxde.__version__, 'ok')" 2>&1 | tail -1

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

QUEUED=$(squeue -u $USER -h -o %j)
for G in 100 250; do
  if [ $G = 100 ]; then TD=03:00:00; TL=04:00:00; TP=06:00:00; else TD=10:00:00; TL=10:00:00; TP=10:00:00; fi
  for V in base no_branch no_gate no_fields; do
    if echo "$QUEUED" | grep -q "don2_${V}_$G"; then echo "don2_${V}_$G already in the queue - not submitted again"; continue; fi
    submit don2_${V}_$G $TD "python scripts/rev/run_deeponet.py --grid $G --variants $V --seeds 1 100"
  done
  submit lookback_$G $TL "python scripts/rev/run_lookback.py --grid $G"
  for V in soft strict; do
    for C in il10 il8; do
      submit pinn_${V}_${C}_$G $TP "python scripts/rev/run_pinn.py --variant $V --grid $G --cytokines $C"
    done
  done
done
echo
squeue -u $USER
