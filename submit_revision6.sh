#!/bin/bash
# Sixth batch (new ABM data): PINN with a residual, sources and boundary condition that match the ABM,
# strict and softened form, three seeds; and the re-implemented ABM cytokine update as a reference.
REPO=$HOME/surrogates_2D
ACCOUNT=gisr115048
cd $REPO || { echo "missing $REPO"; exit 1; }
[ -d "$REPO/preprocessed_abm" ] || { echo "missing preprocessed_abm"; exit 1; }
python -c "import deepxde, tensorflow_probability" 2>/dev/null || { echo "DeepXDE not importable - run: pip install --user deepxde==1.15.0 tensorflow-probability==0.23.0"; exit 1; }
mkdir -p logs

submit () {   # name, time limit, command
  sbatch --parsable --job-name="$1" --partition=gpu_a100 --gpus=1 --cpus-per-task=18 --time="$2" \
         --account=$ACCOUNT --output="logs/abm_$1_%j.out" <<EOT
#!/bin/bash
module load 2023
module load CUDA/12.1.1 cuDNN/8.9.2.26-CUDA-12.1.1
cd $REPO
export TF_CPP_MIN_LOG_LEVEL=2 REV_DATA_ROOT=$REPO/preprocessed_abm REV_OUT=rev_abm
$3
EOT
}

for G in 100 250; do
  for V in strict soft; do
    echo "pinn_${V}_$G  $(submit pinn_${V}_$G 14:00:00 "python scripts/rev/run_pinn_abm.py --variant $V --grid $G")"
  done
done
if ! squeue -u $USER -h -o %j | grep -q "^physics$" && [ ! -f outputs/rev_abm/physics_step.txt ]; then
  echo "physics  $(submit physics 02:00:00 "python scripts/rev/physics_step.py")"
fi
echo
squeue -u $USER
