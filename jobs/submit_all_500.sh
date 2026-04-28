#!/bin/bash
# Submit all 500x500 jobs to Snellius in recommended order.
# Usage:
#   bash jobs/submit_all_500.sh [SCAN_ITER]
#   SCAN_ITER defaults to 0 (scan_iteration_0)
#
# Example - submit iteration 1:
#   bash jobs/submit_all_500.sh 1

set -e
cd $HOME/codes/students/surrogates

SCAN_ITER=${1:-0}
export SCAN_ITER

echo "Submitting 500x500 jobs (scan_iteration_${SCAN_ITER})"

JID_UNET=$(sbatch --parsable --export=ALL,SCAN_ITER=$SCAN_ITER jobs/run_unet_500.sh)
echo "UNet:        $JID_UNET"

JID_DON=$(sbatch --parsable --export=ALL,SCAN_ITER=$SCAN_ITER jobs/run_deeponet_500.sh)
echo "DeepONet:    $JID_DON"

JID_LSTM=$(sbatch --parsable --export=ALL,SCAN_ITER=$SCAN_ITER jobs/run_sta_lstm_500.sh)
echo "STA-LSTM:    $JID_LSTM"

JID_PIDON=$(sbatch --parsable --export=ALL,SCAN_ITER=$SCAN_ITER jobs/run_pi_deeponet_500.sh)
echo "PI-DeepONet: $JID_PIDON"

JID_PINN=$(sbatch --parsable --export=ALL,SCAN_ITER=$SCAN_ITER jobs/run_pinn_500.sh)
echo "PINN:        $JID_PINN"

echo ""
echo "All submitted for scan_iteration_${SCAN_ITER}. Monitor with:"
echo "  squeue -u \$USER"
echo "  tail -f jobs/logs/deeponet_500_0.out"
