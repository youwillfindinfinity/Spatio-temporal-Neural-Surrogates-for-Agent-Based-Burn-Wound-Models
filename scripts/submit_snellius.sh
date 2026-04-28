#!/bin/bash

set -euo pipefail

SCAN_ITER=${1:-0}

ENV_FILE="$(dirname "$0")/../.env"
if [ ! -f "$ENV_FILE" ]; then
    echo "ERROR: .env not found at $ENV_FILE"
    echo "       Copy .env.example to .env and fill in your credentials."
    exit 1
fi

set -a
source "$ENV_FILE"
set +a

for VAR in SNELLIUS_USER SNELLIUS_HOST SNELLIUS_PASS; do
    if [ -z "${!VAR:-}" ]; then
        echo "ERROR: $VAR is not set in .env"
        exit 1
    fi
done

REMOTE_DIR="${SNELLIUS_PROJECT_DIR:-\$HOME/codes/students/surrogates}"

export SSHPASS="$SNELLIUS_PASS"
unset SNELLIUS_PASS # rem original copy

SSH_OPTS="-o StrictHostKeyChecking=no -o BatchMode=no"
REMOTE="${SNELLIUS_USER}@${SNELLIUS_HOST}"

_ssh() { sshpass -e ssh $SSH_OPTS "$REMOTE" "$@"; }
_scp() { sshpass -e scp $SSH_OPTS "$@"; }
_rsync() { sshpass -e rsync -az --no-perms -e "ssh $SSH_OPTS" "$@"; }

echo "Connecting to $SNELLIUS_USER@$SNELLIUS_HOST ..."

if ! command -v sshpass &>/dev/null; then
    echo "ERROR: sshpass not installed. Run: sudo apt install sshpass"
    exit 1
fi

echo ""
echo "Syncing code to Snellius"
_rsync \
    --exclude='.git' \
    --exclude='__pycache__' \
    --exclude='*.pyc' \
    --exclude='.env' \
    --exclude='data/' \
    --exclude='models/' \
    --exclude='*.npy' \
    --exclude='*.h5' \
    --exclude='jobs/logs/' \
    ./ "${REMOTE}:${REMOTE_DIR}/"
echo "Code synced."

PREPROC_LOCAL="data/ICCS2026_v2/scan_iteration_${SCAN_ITER}/preprocessed/500x500"
PREPROC_REMOTE="${REMOTE_DIR}/data/ICCS2026_v2/scan_iteration_${SCAN_ITER}/preprocessed/500x500"

if [ -d "$PREPROC_LOCAL" ]; then
    echo ""
    echo "Syncing preprocessed data for scan_iteration_${SCAN_ITER}"
    _ssh "mkdir -p ${PREPROC_REMOTE}"
    _rsync "$PREPROC_LOCAL/" "${REMOTE}:${PREPROC_REMOTE}/"
    echo "Data synced."
else
    echo "WARNING: $PREPROC_LOCAL not found locally - skipping data sync."
    echo "Make sure the data exists on Snellius before jobs run."
fi

echo ""
echo "Preparing remote directories"
_ssh "cd ${REMOTE_DIR} && mkdir -p jobs/logs models/deeponet_h models/pinn models/pi_deeponet models/sta_lstm models/unet"

echo ""
echo "Submitting jobs (scan_iteration_${SCAN_ITER})"
_ssh "cd ${REMOTE_DIR} && bash jobs/submit_all_500.sh ${SCAN_ITER}"

echo ""
echo "Done. Monitor jobs with:"
echo "  sshpass -e ssh $SSH_OPTS ${REMOTE} 'squeue -u ${SNELLIUS_USER}'"
