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

for VAR in SNELLIUS_USER SNELLIUS_HOST SNELLIUS_PASS SNELLIUS_PROJECT_DIR; do
    if [ -z "${!VAR:-}" ]; then
        echo "ERROR: $VAR is not set in .env"
        exit 1
    fi
done

REMOTE_DIR="${SNELLIUS_PROJECT_DIR}"
# Escape spaces for rsync remote path
REMOTE_DIR_ESC="${REMOTE_DIR// /\\ }"

export SSHPASS="$SNELLIUS_PASS"
unset SNELLIUS_PASS # clear plain-text password from environment

SSH_OPTS="-o StrictHostKeyChecking=no -o BatchMode=no"
REMOTE="${SNELLIUS_USER}@${SNELLIUS_HOST}"

_ssh()   { sshpass -e ssh   $SSH_OPTS "$REMOTE" "$@"; }
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
    --exclude='models/' \
    --exclude='*.npy' \
    --exclude='*.h5' \
    ./ "${REMOTE}:${REMOTE_DIR_ESC}/"
echo "Code synced."

# Sync preprocessed 500x500 data if available
PREPROC_LOCAL="preprocessed/500x500"
if [ -d "$PREPROC_LOCAL" ]; then
    echo ""
    echo "Syncing preprocessed/500x500 data"
    _ssh "mkdir -p '${REMOTE_DIR}/preprocessed/500x500'"
    _rsync "$PREPROC_LOCAL/" "${REMOTE}:${REMOTE_DIR_ESC}/preprocessed/500x500/"
    echo "Data synced."
else
    echo "WARNING: $PREPROC_LOCAL not found locally - skipping data sync."
    echo "         Make sure preprocessed data exists on Snellius before jobs run."
fi

echo ""
echo "Preparing remote directories"
_ssh "
    mkdir -p \
        '${REMOTE_DIR}/models/deeponet_h' \
        '${REMOTE_DIR}/models/pinn' \
        '${REMOTE_DIR}/models/pi_deeponet' \
        '${REMOTE_DIR}/models/sta_lstm' \
        '${REMOTE_DIR}/models/unet' \
        '${REMOTE_DIR}/sensitivity_results' \
        '${REMOTE_DIR}/sensitivity_lookback' && \
    chmod +x '${REMOTE_DIR}/scripts/'*.sh '${REMOTE_DIR}/hpc/'*.sh 2>/dev/null || true
"

echo ""
echo "Done. Monitor jobs with:"
echo "  sshpass -e ssh $SSH_OPTS ${REMOTE} 'squeue -u ${SNELLIUS_USER}'"
