#!/bin/bash
# Sync code and preprocessed data from local to Snellius HPC.
# Usage: bash hpc/sync_to_hpc.sh [--dry-run]
#
# Requires .env at the project root with:
#   SNELLIUS_USER, SNELLIUS_HOST, SNELLIUS_PASS, SNELLIUS_PROJECT_DIR

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(dirname "$SCRIPT_DIR")"

ENV_FILE="$PROJECT_DIR/.env"
if [ ! -f "$ENV_FILE" ]; then
    echo "ERROR: .env not found at $ENV_FILE"
    echo "       Copy .env.example to .env and fill in your credentials."
    exit 1
fi
set -a; source "$ENV_FILE"; set +a

for VAR in SNELLIUS_USER SNELLIUS_HOST SNELLIUS_PASS SNELLIUS_PROJECT_DIR; do
    if [ -z "${!VAR:-}" ]; then
        echo "ERROR: $VAR is not set in .env"
        exit 1
    fi
done

DRY_RUN=""
if [[ "${1:-}" == "--dry-run" ]]; then
    DRY_RUN="--dry-run"
    echo "[DRY RUN] No files will be transferred."
fi

# rsync remote path — escape spaces so the remote shell sees a single path token
REMOTE_PATH="${SNELLIUS_PROJECT_DIR// /\\ }"
REMOTE="${SNELLIUS_USER}@${SNELLIUS_HOST}"

# Build rsync command prefix (with or without sshpass)
if command -v sshpass &>/dev/null; then
    RSYNC=(sshpass -p "${SNELLIUS_PASS}" rsync)
    SSH=(sshpass -p "${SNELLIUS_PASS}" ssh -o StrictHostKeyChecking=no)
else
    echo "WARNING: sshpass not found; falling back to key-based auth."
    RSYNC=(rsync)
    SSH=(ssh)
fi

BASE_OPTS=(-avz --progress --human-readable
    --exclude='__pycache__/'
    --exclude='*.pyc'
    --exclude='.venv/'
    --exclude='.env'
    --exclude='models/**/*.h5'
    --exclude='models/**/*.weights.h5'
    --exclude='*.npy'
)

echo "SYNC LOCAL -> SNELLIUS"
echo "  Local  : $PROJECT_DIR"
echo "  Remote : $REMOTE:$SNELLIUS_PROJECT_DIR"

sync_dir() {
    local src="$1" dest_subdir="$2"
    if [ -d "$PROJECT_DIR/$src" ]; then
        echo ""
        echo "Syncing $src/"
        "${RSYNC[@]}" "${BASE_OPTS[@]}" ${DRY_RUN:+"$DRY_RUN"} \
            "$PROJECT_DIR/$src/" \
            "${REMOTE}:${REMOTE_PATH}/${dest_subdir}/"
    fi
}

sync_dir "scripts"     "scripts"
sync_dir "sensitivity" "sensitivity"
sync_dir "hpc"         "hpc"

echo ""
echo "Syncing requirements.txt"
"${RSYNC[@]}" "${BASE_OPTS[@]}" ${DRY_RUN:+"$DRY_RUN"} \
    "$PROJECT_DIR/requirements.txt" \
    "${REMOTE}:${REMOTE_PATH}/requirements.txt"

# Preprocessed data directories (large; sync only what exists locally)
for PREP_DIR in preprocessed preprocessed_200h preprocessed_p95 preprocessed_noclip; do
    if [ -d "$PROJECT_DIR/$PREP_DIR" ]; then
        echo ""
        echo "Syncing $PREP_DIR/"
        "${RSYNC[@]}" -avz --progress ${DRY_RUN:+"$DRY_RUN"} \
            "$PROJECT_DIR/$PREP_DIR/" \
            "${REMOTE}:${REMOTE_PATH}/${PREP_DIR//' '/'\\ '}/"
    fi
done

# Create output directories on Snellius
echo ""
echo "Creating remote output directories"
"${SSH[@]}" "$REMOTE" "
    mkdir -p \
        '${SNELLIUS_PROJECT_DIR}/results/models/deeponet_h' \
        '${SNELLIUS_PROJECT_DIR}/results/models/pi_deeponet' \
        '${SNELLIUS_PROJECT_DIR}/results/models/pinn' \
        '${SNELLIUS_PROJECT_DIR}/results/models/sta_lstm' \
        '${SNELLIUS_PROJECT_DIR}/results/models/unet' \
        '${SNELLIUS_PROJECT_DIR}/results/models/200hrs/deeponet_h' \
        '${SNELLIUS_PROJECT_DIR}/results/models/200hrs/unet' \
        '${SNELLIUS_PROJECT_DIR}/results/models/200hrs/sta_lstm' \
        '${SNELLIUS_PROJECT_DIR}/results/sensitivity' \
        '${SNELLIUS_PROJECT_DIR}/results/sensitivity_lookback' \
        '${SNELLIUS_PROJECT_DIR}/outputs/figures' \
        '${SNELLIUS_PROJECT_DIR}/outputs/metrics' && \
    chmod +x '${SNELLIUS_PROJECT_DIR}/hpc/'*.sh 2>/dev/null || true && \
    echo 'Remote directories ready.'
"

echo ""
echo "Sync complete: $(date)"
