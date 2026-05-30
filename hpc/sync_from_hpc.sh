#!/bin/bash
# Sync result JSON files from Snellius HPC back to local.
# Usage: bash hpc/sync_from_hpc.sh [models|sensitivity_results|sensitivity_lookback ...]
#   With no arguments, syncs all result directories.
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

# Escape spaces in remote path for rsync
REMOTE_PATH="${SNELLIUS_PROJECT_DIR// /\\ }"
REMOTE="${SNELLIUS_USER}@${SNELLIUS_HOST}"

if command -v sshpass &>/dev/null; then
    RSYNC=(sshpass -p "${SNELLIUS_PASS}" rsync)
else
    echo "WARNING: sshpass not found; falling back to key-based auth."
    RSYNC=(rsync)
fi

BASE_OPTS=(-avz --progress --human-readable)

TARGETS=(results/models results/sensitivity results/sensitivity_lookback)
if [[ $# -gt 0 ]]; then
    TARGETS=("$@")
fi

echo "SYNC SNELLIUS -> LOCAL"
echo "  Remote : $REMOTE:$SNELLIUS_PROJECT_DIR"
echo "  Local  : $PROJECT_DIR"
echo "  Dirs   : ${TARGETS[*]}"
echo ""

for DIR in "${TARGETS[@]}"; do
    echo "Syncing ${DIR}/"
    mkdir -p "$PROJECT_DIR/$DIR"

    case "$DIR" in
        results/models)
            # Only pull JSON results and PNG visualisations; skip large weight files
            "${RSYNC[@]}" "${BASE_OPTS[@]}" \
                --include='*/' \
                --include='res_*.json' \
                --include='time_*.json' \
                --include='vis_*.png' \
                --exclude='*.h5' \
                --exclude='*.weights.h5' \
                --exclude='ckpt_*' \
                --exclude='*.npy' \
                "${REMOTE}:${REMOTE_PATH}/${DIR}/" \
                "$PROJECT_DIR/$DIR/" \
                2>&1 || echo "  [WARN] ${DIR} sync failed or directory missing on remote"
            ;;
        *)
            "${RSYNC[@]}" "${BASE_OPTS[@]}" \
                "${REMOTE}:${REMOTE_PATH}/${DIR}/" \
                "$PROJECT_DIR/$DIR/" \
                2>&1 || echo "  [WARN] ${DIR} sync failed or directory missing on remote"
            ;;
    esac
    echo ""
done

# Sync top-level aggregated CSVs
echo "Syncing top-level CSV files"
"${RSYNC[@]}" "${BASE_OPTS[@]}" \
    --include='*.csv' --exclude='*' \
    "${REMOTE}:${REMOTE_PATH}/" "$PROJECT_DIR/" \
    2>&1 || echo "  [WARN] top-level CSV sync failed"

echo ""
echo "Sync complete: $(date)"
