set -euo pipefail
source "$(dirname "$0")/../.env"

REMOTE="${SNELLIUS_USER}@snellius.surf.nl:${SNELLIUS_DIR}"
LOCAL="$(dirname "$0")/.."

TARGETS=(fits figures sensitivity tables)

if [[ $# -gt 0 ]]; then
    TARGETS=("$@")
fi

echo "Syncing results from Snellius -> local"
echo "  Remote : ${REMOTE}/results/"
echo "  Local  : ${LOCAL}/results/"
echo "  Dirs   : ${TARGETS[*]}"
echo ""

RSYNC_OPTS="-avz --progress --human-readable"
if command -v sshpass &>/dev/null && [[ -n "${SNELLIUS_PASSWORD:-}" ]]; then
    RSYNC_CMD="sshpass -p ${SNELLIUS_PASSWORD} rsync ${RSYNC_OPTS}"
else
    RSYNC_CMD="rsync ${RSYNC_OPTS}" # fall back to key-based auth
fi

for dir in "${TARGETS[@]}"; do
    echo "Syncing results/${dir}/"
    mkdir -p "${LOCAL}/results/${dir}"
    ${RSYNC_CMD} \
        "${REMOTE}/results/${dir}/" \
        "${LOCAL}/results/${dir}/" \
        --exclude="*.tmp" \
        2>&1 || echo "  [WARN] ${dir} sync failed or directory missing on remote"
    echo ""
done

echo "Syncing results/*.json"
${RSYNC_CMD} \
    --include="*.json" --include="*.csv" --exclude="*" \
    "${REMOTE}/results/" "${LOCAL}/results/" \
    2>&1 || echo "  [WARN] top-level results sync failed"
echo ""

echo "Sync complete: $(date)"
