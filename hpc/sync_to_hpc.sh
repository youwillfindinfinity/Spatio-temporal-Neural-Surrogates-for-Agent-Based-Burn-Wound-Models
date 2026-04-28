set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(dirname "$SCRIPT_DIR")"

if [ ! -f "$PROJECT_DIR/.env" ]; then
    echo "ERROR: .env not found at $PROJECT_DIR/.env"
    exit 1
fi
set -a; source "$PROJECT_DIR/.env"; set +a

DRY_RUN=""
if [[ "${1:-}" == "--dry-run" ]]; then
    DRY_RUN="--dry-run"
    echo "[DRY RUN] No files will be transferred."
fi

REMOTE="${SNELLIUS_USER}@${SNELLIUS_HOST}:${SNELLIUS_DIR}"

RSYNC_OPTS="-avz --progress --human-readable --delete"
RSYNC_OPTS="$RSYNC_OPTS --exclude='__pycache__/'"
RSYNC_OPTS="$RSYNC_OPTS --exclude='*.pyc'"
RSYNC_OPTS="$RSYNC_OPTS --exclude=*:Zone.Identifier"
RSYNC_OPTS="$RSYNC_OPTS --exclude=*.Zone.Identifier"
RSYNC_OPTS="$RSYNC_OPTS --exclude='.venv/'"
RSYNC_OPTS="$RSYNC_OPTS --exclude='*.egg-info/'"
RSYNC_OPTS="$RSYNC_OPTS --exclude='.pytest_cache/'"
RSYNC_OPTS="$RSYNC_OPTS --exclude='logs/'"

if command -v sshpass &>/dev/null && [[ -n "${SNELLIUS_PASSWORD:-}" ]]; then
    RSYNC_CMD="sshpass -p ${SNELLIUS_PASSWORD} rsync ${RSYNC_OPTS} ${DRY_RUN}"
else
    RSYNC_CMD="rsync ${RSYNC_OPTS} ${DRY_RUN}"
fi

echo "SYNC LOCAL -> SNELLIUS"
echo "  Local  : $PROJECT_DIR"
echo "  Remote : $REMOTE"

# scr code 
echo ""
echo "Syncing src/"
${RSYNC_CMD} "$PROJECT_DIR/src/" "${REMOTE}/src/"

# HPC scripts
echo ""
echo "Syncing hpc/"
${RSYNC_CMD} \
    --exclude='logs/' \
    "$PROJECT_DIR/hpc/" "${REMOTE}/hpc/"

# config 
echo ""
echo "Syncing config/"
${RSYNC_CMD} "$PROJECT_DIR/config/" "${REMOTE}/config/"

# tests
echo ""
echo "Syncing tests/"
${RSYNC_CMD} "$PROJECT_DIR/tests/" "${REMOTE}/tests/"

# experiments 
echo ""
echo "Syncing experiments/"
${RSYNC_CMD} \
    --exclude='amici_model/' \
    --exclude='results/' \
    --exclude='petab/' \
    "$PROJECT_DIR/experiments/" "${REMOTE}/experiments/"

# warm-start pkl files 
echo ""
echo "Syncing results/fits/*.pkl (warm-starts)"
${RSYNC_CMD} \
    --include='*.pkl' \
    --exclude='*' \
    "$PROJECT_DIR/results/fits/" "${REMOTE}/results/fits/"
if [ -d "$PROJECT_DIR/splitwise" ]; then
    echo "Syncing splitwise/*.pkl (splitwise warm-starts)"
    ${RSYNC_CMD} \
        --include='*.pkl' \
        --include='*.json' \
        --exclude='*' \
        "$PROJECT_DIR/splitwise/" "${REMOTE}/splitwise/"
fi

# top-level files 
echo ""
echo "Syncing top-level files"
for f in CLAUDE.md PROJECT_PLANNING.md FITTING_PLAN.md ODE_EQUATIONS.md \
          pyproject.toml setup.py requirements.txt setup.cfg; do
    if [ -f "$PROJECT_DIR/$f" ]; then
        ${RSYNC_CMD} "$PROJECT_DIR/$f" "${REMOTE}/$f"
    fi
done

# directories (create if missing)
echo ""
echo "Creating remote directories"
sshpass -p "${SNELLIUS_PASSWORD}" ssh "${SNELLIUS_USER}@${SNELLIUS_HOST}" \
    "mkdir -p \
        ${SNELLIUS_DIR}/logs \
        ${SNELLIUS_DIR}/results/fits \
        ${SNELLIUS_DIR}/results/sensitivity \
        ${SNELLIUS_DIR}/results/figures/supplementary \
        ${SNELLIUS_DIR}/results/tables \
        ${SNELLIUS_DIR}/experiments/petab_estimation/results && \
    chmod +x ${SNELLIUS_DIR}/hpc/*.sh 2>/dev/null || true && \
    echo 'Remote directories ready.'"

# .venv on Snellius
echo ""
echo "Setting up .venv on Snellius"
sshpass -p "${SNELLIUS_PASSWORD}" ssh "${SNELLIUS_USER}@${SNELLIUS_HOST}" "
    set -e
    module purge
    module load 2023
    module --ignore_cache load Python/3.11.3-GCCcore-12.3.0
    VENV_DIR='${SNELLIUS_DIR}/.venv'
    if [ ! -f \"\$VENV_DIR/bin/activate\" ]; then
        echo 'Creating .venv on Snellius...'
        python3 -m venv \"\$VENV_DIR\"
    fi
    source \"\$VENV_DIR/bin/activate\"
    pip install --upgrade pip --quiet
    pip install -r '${SNELLIUS_DIR}/requirements.txt' --quiet
    echo '.venv ready: '\$(python3 --version)
"

echo ""
echo "Sync complete: $(date)"