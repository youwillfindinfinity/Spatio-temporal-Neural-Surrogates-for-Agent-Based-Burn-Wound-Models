"""
Central path configuration for the surrogates_2D project.

Import this from any script:

    import sys
    from pathlib import Path
    sys.path.insert(0, str(Path(__file__).resolve().parents[N]))  # N levels up to repo root
    from paths import PREPROCESSED, RESULTS_MODELS, FIGURES, ...

Where N is:
    scripts/<model>/all.py          -> N=2
    scripts/figures.py              -> N=1
    sensitivity/aggregate_results.py -> N=1
"""

from pathlib import Path

ROOT = Path(__file__).resolve().parent

# ── Input data (not committed; user-provided) ─────────────────────────────────
DATA_RAW      = ROOT / "data_raw"          # root for all raw VTK + simulation data
LATTICE_DATA  = DATA_RAW / "100h"          # raw .vtk files (100h multi-grid)
DATA_RAW_200H = DATA_RAW / "200h"          # raw .vtk files (200h single-sim)

# ── Preprocessed arrays ───────────────────────────────────────────────────────
PREPROCESSED          = ROOT / "preprocessed"           # 100h preprocessed
PREPROCESSED_200H     = ROOT / "preprocessed_200h"      # 200h single-sim
PREPROCESSED_P95      = ROOT / "preprocessed_p95"       # sensitivity: fixed p95 clip
PREPROCESSED_NOCLIP   = ROOT / "preprocessed_noclip"    # sensitivity: no clipping
PREPROCESSED_LOOKBACK = ROOT / "preprocessed_lookback"  # lookback sweep (temp)

# ── Results: raw model outputs (JSON metrics, weights, visualisations) ────────
RESULTS             = ROOT / "results"
RESULTS_MODELS      = RESULTS / "models"
RESULTS_ADDITIONAL  = RESULTS / "additional_simulations"
RESULTS_SENS        = RESULTS / "sensitivity"
RESULTS_LOOKBACK    = RESULTS / "sensitivity_lookback"

# ── Outputs: figures, tables, aggregated metrics ──────────────────────────────
OUTPUTS = ROOT / "outputs"
FIGURES = OUTPUTS / "figures"
EDA     = OUTPUTS / "eda"
METRICS = OUTPUTS / "metrics"


def ensure_dirs(*dirs):
    """Create directories if they don't exist."""
    for d in dirs:
        Path(d).mkdir(parents=True, exist_ok=True)
