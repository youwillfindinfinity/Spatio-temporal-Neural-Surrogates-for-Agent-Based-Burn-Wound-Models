"""Directory layout. Scripts import paths from here; ROOT is the repository root."""

from pathlib import Path

ROOT = Path(__file__).resolve().parent

DATA_RAW      = ROOT / "data_raw"          # raw VTK / simulation data (user-provided)
LATTICE_DATA  = DATA_RAW / "100h"          # 100h multi-grid .vtk files
DATA_RAW_200H = DATA_RAW / "200h"          # 200h single-sim .vtk files

PREPROCESSED          = ROOT / "preprocessed"           # 100h arrays
PREPROCESSED_200H     = ROOT / "preprocessed_200h"      # 200h arrays
PREPROCESSED_P95      = ROOT / "preprocessed_p95"       # sensitivity: fixed p95 clip
PREPROCESSED_NOCLIP   = ROOT / "preprocessed_noclip"    # sensitivity: no clipping
PREPROCESSED_LOOKBACK = ROOT / "preprocessed_lookback"  # lookback sweep (temp)
PREPROCESSED_ABM      = ROOT / "preprocessed_abm"       # revision: one folder per ABM run (scripts/rev/preprocess_runs.py)

# model outputs: metrics JSON, weights, visualisations
RESULTS             = ROOT / "results"
RESULTS_MODELS      = RESULTS / "models"
RESULTS_ADDITIONAL  = RESULTS / "additional_simulations"
RESULTS_SENS        = RESULTS / "sensitivity"
RESULTS_LOOKBACK    = RESULTS / "sensitivity_lookback"
RESULTS_REV         = RESULTS / "rev_abm"            # revision: per-model results JSON and predictions (scripts/rev)

# figures, tables, aggregated metrics
OUTPUTS = ROOT / "outputs"
FIGURES = OUTPUTS / "figures"
EDA     = OUTPUTS / "eda"
METRICS = OUTPUTS / "metrics"


def ensure_dirs(*dirs):
    for d in dirs:
        Path(d).mkdir(parents=True, exist_ok=True)
