# Spatio-temporal Neural Surrogates for Agent-Based Burn Wound Models

[![DOI](https://zenodo.org/badge/1223516798.svg)](https://doi.org/10.5281/zenodo.20457662)

> **Spatio-temporal Neural Surrogates for Agent-Based Burn Wound Models**  
> Roland V. Bumbuc\*, Julia Kowalczuk\*, Rezo Getsadze, Ioannis Papapanagiotou, H. Ibrahim Korkmaz, Vivek M. Sheraton  
> *Computers in Biology and Medicine* (submitted)  
> \*Equal contribution

---

## Overview

Agent-based models (ABMs) of the post-burn immune response capture cell-level heterogeneity and spatially resolved cytokine dynamics, but their cost (hundreds to thousands of seconds per trajectory) limits repeated-evaluation workflows such as parameter sweeps, sensitivity analysis, and Bayesian calibration. This repository benchmarks **five neural surrogate architectures** for predicting cytokine concentration fields from a 2D burn wound ABM:

| Architecture | Key characteristic |
|---|---|
| **DeepONet** | Operator learning; best accuracy at 250² |
| **PI-DeepONet** | DeepONet + physics-informed loss |
| **U-Net** | Encoder–decoder CNN; best accuracy/speed balance |
| **STA-LSTM** | Spatial-attention + LSTM; sequential inductive bias |
| **PINN** | Physics-informed NN via DeepXDE |

Models are evaluated on **two cytokines** spanning the field-sparsity spectrum:
- **IL-8** — dense field (\<1% zero pixels)
- **IL-10** — sparse field (up to ~13% non-zero pixels)

at **two grid resolutions**: 100×100 and 250×250.

**Key findings:**
- DeepONet achieves R² > 0.99 at 250² for both cytokines.
- U-Net trains 17–41× faster than DeepONet at comparable accuracy.
- Physics-informed losses do not improve over data-driven baselines on this discrete stochastic ABM.
- Surrogates yield 10²–10⁴× per-evaluation speed-ups, amortising training within ~1 ABM trajectory.

---

## Repository structure

```
surrogates_2D/
├── scripts/
│   ├── preprocessing100.py     # Preprocess 100h VTK data → .npy arrays
│   ├── preprocessing200.py     # Preprocess 200h VTK data → .npy arrays
│   ├── sta_lstm/
│   │   ├── all.py              # Train STA-LSTM (100h data, all grids)
│   │   └── all_500.py          # Train STA-LSTM (500×500 grid)
│   ├── unet/
│   │   ├── all.py              # Train U-Net (100h data)
│   │   └── all_500.py          # Train U-Net (500×500 grid)
│   ├── deeponet/
│   │   ├── all.py              # Train DeepONet (100h data)
│   │   └── all_500.py          # Train DeepONet (500×500 grid)
│   ├── pi_deeponet/
│   │   ├── all.py              # Train PI-DeepONet (100h data)
│   │   └── all_500.py          # Train PI-DeepONet (500×500 grid)
│   ├── pinn/
│   │   ├── all.py              # Train PINN (100h data)
│   │   └── all_500.py          # Train PINN (500×500 grid)
│   ├── figures.py              # Reproduce all paper figures
│   ├── metrics.py              # Shared metric functions
│   ├── lookback.py             # Lookback-window sensitivity study
│   └── submit_snellius.sh      # Submit all jobs to Snellius HPC
├── sensitivity/
│   ├── preprocess_p95.py       # Preprocess with fixed 95th-percentile clip
│   ├── preprocess_no_clip.py   # Preprocess without clipping
│   ├── train_data_models.py    # Train data-driven models (sensitivity)
│   ├── train_physics_models.py # Train physics-informed models (sensitivity)
│   └── aggregate_results.py    # Aggregate sensitivity results → CSV
├── data_raw/
│   ├── 100h/                   # Raw VTK files for 100h multi-grid simulations
│   └── 200h/                   # Raw VTK files for 200h single simulation
├── results/
│   ├── models/                 # Result JSONs per model/grid/seed (100h runs)
│   ├── additional_simulations/ # Results for 200h and 500-grid experiments
│   ├── sensitivity/            # Sensitivity analysis results (clip variants)
│   └── sensitivity_lookback/   # Lookback-window sensitivity results
├── outputs/
│   ├── figures/                # Generated paper figures
│   ├── eda/                    # Exploratory data analysis summaries
│   └── metrics/                # Aggregated metric CSVs
├── paths.py                    # Central path config — imported by all scripts
├── hpc/
│   ├── sync_to_hpc.sh          # Push code + preprocessed data to Snellius
│   └── sync_from_hpc.sh        # Pull result JSONs from Snellius
├── requirements.txt
├── .env.example                # Template for HPC credentials (copy to .env)
└── all_metrics.csv             # Aggregated metrics across all models and seeds
```

---

## Quick start

### 1. Install dependencies

Python 3.10+ is recommended. A GPU with CUDA support is required for full training runs.

```bash
python -m venv .venv
source .venv/bin/activate        # Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

### 2. Prepare your data

The ABM simulation data consists of VTK files produced by the burn wound ABM (CompuCell3D / CPM). Each `.vtk` file is one timestep; 101 or 201 files are expected per simulation run.

Place the VTK files for the 100h simulation in a directory and run preprocessing:

```bash
# Expects a directory tree with subfolders named <GxG>/ (e.g. 100x100/, 250x250/)
# each containing 101 .vtk files.
python scripts/preprocessing100.py
# data_raw/100h/ (default) → preprocessed/{100x100,250x250}/

# For 200h data:
python scripts/preprocessing200.py
# data_raw/200h/ (default) → preprocessed_200h/
```

The preprocessing scripts produce per-grid folders containing:

| File | Shape | Used by |
|---|---|---|
| `Y_target.npy` | (99, G, G, 6) | All models |
| `Y_masks_spatial.npy` | (99, G, G, 5) | All models (eval + loss) |
| `X_lstm.npy` | (99, 2, G, G, 11) | STA-LSTM |
| `X_unet.npy` | (99, G, G, 22) | U-Net |
| `X_branch.npy` | (99, 2, G, G, 11) | DeepONet / PI-DeepONet |
| `X_trunk.npy` | (99, G²,  3) | DeepONet / PI-DeepONet |
| `X_colloc.npy` | (99, G²,  3) | PINN / PI-DeepONet |
| `Y_ic.npy` | (G, G, 6) | PINN |
| `Y_bc_mask.npy` | (G, G) | PINN |
| `Y_raw_phys.npy` | (101, G, G, 6) | PINN (PDE residuals) |
| `metadata.json` | — | All models (scaling params) |

### 3. Train a model

Each training script accepts `--cytokine` (required) and optionally `--grid` and `--seed`.

```bash
# Train U-Net on IL-8 at all available grid sizes (seed 42, with Optuna HP search)
python scripts/unet/all.py --cytokine il8

# Train on a specific grid with a different seed (reuses HP from seed=42 run)
python scripts/unet/all.py --cytokine il10 --grid 250 --seed 1

# Train DeepONet on IL-10 at 100×100
python scripts/deeponet/all.py --cytokine il10 --grid 100

# Train STA-LSTM on IL-8 at all grids
python scripts/sta_lstm/all.py --cytokine il8

# Train PINN (grid argument required)
python scripts/pinn/all.py --cytokine il8 --grid 100

# Train PI-DeepONet
python scripts/pi_deeponet/all.py --cytokine il10 --grid 250
```

Results are written to `results/models/<model_name>/res_<cytokine>_<grid>_<seed>.json`.

### 4. Reproduce paper figures

After training all models, generate paper figures with:

```bash
python scripts/figures.py
# Outputs saved to outputs/figures/
```

---

## Reproducing paper results

The full experiment matrix is:
- **Models**: DeepONet, PI-DeepONet, U-Net, STA-LSTM, PINN
- **Cytokines**: il8, il10
- **Grids**: 100, 250
- **Seeds**: 1, 42, 100

Run all combinations:

```bash
for MODEL in unet sta_lstm deeponet pi_deeponet; do
    for CYT in il8 il10; do
        for SEED in 1 42 100; do
            python scripts/${MODEL}/all.py --cytokine ${CYT} --seed ${SEED}
        done
    done
done

# PINN requires --grid
for CYT in il8 il10; do
    for GRID in 100 250; do
        for SEED in 1 42 100; do
            python scripts/pinn/all.py --cytokine ${CYT} --grid ${GRID} --seed ${SEED}
        done
    done
done
```

Pre-computed results (JSON files) are included in `models/` for all reported combinations.

---

## Running on Snellius HPC

1. Copy and fill the credentials file:

   ```bash
   cp .env.example .env
   # Edit .env: set SNELLIUS_USER, SNELLIUS_PASS, SNELLIUS_PROJECT_DIR
   ```

2. Push code and preprocessed data:

   ```bash
   bash hpc/sync_to_hpc.sh [--dry-run]
   ```

3. Submit jobs:

   ```bash
   bash scripts/submit_snellius.sh
   ```

4. Pull results back:

   ```bash
   bash hpc/sync_from_hpc.sh
   # Or pull specific directories:
   bash hpc/sync_from_hpc.sh models sensitivity_results
   ```

---

## Data format

Inputs are normalised to **[-1, 1]** per cytokine channel using adaptive percentile clipping:

- Kurtosis < 20 → no clipping (100th percentile)
- Kurtosis 20–600 → clip at 99.5th–98.0th percentile

Denormalisation: `u_phys = (u_scaled + 1) / 2 * clip_max[j]`

The clip parameters are stored in `preprocessed/{GxG}/metadata.json` under `scaling.max`.

---

## Sensitivity analyses

The `sensitivity/` directory contains scripts for two robustness checks reported in the paper:

| Study | Script | Output |
|---|---|---|
| Clipping variant (p95, no-clip, adaptive) | `sensitivity/preprocess_{p95,no_clip}.py` + `train_*.py` | `results/sensitivity/` |
| Lookback window (w=1..10) | `scripts/lookback.py` | `results/sensitivity_lookback/` |

---

## Citation

If you use this code or the pre-computed results in your research, please cite:

```bibtex
@article{bumbuc2026surrogates2d,
  title   = {Spatio-temporal Neural Surrogates for Agent-Based Burn Wound Models},
  author  = {Bumbuc, Roland V. and Kowalczuk, Julia and Getsadze, Rezo and
             Papapanagiotou, Ioannis and Korkmaz, H. Ibrahim and Sheraton, Vivek M.},
  journal = {Computers in Biology and Medicine},
  year    = {2026},
  note    = {Submitted}
}
```

---

## License

This repository is released for research reproducibility. Please contact the corresponding author (v.s.muniraj@uva.nl) regarding any other use.
