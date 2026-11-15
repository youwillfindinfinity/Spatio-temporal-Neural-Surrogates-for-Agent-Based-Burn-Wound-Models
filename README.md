# Neural Surrogates for Cytokine Fields in an Agent-Based Burn Wound Model

![Graphical abstract](outputs/figures/GraphicalAbstract.png)

[![DOI](https://zenodo.org/badge/1223516798.svg)](https://doi.org/10.5281/zenodo.20457662)

> **Neural Surrogates for Cytokine Fields in an Agent-Based Burn Wound Model: a Persistence-Referenced Benchmark**  
> Roland V. Bumbuc\*, Julia Kowalczuk\*, Rezo Getsadze, Ioannis Papapanagiotou, H. Ibrahim Korkmaz, Vivek M. Sheraton  
> *Computer Methods and Programs in Biomedicine* (under review)  
> \*Equal contribution

The code of the original submission is kept under the tag `v1.0.0`.

---

## Overview

This repository contains the code and results of a benchmark of neural surrogates for the IL-8 and IL-10 fields of a hybrid agent-based model (ABM) of the post-burn immune response (CompuCell3D coupled to FiPy). The surrogates predict the next hourly field of one cytokine from the two preceding frames (six cytokine fields and five cell-type masks per frame). The cell masks come from the ABM, so the agent dynamics are not learned.

Consecutive hourly frames of the ABM are almost identical, so a model that copies the last frame already reaches R² of 0.9999 for IL-8. All models are therefore compared with this persistence baseline:

```
skill = 1 - MSE_model / MSE_persistence
```

| Model | Role |
|---|---|
| **U-Net** | main benchmark |
| **ConvLSTM** | main benchmark |
| **Fourier neural operator (FNO)** | main benchmark |
| **STA-LSTM** | main benchmark |
| **Point-wise network** | main benchmark (the DeepONet trunk used on its own) |
| **DeepONet** | ablation only |
| **PINN** | physics-based reference |

Each of the five benchmark architectures is trained in two forms: **direct** (the network outputs the next frame) and **increment** (the network outputs a correction that is added to the previous frame). Both forms have their own 20-trial Optuna search and are trained with three seeds, at lattice sizes of 100² and 250².

**Main results**

- Persistence reaches R² of 0.9999 (100²) and 0.9998 (250²) for IL-8, and 0.96 and 0.95 for IL-10.
- In increment form all five architectures beat persistence for IL-8 (skill 0.70-0.88 at 100², 0.59-0.95 at 250²). They stay accurate over 19 h roll-outs and keep their skill on unseen seeds and on runs with a wider boundary margin.
- In direct form the same architectures are worse than persistence for IL-8, although their R² is above 0.99.
- For IL-10, which is produced by a small number of moving M2 macrophages, the gain over persistence is close to zero (skill 0.001-0.006). It rises to 0.07-0.25 when the models also receive the cell masks at the target time.
- With frozen weights, the U-Net is worse than persistence on runs with five times more endothelial cells at 250².
- The PINN, with a residual, sources and boundary condition that match the ABM, reaches R² of 0.91 (100²) and 0.94 (250²) for IL-8 and does not fit IL-10.

---

## Repository structure

```
surrogates_2D/
├── scripts/
│   ├── rev/                      # experiments of the revised manuscript
│   │   ├── preprocess_runs.py    # ABM runs (VTK) -> arrays, one scaling per grid
│   │   ├── common.py             # shared data loading, metrics, persistence, roll-out
│   │   ├── run_main.py           # five architectures x {direct, increment}
│   │   ├── run_multitraj.py      # training on 1, 2 or 3 ABM trajectories
│   │   ├── run_grid_models.py    # U-Net variants (training-set size, weighting)
│   │   ├── run_lookback.py       # look-back window sweep
│   │   ├── run_deeponet.py       # DeepONet ablation
│   │   ├── run_pinn_abm.py       # PINN, strict and softened residual
│   │   ├── run_pinn_dataonly.py  # PINN control without PDE and boundary terms
│   │   ├── physics_step*.py      # re-implemented cytokine update of the ABM
│   │   ├── abm_variability.py    # run-to-run variability of the ABM
│   │   ├── check_*.py            # checks reported in the manuscript
│   │   ├── aggregate*.py         # result JSONs -> tables
│   │   ├── collect_for_paper.py  # values quoted in the manuscript
│   │   └── post_runs.py          # tables and figures of the manuscript
│   ├── unet/, sta_lstm/, deeponet/, pi_deeponet/, pinn/   # original submission
│   └── preprocessing100.py, figures.py, metrics.py, ...   # original submission
├── sensitivity/                  # clipping sensitivity (original submission)
├── results/
│   ├── rev_abm/                  # revised manuscript: all models except the PINN
│   ├── rev_abm_pinn/             # revised manuscript: PINN (strict, soft, data only)
│   ├── rev/, rev_pinn/           # same experiments on the simulations of the original submission
│   └── models/, sensitivity/, ...  # original submission
├── outputs/
│   ├── figures/                  # figures of the revised manuscript
│   ├── rev_abm/                  # tables behind the revised manuscript
│   └── rev/, eda/, metrics/      # earlier analyses
├── submit_revision4.sh           # Snellius: preprocessing, main models, ablation
├── submit_revision5.sh           # Snellius: target-time masks, figures
├── submit_revision6.sh           # Snellius: PINN, physics check
├── paths.py                      # central path configuration
└── requirements.txt
```

`submit_revision.sh`, `submit_revision2.sh` and `submit_revision3.sh` ran the same experiments on the simulations of the original submission (`results/rev`).

---

## Reproducing the revised manuscript

### 1. Install

Python 3.11 with TensorFlow 2.15 was used. A GPU is needed for training.

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

### 2. ABM runs

The experiments use 22 ABM simulations of 100 h (101 hourly frames): the baseline configuration with five seeds, a configuration with five times more endothelial cells (three seeds) and one with a wider boundary margin (three seeds), each at 100² and 250². Each run is a folder `<experiment>_G<grid>_s<seed>/LatticeData/Step_*.vtk`. Baseline seed 1 is the only run used for training.

```bash
python scripts/rev/preprocess_runs.py --runs-root <folder with the runs> --out preprocessed_abm
export REV_DATA_ROOT=$PWD/preprocessed_abm REV_OUT=rev_abm
```

All runs of a grid are scaled with the constants of the training run, so trained models can be applied to the other runs with frozen weights.

### 3. Train and evaluate

```bash
# main benchmark: hyperparameter search, then seeds 42, 1 and 100, both forms, both cytokines
for M in unet convlstm fno stalstm pointwise; do
  for G in 100 250; do
    python scripts/rev/run_main.py --model $M --grid $G
  done
done

# models that also receive the cell masks at the target time
python scripts/rev/run_main.py --model unet --grid 100 --target-masks

# further experiments (U-Net hyperparameters are taken from the main search)
python scripts/rev/run_multitraj.py --grid 100
python scripts/rev/run_grid_models.py --model unet --grid 100 --variants weighted ntrain35 ntrain52
python scripts/rev/run_lookback.py --grid 100
python scripts/rev/run_deeponet.py --grid 100 --variants base no_branch no_gate no_fields --seeds 42 1 100

# PINN
python scripts/rev/run_pinn_abm.py --variant strict --grid 100
python scripts/rev/run_pinn_abm.py --variant soft --grid 100
python scripts/rev/run_pinn_dataonly.py --grid 100
```

Every result JSON contains the one-step metrics for the near (82-91 h) and far (92-100 h) test windows, the persistence baseline, the skill, the 19 h roll-out and the frozen-weight results on the other runs. Finished runs are skipped when a script is started again.

### 4. Tables and figures

```bash
python scripts/rev/aggregate.py
python scripts/rev/aggregate_transfer.py
python scripts/rev/aggregate_pinn.py
python scripts/rev/post_runs.py          # figures -> outputs/figures
python scripts/rev/collect_for_paper.py  # -> outputs/rev_abm/for_paper.txt
```

`Fig2_architectures` and `GraphicalAbstract` in `outputs/figures` were drawn by hand and are not produced by `post_runs.py`.

### Snellius

`submit_revision4.sh`, `submit_revision5.sh` and `submit_revision6.sh` submit the steps above as separate jobs on the `gpu_a100` partition. Set `REPO`, `RUNS` and `ACCOUNT` at the top of each script first.

---

## Data format

Each cytokine channel is clipped at a percentile chosen from its kurtosis (no clipping below a kurtosis of 20, otherwise the 99.5th to 98th percentile) and mapped linearly to [-1, 1]:

```
u_phys = (u_scaled + 1) / 2 * c_max
```

The constants are stored in `metadata.json` of each preprocessed run. Cell-type masks are binary and use the CompuCell3D type identifiers of the model: endothelial cells (1), resting neutrophils (2), activated neutrophils (5), M1 macrophages (8) and M2 macrophages (9).

---

## Citation

```bibtex
@article{bumbuc2026surrogates2d,
  title   = {Neural Surrogates for Cytokine Fields in an Agent-Based Burn Wound Model:
             a Persistence-Referenced Benchmark},
  author  = {Bumbuc, Roland V. and Kowalczuk, Julia and Getsadze, Rezo and
             Papapanagiotou, Ioannis and Korkmaz, H. Ibrahim and Sheraton, Vivek M.},
  journal = {Computer Methods and Programs in Biomedicine},
  year    = {2026},
  note    = {Under review}
}
```

---

## License

This repository is released for research reproducibility. Please contact the corresponding author (v.s.muniraj@uva.nl) regarding any other use.
