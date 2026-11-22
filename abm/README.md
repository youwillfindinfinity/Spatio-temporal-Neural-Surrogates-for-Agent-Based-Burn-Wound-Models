# Burn wound ABM (CompuCell3D + FiPy)

`template/` contains the archived model `combi_clean` of Korkmaz et al. (2024), unchanged. As archived, it runs a 500 × 500 lattice for 200 h, and `endocount` is a parameter-scan placeholder.

`make_variant.py` writes a runnable copy of the model for any grid size, run length and seed:

```bash
python abm/make_variant.py --grid 100 --hours 100 --seed 1 --out abm/runs/G100_h100_s1                    # IL-10 from M2 macrophages
python abm/make_variant.py --grid 100 --hours 100 --seed 1 --il10-source m1 --out abm/runs/G100_published  # as in the published data
```

Run the generated `endothelial.cc3d` with CompuCell3D 4.6.0, for example headless:

```bash
python -m cc3d.run_script -i <run>/endothelial.cc3d -o <out> -f 10000
```

Each run writes `abm_provenance.json` next to `LatticeData/`, recording the grid, seed, IL-10 source, boundary margin and cell-type identifiers. `scripts/rev/preprocess_runs.py` copies this file into `metadata.json`.

## Grid-dependent settings

These settings were changed by hand for the published 50², 100² and 250² data sets. The edited copies were not archived, so the values below are reconstructed from the published `LatticeData`.

| Setting | 500² template | Variant for grid G | Evidence in the published data |
|---|---|---|---|
| `nx`, `<Dimensions>` | 500 | G | VTK `DIMENSIONS` |
| `boundaryat` (boundary margin b) | 50 | G // 10 | endothelial cells span [G/10 + 2, 9G/10 − 2] |
| `<Steps>` | 2,000,001 | hours × 10,000 + 1 | 101 frames for 100 h |
| seeding and top-up counts | 1000 / 100 / 25 / 10 / 3 | `max(1, round(k·G/500))` | counts at t = 0 and the jumps every 10 h scale as G/500 |
| lattice bounds | `500`, `499`, `250000` | `nx`, `nx − 1`, `nx·ny` | required for G < 500 |

`boundaryat` sets the length of the four zero-concentration boundary segments of the cytokine fields (one per side, next to a corner; `Simulation/test.py`) and where cells are placed (`randopos` in `endothelialSteppables.py`). Endothelial cells, fibroblasts, myofibroblasts and the neutrophils added every 10 h are placed at random in the central square more than `boundaryat` from the edges. Monocytes and most of the initial neutrophils are placed in the strips of width `boundaryat` along the edges.

The domain is always 5 × 5 cm, and cells occupy one or two voxels (side 5/G cm, depth 1 cm). Grids therefore differ in physical cell size and cell density, not only in resolution.

## Changes to the template

- **IL-10 source.** The template multiplies `km2il10` by `cellpresentm1`, so M1 macrophages secrete IL-10, although the parameter name and the model description assign it to M2 macrophages. `--il10-source m2` (the default) uses M2; `m1` reproduces the published data sets.
- **Seed.** The template sets no seed. `--seed` writes one value to three places:
  - `<RandomSeed>` in the Potts section of `endothelial.xml`, which drives cell motion;
  - the main script `endothelial.py`, which seeds `random` and `numpy.random` before any model module is imported;
  - `endothelialSteppables.py`, which reseeds with the same value on import, before the first draw.

  The steppables run `import random` and then `from numpy import *`, which rebinds `random` to `numpy.random`. NumPy's generator therefore drives cell placement, life spans and phenotype switching.

## Properties of the template kept unchanged

- **Chemotaxis field names.** Chemotaxis is attached to fields named `IL8`, `IL1`, `TNF` and `TGF`, but the fields are created as `il8`, `il1`, … The XML Chemotaxis plugin lists only `cytokine`, a field that stays at zero. Whether chemotaxis acts has not been checked.
- **Cytokine time step.** Each hourly frame (`relaxationmcs = 10000` Monte Carlo steps) advances the cytokine equations by a single implicit FiPy step with `dt = 1`. The authors confirmed that this coupling is intended.
- **Cell division.** It is never triggered: its condition, `randint(0,1000) == 5000`, cannot be met.

## Simulator version

The published data were produced with Python 3.10 (the archived model contains `cpython-310` bytecode) and VTK 9.1 or later (the legacy VTK headers state version 5.1). This points to a CompuCell3D release between 4.4 and 4.6. The "4.2.5" in the XML is the version the project was created with.

- **CompuCell3D 4.7.0 does not reproduce the model.** Builds with Python 3.13 and 3.10 both removed every motile one-pixel cell within the first simulated hour. The XML has no Volume plugin, so older releases presumably kept one-pixel cells alive in another way.
- **CompuCell3D 4.6.0 reproduces the early dynamics.** This is the conda package `cc3d=4.6.0` from the `compucell3d` channel, with Python 3.10.
  - At 100² with seed 99, neutrophils went 184 → 176 → 158 over hours 0-2 (published: 194 → 177 → 163) and monocytes 172 → 146 → 111 (published: 171 → 136 → 112).
  - The steppables run at Monte Carlo step 0 by default, as in the published data.
  - Two runs with the same seed gave identical results.

## Simulation runs

`experiments/design/build_experiment.py` defines 22 runs of 100 h at 100² and 250², all with IL-10 secreted by M2 macrophages:

| Experiment | Change from the baseline | Seeds |
|---|---|---|
| `E3_il10_from_m2` | none (baseline) | 1-5 |
| `E2a_endothelial_x5_m2` | five times more endothelial cells | 1-3 |
| `E2b_geometry_margin_G5_m2` | boundary margin `boundaryat` = G/5 instead of G/10 | 1-3 |

- **Running.** `hpc/submit_abm.sbatch` runs the 22 runs as a SLURM array (rome partition, 16 CPUs), one run per task, through `hpc/run_abm.py`.
- **Provenance.** `run_abm.py` writes `run_provenance.json` for each run. It records the seed, settings, simulator and package versions, CPU and node, SLURM IDs, wall time, simulation speed, per-frame timings, peak memory and checksums of the model files. `experiments/design/provenance_index.py` collects these records into `provenance_index.csv` and `speed_summary.csv`.
- **Cost.** On Snellius (AMD EPYC 7H12, `OMP_NUM_THREADS=4`), the median wall time per simulated hour was 5.3 s at 100² (about 9 min per run, peak memory about 0.3 GB) and 25.5 s at 250² (about 43 min, about 0.9 GB).
- **Seeding of these runs.** They were seeded through `endothelial.xml` and `endothelialSteppables.py`; the seed in `endothelial.py` was added afterwards.
- **Reproducibility.** Baseline seed 1 was rerun on other compute nodes with `hpc/repro_check.sbatch`:
  - with the original model files: twice at 100² and once at 250²;
  - with the seed also set in `endothelial.py`: once at each grid size.

  All 101 frames and all data files were byte-identical to the production runs. The only file that differed was `StepLDF.dml`, which records the output path.
- **Fallback writer.** `make_variant.py` also contains an optional Python lattice writer, for CompuCell3D builds whose VTK output fails. These runs used CompuCell3D's own writer.

The simulations, the model files of every run and the provenance records are archived on Zenodo (doi:10.5281/zenodo.24355550).

## Reference

Korkmaz HI, Sheraton VM, Bumbuc RV, et al. An in silico modeling approach to understanding the dynamics of the post-burn immune response. *Frontiers in Immunology* 15 (2024) 1303776. doi:10.3389/fimmu.2024.1303776
