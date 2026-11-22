"""
Simulation runs of the burn-wound ABM with IL-10 secreted by M2 macrophages.

Builds one CompuCell3D variant per run with make_variant.py and writes
  <root>/experiment.json   design and purpose of each experiment
  <root>/manifest.csv      one row per run (read by hpc/submit_abm.sbatch)
  <root>/runs/<run_id>/    ready-to-run CC3D project + abm_provenance.json

Usage:
  python build_experiment.py --root <dir>
"""
import argparse
import csv
import datetime
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from make_variant import make_variant  # noqa: E402

HOURS = 100
GRIDS = (100, 250)

# All runs use IL-10 secreted by M2 macrophages. Runs E1/E2a/E2b with the M1
# source of the archived model were started on 2026-11-11 and cancelled after
# ~5 min; their partial output was not used.
EXPERIMENTS = {
    "E3_il10_from_m2": dict(
        il10_source="m2", endo_factor=1.0, boundary_frac=10, seeds=(1, 2, 3, 4, 5),
        purpose="Baseline with IL-10 secreted by M2 macrophages (the archived model multiplies "
                "km2il10 by the M1 array): run-to-run variability under identical settings and "
                "different random seeds; wall-clock cost and hardware."),
    "E2a_endothelial_x5_m2": dict(
        il10_source="m2", endo_factor=5.0, boundary_frac=10, seeds=(1, 2, 3),
        purpose="Changed parameter: initial endothelial cell count x5 "
                "(as in the existing 500^2 parameter scan, endocount 1000 -> 5000)."),
    "E2b_geometry_margin_G5_m2": dict(
        il10_source="m2", endo_factor=1.0, boundary_frac=5, seeds=(1, 2, 3),
        purpose="Changed wound geometry: tissue margin / boundary-patch "
                "width G/5 instead of G/10 (smaller tissue square, wider entry strip)."),
}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", required=True)
    ap.add_argument("--append", action="store_true",
                    help="add runs missing from an existing root; new rows go to --manifest-out")
    ap.add_argument("--manifest-out", default="manifest.csv")
    a = ap.parse_args()
    root = Path(a.root)
    (root / "runs").mkdir(parents=True, exist_ok=a.append)

    rows = []
    for exp, cfg in EXPERIMENTS.items():
        for grid in GRIDS:
            for seed in cfg["seeds"]:
                run_id = f"{exp}_G{grid}_s{seed}"
                if a.append and (root / "runs" / run_id).exists():
                    continue
                make_variant(grid, HOURS, seed, cfg["il10_source"], root / "runs" / run_id,
                             cfg["endo_factor"], cfg["boundary_frac"],
                             python_vtk_writer=False)  # CC3D 4.6.0 native lattice output
                rows.append(dict(run_id=run_id, experiment=exp, grid=grid, hours=HOURS,
                                 seed=seed, il10_source=cfg["il10_source"],
                                 endo_factor=cfg["endo_factor"],
                                 boundary_frac=cfg["boundary_frac"],
                                 run_dir=f"runs/{run_id}"))

    if not rows:
        print("nothing to add"); return
    with open(root / a.manifest_out, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader(); w.writerows(rows)
    design = {
        "created": datetime.datetime.now().isoformat(timespec="seconds"),
        "hours": HOURS, "grids": GRIDS, "frames_per_run": HOURS + 1,
        "output_frequency_mcs": 10000,
        "simulator": {
            "cc3d": "4.6.0 (conda, channel compucell3d, py310hb93dfd8_0)", "python": "3.10",
            "environment": "~/miniforge3/envs/cc3d_460 on Snellius (no environment modules)",
            "why_this_version": (
                "The published data were produced with Python 3.10 (cpython-310 bytecode in the "
                "archived model) and VTK >= 9.1 (legacy VTK 5.1 headers), i.e. a CC3D 4.4-4.6 "
                "release. CC3D 4.7.0 (source builds on Snellius and locally) removes all motile "
                "1-pixel cells within the first simulated hour; 4.6.0 reproduces the published "
                "early dynamics at 100^2."),
            "validation_100x100_seed99_vs_published": {
                "neutrophils_h0_h1_h2": {"published": [194, 177, 163], "cc3d_4.6.0": [184, 176, 158],
                                          "cc3d_4.7.0": [69, 0, 0]},
                "monocytes_h0_h1_h2": {"published": [171, 136, 112], "cc3d_4.6.0": [172, 146, 111],
                                        "cc3d_4.7.0": [85, 0, 0]},
                "resting_monocytes_h0_h1_h2": {"published": [35, 52, 55], "cc3d_4.6.0": [36, 47, 66]},
                "activated_neutrophils_h0_h1_h2": {"published": [7, 14, 22], "cc3d_4.6.0": [5, 8, 16]},
                "fibroblasts_h0_h1_h2": {"published": [22, 22, 22], "cc3d_4.6.0": [22, 22, 21]}},
            "steppable_at_mcs_0": "default in 4.6.0, as in the published data (frame 0 already "
                                  "contains IL-8 and converted monocytes)",
            "seed_reproducibility": "two 4.6.0 runs with seed 99 gave identical cell counts"},
        "experiments": {k: {kk: (list(vv) if isinstance(vv, tuple) else vv)
                            for kk, vv in v.items()} for k, v in EXPERIMENTS.items()},
        "n_runs": len(rows),
    }
    if a.append and (root / "experiment.json").exists():
        old = json.loads((root / "experiment.json").read_text())
        old.setdefault("history", []).append(
            {"date": design["created"], "manifest": a.manifest_out, "n_runs_added": len(rows),
             "note": "all runs use il10_source=m2; M1 runs E1/E2a/E2b cancelled, output not used"})
        old["experiments"] = design["experiments"]
        design = old
    (root / "experiment.json").write_text(json.dumps(design, indent=2))
    print(f"{len(rows)} runs written to {root} ({a.manifest_out})")


if __name__ == "__main__":
    main()
