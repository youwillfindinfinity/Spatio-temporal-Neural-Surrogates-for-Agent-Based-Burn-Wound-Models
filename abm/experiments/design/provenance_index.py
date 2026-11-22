"""
One-row-per-run provenance index of the ABM runs.

Reads <root>/output/*/run_provenance.json (written by hpc/run_abm.py) and writes
  <root>/provenance_index.csv    seed, settings, simulator, hardware, SLURM ids, speed
  <root>/speed_summary.csv       wall seconds per simulated hour by experiment and grid

Usage:
  python provenance_index.py --root <experiment root>
"""
import argparse
import csv
import json
import statistics
from collections import defaultdict
from pathlib import Path


def row(p):
    j = json.loads(p.read_text())
    m, hw, sp, out, sw, sl = (j["model"], j["hardware"], j["speed"], j["output"],
                              j["software"], j["slurm"])
    return dict(
        run_id=j["run_id"], experiment=j["run_id"].rsplit("_G", 1)[0],
        grid=m["grid"], hours=m["hours"], seed=j["seed"], il10_source=m["il10_source"],
        endocount=m["endocount"], endo_factor=m["endo_factor"],
        boundaryat=m["boundaryat"], boundary_frac=m["boundary_frac"],
        cc3d=sw.get("cc3d"), python=sw.get("python"), numpy=sw.get("numpy"),
        fipy=sw.get("fipy"), scipy=sw.get("scipy"),
        omp_threads=j["environment"].get("OMP_NUM_THREADS"),
        node=hw.get("hostname"), cpu=hw.get("Model name"),
        cpus_available=hw.get("cpus_available_to_job"),
        slurm_job=f'{sl.get("SLURM_ARRAY_JOB_ID")}_{sl.get("SLURM_ARRAY_TASK_ID")}',
        partition=sl.get("SLURM_JOB_PARTITION"),
        start=j["start"], end=j["end"], exit_code=j["exit_code"],
        wall_seconds=sp["wall_seconds"], mcs_per_second=sp["mcs_per_second"],
        wall_s_per_sim_hour=sp["wall_seconds_per_simulated_hour"],
        frame_s_mean=sp["per_frame_wall_seconds"]["mean"],
        frame_s_min=sp["per_frame_wall_seconds"]["min"],
        frame_s_max=sp["per_frame_wall_seconds"]["max"],
        peak_rss_MB=round(j["peak_rss_children_MB"], 1),
        n_vtk=out["n_vtk"], complete=out["complete"], vtk_bytes=out["bytes"],
        template_md5=m.get("template_md5"))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", required=True)
    root = Path(ap.parse_args().root)
    rows = [row(p) for p in sorted(root.glob("output/*/run_provenance.json"))]
    with open(root / "provenance_index.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader(); w.writerows(rows)

    groups = defaultdict(list)
    for r in rows:
        groups[(r["experiment"], r["grid"])].append(r["wall_s_per_sim_hour"])
    with open(root / "speed_summary.csv", "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["experiment", "grid", "n_runs", "wall_s_per_sim_hour_median",
                    "min", "max", "wall_min_per_100h_run_median"])
        for (exp, grid), v in sorted(groups.items()):
            med = statistics.median(v)
            w.writerow([exp, grid, len(v), med, min(v), max(v), round(med * 100 / 60, 1)])
    print(f"{len(rows)} runs, {sum(r['complete'] for r in rows)} complete -> {root}")


if __name__ == "__main__":
    main()
