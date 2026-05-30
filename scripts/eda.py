import sys
import os
import re
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from paths import LATTICE_DATA, EDA

import numpy as np
import pyvista as pv

DATA_DIR = LATTICE_DATA
OUTPUT_DIR = EDA

CYTOKINES = ["il8", "il1", "il6", "il10", "tnf", "tgf"]
TARGET_GRIDS = [50, 100, 250, 500]

def extract_grid_size(name):
    match = re.search(r'(\d+)x(\d+)', name)
    if match: return int(match.group(1))
    match = re.search(r'(\d+)', name)
    if match: return int(match.group(1))
    return None

if __name__ == "__main__":
    rows = []
    folders = sorted([f for f in os.listdir(DATA_DIR) if (DATA_DIR / f).is_dir()])

    for folder in folders:
        grid = extract_grid_size(folder)
        if grid not in TARGET_GRIDS:
            continue

        case_path = DATA_DIR / folder
        vtk_files = sorted([f for f in os.listdir(case_path) if f.endswith(".vtk")])
        if not vtk_files:
            continue

        print(f"Grid {grid}x{grid}: {len(vtk_files)} time-steps.")

        global_min = {c: np.inf for c in CYTOKINES}
        global_max = {c: -np.inf for c in CYTOKINES}
        nz_pct_sum = {c: 0.0 for c in CYTOKINES}
        n_files = 0

        step = 10 if grid >= 500 else (5 if grid >= 250 else 1)
        for i in range(0, len(vtk_files), step):
            mesh = pv.read(str(case_path / vtk_files[i]))
            n_files += 1
            for c in CYTOKINES:
                if c not in mesh.point_data:
                    continue
                vals = mesh.point_data[c]
                global_min[c] = min(global_min[c], float(np.min(vals)))
                global_max[c] = max(global_max[c], float(np.max(vals)))
                nz_pct_sum[c] += np.count_nonzero(vals > 0) / vals.size

        for c in CYTOKINES:
            if global_max[c] == -np.inf:
                continue
            rows.append({
                "Grid": grid,
                "Cytokine": c.upper(),
                "Global_Min": f"{global_min[c]:.4e}",
                "Global_Max": f"{global_max[c]:.4e}",
                "Time_Avg_Sparsity_Pct": round((1.0 - nz_pct_sum[c] / n_files) * 100, 4),
            })

    if rows:
        import csv
        out_path = OUTPUT_DIR / "summary.csv"
        with open(out_path, "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=["Grid", "Cytokine", "Global_Min", "Global_Max", "Time_Avg_Sparsity_Pct"])
            w.writeheader()
            w.writerows(rows)
        print(f"\nSaved: {out_path}")