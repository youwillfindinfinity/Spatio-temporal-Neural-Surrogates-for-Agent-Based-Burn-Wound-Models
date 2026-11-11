"""Active pixel-fraction analysis across grid resolutions."""
import sys, json, csv
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from paths import PREPROCESSED, EDA, FIGURES, ensure_dirs

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

CYTS      = {"il8": 0, "il10": 3}
SOURCES   = {"il8": [0, 1, 2], "il10": [4]}      # mask channels: EC, NN, NA, M1, M2
TRUE_SIZE = 5.0                                   # domain side length, as in pinn/all.py
WINDOW    = 2
SPLITS    = {"train": (0, 70), "val": (70, 80), "near": (80, 90), "far": (90, 99), "all": (0, None)}


def main():
    ensure_dirs(EDA, FIGURES)
    rows, curves = [], {}
    grids = sorted(int(d.name.split("x")[0]) for d in PREPROCESSED.iterdir()
                   if d.is_dir() and (d / "Y_raw_phys.npy").exists())
    for G in grids:
        p = PREPROCESSED / f"{G}x{G}"
        meta  = json.load(open(p / "metadata.json"))
        masks = np.load(p / "Y_masks_spatial.npy", mmap_mode="r")
        N     = masks.shape[0]
        raw   = np.load(p / "Y_raw_phys.npy", mmap_mode="r")[WINDOW:WINDOW + N]
        px_area = (TRUE_SIZE / G) ** 2
        cells = np.asarray(masks).max(axis=-1) > 0.5
        for cyt, j in CYTS.items():
            u    = np.asarray(raw[..., j], dtype=np.float64)
            thr  = 0.05 * float(meta["scaling"]["max"][j])
            src  = np.asarray(masks[..., SOURCES[cyt]]).max(axis=-1) > 0.5
            regions = {"nonzero": u > 0, "above_dice_thr": u > thr,
                       "cell_occupied": cells, "source_cells": src}
            per_t = {k: v.reshape(N, -1).mean(axis=1) for k, v in regions.items()}
            curves[(G, cyt)] = per_t
            for split, (a, b) in SPLITS.items():
                row = {"grid": G, "cytokine": cyt, "window": split, "dice_thr": f"{thr:.3e}"}
                for k, f in per_t.items():
                    frac = float(f[a:b].mean())
                    row[f"{k}_frac_pct"] = round(100 * frac, 3)
                    row[f"{k}_pixels"]   = int(round(frac * G * G))
                    row[f"{k}_area"]     = round(frac * G * G * px_area, 4)
                rows.append(row)

    out = EDA / "active_fraction.csv"
    with open(out, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys())); w.writeheader(); w.writerows(rows)
    print(f"Saved {out}")

    fig, axes = plt.subplots(2, 2, figsize=(11, 7), sharex=True)
    for r, cyt in enumerate(CYTS):
        for c, key in enumerate(["above_dice_thr", "nonzero"]):
            ax = axes[r, c]
            for G in grids:
                f = curves[(G, cyt)][key]
                ax.plot(np.arange(WINDOW, WINDOW + len(f)), 100 * f, label=f"${G}^2$")
            ax.axvspan(82, 100, alpha=0.08, color="k")
            ax.set_title(f"{cyt.upper()} - {key.replace('_', ' ')}")
            ax.set_ylabel("% of grid points")
            if r == 1: ax.set_xlabel("time (h)")
    axes[0, 0].legend(title="grid")
    fig.tight_layout(); fig.savefig(FIGURES / "active_fraction.png", dpi=200)
    print(f"Saved {FIGURES / 'active_fraction.png'}")


if __name__ == "__main__":
    main()
