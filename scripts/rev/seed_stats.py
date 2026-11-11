"""Seed statistics from existing result JSONs: training-time variance, metric mean +/- SD
for both horizons, and per-time-step R2 (error growth with forecast distance)."""
import json, sys, csv
from pathlib import Path
import numpy as np
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = Path(sys.argv[1]) if len(sys.argv) > 1 else Path("results/models")
OUT  = Path(sys.argv[2]) if len(sys.argv) > 2 else Path("outputs/metrics")
OUT.mkdir(parents=True, exist_ok=True)
MODELS = {"deeponet_h": "DeepONet", "pi_deeponet": "PI-DeepONet", "unet": "U-Net", "sta_lstm": "STA-LSTM", "pinn": "PINN"}
SEEDS, CYTS, GRIDS = [1, 42, 100], ["il8", "il10"], [100, 250]
METRICS = ["Global_R2", "SSIM", "Avg_Dice", "Spatial_Correlation", "Masked_RMSE"]

time_rows, metric_rows, pert_rows, curves = [], [], [], {}
for m, name in MODELS.items():
    for c in CYTS:
        for g in GRIDS:
            runs = [json.load(open(ROOT / m / f"res_{c}_{g}_{s}.json")) for s in SEEDS]
            t = np.array([r["train_time_seconds"] for r in runs], float)
            time_rows.append({"model": name, "cytokine": c, "grid": g, **{f"seed_{s}": round(x, 1) for s, x in zip(SEEDS, t)},
                              "mean_s": round(t.mean(), 1), "sd_s": round(t.std(ddof=1), 1),
                              "cv_pct": round(100 * t.std(ddof=1) / t.mean(), 1), "max_over_min": round(t.max() / t.min(), 2)})
            per_t = []
            for hk in runs[0]["results"]:
                hor = "near" if hk.startswith("Near") else "far"
                row = {"model": name, "cytokine": c, "grid": g, "horizon": hor}
                for k in METRICS:
                    v = np.array([r["results"][hk][k] for r in runs], float)
                    row[f"{k}_mean"], row[f"{k}_sd"] = float(f"{v.mean():.4g}"), float(f"{v.std(ddof=1):.2g}")
                metric_rows.append(row)
                per_t.append(np.array([r["results"][hk]["Per_Timestep_R2"] for r in runs], float))
            pt = np.concatenate(per_t, axis=1)                      # (seeds, 19) -> t = 82..100
            curves[(name, c, g)] = pt
            for i in range(pt.shape[1]):
                pert_rows.append({"model": name, "cytokine": c, "grid": g, "t_hours": 82 + i,
                                  "R2_mean": round(float(np.nanmean(pt[:, i])), 4), "R2_sd": round(float(np.nanstd(pt[:, i], ddof=1)), 4)})

for fn, rows in [("train_time_by_seed.csv", time_rows), ("metrics_mean_sd.csv", metric_rows), ("per_timestep_r2.csv", pert_rows)]:
    with open(OUT / fn, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys())); w.writeheader(); w.writerows(rows)

fig, axes = plt.subplots(2, 2, figsize=(11, 7), sharex=True)
tt = np.arange(82, 101)
for r, c in enumerate(CYTS):
    for k, g in enumerate(GRIDS):
        ax = axes[r, k]
        for name in MODELS.values():
            pt = curves[(name, c, g)]; mu, sd = np.nanmean(pt, 0), np.nanstd(pt, 0, ddof=1)
            ax.plot(tt, mu, marker="o", ms=3, label=name); ax.fill_between(tt, mu - sd, mu + sd, alpha=0.2)
        ax.axvline(91.5, color="k", ls=":", lw=0.8); ax.set_title(f"{c.upper()}, ${g}^2$"); ax.set_ylabel("$R^2$ per frame")
        lo = min(np.nanmin(np.nanmean(curves[(n, c, g)], 0)) for n in MODELS.values() if n != "PINN")
        ax.set_ylim(max(-0.05, lo - 0.1) if c == "il10" else lo - 0.05, 1.005)
        if r == 1: ax.set_xlabel("simulation time (h); training ends at t = 71 h")
axes[0, 0].legend(fontsize=8)
fig.tight_layout(); fig.savefig(OUT / "per_timestep_r2.png", dpi=200)
print("saved to", OUT)
