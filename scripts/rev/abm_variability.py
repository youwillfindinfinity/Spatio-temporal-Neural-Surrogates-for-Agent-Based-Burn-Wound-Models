"""Run-to-run variability of the ABM: surrogate metrics between pairs of ABM runs with the same settings
and different seeds (noise floor), and between the baseline and modified-setting runs.
Fields are clipped at the training run's c_max. Output: outputs/<REV_OUT>/abm_variability.csv and .txt"""
import csv, itertools, json, sys, warnings
import numpy as np
import common as C
warnings.filterwarnings("ignore")
from paths import OUTPUTS

DST = OUTPUTS / C.os.environ.get("REV_OUT", "rev"); DST.mkdir(parents=True, exist_ok=True)
rows, stats = [], []
import re
grids = sorted({int(re.search(r"_G(\d+)_s\d+$", d.name).group(1)) for d in C.DATA_ROOT.iterdir() if d.is_dir() and re.search(r"_G(\d+)_s\d+$", d.name)})
for grid in grids:
    runs = {C.train_run(grid): C.TRAIN_EXP, **{r: e for r, e in C.test_runs(grid).items() if "-ownscale" not in r}}
    for cyt in ("il10", "il8"):
        idx = C.CYTS.index(cyt)
        data = {r: C.load(grid, cyt, r) for r in runs}
        cmax = data[C.train_run(grid)]["cmax"]
        f = {r: C.den(d["Y"], cmax) for r, d in data.items()}
        base = [r for r, e in runs.items() if e == C.TRAIN_EXP]
        pairs = [("same settings, different seed", a, b) for a, b in itertools.combinations(base, 2)]
        pairs += [(f"baseline vs {e}", C.train_run(grid), r) for r, e in runs.items() if e != C.TRAIN_EXP]
        for label, a, b in pairs:
            for wname, sl in (("t82-100", slice(C.START, C.END)), ("t2-100", slice(0, 99))):
                m = C._paper(f[a][sl], f[b][sl], data[a]["M"][sl], cmax)
                rows.append({"grid": grid, "cytokine": cyt, "comparison": label, "run_a": a, "run_b": b, "window": wname,
                             **{k: float(f"{v:.5g}") for k, v in m.items()}, "RMSE_rel_cmax": float(f"{np.sqrt(np.mean((f[a][sl] - f[b][sl]) ** 2)) / cmax:.5g}")})
        for t in (25, 50, 75, 100):
            m = np.array([f[r][t - 2].mean() for r in base])
            stats.append((grid, cyt, t, len(base), float(m.mean()), float(m.std(ddof=1) / m.mean()) if len(base) > 1 and m.mean() > 0 else float("nan")))
with open(DST / "abm_variability.csv", "w", newline="") as fh:
    w = csv.DictWriter(fh, fieldnames=list(rows[0].keys())); w.writeheader(); w.writerows(rows)
lines = ["ABM RUN-TO-RUN VARIABILITY (mean over run pairs; metrics as for the surrogates)", ""]
for grid in grids:
    for cyt in ("il10", "il8"):
        lines.append(f"=== {cyt.upper()} {grid}x{grid} ===")
        lines.append(f"{'comparison':44s}{'window':>9s}{'pairs':>6s}{'R2':>9s}{'SSIM':>8s}{'Dice':>8s}{'Corr':>8s}{'RMSE/cmax':>11s}")
        sub = [r for r in rows if r["grid"] == grid and r["cytokine"] == cyt]
        for label in dict.fromkeys(r["comparison"] for r in sub):
            for wname in ("t82-100", "t2-100"):
                s = [r for r in sub if r["comparison"] == label and r["window"] == wname]
                mean = lambda k: float(np.mean([r[k] for r in s]))
                lines.append(f"{label:44s}{wname:>9s}{len(s):>6d}{mean('Global_R2'):9.4f}{mean('SSIM'):8.4f}{mean('Avg_Dice'):8.4f}{mean('Spatial_Correlation'):8.4f}{mean('RMSE_rel_cmax'):11.4f}")
        lines.append("")
lines.append("Domain-mean concentration across baseline seeds: mean and coefficient of variation")
for g, c, t, n, mu, cv in stats:
    lines.append(f"  {c:5s} {g:4d}  t = {t:3d} h  n = {n}  mean {mu:.3e}  CV {100 * cv:.2f} %")
(DST / "abm_variability.txt").write_text("\n".join(lines) + "\n"); print("\n".join(lines))
