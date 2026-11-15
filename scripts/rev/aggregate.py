"""Collect all results/rev/**/res_*.json into CSV tables and a readable summary (outputs/rev/)."""
import json, csv, sys, warnings
warnings.filterwarnings("ignore", category=RuntimeWarning)
from pathlib import Path
import numpy as np
ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from paths import RESULTS, OUTPUTS

import os
TAG = os.environ.get("REV_OUT", "rev")
SRC, DST = RESULTS / TAG, OUTPUTS / TAG
DST.mkdir(parents=True, exist_ok=True)
runs = [json.load(open(p)) for p in sorted(SRC.glob("*/*/res_*.json"))]
if not runs:
    sys.exit(f"no results found in {SRC}")
groups = {}
for r in runs:
    groups.setdefault((r["cytokine"], r["grid"], r["model"], r["variant"]), []).append(r)

ms = lambda v: (float(np.nanmean(v)), float(np.nanstd(v, ddof=1)) if len(v) > 1 else float("nan"))
PM = ["Global_R2", "SSIM", "Avg_Dice", "Spatial_Correlation", "Masked_RMSE"]
one, reg, roll, corr = [], [], [], []
for (cyt, grid, model, var), rs in sorted(groups.items()):
    key = {"cytokine": cyt, "grid": grid, "model": model, "variant": var, "n_seeds": len(rs)}
    row = dict(key)
    for w in ("near", "far"):
        for k in PM:
            m, s = ms([r["windows"][w]["model"][k] for r in rs]); row[f"{w}_{k}"] = f"{m:.5g}"; row[f"{w}_{k}_sd"] = f"{s:.2g}"
            row[f"{w}_{k}_persistence"] = f"{rs[0]['windows'][w]['persistence'][k]:.5g}"
        m, s = ms([r["windows"][w]["skill_vs_persistence"] for r in rs]); row[f"{w}_skill"] = f"{m:.4g}"; row[f"{w}_skill_sd"] = f"{s:.2g}"
    tt = [r["train_time_seconds"] for r in rs if r.get("train_time_seconds")]
    row["train_s"], row["train_s_sd"] = (f"{ms(tt)[0]:.1f}", f"{ms(tt)[1]:.1f}") if tt else ("", "")
    row["n_parameters"] = rs[0].get("n_parameters", "")
    one.append(row)
    for w in ("near", "far"):
        rr = dict(key, window=w, band_halfwidth_px=rs[0]["windows"][w]["regions_model"]["band_halfwidth_px"])
        for region in ("front", "interior", "exterior"):
            rr[f"{region}_pixel_share"] = f"{rs[0]['windows'][w]['regions_model'][region + '_pixel_share']:.4f}"
            for who in ("model", "persistence"):
                for q in ("rmse_rel_cmax", "error_share"):
                    rr[f"{region}_{q}_{who}"] = f"{ms([r['windows'][w]['regions_' + who][region + '_' + q] for r in rs])[0]:.4g}"
        reg.append(rr)
    if rs[0]["rollout"]:
        ro = dict(key)
        for lead in (1, 3, 5, 10, 15, 19):
            for q in ("R2_gt", "R2_frozen", "R2_persist_frozen"):
                ro[f"lead{lead}_{q}"] = f"{ms([r['rollout'][q][lead - 1] for r in rs])[0]:.4f}"
        roll.append(ro)
    cm = np.array([r["per_frame"]["corr_model"] for r in rs], float)
    for k, t in enumerate(rs[0]["per_frame"]["t_hours"]):
        corr.append(dict(key, t_hours=t, corr_model_mean=f"{np.nanmean(cm[:, k]):.5f}", corr_model_min=f"{np.nanmin(cm[:, k]):.5f}",
                         corr_persistence=f"{rs[0]['per_frame']['corr_persistence'][k]:.5f}",
                         R2_model_mean=f"{np.nanmean([r['per_frame']['R2_model'][k] for r in rs]):.5f}",
                         R2_persistence=f"{rs[0]['per_frame']['R2_persistence'][k]:.5f}"))

for name, rows in (("summary_one_step.csv", one), ("summary_regions.csv", reg), ("summary_rollout.csv", roll), ("per_frame_corr.csv", corr)):
    if rows:
        with open(DST / name, "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=list(rows[0].keys())); w.writeheader(); w.writerows(rows)

lines = [f"REVISION EXPERIMENTS - SUMMARY ({len(runs)} runs)", "skill = 1 - MSE_model / MSE_persistence (one step ahead); > 0 means better than copying the previous frame",
         "front = +/- band around the 0.05 c_max contour of the true field; values are RMSE / c_max", ""]
rmap = {(r["cytokine"], r["grid"], r["model"], r["variant"]): r for r in reg if r["window"] == "near"}
omap = {(r["cytokine"], r["grid"], r["model"], r["variant"]): r for r in roll}
for cyt, grid in sorted({(r["cytokine"], r["grid"]) for r in one}):
    sub = [r for r in one if r["cytokine"] == cyt and r["grid"] == grid]
    lines.append(f"=== {cyt.upper()}  {grid}x{grid} ===")
    lines.append(f"{'model/variant':28s}{'n':>2s}{'R2 near':>10s}{'SSIM':>8s}{'Dice':>8s}{'skill n':>9s}{'skill f':>9s}{'front':>9s}{'inter.':>9s}{'roll19':>8s}{'roll19fz':>9s}{'train s':>9s}")
    p, pr = sub[0], rmap[(cyt, grid, sub[0]["model"], sub[0]["variant"])]
    po = next((omap[k] for k in omap if k[0] == cyt and k[1] == grid), None)
    lines.append(f"{'persistence (prev. frame)':28s}{'':>2s}{float(p['near_Global_R2_persistence']):10.4f}{float(p['near_SSIM_persistence']):8.4f}{float(p['near_Avg_Dice_persistence']):8.4f}"
                 f"{0:9.3f}{0:9.3f}{float(pr['front_rmse_rel_cmax_persistence']):9.4f}{float(pr['interior_rmse_rel_cmax_persistence']):9.4f}"
                 + (f"{float(po['lead19_R2_persist_frozen']):8.3f}{float(po['lead19_R2_persist_frozen']):9.3f}" if po else f"{'':>17s}") + f"{'':>9s}")
    for r in sub:
        k = (cyt, grid, r["model"], r["variant"]); rg = rmap[k]; ro = omap.get(k)
        lines.append(f"{(r['model'] + '/' + r['variant'])[:27]:28s}{r['n_seeds']:>2d}{float(r['near_Global_R2']):10.4f}{float(r['near_SSIM']):8.4f}{float(r['near_Avg_Dice']):8.4f}"
                     f"{float(r['near_skill']):9.3f}{float(r['far_skill']):9.3f}{float(rg['front_rmse_rel_cmax_model']):9.4f}{float(rg['interior_rmse_rel_cmax_model']):9.4f}"
                     + (f"{float(ro['lead19_R2_gt']):8.3f}{float(ro['lead19_R2_frozen']):9.3f}" if ro else f"{'':>17s}") + f"{r['train_s']:>9s}")
    lines.append("")
lines.append("Per-frame spatial correlation, lowest frame per configuration (model mean over seeds):")
for (cyt, grid, model, var) in sorted(groups):
    c = [float(x["corr_model_mean"]) for x in corr if (x["cytokine"], x["grid"], x["model"], x["variant"]) == (cyt, grid, model, var)]
    lines.append(f"  {cyt} {grid} {model}/{var}: min {min(c):.4f}  median {float(np.median(c)):.4f}  max {max(c):.4f}")
(DST / "summary.txt").write_text("\n".join(lines) + "\n")
print("\n".join(lines[:60])); print(f"\nwritten to {DST}")
