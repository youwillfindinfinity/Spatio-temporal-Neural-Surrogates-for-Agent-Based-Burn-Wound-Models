"""Frozen-weight results on the other ABM runs (other seeds, modified settings): mean over training
seeds and over the test runs of each experiment. Output: outputs/<REV_OUT>/summary_transfer.txt/.csv"""
import csv, json, os, sys, warnings
from pathlib import Path
import numpy as np
warnings.filterwarnings("ignore")
ROOT = Path(__file__).resolve().parents[2]; sys.path.insert(0, str(ROOT))
from paths import RESULTS, OUTPUTS
TAG = os.environ.get("REV_OUT", "rev"); SRC, DST = RESULTS / TAG, OUTPUTS / TAG; DST.mkdir(parents=True, exist_ok=True)

groups = {}
for p in sorted(SRC.glob("*/*/res_*.json")):
    r = json.load(open(p))
    for run, t in (r.get("transfer") or {}).items():
        groups.setdefault((r["cytokine"], r["grid"], t["experiment"], r["model"], r["variant"]), []).append(t)
if not groups: sys.exit("no transfer results found")
rows = []
for (cyt, grid, exp, model, var), ts in sorted(groups.items()):
    m = lambda f: float(np.nanmean([f(t) for t in ts]))
    rows.append({"cytokine": cyt, "grid": grid, "test_experiment": exp, "model": model, "variant": var, "n": len(ts),
                 "win_R2": m(lambda t: t["window"]["model"]["Global_R2"]), "win_R2_pers": m(lambda t: t["window"]["persistence"]["Global_R2"]),
                 "win_SSIM": m(lambda t: t["window"]["model"]["SSIM"]), "win_Dice": m(lambda t: t["window"]["model"]["Avg_Dice"]),
                 "win_skill": m(lambda t: t["window"]["skill_vs_persistence"]),
                 "full_R2": m(lambda t: t["full"]["R2_model"]), "full_R2_pers": m(lambda t: t["full"]["R2_persistence"]),
                 "full_skill": m(lambda t: t["full"]["skill_vs_persistence"]),
                 "early_skill_t2_20": m(lambda t: np.nanmean(t["full"]["per_frame_skill"][:19])),
                 "roll19_R2": m(lambda t: t["rollout_from_t81"]["R2_model"][-1]), "roll19_pers": m(lambda t: t["rollout_from_t81"]["R2_persist_frozen"][-1]),
                 **{f"start_roll{L}_R2": m(lambda t, L=L: t["rollout_from_start"]["R2_model"][L - 1]) for L in (10, 25, 50, 99)},
                 **{f"start_roll{L}_pers": m(lambda t, L=L: t["rollout_from_start"]["R2_persist_frozen"][L - 1]) for L in (10, 25, 50, 99)}})
with open(DST / "summary_transfer.csv", "w", newline="") as f:
    w = csv.DictWriter(f, fieldnames=list(rows[0].keys())); w.writeheader(); w.writerows([{k: (f"{v:.5g}" if isinstance(v, float) else v) for k, v in r.items()} for r in rows])
L = ["FROZEN-WEIGHT TRANSFER TO OTHER ABM RUNS (trained on the baseline run, seed 1)",
     "win = t 82-100 h; full = t 2-100 h; early = mean per-frame skill for t 2-20 h; roll19 = roll-out from t 81 h;",
     "start-roll = roll-out from the first two frames, R2 after 10 / 25 / 50 / 99 h (persistence = frame at t = 1 h held). n = training seeds x test runs", ""]
for cyt, grid, exp in sorted({(r["cytokine"], r["grid"], r["test_experiment"]) for r in rows}):
    sub = [r for r in rows if (r["cytokine"], r["grid"], r["test_experiment"]) == (cyt, grid, exp)]
    L.append(f"=== {cyt.upper()} {grid}x{grid} | test runs: {exp} ===")
    L.append(f"{'model/variant':30s}{'n':>3s}{'win R2':>8s}{'skill':>8s}{'full R2':>9s}{'skill':>8s}{'early':>8s}{'roll19':>8s} | start-roll R2 {'10h':>6s}{'25h':>7s}{'50h':>7s}{'99h':>7s}")
    s = sub[0]
    L.append(f"{'persistence':30s}{'':>3s}{s['win_R2_pers']:8.4f}{0:8.3f}{s['full_R2_pers']:9.4f}{0:8.3f}{0:8.3f}{s['roll19_pers']:8.3f} |              {s['start_roll10_pers']:7.3f}{s['start_roll25_pers']:7.3f}{s['start_roll50_pers']:7.3f}{s['start_roll99_pers']:7.3f}")
    for r in sub:
        L.append(f"{(r['model'] + '/' + r['variant'])[:29]:30s}{r['n']:>3d}{r['win_R2']:8.4f}{r['win_skill']:8.3f}{r['full_R2']:9.4f}{r['full_skill']:8.3f}{r['early_skill_t2_20']:8.3f}{r['roll19_R2']:8.3f} |              "
                 f"{r['start_roll10_R2']:7.3f}{r['start_roll25_R2']:7.3f}{r['start_roll50_R2']:7.3f}{r['start_roll99_R2']:7.3f}")
    L.append("")
(DST / "summary_transfer.txt").write_text("\n".join(L) + "\n"); print("\n".join(L))
