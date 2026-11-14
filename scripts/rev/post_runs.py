"""Training-free analyses and manuscript figures for the experiments on the revision ABM runs.
Reads results/<REV_OUT> and the preprocessed runs (REV_DATA_ROOT); writes tables and figures
(PNG and PDF) to outputs/<REV_OUT>/figures. CPU only; safe to re-run at any time.

  T1  active pixel fraction per grid, cytokine and window (baseline seeds)
  T2  errors relative to the front of the cytokine field (contour at 0.05 c_max)
  F1  data overview: domain-mean concentrations, cell-type counts, example fields
  F2  one-step skill relative to persistence, increment and direct models
  F3  autoregressive roll-out versus frozen persistence
  F4  frozen-weight transfer to unseen seeds and modified configurations
  F5  operator-network ablation
  F6  effect of training-set size
  F7  error maps with the front contour
  F8  per-frame spatial correlation
  F9  effect of supplying the cell masks at the target time (if those runs exist)
"""
import csv, json, sys, warnings
from pathlib import Path
import numpy as np
from scipy import ndimage
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
import common as C
from paths import OUTPUTS
warnings.filterwarnings("ignore")

SRC = C.OUT
DST = OUTPUTS / C.os.environ.get("REV_OUT", "rev") / "figures"; DST.mkdir(parents=True, exist_ok=True)
plt.rcParams.update({"font.size": 8, "axes.titlesize": 9, "legend.fontsize": 7, "figure.dpi": 120})
ARCH = [("unet", "U-Net"), ("convlstm", "ConvLSTM"), ("fno", "FNO"), ("pointwise", "Point-wise"), ("stalstm", "STA-LSTM")]
# Model colours of the original manuscript (scripts/figures.py), extended along the same palette for the new models.
# The point-wise network is the DeepONet trunk and keeps the DeepONet colour; PI-DeepONet (#ffb14e) and PINN (#cd34b5)
# keep theirs and are not reused.
P_ORIG = {"DeepONet": "#ffd700", "PI-DeepONet": "#ffb14e", "U-Net": "#fa8775", "STA-LSTM": "#ea5f94", "PINN": "#cd34b5"}
COL = {"unet": P_ORIG["U-Net"], "stalstm": P_ORIG["STA-LSTM"], "pointwise": P_ORIG["DeepONet"], "convlstm": "#9d02d7", "fno": "#0000ff"}
CC = ["#e6194b", "#f58231", "#3cb44b", "#4363d8", "#911eb4", "#42d4f4"]          # cytokine colours of the original figures
CMAP_FIELD = "magma"; CMAP_DIFF = "magma"
EDGE = dict(edgecolor="k", linewidth=0.4)
CYT = [("il8", "IL-8"), ("il10", "IL-10")]
GRIDS = sorted({int(p.stem.split("_")[2]) for p in SRC.glob("main_*/*/res_*.json")}) or [100, 250]
lines = []


def P(*a):
    s = " ".join(str(x) for x in a); print(s, flush=True); lines.append(s)


def runs(model, variant, cyt, grid):
    return [json.load(open(p)) for p in sorted((SRC / model / variant).glob(f"res_{cyt}_{grid}_*.json"))]


def ms(v):
    v = np.asarray(v, float)
    return float(np.nanmean(v)), (float(np.nanstd(v, ddof=1)) if len(v) > 1 else 0.0)


def save(fig, name):
    fig.savefig(DST / f"{name}.png", dpi=300, bbox_inches="tight"); fig.savefig(DST / f"{name}.pdf", bbox_inches="tight"); plt.close(fig)
    P(f"  figure {name}")


def guarded(fn):
    try:
        fn()
    except Exception as e:                       # one failing figure must not stop the others
        import traceback; P(f"  SKIPPED {fn.__name__}: {e!r}"); traceback.print_exc()


def phys(grid, cyt, run=None):
    d = C.load(grid, cyt, run)
    return C.den(d["Y"], d["cmax"]), C.den(d["Xu"][..., 11 + d["idx"]], d["cmax"]), d


def signed_distance(active):
    if not active.any() or active.all(): return None
    return np.where(active, ndimage.distance_transform_edt(active) - 0.5, -(ndimage.distance_transform_edt(~active) - 0.5))


# ------------------------------------------------------------------ T1
def t1_active_fraction():
    rows = []
    P("\nT1 ACTIVE PIXEL FRACTION (baseline runs, mean over seeds; % of grid points)")
    P(f"{'grid':>5s} {'cyt':5s} {'window':8s}{'non-zero':>10s}{'> 5% cmax':>11s}{'px > thr':>10s}{'cells':>8s}{'source':>8s}{'px src':>8s}")
    wins = {"t2-20": (0, 19), "train": (0, 70), "near": (80, 90), "far": (90, 99)}
    for grid in GRIDS:
        seeds = [C.train_run(grid)] + [r for r, e in C.test_runs(grid).items() if e == C.TRAIN_EXP]
        for cyt, lab in CYT:
            acc = {w: [] for w in wins}
            for r in seeds:
                truth, _, d = phys(grid, cyt, r)
                raw = np.load(C.run_dir(grid, r) / "Y_raw_phys.npy", mmap_mode="r")[2:, ..., d["idx"]]
                src = d["M"][..., 4] if cyt == "il10" else d["M"][..., 0]        # M2 / endothelial cells
                cells = d["M"].max(-1)
                for w, (a, b) in wins.items():
                    acc[w].append([np.mean(np.asarray(raw[a:b]) > 0), np.mean(truth[a:b] > 0.05 * d["cmax"]), np.mean(cells[a:b] > 0.5), np.mean(src[a:b] > 0.5)])
            for w in wins:
                m = np.mean(acc[w], 0) * 100
                rows.append(dict(grid=grid, cytokine=cyt, window=w, n_seeds=len(seeds), nonzero_pct=round(m[0], 3), above_thr_pct=round(m[1], 3),
                                 above_thr_pixels=int(round(m[1] / 100 * grid * grid)), cell_occupied_pct=round(m[2], 3), source_cells_pct=round(m[3], 4),
                                 source_cells_pixels=round(m[3] / 100 * grid * grid, 1)))
                P(f"{grid:>5d} {cyt:5s} {w:8s}{m[0]:10.2f}{m[1]:11.2f}{rows[-1]['above_thr_pixels']:10d}{m[2]:8.2f}{m[3]:8.3f}{rows[-1]['source_cells_pixels']:8.1f}")
    with open(DST.parent / "active_fraction.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys())); w.writeheader(); w.writerows(rows)


# ------------------------------------------------------------------ T2 + F7
SHOW = [("main_unet", "increment", "U-Net incr."), ("main_convlstm", "increment", "ConvLSTM incr."), ("main_fno", "increment", "FNO incr."),
        ("main_pointwise", "increment", "Point-wise incr."), ("main_unet", "direct", "U-Net direct"), ("deeponet", "base", "DeepONet")]


def t2_front_and_maps():
    rows = []
    P("\nT2 ERROR RELATIVE TO THE FIELD FRONT (test window, RMSE / c_max; share of squared error)")
    for grid in GRIDS:
        for cyt, lab in CYT:
            truth, pers, d = phys(grid, cyt); truth, pers = truth[C.START:C.END], pers[C.START:C.END]; cmax = d["cmax"]
            cand = [("Persistence", [pers])]
            for m, v, name in SHOW:
                ps = [np.load(p)["one_step"].astype(np.float64) for p in sorted((SRC / m / v).glob(f"pred_{cyt}_{grid}_*.npz"))]
                if ps: cand.append((name, ps))
            P(f"=== {lab} {grid}x{grid} ===")
            P(f"{'model':18s}{'front':>9s}{'inside':>9s}{'outside':>9s}{'px front':>10s}{'px inside':>10s}{'err front':>10s}{'err inside':>11s}")
            for name, ps in cand:
                out = []
                for pred in ps:
                    acc = {k: [0.0, 0] for k in ("front", "inside", "outside")}
                    for t in range(truth.shape[0]):
                        sd = signed_distance(truth[t] > 0.05 * cmax)
                        if sd is None: continue
                        se = (truth[t] - pred[t]) ** 2
                        for k, msk in (("front", np.abs(sd) <= 1), ("inside", sd > 1), ("outside", sd < -1)):
                            acc[k][0] += se[msk].sum(); acc[k][1] += msk.sum()
                    tot = sum(v[0] for v in acc.values()); n = sum(v[1] for v in acc.values())
                    out.append([np.sqrt(acc[k][0] / max(acc[k][1], 1)) / cmax for k in acc] + [acc[k][1] / max(n, 1) for k in ("front", "inside")] + [acc[k][0] / tot if tot else np.nan for k in ("front", "inside")])
                o = np.nanmean(out, 0)
                rows.append(dict(grid=grid, cytokine=cyt, model=name, n=len(ps), front_rmse_rel=o[0], inside_rmse_rel=o[1], outside_rmse_rel=o[2],
                                 front_pixel_share=o[3], inside_pixel_share=o[4], front_error_share=o[5], inside_error_share=o[6]))
                P(f"{name:18s}{o[0]:9.4f}{o[1]:9.4f}{o[2]:9.4f}{o[3]:10.3f}{o[4]:10.3f}{o[5]:10.3f}{o[6]:11.3f}")
            # error maps at t = 88 h, first seed
            k = 88 - 82; panels = [(n, p[0][k]) for n, p in cand]
            fig, ax = plt.subplots(1, len(panels) + 1, figsize=(2.3 * (len(panels) + 1), 2.6))
            ax[0].imshow(truth[k].T, origin="lower", cmap=CMAP_FIELD, vmax=np.percentile(truth[k], 99.5)); ax[0].set_title("ABM")
            vmax = max(np.percentile(np.abs(truth[k] - q), 99.5) for _, q in panels[1:]) if len(panels) > 1 else None
            for a, (n, q) in zip(ax[1:], panels):
                im = a.imshow(np.abs(truth[k] - q).T, origin="lower", cmap=CMAP_DIFF, vmin=0, vmax=vmax); a.set_title(n)
            for a in ax:
                a.contour((truth[k] > 0.05 * cmax).T.astype(float), levels=[0.5], colors="cyan", linewidths=0.5); a.set_xticks([]); a.set_yticks([])
            fig.colorbar(im, ax=ax[1:], fraction=0.012, pad=0.01, label="|error|")
            fig.suptitle(f"{lab}, {grid}$^2$, t = 88 h; cyan: contour at 5 % of c_max", y=1.02)
            save(fig, f"F7_error_maps_{cyt}_{grid}")
    with open(DST.parent / "front_analysis.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys())); w.writeheader(); w.writerows(rows)


# ------------------------------------------------------------------ F1
def f1_data_overview():
    fig, ax = plt.subplots(2, 2 * len(GRIDS), figsize=(4.2 * 2 * len(GRIDS), 5.6), squeeze=False)
    t = np.arange(2, 101); names = ["EC", "resting neutr.", "activated neutr.", "M1", "M2"]
    for gi, grid in enumerate(GRIDS):
        seeds = [C.train_run(grid)] + [r for r, e in C.test_runs(grid).items() if e == C.TRAIN_EXP]
        a = ax[0, 2 * gi]
        for (cyt, lab), c in zip(CYT, (CC[C.CYTS.index("il8")], CC[C.CYTS.index("il10")])):
            m = np.array([phys(grid, cyt, r)[0].mean((1, 2)) for r in seeds]); mu, sd = m.mean(0), m.std(0)
            a.plot(t, mu / mu.max(), color=c, label=f"{lab} (max {mu.max():.1e})"); a.fill_between(t, (mu - sd) / mu.max(), (mu + sd) / mu.max(), color=c, alpha=0.25)
        a.set_title(f"{grid}$^2$: domain mean, {len(seeds)} seeds"); a.set_xlabel("time (h)"); a.set_ylabel("normalised concentration"); a.legend()
        a = ax[0, 2 * gi + 1]; M = C.load(grid, "il8")["M"]
        for j, n in enumerate(names): a.plot(t, M[..., j].sum((1, 2)), label=n)
        a.set_yscale("symlog", linthresh=10); a.set_title(f"{grid}$^2$: lattice sites per cell type (seed 1)"); a.set_xlabel("time (h)"); a.legend(ncol=2)
        for ci, (cyt, lab) in enumerate(CYT):
            truth, _, d = phys(grid, cyt); k = 86; a = ax[1, 2 * gi + ci]
            im = a.imshow(truth[k].T, origin="lower", cmap=CMAP_FIELD, vmax=np.percentile(truth[k], 99.5))
            ys, xs = np.where(d["M"][k, ..., 4 if cyt == "il10" else 0].T > 0.5); a.plot(xs, ys, ".", color="white", ms=1.5, alpha=0.8)
            a.set_title(f"{lab}, {grid}$^2$, t = 88 h; dots: {'M2' if cyt == 'il10' else 'endothelial'} cells"); a.set_xticks([]); a.set_yticks([]); fig.colorbar(im, ax=a, fraction=0.046)
    fig.tight_layout(); save(fig, "F1_data_overview")


# ------------------------------------------------------------------ F2
def f2_skill():
    fig, ax = plt.subplots(2, 2 * len(GRIDS), figsize=(3.4 * 2 * len(GRIDS), 5.4), squeeze=False)
    P("\nF2 ONE-STEP SKILL (near horizon, mean +/- SD over training seeds)")
    col = 0
    for cyt, lab in CYT:
        for grid in GRIDS:
            for row, form in enumerate(("increment", "direct")):
                a = ax[row, col]; vals = []
                for i, (m, n) in enumerate(ARCH):
                    rs = runs(f"main_{m}", form, cyt, grid)
                    if not rs: continue
                    mu, sd = ms([r["windows"]["near"]["skill_vs_persistence"] for r in rs]); vals.append(mu)
                    a.bar(i, mu, yerr=sd, color=COL[m], capsize=2, **EDGE); P(f"  {lab} {grid} {n:11s} {form:9s} skill {mu:+9.3f} +/- {sd:.3f} (n={len(rs)})")
                a.axhline(0, color="k", lw=0.8); a.set_xticks(range(len(ARCH))); a.set_xticklabels([n for _, n in ARCH], rotation=35, ha="right")
                if form == "direct" or (vals and min(vals) < -1.5): a.set_yscale("symlog", linthresh=1)
                a.set_title(f"{lab}, {grid}$^2$ - {form}")
                if col == 0: a.set_ylabel("skill vs persistence")
            col += 1
    fig.tight_layout(); save(fig, "F2_skill")


# ------------------------------------------------------------------ F3
def f3_rollout():
    fig, ax = plt.subplots(1, 2 * len(GRIDS), figsize=(3.4 * 2 * len(GRIDS), 3.0), squeeze=False); lead = np.arange(1, 20); col = 0
    P("\nF3 ROLL-OUT, R2 after 19 h (other cytokines frozen at t = 81 h | supplied by the ABM)")
    for cyt, lab in CYT:
        for grid in GRIDS:
            a = ax[0, col]; done = False
            for m, n in ARCH:
                rs = [r for r in runs(f"main_{m}", "increment", cyt, grid) if r.get("rollout")]
                if not rs: continue
                if not done:
                    a.plot(lead, rs[0]["rollout"]["R2_persist_frozen"], "k--", label="frozen persistence"); done = True
                    P(f"  {lab} {grid} frozen persistence {rs[0]['rollout']['R2_persist_frozen'][-1]:.3f}")
                fz = np.nanmean([r["rollout"]["R2_frozen"] for r in rs], 0); gt = np.nanmean([r["rollout"]["R2_gt"] for r in rs], 0)
                a.plot(lead, fz, color=COL[m], label=n); P(f"  {lab} {grid} {n:11s} {fz[-1]:.3f} | {gt[-1]:.3f}")
            a.set_title(f"{lab}, {grid}$^2$"); a.set_xlabel("lead time (h)")
            if col == 0: a.set_ylabel("per-frame $R^2$"); a.legend()
            col += 1
    fig.tight_layout(); save(fig, "F3_rollout")


# ------------------------------------------------------------------ F4
def f4_transfer():
    P("\nF4 FROZEN-WEIGHT TRANSFER, skill in the window t 82-100 h (increment models)")
    fig, ax = plt.subplots(1, 2 * len(GRIDS), figsize=(3.6 * 2 * len(GRIDS), 3.2), squeeze=False); col = 0
    for cyt, lab in CYT:
        for grid in GRIDS:
            a = ax[0, col]; groups = None
            for i, (m, n) in enumerate(ARCH):
                rs = runs(f"main_{m}", "increment", cyt, grid)
                if not rs or not rs[0].get("transfer"): continue
                by = {"training run": [r["windows"]["near"]["skill_vs_persistence"] * 10 / 19 + r["windows"]["far"]["skill_vs_persistence"] * 9 / 19 for r in rs]}
                for r in rs:
                    for run, t in r["transfer"].items(): by.setdefault(t["experiment"], []).append(t["window"]["skill_vs_persistence"])
                groups = groups or list(by)
                for j, g in enumerate(groups):
                    if g in by:
                        a.bar(j + (i - 2) * 0.16, np.mean(by[g]), 0.16, color=COL[m], label=n if j == 0 else None, **EDGE)
                        P(f"  {lab} {grid} {n:11s} {g:36s} {np.mean(by[g]):+.3f} (n={len(by[g])})")
            if groups:
                a.set_xticks(range(len(groups))); a.set_xticklabels([g.replace("_il10_from_m2", " seeds").replace("_m2", "").replace("_", " ") for g in groups], rotation=30, ha="right", fontsize=6)
            a.axhline(0, color="k", lw=0.8); a.set_title(f"{lab}, {grid}$^2$")
            if col == 0: a.set_ylabel("skill vs persistence"); a.legend()
            col += 1
    fig.tight_layout(); save(fig, "F4_transfer")


# ------------------------------------------------------------------ F5
def f5_ablation():
    V = [("base", "full"), ("no_branch", "no branch"), ("no_gate", "plain trunk"), ("no_fields", "no local fields")]
    fig, ax = plt.subplots(2, 2 * len(GRIDS), figsize=(3.2 * 2 * len(GRIDS), 5.0), squeeze=False); col = 0
    P("\nF5 OPERATOR-NETWORK ABLATION: near R2 | skill | roll-out R2 at 19 h (others frozen)")
    for cyt, lab in CYT:
        for grid in GRIDS:
            for i, (v, n) in enumerate(V):
                rs = runs("deeponet", v, cyt, grid)
                if not rs: continue
                r2 = ms([r["windows"]["near"]["model"]["Global_R2"] for r in rs]); sk = ms([r["windows"]["near"]["skill_vs_persistence"] for r in rs])
                ro = ms([r["rollout"]["R2_frozen"][-1] for r in rs if r.get("rollout")])
                ax[0, col].bar(i, sk[0], yerr=sk[1], color=P_ORIG["DeepONet"], capsize=2, **EDGE); ax[1, col].bar(i, ro[0], yerr=ro[1], color=P_ORIG["DeepONet"], capsize=2, **EDGE)
                P(f"  {lab} {grid} {n:16s} {r2[0]:.4f} | {sk[0]:+9.3f} | {ro[0]:+.3f} (n={len(rs)})")
            for r in (0, 1):
                ax[r, col].set_xticks(range(len(V))); ax[r, col].set_xticklabels([n for _, n in V], rotation=30, ha="right"); ax[r, col].axhline(0, color="k", lw=0.8)
            ax[0, col].set_yscale("symlog", linthresh=1); ax[0, col].set_title(f"{lab}, {grid}$^2$"); ax[1, col].set_ylim(-1, 1.02)
            if col == 0: ax[0, col].set_ylabel("skill vs persistence"); ax[1, col].set_ylabel("roll-out $R^2$ at 19 h")
            col += 1
    fig.tight_layout(); save(fig, "F5_ablation")


# ------------------------------------------------------------------ F6
def f6_data_size():
    fig, ax = plt.subplots(1, 2 * len(GRIDS), figsize=(3.2 * 2 * len(GRIDS), 2.9), squeeze=False); col = 0
    P("\nF6 TRAINING-SET SIZE (U-Net): skill on the training run | window skill on unseen baseline seeds")
    for cyt, lab in CYT:
        for grid in GRIDS:
            a = ax[0, col]
            for form, ls in (("increment", "-"), ("direct", "--")):
                own, new = [], []
                for n in (1, 2, 3):
                    rs = runs("multitraj_unet", f"{form}_{n}traj", cyt, grid)
                    if not rs: own.append(np.nan); new.append(np.nan); continue
                    own.append(np.mean([r["windows"]["near"]["skill_vs_persistence"] for r in rs]))
                    tr = [t["window"]["skill_vs_persistence"] for r in rs for run, t in (r.get("transfer") or {}).items() if t["experiment"] == C.TRAIN_EXP and run[-2:] in ("s4", "s5")]
                    new.append(np.mean(tr) if tr else np.nan)
                    P(f"  {lab} {grid} {form:9s} {n} traj: {own[-1]:+9.3f} | {new[-1]:+9.3f}")
                a.plot([1, 2, 3], new, ls, marker="o", color=COL["unet"], label=f"{form}, unseen seeds 4-5")
            a.set_yscale("symlog", linthresh=1); a.axhline(0, color="k", lw=0.8); a.set_xticks([1, 2, 3]); a.set_xlabel("training trajectories"); a.set_title(f"{lab}, {grid}$^2$")
            if col == 0: a.set_ylabel("skill vs persistence"); a.legend()
            col += 1
    fig.tight_layout(); save(fig, "F6_data_size")


# ------------------------------------------------------------------ F8
def f8_correlation():
    fig, ax = plt.subplots(1, 2 * len(GRIDS), figsize=(3.4 * 2 * len(GRIDS), 3.2), squeeze=False); col = 0
    P("\nF8 PER-FRAME SPATIAL CORRELATION (19 test frames x seeds): min / median / max")
    for cyt, lab in CYT:
        for grid in GRIDS:
            vals, labs = [], []
            for m, v, n in SHOW:
                rs = runs(m, v, cyt, grid)
                if not rs: continue
                if not vals: vals.append(np.array(rs[0]["per_frame"]["corr_persistence"])); labs.append("Persistence")
                c = np.concatenate([r["per_frame"]["corr_model"] for r in rs]); vals.append(c); labs.append(n)
            for n, c in zip(labs, vals): P(f"  {lab} {grid} {n:18s} {np.nanmin(c):.4f} / {np.nanmedian(c):.4f} / {np.nanmax(c):.4f}")
            if vals:
                ax[0, col].boxplot(vals, vert=False, whis=(0, 100)); ax[0, col].set_yticklabels(labs)
            ax[0, col].set_title(f"{lab}, {grid}$^2$"); ax[0, col].set_xlabel("per-frame spatial correlation"); col += 1
    fig.tight_layout(); save(fig, "F8_per_frame_correlation")


# ------------------------------------------------------------------ F9
def f9_target_masks():
    if not list(SRC.glob("tmask_*/*/res_*.json")): P("\nF9: no target-time-mask runs found"); return
    P("\nF9 CELL MASKS AT THE TARGET TIME: skill near | far | unseen baseline seeds (window) | roll-out R2 19 h (others frozen)")
    fig, ax = plt.subplots(1, 2 * len(GRIDS), figsize=(3.4 * 2 * len(GRIDS), 3.0), squeeze=False); col = 0
    for cyt, lab in CYT:
        for grid in GRIDS:
            a = ax[0, col]
            for i, (m, n) in enumerate(ARCH):
                for j, (pre, hatch) in enumerate((("main", ""), ("tmask", "//"))):
                    for form in ("increment", "direct"):
                        rs = runs(f"{pre}_{m}", form, cyt, grid)
                        if not rs: continue
                        nr = ms([r["windows"]["near"]["skill_vs_persistence"] for r in rs]); fr = ms([r["windows"]["far"]["skill_vs_persistence"] for r in rs])
                        tr = [t["window"]["skill_vs_persistence"] for r in rs for t in (r.get("transfer") or {}).values() if t["experiment"] == C.TRAIN_EXP]
                        ro = np.nanmean([r["rollout"]["R2_frozen"][-1] for r in rs if r.get("rollout")])
                        P(f"  {lab} {grid} {n:11s} {form:9s} {'with' if pre == 'tmask' else 'without':7s} {nr[0]:+8.3f} | {fr[0]:+8.3f} | {np.mean(tr) if tr else float('nan'):+8.3f} | {ro:+.3f}")
                        if form == "increment":
                            a.bar(i + (j - 0.5) * 0.36, nr[0], 0.36, yerr=nr[1], color=COL[m], hatch=hatch, capsize=2, **EDGE)
            a.axhline(0, color="k", lw=0.8); a.set_xticks(range(len(ARCH))); a.set_xticklabels([n for _, n in ARCH], rotation=35, ha="right")
            a.set_title(f"{lab}, {grid}$^2$ (hatched: masks at target time)")
            if col == 0: a.set_ylabel("skill vs persistence (increment)")
            col += 1
    fig.tight_layout(); save(fig, "F9_target_time_masks")


if __name__ == "__main__":
    P(f"results: {SRC}\noutput:  {DST}\ngrids:   {GRIDS}")
    for fn in (t1_active_fraction, f1_data_overview, f2_skill, f3_rollout, f4_transfer, f5_ablation, f6_data_size, t2_front_and_maps, f8_correlation, f9_target_masks):
        guarded(fn)
    (DST.parent / "post_runs_summary.txt").write_text("\n".join(lines) + "\n")
    print(f"\nsummary written to {DST.parent / 'post_runs_summary.txt'}")
