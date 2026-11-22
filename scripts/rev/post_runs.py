"""Training-free analyses and figures for the paper from the ABM-run experiments.
Reads results/<REV_OUT> and the preprocessed runs (REV_DATA_ROOT); writes tables and figures
(PNG and PDF) to outputs/figures, named after their number in the paper (FigA*: appendix). CPU only.

  Table 2    active pixel fraction per grid, cytokine and window (baseline seeds)
  Fig. 1     data overview: domain-mean concentrations, cell-type counts, example fields
  Fig. 3     one-step skill relative to persistence, increment and direct models
  Fig. 4     autoregressive roll-out versus frozen persistence
  Fig. 5     frozen-weight transfer to unseen seeds and modified configurations
  Fig. 6     effect of supplying the cell masks at the target time (if those runs exist)
  Fig. 7     operator-network ablation
  Fig. A.1   per-frame spatial correlation
  Fig. A.2   effect of training-set size
  Fig. A.3   errors relative to the front of the cytokine field (contour at 0.05 c_max); error maps of both cytokines and grids
Fig. 2 and the graphical abstract are drawn by hand.
"""
import csv, json, sys, warnings
from pathlib import Path
import numpy as np
from scipy import ndimage
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm, Normalize
from matplotlib.ticker import MaxNLocator
import common as C
from paths import OUTPUTS
warnings.filterwarnings("ignore")

SRC = C.OUT
DST = OUTPUTS / "figures"; DST.mkdir(parents=True, exist_ok=True)
TABLES = OUTPUTS / SRC.name; TABLES.mkdir(parents=True, exist_ok=True)   # tables and summary
plt.rcParams.update({"font.size": 8, "axes.titlesize": 9, "legend.fontsize": 7, "figure.dpi": 120})
ARCH = [("unet", "U-Net"), ("convlstm", "ConvLSTM"), ("fno", "FNO"), ("pointwise", "Point-wise"), ("stalstm", "STA-LSTM")]
# colours as in scripts/figures.py; the point-wise network keeps the DeepONet colour
P_ORIG = {"DeepONet": "#ffd700", "PI-DeepONet": "#ffb14e", "U-Net": "#fa8775", "STA-LSTM": "#ea5f94", "PINN": "#cd34b5"}
COL = {"unet": P_ORIG["U-Net"], "stalstm": P_ORIG["STA-LSTM"], "pointwise": P_ORIG["DeepONet"], "convlstm": "#9d02d7", "fno": "#0000ff"}
CC = ["#e6194b", "#f58231", "#3cb44b", "#4363d8", "#911eb4", "#42d4f4"]          # cytokine colours as in scripts/figures.py
CMAP_FIELD = "magma"; CMAP_DIFF = "magma"
FIELD_RANGE = (1e-9, 1e-3)                       # shared colour range (pg/mL) of the concentration maps in Fig. 1
def pg_per_ml(grid):
    """ABM fields are amounts per voxel (pg); a voxel is (5/G cm)^2 x 1 cm = 25/G^2 mL."""
    return grid ** 2 / 25.0
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


def table2_active_fraction():
    rows = []
    P("\nTABLE 2  ACTIVE PIXEL FRACTION (baseline runs, mean over seeds; % of grid points)")
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
    with open(TABLES / "active_fraction.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys())); w.writeheader(); w.writerows(rows)


MAP_SEED = 1                                     # seed shown in the error maps (Fig. A.3)
SHOW = [("main_unet", "increment", "U-Net incr."), ("main_convlstm", "increment", "ConvLSTM incr."), ("main_fno", "increment", "FNO incr."),
        ("main_pointwise", "increment", "Point-wise incr."), ("main_unet", "direct", "U-Net direct"), ("deeponet", "base", "DeepONet")]


def front_and_error_maps():
    rows, maps = [], []
    P("\nFIG. A.3  ERROR RELATIVE TO THE FIELD FRONT (test window, RMSE / c_max; share of squared error)")
    missing = [f for m, v, _ in SHOW for f in (SRC / m / v).glob("res_*.json") if not f.with_name(f.name.replace("res_", "pred_").replace(".json", ".npz")).exists()]
    if missing:
        P(f"  predictions of {len(missing)} trained models not found: front statistics as stored in {(TABLES / 'front_analysis.csv').relative_to(C.ROOT)}; maps redrawn")
        if (TABLES / "front_analysis.csv").exists():
            with open(TABLES / "front_analysis.csv") as f:
                stored = list(csv.DictReader(f))
            for grid in GRIDS:
                for cyt, lab in CYT:
                    P(f"=== {lab} {grid}x{grid} ===")
                    P(f"{'model':18s}{'front':>9s}{'inside':>9s}{'outside':>9s}{'px front':>10s}{'px inside':>10s}{'err front':>10s}{'err inside':>11s}")
                    for r in (r for r in stored if r["grid"] == str(grid) and r["cytokine"] == cyt):
                        o = [float(r[k]) for k in ("front_rmse_rel", "inside_rmse_rel", "outside_rmse_rel", "front_pixel_share", "inside_pixel_share", "front_error_share", "inside_error_share")]
                        P(f"{r['model']:18s}{o[0]:9.4f}{o[1]:9.4f}{o[2]:9.4f}{o[3]:10.3f}{o[4]:10.3f}{o[5]:10.3f}{o[6]:11.3f}")
        redraw_error_maps(); return
    for grid in GRIDS:
        for cyt, lab in CYT:
            truth, pers, d = phys(grid, cyt); truth, pers = truth[C.START:C.END], pers[C.START:C.END]; cmax = d["cmax"]
            cand = [("Persistence", [pers])]
            for m, v, name in SHOW:
                files = sorted((SRC / m / v).glob(f"pred_{cyt}_{grid}_*.npz"), key=lambda f: f.stem.rsplit("_", 1)[1] != str(MAP_SEED))   # MAP_SEED first
                ps = [np.load(p)["one_step"].astype(np.float64) for p in files]
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
            # error maps at t = 88 h (pg/mL)
            k = 88 - 82
            maps.append(dict(grid=grid, cyt=cyt, lab=lab, truth=truth[k], cmax=cmax,
                             panels=[(n, np.abs(truth[k] - p[0][k]) * pg_per_ml(grid)) for n, p in cand]))
    with open(TABLES / "front_analysis.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys())); w.writeheader(); w.writerows(rows)
    plot_error_maps(maps)


def redraw_error_maps(pred_root=None):
    """Redraw Fig. A.3 only, from the predictions of seed MAP_SEED."""
    root, maps, k = Path(pred_root) if pred_root else SRC, [], 88 - 82
    for grid in GRIDS:
        for cyt, lab in CYT:
            truth, pers, d = phys(grid, cyt); truth, pers = truth[C.START:C.END], pers[C.START:C.END]
            panels = [("Persistence", np.abs(truth[k] - pers[k]) * pg_per_ml(grid))]
            for m, v, n in SHOW:
                f = root / m / v / f"pred_{cyt}_{grid}_{MAP_SEED}.npz"
                if f.exists(): panels.append((n, np.abs(truth[k] - np.load(f)["one_step"][k].astype(np.float64)) * pg_per_ml(grid)))
                else: P(f"  missing {f.relative_to(C.ROOT) if f.is_relative_to(C.ROOT) else f}")
            maps.append(dict(grid=grid, cyt=cyt, lab=lab, truth=truth[k], cmax=d["cmax"], panels=panels))
    plot_error_maps(maps)


def plot_error_maps(maps):
    """Fig. A.3: absolute error at t = 88 h (pg/mL). Rows: models; columns: cytokine and grid; one colour scale from 0."""
    order = {c: i for i, (c, _) in enumerate(CYT)}
    maps = sorted(maps, key=lambda m: (order[m["cyt"]], m["grid"]))
    errs = [e for m in maps for _, e in m["panels"]]
    if not errs: return
    norm = Normalize(0.0, max(float(e.max()) for e in errs))
    P(f"  error maps: one linear colour scale 0 - {norm.vmax:.2e} pg/mL (largest error in any panel)")
    names = [n for n, _ in maps[0]["panels"]]
    nr, nc = len(names), len(maps)
    left, right, bottom, top, cell = 0.13, 0.86, 0.01, 0.97, 1.28             # square panels, no gaps
    fig, ax = plt.subplots(nr, nc, figsize=(cell * nc / (right - left), cell * nr / (top - bottom)), squeeze=False)
    fig.subplots_adjust(left=left, right=right, bottom=bottom, top=top, wspace=0.02, hspace=0.02)
    for j, m in enumerate(maps):
        panels = dict(m["panels"]); front = (m["truth"] > 0.05 * m["cmax"]).T.astype(float)
        lw, alpha = (0.2, 0.6) if m["cyt"] == "il10" else (0.5, 1.0)   # thinner contours for IL-10
        for i, n in enumerate(names):
            a = ax[i, j]
            if n in panels: im = a.imshow(panels[n].T, origin="lower", cmap=CMAP_DIFF, norm=norm)
            a.contour(front, levels=[0.5], colors="cyan", linewidths=lw, alpha=alpha); a.set_xticks([]); a.set_yticks([])
            if i == 0: a.set_title(f"{m['lab']}, {m['grid']}$^2$", fontsize=9)
            if j == 0: a.set_ylabel(n, fontsize=8.5)
    p0, p1 = ax[0, -1].get_position(), ax[-1, -1].get_position()
    cax = fig.add_axes([p0.x1 + 0.015, p1.y0, 0.022, p0.y1 - p1.y0])          # one colour bar
    fig.colorbar(im, cax=cax, label="|error| (pg/mL)")
    save(fig, "FigA3_error_maps")

def fig1_data_overview():
    """Fig. 1: mean concentrations and cell counts over the baseline seeds, and the fields of seed 1 at t = 88 h."""
    fig = plt.figure(figsize=(17, 7.2), layout="constrained")
    gs = fig.add_gridspec(2, 4, height_ratios=[1.15, 1])
    names = ["EC", "resting neutr.", "activated neutr.", "M1", "M2"]
    for gi, grid in enumerate(GRIDS):
        seeds = [C.train_run(grid)] + [r for r, e in C.test_runs(grid).items() if e == C.TRAIN_EXP]
        raw = [np.load(C.run_dir(grid, r) / "Y_raw_phys.npy", mmap_mode="r") for r in seeds]     # (101, G, G, 6), t = 0-100 h
        sub = gs[0, gi].subgridspec(2, 1, height_ratios=[3, 1.1])
        a = fig.add_subplot(sub[0]); b = fig.add_subplot(sub[1], sharex=a)
        t = np.arange(1, raw[0].shape[0]); finals = []
        for cyt, lab in CYT:
            j = C.CYTS.index(cyt); m = np.array([x[1:, ..., j].mean((1, 2)) for x in raw]) * pg_per_ml(grid)
            mu, sd = m.mean(0), m.std(0, ddof=1); finals.append(mu[-1])
            a.plot(t, mu, color=CC[j], label=lab); a.fill_between(t, np.maximum(mu - sd, mu * 1e-3), mu + sd, color=CC[j], alpha=0.3, lw=0)
            ok = mu > 0; rel = np.where(ok, 100 * sd / np.where(ok, mu, 1), np.nan)
            b.fill_between(t, -rel, rel, color=CC[j], alpha=0.35, lw=0); b.plot(t, rel, color=CC[j], lw=0.6); b.plot(t, -rel, color=CC[j], lw=0.6)
        a.set_yscale("log"); a.set_ylim(min(finals) * 1e-3, None); a.tick_params(labelbottom=False)
        a.set_title(f"{grid}$^2$: domain-mean concentration, {len(seeds)} seeds"); a.set_ylabel("concentration (pg/mL)"); a.legend()
        b.axhline(0, color="k", lw=0.5); b.set_yscale("symlog", linthresh=1); b.set_ylim(-50, 50)
        b.set_yticks([-10, -1, 0, 1, 10]); b.set_yticklabels(["\u221210", "\u22121", "0", "1", "10"])
        b.set_ylabel("\u00b1SD (% of mean)"); b.set_xlabel("time (h)")
        a = fig.add_subplot(gs[0, 2 + gi])
        M = np.array([np.load(C.run_dir(grid, r) / "Y_masks_spatial.npy", mmap_mode="r").sum((1, 2)) for r in seeds])   # (seeds, 99, 5)
        tm = np.arange(2, 2 + M.shape[1])
        for j, n in enumerate(names):
            mu, sd = M[..., j].mean(0), M[..., j].std(0, ddof=1)
            a.plot(tm, mu, label=n); a.fill_between(tm, np.maximum(mu - sd, 0), mu + sd, alpha=0.3, lw=0)
        a.set_yscale("symlog", linthresh=10); a.set_title(f"{grid}$^2$: voxels occupied per cell type, {len(seeds)} seeds")
        a.set_xlabel("time (h)"); a.set_ylabel("voxels occupied"); a.legend(ncol=2)
    vmin, vmax = FIELD_RANGE; maps = []
    for gi, grid in enumerate(GRIDS):
        raw = np.load(C.run_dir(grid) / "Y_raw_phys.npy", mmap_mode="r"); M = np.load(C.run_dir(grid) / "Y_masks_spatial.npy", mmap_mode="r")
        for ci, (cyt, lab) in enumerate(CYT):
            a = fig.add_subplot(gs[1, 2 * gi + ci]); maps.append(a); f = np.clip(raw[88, ..., C.CYTS.index(cyt)] * pg_per_ml(grid), vmin, None)
            im = a.imshow(f.T, origin="lower", cmap=CMAP_FIELD, norm=LogNorm(vmin, vmax))
            ys, xs = np.where(M[86, ..., 4 if cyt == "il10" else 0].T > 0.5); a.plot(xs, ys, ".", color="white", ms=1.5, alpha=0.8)   # sample 86 = t 88 h
            a.set_title(f"{lab}, {grid}$^2$, t = 88 h; dots: {'M2' if cyt == 'il10' else 'endothelial'} cells"); a.set_xticks([]); a.set_yticks([])
    fig.colorbar(im, ax=maps, label="concentration (pg/mL)", fraction=0.015, pad=0.01)
    save(fig, "Fig1_data_overview")


def mean_sd_label(a, x, mu, sd, lo, hi, bg=False):
    """Write mean +/- SD above (positive) or below (negative) the bar and its seed dots (lo, hi), rounded as in Table 3:
    mean to three significant digits (at most four decimals), SD to two significant digits."""
    d = min(4, max(0, 2 - int(np.floor(np.log10(abs(mu)))))) if mu else 3
    dsd = max(0, 1 - int(np.floor(np.log10(sd)))) if sd > abs(mu) else d      # SD larger than the mean: two digits of its own
    sd2 = float(f"{sd:.2g}")
    up = mu >= 0
    a.annotate(f"{mu:.{d}f}\n±{sd2:.{dsd}f}".replace("-", "−"), (x, max(mu, hi) if up else min(mu, lo)),
               xytext=(0, 3 if up else -3), textcoords="offset points", ha="center",
               va="bottom" if up else "top", fontsize=7, linespacing=0.95,
               bbox=dict(fc="white", ec="none", alpha=0.8, pad=0.4) if bg else None, zorder=4)


def fit_axis(a, vals, log):
    """Symmetric-log axis if asked, and room above and below the bars for the mean +/- SD labels; no ticks above 1."""
    if log: a.set_yscale("symlog", linthresh=1)
    lo, hi = a.get_ylim()
    if log:
        a.set_ylim(lo * 15 if lo < -1 else lo - 1, (hi * 15 if hi > 1 else hi + 1) if max(vals) >= 0 else hi)
    else:
        a.set_ylim(lo - 0.3 * (hi - lo) if min(vals) < 0 else lo, hi + 0.3 * (hi - lo))
        lo, hi = a.get_ylim()
        a.set_yticks([t for t in a.get_yticks() if lo <= t <= min(hi, 1)])   # skill and R2 cannot exceed 1


def fig3_skill():
    fig, ax = plt.subplots(2, 2 * len(GRIDS), figsize=(3.4 * 2 * len(GRIDS), 5.4), squeeze=False)
    P("\nFIG. 3  ONE-STEP SKILL (near horizon, mean +/- SD over training seeds)")
    col = 0
    for cyt, lab in CYT:
        for grid in GRIDS:
            for row, form in enumerate(("increment", "direct")):
                a = ax[row, col]; vals = []
                for i, (m, n) in enumerate(ARCH):
                    rs = runs(f"main_{m}", form, cyt, grid)
                    if not rs: continue
                    seeds = [r["windows"]["near"]["skill_vs_persistence"] for r in rs]; mu, sd = ms(seeds); vals.append(mu)
                    a.bar(i, mu, color=COL[m], **EDGE); P(f"  {lab} {grid} {n:11s} {form:9s} skill {mu:+9.3f} +/- {sd:.3f} (n={len(rs)})")
                    a.plot(i + np.linspace(-0.2, 0.2, len(seeds)), seeds, "o", ms=3, mfc="white", mec="k", mew=0.7, zorder=3)   # one dot per seed
                    mean_sd_label(a, i, mu, sd, min(seeds), max(seeds))
                a.axhline(0, color="k", lw=0.8); a.set_xticks(range(len(ARCH))); a.set_xticklabels([n for _, n in ARCH], rotation=35, ha="right")
                fit_axis(a, vals, log=form == "direct" or (vals and min(vals) < -1.5))
                a.set_title(f"{lab}, {grid}$^2$ - {form}")
                if col == 0: a.set_ylabel("skill relative to persistence")
            col += 1
    fig.tight_layout(); save(fig, "Fig3_skill")


def fig4_rollout():
    fig, ax = plt.subplots(1, 2 * len(GRIDS), figsize=(3.4 * 2 * len(GRIDS), 3.0), squeeze=False); lead = np.arange(1, 20); col = 0
    P("\nFIG. 4  ROLL-OUT, R2 after 19 h (other cytokines frozen at t = 81 h | supplied by the ABM)")
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
    fig.tight_layout(); save(fig, "Fig4_rollout")


TRANSFER_SETS = [("training", "Training run"), (C.TRAIN_EXP, "Unseen seeds"), ("E2a_endothelial_x5_m2", "Five-fold endothelial"),
                 ("E2a_endothelial_x5_m2-ownscale", "Five-fold endothelial,\nown scaling"), ("E2b_geometry_margin_G5_m2", "Wider margin")]   # order and names of Table 5


def fig5_transfer():
    P("\nFIG. 5  FROZEN-WEIGHT TRANSFER, skill in the window t 82-100 h (increment models)")
    fig, ax = plt.subplots(1, 2 * len(GRIDS), figsize=(3.6 * 2 * len(GRIDS), 3.2), squeeze=False); col = 0
    for cyt, lab in CYT:
        for grid in GRIDS:
            a = ax[0, col]
            for i, (m, n) in enumerate(ARCH):
                rs = runs(f"main_{m}", "increment", cyt, grid)
                if not rs or not rs[0].get("transfer"): continue
                # training run: skill over t = 82-100 h, as for the test runs
                by = {"training": [1 - np.sum(np.square(r["per_frame"]["RMSE_model"])) / np.sum(np.square(r["per_frame"]["RMSE_persistence"])) for r in rs]}
                for r in rs:
                    for run, t in r["transfer"].items(): by.setdefault(t["experiment"], []).append(t["window"]["skill_vs_persistence"])
                for j, (g, name) in enumerate(TRANSFER_SETS):
                    if g in by:
                        a.bar(j + (i - 2) * 0.16, np.mean(by[g]), 0.16, color=COL[m], label=n if j == 0 else None, **EDGE)
                        P(f"  {lab} {grid} {n:11s} {name.replace(chr(10), ' '):36s} {np.mean(by[g]):+.3f} (n={len(by[g])})")
            a.set_xticks(range(len(TRANSFER_SETS))); a.set_xticklabels([name for _, name in TRANSFER_SETS], rotation=30, ha="right", fontsize=6.5, linespacing=0.95)
            a.axhline(0, color="k", lw=0.8); a.set_title(f"{lab}, {grid}$^2$")
            if col == 0: a.set_ylabel("skill relative to persistence")
            col += 1
    fig.tight_layout()
    handles, labels = ax[0, 0].get_legend_handles_labels()            # legend above the panels
    fig.legend(handles, labels, loc="lower center", bbox_to_anchor=(0.5, 1.0), ncol=len(labels), frameon=False)
    save(fig, "Fig5_transfer")


def fig7_ablation():
    V = [("base", "Full DeepONet"), ("no_branch", "Without branch"), ("no_gate", "Plain trunk"), ("no_fields", "Trunk without fields")]   # row names of Table 7
    fig, ax = plt.subplots(2, 2 * len(GRIDS), figsize=(3.4 * 2 * len(GRIDS), 5.4), squeeze=False); col = 0
    P("\nFIG. 7  OPERATOR-NETWORK ABLATION: near R2 | skill | roll-out R2 at 19 h (others frozen)")
    for cyt, lab in CYT:
        for grid in GRIDS:
            vals, pers = ([], []), None
            for i, (v, n) in enumerate(V):
                rs = runs("deeponet", v, cyt, grid)
                if not rs: continue
                r2 = ms([r["windows"]["near"]["model"]["Global_R2"] for r in rs])
                seeds = ([r["windows"]["near"]["skill_vs_persistence"] for r in rs], [r["rollout"]["R2_frozen"][-1] for r in rs if r.get("rollout")])
                pers = pers if pers is not None else next((r["rollout"]["R2_persist_frozen"][-1] for r in rs if r.get("rollout")), None)
                for row in (0, 1):
                    a = ax[row, col]; mu, sd = ms(seeds[row]); vals[row].append(mu)
                    a.bar(i, mu, color=COL["pointwise"], hatch="" if v == "base" else "//", **EDGE)   # hatched: ablated variants
                    a.plot(i + np.linspace(-0.2, 0.2, len(seeds[row])), seeds[row], "o", ms=3, mfc="white", mec="k", mew=0.7, zorder=3)
                    mean_sd_label(a, i, mu, sd, min(seeds[row]), max(seeds[row]), bg=True)
                P(f"  {lab} {grid} {n:16s} {r2[0]:.4f} | {np.mean(seeds[0]):+9.3f} | {np.mean(seeds[1]):+.3f} (n={len(rs)})")
            for row in (0, 1):
                a = ax[row, col]
                if row == 1 and pers is not None: a.axhline(pers, color="k", ls="--", lw=0.8, label="frozen persistence")
                a.axhline(0, color="k", lw=0.8); a.set_xticks(range(len(V))); a.set_xticklabels([n for _, n in V], rotation=35, ha="right")
                fit_axis(a, vals[row], log=bool(vals[row]) and min(vals[row]) < -1.5)
            ax[0, col].set_title(f"{lab}, {grid}$^2$ - skill"); ax[1, col].set_title(f"{lab}, {grid}$^2$ - roll-out")
            if col == 0:
                ax[0, col].set_ylabel("skill relative to persistence"); ax[1, col].set_ylabel("$R^2$ after 19 h")
            col += 1
    fig.tight_layout()
    handles, labels = ax[1, 0].get_legend_handles_labels()            # legend below the panels
    fig.legend(handles, labels, loc="upper center", bbox_to_anchor=(0.5, 0.0), frameon=False)
    save(fig, "Fig7_ablation")


def figA2_data_size():
    fig, ax = plt.subplots(1, 2 * len(GRIDS), figsize=(3.2 * 2 * len(GRIDS), 2.9), squeeze=False); col = 0
    P("\nFIG. A.2  TRAINING-SET SIZE (U-Net): skill on the training run | window skill on unseen baseline seeds")
    for cyt, lab in CYT:
        for grid in GRIDS:
            a = ax[0, col]; shown = []
            # increment: U-Net colour; direct: darker shade, dashed
            for form, ls, colr, mfc in (("increment", "-", COL["unet"], COL["unet"]), ("direct", "--", "#a63d2a", "white")):
                own, new = [], []
                for n in (1, 2, 3):
                    rs = runs("multitraj_unet", f"{form}_{n}traj", cyt, grid)
                    if not rs: own.append(np.nan); new.append(np.nan); continue
                    own.append(np.mean([r["windows"]["near"]["skill_vs_persistence"] for r in rs]))
                    tr = [t["window"]["skill_vs_persistence"] for r in rs for run, t in (r.get("transfer") or {}).items() if t["experiment"] == C.TRAIN_EXP and run[-2:] in ("s4", "s5")]
                    new.append(np.mean(tr) if tr else np.nan)
                    P(f"  {lab} {grid} {form:9s} {n} traj: {own[-1]:+9.3f} | {new[-1]:+9.3f}")
                a.plot([1, 2, 3], new, ls, marker="o", color=colr, mfc=mfc, mew=1.2, lw=1.6, label=f"{form}, unseen seeds 4-5"); shown += new
            if np.nanmin(shown) < -10: a.set_yscale("symlog", linthresh=1)   # log axis only for wide ranges
            a.axhline(0, color="k", lw=0.8); a.set_xticks([1, 2, 3]); a.set_xlabel("training trajectories"); a.set_title(f"{lab}, {grid}$^2$")
            if col == 0: a.set_ylabel("skill relative to persistence")
            col += 1
    fig.tight_layout()
    handles, labels = ax[0, 0].get_legend_handles_labels()            # legend below the panels
    fig.legend(handles, labels, loc="upper center", bbox_to_anchor=(0.5, 0.0), ncol=len(labels), frameon=False, handlelength=3.5)
    save(fig, "FigA2_data_size")


def figA1_correlation():
    fig, ax = plt.subplots(1, 2 * len(GRIDS), figsize=(3.4 * 2 * len(GRIDS), 3.2), squeeze=False); col = 0
    P("\nFIG. A.1  PER-FRAME SPATIAL CORRELATION (19 test frames x seeds): min / median / max")
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
                ax[0, col].xaxis.set_major_locator(MaxNLocator(nbins=4, steps=[1, 2, 5, 10], min_n_ticks=3))   # at most 3 decimals
            ax[0, col].set_title(f"{lab}, {grid}$^2$"); ax[0, col].set_xlabel("per-frame spatial correlation"); col += 1
    fig.tight_layout(); save(fig, "FigA1_per_frame_correlation")


def fig6_target_masks():
    if not list(SRC.glob("tmask_*/*/res_*.json")): P("\nFIG. 6: no target-time-mask runs found"); return
    P("\nFIG. 6  CELL MASKS AT THE TARGET TIME: skill near | far | unseen baseline seeds (window) | roll-out R2 19 h (others frozen)")
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
            if col == 0: a.set_ylabel("skill relative to persistence (increment)")
            col += 1
    fig.tight_layout(); save(fig, "Fig6_target_time_masks")


if __name__ == "__main__":
    P(f"results: {SRC.relative_to(C.ROOT)}\noutput:  {DST.relative_to(C.ROOT)}, {TABLES.relative_to(C.ROOT)}\ngrids:   {GRIDS}")
    for fn in (table2_active_fraction, fig1_data_overview, fig3_skill, fig4_rollout, fig5_transfer, fig6_target_masks, fig7_ablation,
               figA1_correlation, figA2_data_size, front_and_error_maps):
        guarded(fn)
    (TABLES / "post_runs_summary.txt").write_text("\n".join(lines) + "\n")
    print(f"\nsummary written to {TABLES / 'post_runs_summary.txt'}")
