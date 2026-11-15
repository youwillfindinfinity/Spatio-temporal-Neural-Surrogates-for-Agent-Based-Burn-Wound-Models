"""Training-free analyses from the saved predictions and result JSONs (results/rev):
  1. errors relative to the cytokine front: 1-pixel band around the 0.05 c_max contour of the
     true field, inside and outside, plus the error profile versus distance to the contour
  2. absolute-error maps at t = 88 h with the contour overlaid
  3. distribution of per-frame spatial correlation
  4. autoregressive roll-out curves against frozen persistence
Outputs go to outputs/rev/ (CSV, PNG and post_summary.txt). CPU only.
"""
import json, csv, sys, warnings
from pathlib import Path
import numpy as np
from scipy import ndimage
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
warnings.filterwarnings("ignore")
ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from paths import PREPROCESSED, RESULTS, OUTPUTS

SRC, DST = RESULTS / "rev", OUTPUTS / "rev"; DST.mkdir(parents=True, exist_ok=True)
CYTS = ["il8", "il1", "il6", "il10", "tnf", "tgf"]
START, END = 80, 99
SHOW = [("unet", "base", "U-Net"), ("unet", "residual", "U-Net residual"), ("convlstm", "base", "ConvLSTM"), ("fno", "base", "FNO"),
        ("stalstm", "base", "STA-LSTM"), ("deeponet", "base", "DeepONet"), ("deeponet", "no_branch", "DeepONet w/o branch")]
GRIDS = sorted(int(d.name.split("x")[0]) for d in PREPROCESSED.iterdir() if (d / "X_unet.npy").exists())
GRIDS = [g for g in GRIDS if any((SRC / m / v / f"res_il10_{g}_42.json").exists() or (SRC / m / v / f"res_il8_{g}_42.json").exists() for m, v, _ in SHOW)]
DIST = np.arange(-8, 9)


def data(grid, cyt):
    idx = CYTS.index(cyt); dp = PREPROCESSED / f"{grid}x{grid}"
    cmax = float(json.load(open(dp / "metadata.json"))["scaling"]["max"][idx])
    den = lambda a: (np.asarray(a, np.float64) + 1) / 2 * cmax
    truth = den(np.load(dp / "Y_target.npy", mmap_mode="r")[START:END, ..., idx])
    pers = den(np.load(dp / "X_unet.npy", mmap_mode="r")[START:END, ..., 11 + idx])
    return truth, pers, cmax


def seeds_of(m, v, cyt, grid):
    return sorted(int(p.stem.split("_")[-1]) for p in (SRC / m / v).glob(f"pred_{cyt}_{grid}_*.npz"))


def signed_distance(active):
    """> 0 inside the above-threshold region, < 0 outside, in pixels; 0-crossing = contour."""
    if not active.any() or active.all(): return None
    return np.where(active, ndimage.distance_transform_edt(active) - 0.5, -(ndimage.distance_transform_edt(~active) - 0.5))


def front_stats(truth, pred, cmax):
    acc = {k: [0.0, 0] for k in ("front", "inside", "outside")}
    prof = np.zeros((len(DIST), 2))
    for t in range(truth.shape[0]):
        sd = signed_distance(truth[t] > 0.05 * cmax)
        if sd is None: continue
        ae = np.abs(truth[t] - pred[t]); se = ae ** 2
        reg = {"front": np.abs(sd) <= 1.0, "inside": sd > 1.0, "outside": sd < -1.0}
        for k, m in reg.items():
            acc[k][0] += float(se[m].sum()); acc[k][1] += int(m.sum())
        b = np.clip(np.round(sd).astype(int), DIST[0], DIST[-1]) - DIST[0]
        prof[:, 0] += np.bincount(b.ravel(), weights=ae.ravel(), minlength=len(DIST)); prof[:, 1] += np.bincount(b.ravel(), minlength=len(DIST))
    tot_se = sum(v[0] for v in acc.values()); tot_n = sum(v[1] for v in acc.values())
    out = {}
    for k, (se, n) in acc.items():
        out[f"{k}_rmse_rel"] = float(np.sqrt(se / n) / cmax) if n else float("nan")
        out[f"{k}_pixel_share"] = n / tot_n if tot_n else float("nan")
        out[f"{k}_error_share"] = se / tot_se if tot_se else float("nan")
    return out, np.where(prof[:, 1] > 0, prof[:, 0] / np.maximum(prof[:, 1], 1) / cmax, np.nan)


rows, lines = [], ["POST-HOC ANALYSES", "front = pixels within 1 px of the 0.05 c_max contour of the true field; values: RMSE / c_max, share of pixels, share of squared error", ""]
configs = [(c, g) for c in ("il10", "il8") for g in GRIDS]

# ---------------------------------------------------------------- 1. front analysis + distance profile
fig, axes = plt.subplots(len(configs), 2, figsize=(12, 3.2 * len(configs)), squeeze=False)
for r, (cyt, grid) in enumerate(configs):
    truth, pers, cmax = data(grid, cyt)
    todo = [("persistence", "", "Persistence", [None])] + [(m, v, lab, seeds_of(m, v, cyt, grid)) for m, v, lab in SHOW]
    lines.append(f"=== {cyt.upper()} {grid}x{grid} ===")
    lines.append(f"{'model':22s}{'n':>2s}" + "".join(f"{h:>11s}" for h in ("front", "inside", "outside", "px front", "px inside", "err front", "err inside", "err outs.")))
    for wname, sl in (("near", slice(0, 10)), ("far", slice(10, 19))):
        for m, v, lab, seeds in todo:
            if not seeds: continue
            st, pf = [], []
            for s in seeds:
                pred = pers if s is None else np.load(SRC / m / v / f"pred_{cyt}_{grid}_{s}.npz")["one_step"].astype(np.float64)
                a, b = front_stats(truth[sl], pred[sl], cmax); st.append(a); pf.append(b)
            mean = {k: float(np.nanmean([x[k] for x in st])) for k in st[0]}
            rows.append(dict(cytokine=cyt, grid=grid, window=wname, model=lab, n_seeds=len(seeds), **{k: f"{x:.5g}" for k, x in mean.items()}))
            if wname == "near":
                lines.append(f"{lab:22s}{len(seeds):>2d}" + "".join(f"{mean[k]:11.4f}" for k in ("front_rmse_rel", "inside_rmse_rel", "outside_rmse_rel", "front_pixel_share",
                             "inside_pixel_share", "front_error_share", "inside_error_share", "outside_error_share")))
                axes[r, 0].plot(DIST, np.nanmean(pf, 0), marker="o", ms=3, label=lab, ls="--" if lab == "Persistence" else "-", color="k" if lab == "Persistence" else None)
    lines.append("")
    axes[r, 0].axvline(0, color="gray", lw=0.8, ls=":"); axes[r, 0].set_yscale("log")
    axes[r, 0].set_title(f"{cyt.upper()} {grid}$^2$: error vs distance to the 5% contour (near horizon)")
    axes[r, 0].set_xlabel("signed distance to contour (px; > 0 inside)"); axes[r, 0].set_ylabel("mean |error| / c_max")
    if r == 0: axes[r, 0].legend(fontsize=7, ncol=2)
    # ------------------------------------------------------------ 4. roll-out curves
    ax = axes[r, 1]; lead = np.arange(1, 20); pf_done = False
    for m, v, lab in SHOW:
        js = [json.load(open(p)) for p in sorted((SRC / m / v).glob(f"res_{cyt}_{grid}_*.json"))]
        js = [j for j in js if j.get("rollout")]
        if not js: continue
        if not pf_done:
            ax.plot(lead, js[0]["rollout"]["R2_persist_frozen"], "k--", label="Frozen persistence"); pf_done = True
        ax.plot(lead, np.nanmean([j["rollout"]["R2_gt"] for j in js], 0), marker="o", ms=3, label=lab)
    ax.set_ylim(0.4, 1.005); ax.set_title(f"{cyt.upper()} {grid}$^2$: autoregressive roll-out from t = 81 h")
    ax.set_xlabel("lead time (h)"); ax.set_ylabel("per-frame $R^2$")
    if r == 0: ax.legend(fontsize=7, ncol=2)
fig.tight_layout(); fig.savefig(DST / "fig_front_profile_and_rollout.png", dpi=200); plt.close(fig)
with open(DST / "front_analysis.csv", "w", newline="") as f:
    w = csv.DictWriter(f, fieldnames=list(rows[0].keys())); w.writeheader(); w.writerows(rows)

# ---------------------------------------------------------------- 2. error maps at t = 88 h
k88 = 88 - 82
for cyt, grid in configs:
    truth, pers, cmax = data(grid, cyt)
    panels = [("Persistence", pers[k88])]
    for m, v, lab in SHOW:
        p = SRC / m / v / f"pred_{cyt}_{grid}_42.npz"
        if p.exists(): panels.append((lab, np.load(p)["one_step"][k88].astype(np.float64)))
    fig, ax = plt.subplots(1, len(panels) + 1, figsize=(2.6 * (len(panels) + 1), 3.0))
    ax[0].imshow(truth[k88].T, origin="lower", cmap="viridis", vmax=np.percentile(truth[k88], 99.5)); ax[0].set_title("ABM (truth)", fontsize=8)
    vmax = max(np.percentile(np.abs(truth[k88] - q), 99.5) for _, q in panels[1:]) if len(panels) > 1 else None
    for a, (lab, q) in zip(ax[1:], panels):
        im = a.imshow(np.abs(truth[k88] - q).T, origin="lower", cmap="magma", vmin=0, vmax=vmax)
        a.set_title(f"|error| {lab}", fontsize=8)
    for a in ax:
        a.contour((truth[k88] > 0.05 * cmax).T.astype(float), levels=[0.5], colors="cyan", linewidths=0.6); a.set_xticks([]); a.set_yticks([])
    fig.colorbar(im, ax=ax[1:], fraction=0.012, pad=0.01, label="|error| (physical units)")
    fig.suptitle(f"{cyt.upper()} {grid}x{grid}, t = 88 h, seed 42; cyan: 0.05 c_max contour of the true field", fontsize=9)
    fig.savefig(DST / f"fig_error_maps_{cyt}_{grid}.png", dpi=200, bbox_inches="tight"); plt.close(fig)

# ---------------------------------------------------------------- 3. per-frame correlation distribution
fig, axes = plt.subplots(1, len(configs), figsize=(4.2 * len(configs), 3.6), squeeze=False)
lines.append("Per-frame spatial correlation over 19 test frames x seeds: min / 5th pct / median / max / share of frames below 0.95")
for a, (cyt, grid) in zip(axes[0], configs):
    vals, labs = [], []
    truth, pers, cmax = data(grid, cyt)
    for m, v, lab in SHOW:
        js = [json.load(open(p)) for p in sorted((SRC / m / v).glob(f"res_{cyt}_{grid}_*.json"))]
        if not js: continue
        c = np.concatenate([j["per_frame"]["corr_model"] for j in js]); vals.append(c); labs.append(lab)
        lines.append(f"  {cyt} {grid} {lab:20s} {np.nanmin(c):.4f} / {np.nanpercentile(c, 5):.4f} / {np.nanmedian(c):.4f} / {np.nanmax(c):.4f} / {np.mean(c < 0.95):.2f}")
        if lab == "U-Net":
            c0 = np.array(js[0]["per_frame"]["corr_persistence"]); vals.insert(0, c0); labs.insert(0, "Persistence")
    a.boxplot(vals, vert=False, whis=(0, 100)); a.set_yticklabels(labs, fontsize=7)
    for i, c in enumerate(vals, 1): a.plot(c, np.full_like(c, i) + np.random.default_rng(0).uniform(-0.18, 0.18, len(c)), ".", ms=3, alpha=0.6)
    a.set_title(f"{cyt.upper()} {grid}$^2$", fontsize=9); a.set_xlabel("per-frame spatial correlation")
fig.tight_layout(); fig.savefig(DST / "fig_per_frame_correlation.png", dpi=200); plt.close(fig)

(DST / "post_summary.txt").write_text("\n".join(lines) + "\n")
print("\n".join(lines)); print(f"\nfigures and tables written to {DST}")
