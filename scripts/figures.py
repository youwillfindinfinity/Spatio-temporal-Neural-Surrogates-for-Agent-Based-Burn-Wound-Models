#!/usr/bin/env python3

import os, json, argparse, csv, glob, warnings
from pathlib import Path
import numpy as np
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Patch
from matplotlib.lines import Line2D

FIGDIR = Path("./figures")
CN = ["il8","il1","il6","il10","tnf","tgf"]
CL = ["IL-8",r"IL-1$\beta$","IL-6","IL-10",r"TNF-$\alpha$",r"TGF-$\beta$"]
P = {"DeepONet":"#ffd700","PI-DeepONet":"#ffb14e","U-Net":"#fa8775",
     "STA-LSTM":"#ea5f94","PINN":"#cd34b5"}
MO_100 = ["DeepONet","PI-DeepONet","U-Net","STA-LSTM","PINN"]
W100 = {"DeepONet":"models/deeponet_h","PI-DeepONet":"models/pi_deeponet",
        "U-Net":"models/unet","STA-LSTM":"models/sta_lstm","PINN":"models/pinn"}
MK = {"DeepONet":"o","PI-DeepONet":"P","U-Net":"s","STA-LSTM":"^","PINN":"D"}
LS = {"DeepONet":"-","PI-DeepONet":"--","U-Net":"-","STA-LSTM":"-.","PINN":":"}
CC = ["#e6194b","#f58231","#3cb44b","#4363d8","#911eb4","#42d4f4"]
CMAP_FIELD = "magma"; CMAP_DIFF = "magma"
SC = {"Train":"green","Val":"orange","Near":"dodgerblue","Far":"red"}
SEED = 42; GRIDS_100 = [100,250]; EVAL_CHUNK = 4096

# Measured ABM (CompuCell3D / CPM) wall-clock runtimes for one 100h simulation
# at each grid resolution. Used as the reference cost when computing
# surrogate inference speed-ups. Values reported in seconds.
ABM_RUNTIME_S = {
    50:  225.92,
    100: 541.27,
    250: 2502.84,
}

def setup():
    import shutil
    use_tex = shutil.which("latex") is not None
    plt.rcParams.update({"text.usetex":use_tex,"font.family":"serif",
        "font.serif":["Computer Modern Roman","DejaVu Serif","Times New Roman"],
        "font.size":10,"axes.labelsize":10,"axes.titlesize":11,"legend.fontsize":8,
        "xtick.labelsize":9,"ytick.labelsize":9,
        "figure.dpi":300,"savefig.dpi":300,"savefig.bbox":"tight",
        "axes.spines.top":False,"axes.spines.right":False})
    if use_tex: print("  Using LaTeX rendering")
    else: print("  LaTeX not found, using mathtext")

def sf(fig,n): fig.savefig(FIGDIR/f"{n}.png"); plt.close(fig); print(f"    -> {n}.png")
def lr(g):
    p=Path(f"./preprocessed/{g}x{g}/Y_raw_phys.npy")
    return np.load(p) if p.exists() else None
def lj(wd,c,g,s):
    p=Path(wd)/f"res_{c}_{g}_{s}.json"
    if p.exists():
        with open(p) as f: return json.load(f)
    return None
def get_near(r):
    if not r: return {}
    for k in r.get("results",{}):
        if "near" in k.lower(): return r["results"][k]
    rr=r.get("results",{});return rr[list(rr.keys())[0]] if rr else {}
def get_far(r):
    if not r: return {}
    for k in r.get("results",{}):
        if "far" in k.lower(): return r["results"][k]
    return {}
def gnr(r): return get_near(r).get("Global_R2")
def gtt(r): return r.get("train_time_seconds") if r else None
def gpt(r): return r.get("pred_time_seconds") if r else None

# Seeds used across runs - mean times computed over these
SEEDS = [1, 42, 100]

def gtt_mean(wd, c, g, seeds=SEEDS):
    vals = []
    for s in seeds:
        r = lj(wd, c, g, s)
        v = gtt(r)
        if v is not None and v > 0: vals.append(v)
    if not vals: return None, 0
    return float(np.mean(vals)), len(vals)

def gtt_mean_std(wd, c, g, seeds=SEEDS):
    """Return (mean, std, n) of train time across seeds."""
    vals = []
    for s in seeds:
        r = lj(wd, c, g, s)
        v = gtt(r)
        if v is not None and v > 0: vals.append(v)
    if not vals: return None, None, 0
    if len(vals) == 1: return float(vals[0]), 0.0, 1
    return float(np.mean(vals)), float(np.std(vals, ddof=0)), len(vals)

def gpt_mean(model, c, g, wdirs, seeds=SEEDS):
    vals = []
    for s in seeds:
        pt = _get_pred_time(model, c, g, s, wdirs)
        if pt is not None and pt > 0: vals.append(pt)
    if not vals: return None, 0
    return float(np.mean(vals)), len(vals)

def gpt_mean_std(model, c, g, wdirs, seeds=SEEDS):
    """Return (mean, std, n) of pred time across seeds."""
    vals = []
    for s in seeds:
        pt = _get_pred_time(model, c, g, s, wdirs)
        if pt is not None and pt > 0: vals.append(pt)
    if not vals: return None, None, 0
    if len(vals) == 1: return float(vals[0]), 0.0, 1
    return float(np.mean(vals)), float(np.std(vals, ddof=0)), len(vals)

# Seed-aggregated metric helpers (for error bars on result plots)
def metric_across_seeds(wd, cyt, grid, metric_key, horizon="near", seeds=SEEDS):
    """Collect a single metric across seeds for a (wd, cyt, grid, horizon).
    Returns list of valid float values."""
    getter = get_near if horizon == "near" else get_far
    vals = []
    for s in seeds:
        r = lj(wd, cyt, grid, s)
        if not r: continue
        h = getter(r)
        if not h: continue
        v = h.get(metric_key)
        if v is None: continue
        if isinstance(v, float) and v != v: continue  # NaN
        vals.append(float(v))
    return vals

def metric_mean_std(wd, cyt, grid, metric_key, horizon="near", seeds=SEEDS):
    """Return (mean, std, n) for a metric across seeds. std=0 if only 1 seed."""
    vals = metric_across_seeds(wd, cyt, grid, metric_key, horizon, seeds)
    if not vals: return None, None, 0
    if len(vals) == 1: return vals[0], 0.0, 1
    return float(np.mean(vals)), float(np.std(vals, ddof=0)), len(vals)

def per_timestep_r2_mean_std(wd, cyt, grid, seeds=SEEDS):
    """For temporal panel: collect per-time-step R^2 arrays across seeds and
    return (times, mean_r2, std_r2) on the union of available time points.
    Returns lists with NaN for missing entries."""
    per_seed = {}
    for s in seeds:
        r = lj(wd, cyt, grid, s)
        if not r: continue
        rr = r.get("results", {})
        for k in rr:
            kl = k.lower()
            ts = None
            if "near" in kl: ts = 82  # first eval step in near window (t=82..91)
            elif "far" in kl: ts = 92  # first eval step in far window (t=92..100)
            else: continue
            ptr = rr[k].get("Per_Timestep_R2", [])
            for i, v in enumerate(ptr):
                if v is None: continue
                try:
                    fv = float(v)
                    if fv != fv: continue
                except Exception:
                    continue
                t = ts + i
                per_seed.setdefault(t, []).append(fv)
    if not per_seed:
        return np.array([]), np.array([]), np.array([])
    ts_sorted = sorted(per_seed.keys())
    means = np.array([np.mean(per_seed[t]) for t in ts_sorted])
    # std=0 if only 1 seed contributed at that time-step
    stds = np.array([np.std(per_seed[t], ddof=0) if len(per_seed[t]) > 1 else 0.0 for t in ts_sorted])
    return np.array(ts_sorted), means, stds


def abm_time(grid):
    """Reference ABM wall-clock for one 100h simulation at the given grid resolution.
    Returns None if no measured value is available for that grid."""
    return ABM_RUNTIME_S.get(grid)


# FIG 1: Concentrations + sparsity
# 1a: 100h with 2 lines (100^2 + 250^2) per cytokine panel
# 1b: Sparsity heatmaps (IL-8, IL-10 at 250, t=80)

def fig1():
    sp_100h = {"Train":(0,72),"Val":(72,82),"Near":(82,91),"Far":(91,101)}

    r100 = lr(100); r250 = lr(250)
    if r100 is not None or r250 is not None:
        fig,axes=plt.subplots(2,3,figsize=(7.2,4.2))
        for i in range(6):
            ax=axes[i//3,i%3]
            for grid,raw,ls,lw in [(100,r100,"-",1.4),(250,r250,"--",1.2)]:
                if raw is None: continue
                T=raw.shape[0]; t=np.arange(T); f=raw[:,:,:,i]; mc=f.reshape(T,-1).mean(1)
                ax.plot(t,mc,color=CC[i],ls=ls,lw=lw)
                ax.text(t[-1]+1, mc[-1], f"${grid}^2$", fontsize=6, color=CC[i],
                       va="center", ha="left", fontweight="bold", clip_on=False)
            for lb,(t0,t1) in sp_100h.items():
                ax.axvspan(t0,t1,alpha=0.06 if lb=="Train" else 0.10,color=SC[lb])
            ax.set_title(CL[i],fontsize=10)
            if i>=3: ax.set_xlabel("Time (h)")
            if i%3==0: ax.set_ylabel("Mean conc.")
            ax.set_xlim(0,112)
            ax.ticklabel_format(axis='y',style='scientific',scilimits=(-2,2))
        split_h=[Patch(fc=SC[s],alpha=0.3,label=s) for s in sp_100h]
        fig.legend(handles=split_h,loc="lower center",ncol=4,
                  fontsize=7,bbox_to_anchor=(0.5,-0.04),framealpha=0.9,edgecolor="gray")
        fig.tight_layout(rect=[0,0.05,1,1])
        sf(fig,"Fig1_concentrations_100h")

    # sparsity heatmaps (250, 100h, t=80)
    r250 = lr(250)
    if r250 is not None:
        fig,axes=plt.subplots(1,2,figsize=(7.2,3.0))
        for ax,ci,cl in zip(axes,[0,3],["IL-8","IL-10"]):
            f=r250[80,:,:,ci]; vm=np.percentile(f[f>0],99.5) if (f>0).any() else 1e-15
            im=ax.imshow(f,cmap=CMAP_FIELD,vmin=0,vmax=vm,origin="lower")
            ax.set_title(cl); ax.set_xticks([]); ax.set_yticks([])
            spar=(f==0).sum()/f.size*100
            ax.text(0.03,0.97,f"Sparsity: {spar:.1f}%",transform=ax.transAxes,fontsize=9,va="top",color="white",
                    bbox=dict(boxstyle="round,pad=0.2",fc="black",alpha=0.7))
            fig.colorbar(im,ax=ax,fraction=0.046,format="%.1e")
        fig.tight_layout(); sf(fig,"Fig1_sparsity_100h")


# FIG 2a: R2 bars (RQ1) - with error bars (mean +/- std across seeds)
def fig2():
    fig,axes=plt.subplots(2,2,figsize=(4.8,5.5),sharey=True)
    av=[]
    for row,cyt,cl in [(0,"il8","IL-8"),(1,"il10","IL-10")]:
        for col,g in [(0,100),(1,250)]:
            ax=axes[row,col]
            means=[]; stds=[]; ns=[]
            for m in MO_100:
                mu, sd, n = metric_mean_std(W100[m], cyt, g, "Global_R2", "near")
                means.append(mu); stds.append(sd); ns.append(n)
            av.extend([v for v in means if v is not None])
            x=np.arange(len(MO_100))
            bv=[v if v is not None else 0 for v in means]
            ev=[s if s is not None else 0 for s in stds]
            bars=ax.bar(x,bv,color=[P[m] for m in MO_100],ec="black",lw=.4,alpha=.9,
                        yerr=ev, capsize=3,
                        error_kw=dict(ecolor='black', elinewidth=0.8, capthick=0.8))
            ax.set_xticks(x); ax.set_xticklabels(MO_100,rotation=30,ha="right",fontsize=8)
            if col==0: ax.set_ylabel(r"$R^2$")
            ax.set_title(f"{cl}, ${g}^2$")
            ax.axhline(1,color="gray",ls=":",alpha=.3)
            for b,v,s,n in zip(bars,means,stds,ns):
                if v is not None:
                    y=max(v,0)+(s if s else 0)+0.02 if v>=0 else v-0.03
                    lbl = f"{v:.3f}\n$\\pm${(s or 0):.3f}"
                    ax.text(b.get_x()+b.get_width()/2,y,lbl,
                            ha="center",fontsize=5.5,fontweight="bold")
    vm=min(av) if av else 0
    for r in axes:
        for ax in r: ax.set_ylim(min(vm-0.10,-0.05),1.15); ax.axhline(0,color="black",ls="-",lw=0.6,alpha=.3)
    # caption note about error bars
    fig.text(0.5, -0.02, f"Error bars: $\\pm$1 s.d. across {len(SEEDS)} seeds ({', '.join(map(str,SEEDS))}).",
             ha="center", fontsize=7, style="italic", color="#444")
    fig.tight_layout(); sf(fig,"Fig2a_R2_100h")


# FIG 2b: Pareto + Speedup combined (RQ1)
_PRED_TIME_CACHE = {}
_PRED_TIMES_CSV = Path("./figures/pred_times.csv")

def _load_pred_time_cache():
    if _PRED_TIMES_CSV.exists():
        import csv as _csv
        with open(_PRED_TIMES_CSV) as f:
            for row in _csv.DictReader(f):
                key = (row["model"], row["cytokine"], int(row["grid"]), int(row["seed"]))
                _PRED_TIME_CACHE[key] = float(row["pred_time_s"])
        print(f"    Loaded {len(_PRED_TIME_CACHE)} cached pred times from {_PRED_TIMES_CSV}")

def _save_pred_time_cache():
    if not _PRED_TIME_CACHE: return
    import csv as _csv
    FIGDIR.mkdir(parents=True, exist_ok=True)
    with open(_PRED_TIMES_CSV, "w", newline="") as f:
        w = _csv.DictWriter(f, fieldnames=["model","cytokine","grid","seed","pred_time_s"])
        w.writeheader()
        for (m,c,g,s), t in sorted(_PRED_TIME_CACHE.items()):
            w.writerow({"model":m,"cytokine":c,"grid":g,"seed":s,"pred_time_s":round(t,4)})
    print(f"    Saved {len(_PRED_TIME_CACHE)} pred times to {_PRED_TIMES_CSV}")

def _get_pred_time(model, cyt, grid, seed, wdirs):
    r = lj(wdirs[model], cyt, grid, seed)
    pt = gpt(r)
    if pt and pt > 0: return pt
    key = (model, cyt, grid, seed)
    if key in _PRED_TIME_CACHE: return _PRED_TIME_CACHE[key]
    return None

def fig2b():
    fig, axes = plt.subplots(2, 3, figsize=(7.2, 5.5))

    for row, (cyt, cl) in enumerate([("il8","IL-8"),("il10","IL-10")]):
        # (a) Pareto: R2 vs total time (mean +/- std across seeds for BOTH axes)
        ax = axes[row, 0]
        ar = []
        for model in MO_100:
            pts = []  # (grid, t_mean, t_std, r2_mean, r2_std)
            for grid in [100, 250]:
                # accuracy across seeds
                r2_mu, r2_sd, n_r2 = metric_mean_std(W100[model], cyt, grid, "Global_R2", "near")
                tt_mu, tt_sd, n_tt = gtt_mean_std(W100[model], cyt, grid)
                pt_mu, pt_sd, n_pt = gpt_mean_std(model, cyt, grid, W100)
                if r2_mu is None or tt_mu is None or pt_mu is None: continue
                t_mu = tt_mu + pt_mu
                # combine independent std's in quadrature
                t_sd = np.sqrt((tt_sd or 0)**2 + (pt_sd or 0)**2)
                pts.append((grid, t_mu, t_sd, r2_mu, r2_sd or 0))
                ar.append(r2_mu)
            if not pts: continue
            ts_arr = [p[1] for p in pts]
            r2_arr = [p[3] for p in pts]
            for g, t_mu, t_sd, r2_mu, r2_sd in pts:
                ax.scatter(t_mu, r2_mu, c=P[model], s=40 if g==100 else 100,
                          marker=MK[model], ec="black", lw=.5, zorder=5, clip_on=False)
            if len(pts) > 1:
                ax.plot(ts_arr, r2_arr, c=P[model], lw=1.2, alpha=.4,
                        ls=LS[model], zorder=4)
        ax.set_xscale("log"); ax.set_xlabel("Total time (s, mean over seeds)")
        ax.set_ylabel(r"$R^2$"); ax.grid(True, alpha=0.12)
        ax.set_title(f"{cl} - Accuracy vs time", fontsize=9)
        vm = min(ar) if ar else 0
        ax.set_ylim(min(vm-0.08, -0.08), 1.05)
        ax.axhline(0, color="black", ls="-", lw=0.6, alpha=.3)

        # (b) Prediction time (mean +/- std across seeds)
        ax = axes[row, 1]
        models_with_time = []; pred_vals = []; pred_errs = []
        for m in MO_100:
            pt_mu, pt_sd, n_pt = gpt_mean_std(m, cyt, 250, W100)
            if pt_mu and pt_mu > 0:
                models_with_time.append(m); pred_vals.append(pt_mu)
                pred_errs.append(pt_sd if pt_sd is not None else 0)
        if models_with_time:
            x = np.arange(len(models_with_time))
            bars = ax.bar(x, pred_vals, color=[P[m] for m in models_with_time], ec="black", lw=.4,
                          yerr=pred_errs, capsize=3,
                          error_kw=dict(ecolor='black', elinewidth=0.8, capthick=0.8))
            ax.set_xticks(x); ax.set_xticklabels(models_with_time, rotation=35, ha="right", fontsize=7)
            ax.set_ylabel("Pred time (s, mean)"); ax.set_yscale("log")
            ax.set_ylim(top=max(pred_vals)*15)
            for b, pt_, pe_ in zip(bars, pred_vals, pred_errs):
                y_top = pt_ + (pe_ if pe_ else 0)
                y_lab = y_top * 1.2
                lbl = f"{pt_:.1f}s\n$\\pm${(pe_ or 0):.1f}s"
                ax.text(b.get_x()+b.get_width()/2, y_lab, lbl,
                       ha="center", va="bottom", fontsize=5.5, fontweight="bold")
        ax.set_title(f"{cl} - Prediction ($250^2$)", fontsize=9)

        # (c) Inference speedup vs ABM at 250^2 (with error bars derived from pred-time std)
        ax = axes[row, 2]
        abm_t_250 = abm_time(250)
        ms = []; su = []; su_err = []
        for m in MO_100:
            pt_mu, pt_sd, n_pt = gpt_mean_std(m, cyt, 250, W100)
            if pt_mu and pt_mu > 0 and abm_t_250 is not None:
                speedup = abm_t_250 / pt_mu
                # error propagation: d(speedup) = abm_t * pt_sd / pt_mu^2
                if pt_sd is not None and pt_sd > 0:
                    sp_err = abm_t_250 * pt_sd / (pt_mu ** 2)
                else:
                    sp_err = 0.0
                ms.append(m); su.append(speedup); su_err.append(sp_err)
        if ms:
            x = np.arange(len(ms))
            bars = ax.bar(x, su, color=[P[m] for m in ms], ec="black", lw=.4,
                          yerr=su_err, capsize=3,
                          error_kw=dict(ecolor='black', elinewidth=0.8, capthick=0.8))
            ax.set_xticks(x); ax.set_xticklabels(ms, rotation=35, ha="right", fontsize=7)
            ax.set_ylabel(r"Inference speedup ($\times$)")
            ax.set_yscale("log"); ax.set_ylim(top=max(su)*15)
            def _fmt_su(v):
                if v >= 1e4: return f"{v/1000:.1f}k$\\times$"
                return f"{v:.0f}$\\times$"
            for b, s_, se_ in zip(bars, su, su_err):
                y_top = s_ + (se_ if se_ else 0)
                y_lab = y_top * 1.2
                lbl = f"{_fmt_su(s_)}\n$\\pm${_fmt_su(se_ or 0)}"
                ax.text(b.get_x()+b.get_width()/2, y_lab, lbl,
                       ha="center", va="bottom", fontsize=5.5, fontweight="bold")
        ax.set_title(f"{cl} - Inference speedup vs ABM ($250^2$)", fontsize=9)

    # Legend
    handles = [Line2D([0],[0], marker=MK[m], color=P[m], ls=LS[m], lw=1.2,
               markersize=6, mec="black", mew=.5, label=m) for m in MO_100]
    handles += [Line2D([0],[0], marker="o", color="gray", ls="", markersize=s, mec="black",
                label=lb) for s, lb in [(4,r"$100^2$"),(7,r"$250^2$")]]
    fig.legend(handles=handles, loc="lower center", bbox_to_anchor=(0.5,-0.04),
              fontsize=6, framealpha=0.9, edgecolor="gray", ncol=4)
    fig.text(0.5, -0.07, f"Error bars: $\\pm$1 s.d. across {len(SEEDS)} seeds ({', '.join(map(str,SEEDS))}).",
             ha="center", fontsize=7, style="italic", color="#444")
    fig.tight_layout(); fig.subplots_adjust(bottom=0.17)
    sf(fig, "Fig2b_pareto_speedup_100h")


# FIG 3: RQ3 composite - multi-metric + scalability + temporal
def fig3():
    mets=[("Global_R2",r"$R^2$"),("Spatial_Correlation","Corr"),("SSIM","SSIM"),("Avg_Dice","Dice")]
    hatches=["","//","..","xx"]

    models = MO_100
    wdirs = W100
    grid = 250

    fig,axes=plt.subplots(2,3,figsize=(7.2,5.5))
    panel_data = {(r,c): [] for r in range(2) for c in range(3)}
    for row,(cyt,cl) in enumerate([("il8","IL-8"),("il10","IL-10")]):
        # (a) Multi-metric bars with error bars (mean +/- std across seeds)
        ax=axes[row,0]; nm=len(models); w=0.18; x=np.arange(nm)
        for j,(met,ml) in enumerate(mets):
            means=[]; stds=[]
            for m in models:
                mu, sd, n = metric_mean_std(wdirs[m], cyt, grid, met, "near")
                means.append(mu); stds.append(sd)
            panel_data[(row,0)].extend([v for v in means if v is not None])
            bv=[v if v is not None else 0 for v in means]
            ev=[s if s is not None else 0 for s in stds]
            offset=(j-len(mets)/2+0.5)*w
            bars=ax.bar(x+offset,bv,w,color=[P[m] for m in models],
                       ec="black",lw=.3,alpha=0.65+j*0.08,hatch=hatches[j],
                       yerr=ev, capsize=1.5,
                       error_kw=dict(ecolor='black', elinewidth=0.5, capthick=0.5))
            for b,v in zip(bars,means):
                if v is None:
                    cx=b.get_x()+b.get_width()/2
                    ax.text(cx,0.01,"N/A",ha="center",fontsize=4,color="black",rotation=90,va="bottom")
        ax.set_xticks(x); ax.set_xticklabels(models,rotation=35,ha="right",fontsize=7)
        ax.set_ylabel("Score")
        ax.axhline(1,color="gray",ls=":",alpha=.3)
        ax.set_title(f"{cl} - Metrics",fontsize=9)

        # (b) Scalability across grid resolutions - now with errorbars
        ax=axes[row,1]
        for model in models:
            gs=[]; mus=[]; sds=[]
            for g in [100,250]:
                mu, sd, n = metric_mean_std(wdirs[model], cyt, g, "Global_R2", "near")
                if mu is not None:
                    gs.append(g); mus.append(mu); sds.append(sd if sd is not None else 0)
                    panel_data[(row,1)].append(mu)
            if gs:
                ax.errorbar(gs, mus, yerr=sds, color=P[model], lw=1.5,
                            marker=MK[model], ms=7, mec="black", mew=.6,
                            ls=LS[model], capsize=3, elinewidth=0.8, capthick=0.8,
                            clip_on=False)
        ax.set_xlabel("Grid resolution"); ax.set_xticks([100,250])
        ax.set_ylabel(r"$R^2$"); ax.grid(True,alpha=0.12)
        ax.set_title(f"{cl} - Scalability",fontsize=9)

        # (c) Temporal - per-timestep R^2 with shaded std band across seeds
        ax=axes[row,2]
        nr_range = (82, 91)
        for model in models:
            ts, mu, sd = per_timestep_r2_mean_std(wdirs[model], cyt, grid)
            if len(ts) == 0: continue
            # for PINN, gentle smoothing if very noisy (preserves original behaviour)
            mu_plot = mu.copy()
            if model=="PINN" and len(mu)>5 and np.std(np.diff(mu))>0.03:
                k5=np.ones(5)/5; mu_plot=np.convolve(mu,k5,mode="same")
                mu_plot[:2]=mu[:2]; mu_plot[-2:]=mu[-2:]
            ax.plot(ts, mu_plot, color=P[model], lw=1.2, ls=LS[model], label=model)
            # shaded band only where we actually have >1 seed worth of variability
            if np.any(sd > 0):
                ax.fill_between(ts, mu_plot - sd, mu_plot + sd,
                                color=P[model], alpha=0.18, lw=0)
            panel_data[(row,2)].extend(mu.tolist())
        bnd = nr_range[1] + 1  # boundary between near (..91) and far (92..)
        ax.axvline(bnd-0.5, color="#999", ls="--", alpha=0.5, lw=1)
        ax.text(bnd-1, 0.97, "Near", ha="right", fontsize=6, color="#555",
                fontweight="bold", transform=ax.get_xaxis_transform())
        ax.text(bnd, 0.97, "Far", ha="left", fontsize=6, color="#555",
                fontweight="bold", transform=ax.get_xaxis_transform())
        ax.set_xlabel("Time (h)"); ax.set_ylabel(r"$R^2$")
        ax.set_title(f"{cl} - Temporal",fontsize=9); ax.grid(True,alpha=0.12)

    for r in range(2):
        for c in range(3):
            data = panel_data[(r,c)]
            has_neg = any(v < -0.001 for v in data) if data else False
            if has_neg:
                ymin = min(min(data) - 0.10, -0.20)
                ymax = 1.15
                axes[r, c].axhline(0, color="black", ls="-", lw=0.5, alpha=.4)
            else:
                ymin = 0
                ymax = 1.15
            axes[r, c].set_ylim(ymin, ymax)

    # legend
    met_handles=[Patch(fc="lightgray",ec="black",hatch=hatches[j],label=mets[j][1]) for j in range(len(mets))]
    model_handles=[Line2D([0],[0],color=P[m],lw=1.5,ls=LS.get(m,"-"),
                   marker=MK.get(m,"o"),ms=5,mec="black",mew=.5,label=m) for m in models]
    all_handles=met_handles+model_handles
    fig.legend(handles=all_handles,loc="lower center",bbox_to_anchor=(0.5,-0.06),
              fontsize=6,framealpha=0.9,edgecolor="gray",ncol=len(all_handles))
    fig.text(0.5, -0.09,
             f"Error bars / shaded bands: $\\pm$1 s.d. across {len(SEEDS)} seeds ({', '.join(map(str,SEEDS))}).",
             ha="center", fontsize=7, style="italic", color="#444")
    fig.tight_layout(); fig.subplots_adjust(bottom=0.15)
    sf(fig,"Fig3_rq3_composite_100h")

# FIG 6: Spatial maps (RQ2) - needs TensorFlow
# 6a: GT | Pred | |Diff| for DeepONet, 250, both cytokines
# 6b: |Diff| only, all models, both grids, both cytokines

def fig6(hour=88):
    import tensorflow as tf; import matplotlib.colors as mcolors
    os.environ["TF_CPP_MIN_LOG_LEVEL"]="3"; warnings.filterwarnings("ignore")
    tf.config.optimizer.set_jit(False)
    for gpu in tf.config.list_physical_devices("GPU"):
        try: tf.config.experimental.set_memory_growth(gpu,True)
        except: pass

    BASE=Path(".").resolve(); PREPRO=BASE/"preprocessed"; MODELS=BASE/"models"
    MODEL_ORDER=MO_100
    MODEL_DIRS={"DeepONet":"deeponet_h","PI-DeepONet":"pi_deeponet",
                "U-Net":"unet","STA-LSTM":"sta_lstm","PINN":"pinn"}
    CYTS=[("il8","IL-8",0),("il10","IL-10",3)]; t_idx=hour-2

    # Architectures
    def _cb(x,f):
        x=tf.keras.layers.Conv2D(f,3,padding="same",activation="relu")(x)
        x=tf.keras.layers.BatchNormalization()(x)
        x=tf.keras.layers.Conv2D(f,3,padding="same",activation="relu")(x)
        return tf.keras.layers.BatchNormalization()(x)
    def build_unet(G,bf=32,depth=4,do=0.0):
        inp=tf.keras.Input(shape=(G,G,22));sk=[];x=inp
        for i in range(depth): s=_cb(x,bf*(2**i));sk.append(s);x=tf.keras.layers.MaxPooling2D(2,padding="same")(s)
        x=_cb(x,bf*(2**depth))
        if do>0: x=tf.keras.layers.Dropout(do)(x)
        for i in reversed(range(depth)):
            x=tf.keras.layers.Conv2DTranspose(bf*(2**i),2,strides=2,padding="same",activation="relu")(x)
            s=sk[i]
            if x.shape[1]!=s.shape[1] or x.shape[2]!=s.shape[2]: x=tf.keras.layers.Resizing(s.shape[1],s.shape[2])(x)
            x=tf.keras.layers.Concatenate()([x,s]);x=_cb(x,bf*(2**i))
        return tf.keras.Model(inp,tf.keras.layers.Conv2D(1,1,padding="same")(x))
    class SpatialAttention(tf.keras.layers.Layer):
        def __init__(s,**kw): super().__init__(**kw);s.attn_conv=tf.keras.layers.Conv2D(1,1,padding="same",activation="sigmoid")
        def call(s,x): return x*s.attn_conv(x)
    class STALSTM(tf.keras.Model):
        def __init__(s,G,filters=64,lstm_units=128):
            super().__init__();s.grid_size=G;s.filters=filters;s.lstm_units=lstm_units;s.latent_size=max(G//4,8)
            s.enc=tf.keras.layers.TimeDistributed(tf.keras.Sequential([tf.keras.layers.Conv2D(filters,3,strides=2,padding="same",activation="relu"),tf.keras.layers.Conv2D(filters,3,strides=2,padding="same",activation="relu")]),name="encoder")
            s.spatial_attn=tf.keras.layers.TimeDistributed(SpatialAttention(),name="spatial_attention")
            s.gap=tf.keras.layers.TimeDistributed(tf.keras.layers.GlobalAveragePooling2D(),name="gap")
            s.lstm=tf.keras.layers.LSTM(lstm_units,return_sequences=False,name="lstm")
            s.relu=tf.keras.layers.Activation("relu")
            s.fc=tf.keras.layers.Dense(s.latent_size**2*filters,activation="relu")
            s.reshape_latent=tf.keras.layers.Reshape((s.latent_size,s.latent_size,filters))
            s.deconv1=tf.keras.layers.Conv2DTranspose(filters//2,3,strides=2,padding="same",activation="relu")
            s.deconv2=tf.keras.layers.Conv2DTranspose(filters//4,3,strides=2,padding="same",activation="relu")
            s.out_conv=tf.keras.layers.Conv2D(1,3,padding="same",activation="linear")
            s.out_resize=tf.keras.layers.Resizing(G,G)
        def call(s,x):
            h=s.enc(x);h=s.spatial_attn(h);h=s.gap(h);h=s.lstm(h);h=s.relu(h);h=s.fc(h);h=s.reshape_latent(h)
            return s.out_resize(s.out_conv(s.deconv2(s.deconv1(h))))
    class Branch(tf.keras.layers.Layer):
        def __init__(s,h,p,**kw): super().__init__(**kw);s.fc1=tf.keras.layers.Dense(h,activation="relu");s.fc2=tf.keras.layers.Dense(p)
        def call(s,x,training=False): return s.fc2(s.fc1(x))
    class Trunk(tf.keras.layers.Layer):
        def __init__(s,h,p,**kw):
            super().__init__(**kw);s.U=tf.keras.layers.Dense(h,activation="tanh");s.V=tf.keras.layers.Dense(h,activation="tanh")
            s.W1a=tf.keras.layers.Dense(h,activation="relu");s.W1b=tf.keras.layers.Dense(h)
            s.W2a=tf.keras.layers.Dense(h,activation="relu");s.W2b=tf.keras.layers.Dense(h);s.out=tf.keras.layers.Dense(p)
        def call(s,x):
            u=s.U(x);v=s.V(x);h=s.W1b(s.W1a(x));h=h*u+(1-h)*v;h=s.W2b(s.W2a(h));h=h*u+(1-h)*v;return s.out(h)
    class DON(tf.keras.Model):
        def __init__(s,h,p): super().__init__();s.branch=Branch(h,p);s.trunk=Trunk(h,p);s.bias=s.add_weight(shape=(1,),initializer="zeros",trainable=True)
        def call(s,inp,training=False): b=s.branch(inp[0],training=training);t=s.trunk(inp[1]);return tf.expand_dims(tf.einsum("bp,bnp->bn",b,t)+s.bias,-1)
    class PIDON(tf.keras.Model):
        def __init__(s,h,p): super().__init__();s.branch=Branch(h,p);s.trunk=Trunk(h,p);s.bias=s.add_weight(shape=(1,),initializer="zeros",trainable=True)
        def call_data(s,xb,xt,training=False): b=s.branch(xb,training=training);t=s.trunk(xt);return tf.expand_dims(tf.einsum("bp,bnp->bn",b,t)+s.bias,-1)
        def call(s,inp,training=False): return s.call_data(inp[0],inp[1],training=training)

    def denorm(s,cm): return (np.asarray(s,np.float64)+1)/2*cm
    def branch_inp(Xb,Xt,ci):
        N,_,G,_,_=Xb.shape;f0=Xb[:,0,:,:,ci];mask=(Xb[:,0,:,:,6:].max(-1)>0.5).astype(np.float32)
        xs=np.linspace(0,1,G,dtype=np.float32);xx,yy=np.meshgrid(xs,xs,indexing='ij')
        o=np.zeros((N,7),np.float32)
        for i in range(N):
            f=f0[i];m=mask[i];na=float(np.sum(m))+1e-6
            o[i]=[(float(np.max(f))+1)/2,(float(np.mean(f))+1)/2,float(np.std(f)),float(np.sum(xx*m)/na),float(np.sum(yy*m)/na),na/(G*G),float(Xt[i,0,2])]
        return o
    def trunk_inp(Xb,Xt):
        N,_,G,_,_=Xb.shape;return np.concatenate([Xt[:,:,:2].astype(np.float32),Xb.transpose(0,2,3,1,4).reshape(N,G*G,22).astype(np.float32)],-1)
    def load_predict(mname,cyt,ci,t_idx,G):
        d=MODELS/MODEL_DIRS[mname];dp=PREPRO/f"{G}x{G}";rp=d/f"res_{cyt}_{G}_{SEED}.json"
        if not rp.exists(): return None
        with open(rp) as f: res=json.load(f)
        bp=res["best_params"]
        with open(dp/"metadata.json") as f: meta=json.load(f)
        cm=float(meta["scaling"]["max"][ci]);G2=G*G;tf.keras.backend.clear_session()
        if mname=="U-Net":
            X=np.load(dp/"X_unet.npy").astype(np.float32);m=build_unet(G,bp["base_filters"],bp["depth"],bp.get("dropout",0))
            w=d/f"weights_{cyt}_{G}_{SEED}.weights.h5"
            if not w.exists(): return None
            m.load_weights(str(w));return np.maximum(denorm(m.predict(X[t_idx:t_idx+1],verbose=0)[0,...,0],cm),0)
        elif mname=="STA-LSTM":
            X=np.load(dp/"X_lstm.npy").astype(np.float32);m=STALSTM(G,bp["filters"],bp["lstm_units"]);_=m(X[:1])
            w=d/f"weights_{cyt}_{G}_{SEED}.weights.h5"
            if not w.exists(): return None
            m.load_weights(str(w));return np.maximum(denorm(m.predict(X[t_idx:t_idx+1],verbose=0)[0,...,0],cm),0)
        elif mname in ("DeepONet","PI-DeepONet"):
            Xb=np.load(dp/"X_branch.npy").astype(np.float32);Xt=np.load(dp/"X_trunk.npy").astype(np.float32)
            xbr=branch_inp(Xb,Xt,ci);xtr=trunk_inp(Xb,Xt);is_pi=mname=="PI-DeepONet"
            if is_pi: m=PIDON(bp["hidden"],bp["p"]);_=m.call_data(tf.constant(xbr[:1]),tf.constant(xtr[:1,:10,:]))
            else: m=DON(bp["hidden"],bp["p"]);_=m([tf.constant(xbr[:1]),tf.constant(xtr[:1,:10,:])])
            w=d/f"weights_{cyt}_{G}_{SEED}.weights.h5"
            if not w.exists(): return None
            m.load_weights(str(w));out=np.zeros((1,G2,1),np.float32);xb=tf.constant(xbr[t_idx:t_idx+1])
            for s in range(0,G2,EVAL_CHUNK):
                e=min(s+EVAL_CHUNK,G2);xt=tf.constant(xtr[t_idx:t_idx+1,s:e])
                if is_pi: out[0,s:e]=m.call_data(xb,xt,training=False).numpy()[0]
                else: out[0,s:e]=m([xb,xt],training=False).numpy()[0]
            return np.maximum(denorm(out.reshape(G,G),cm),0)
        elif mname=="PINN":
            import deepxde as dde;dde.config.set_default_float("float64");dde.config.disable_xla_jit()
            net=dde.maps.FNN([3]+[bp["hidden"]]*bp["n_layers"]+[1],"tanh","Glorot uniform")
            net.apply_output_transform(lambda x,y:y);net(tf.constant(np.zeros((1,3),np.float64)))
            wp=str(d/f"weights_{cyt}_{G}_{SEED}-*.weights.h5");wfs=sorted(glob.glob(wp))
            if not wfs:
                wp2=str(d/f"weights_{cyt}_{G}_{SEED}.weights.h5")
                if os.path.exists(wp2): wfs=[wp2]
            if not wfs: return None
            net.load_weights(wfs[-1]);xs=np.linspace(-1,1,G,np.float64);xx,yy=np.meshgrid(xs,xs,indexing="ij")
            xy=np.stack([xx.ravel(),yy.ravel()],1);tn=np.linspace(-1,1,101,np.float64)
            tt=np.full((G2,1),tn[t_idx+2],np.float64)
            return np.maximum(denorm(net(tf.constant(np.hstack([xy,tt]))).numpy().reshape(G,G),cm),0)
        return None

    print(f"  Loading predictions for t={hour}h...")
    gts={};all_preds={}
    for cyt,cl,ci in CYTS:
        gts[cyt]={}
        for G in GRIDS_100: Y=np.load(PREPRO/f"{G}x{G}/Y_raw_phys.npy");gts[cyt][G]=Y[hour,:,:,ci]
        all_preds[cyt]={}
        for mn in MODEL_ORDER:
            all_preds[cyt][mn]={}
            for G in GRIDS_100:
                print(f"    {cl} | {mn} @ {G}^2...",end=" ",flush=True)
                p=load_predict(mn,cyt,ci,t_idx,G)
                if p is not None: all_preds[cyt][mn][G]=p;print("OK")
                else: print("SKIP")

    # GLOBAL scales per cytokine
    gt_norms = {}
    diff_vmax = {}
    for cyt, cl, ci in CYTS:
        # GT vmax: from raw physical data
        all_gt_vals = []
        for G in GRIDS_100:
            g = gts[cyt][G]
            if (g > 0).any(): all_gt_vals.append(np.percentile(g[g>0], 99.5))
        vmax_gt = max(all_gt_vals) if all_gt_vals else 1e-15
        gt_norms[cyt] = mcolors.PowerNorm(gamma=0.5, vmin=0, vmax=vmax_gt)

        # Diff vmax: Y_target (clipped, same as model) for fair comparison
        all_diff_vals = []
        for mn in MODEL_ORDER:
            for G in GRIDS_100:
                if G in all_preds[cyt].get(mn, {}):
                    dp=PREPRO/f"{G}x{G}"
                    with open(dp/"metadata.json") as fmeta: meta_loc=json.load(fmeta)
                    cm=float(meta_loc["scaling"]["max"][ci])
                    Y_tgt=np.load(dp/"Y_target.npy").astype(np.float64)
                    gt_m=np.maximum((Y_tgt[hour-2,:,:,ci]+1)/2*cm, 0)
                    d = np.abs(all_preds[cyt][mn][G] - gt_m)
                    all_diff_vals.append(d.ravel())
        if all_diff_vals:
            ad = np.concatenate(all_diff_vals)
            diff_vmax[cyt] = float(np.percentile(ad, 99))
        else:
            diff_vmax[cyt] = 1e-15
        if diff_vmax[cyt] <= 0: diff_vmax[cyt] = 1e-15

    # 6a: GT | Pred | |Diff| for DeepONet, 250
    G=250; fig,axes=plt.subplots(2,3,figsize=(7.2,5.0))
    col_titles=["Ground Truth","Prediction",r"$\vert$Pred - GT$\vert$"]
    for row,(cyt,cl,ci) in enumerate(CYTS):
        pred=all_preds[cyt].get("DeepONet",{}).get(G)
        if pred is None: continue
        # Y_target (clipped+normalized) denormalized with same clip_max as model
        dp=PREPRO/f"{G}x{G}"
        with open(dp/"metadata.json") as fmeta: meta_loc=json.load(fmeta)
        cm=float(meta_loc["scaling"]["max"][ci])
        Y_tgt=np.load(dp/"Y_target.npy").astype(np.float64)
        gt_matched=np.maximum((Y_tgt[hour-2,:,:,ci]+1)/2*cm, 0)
        gt_raw=gts[cyt][G]

        diff=np.abs(pred-gt_matched); norm=gt_norms[cyt]; dmax=diff_vmax[cyt]
        for col,(data,cmap,vnorm) in enumerate([
            (gt_raw,CMAP_FIELD,norm),(pred,CMAP_FIELD,norm),(diff,CMAP_DIFF,None)]):
            ax=axes[row,col]
            if vnorm: im=ax.imshow(data,cmap=cmap,norm=vnorm,origin="lower")
            else: im=ax.imshow(data,cmap=cmap,vmin=0,vmax=dmax,origin="lower")
            ax.set_xticks([]);ax.set_yticks([]);plt.colorbar(im,ax=ax,fraction=0.046,pad=0.04)
            ax.set_title(col_titles[col],fontsize=10)
            if col==0: ax.set_ylabel(cl,fontsize=10)
            if col==2:
                ssr=np.sum((pred-gt_matched)**2);sst=np.sum((gt_matched-gt_matched.mean())**2)
                r2=1-ssr/sst if sst>0 else 0
                ax.text(0.95,0.05,f"$R^2={r2:.4f}$",transform=ax.transAxes,fontsize=16,ha="right",va="bottom",
                       bbox=dict(boxstyle="round,pad=0.3",fc="white",alpha=0.85,ec="gray",lw=0.5))
    fig.tight_layout(); sf(fig,f"Fig6a_GT_Pred_Diff_t{hour}")

    # 6b: Absolute diff all models
    rows_def=[(cyt,cl,G) for cyt,cl,_ in CYTS for G in GRIDS_100]
    nr=len(rows_def);nc=len(MODEL_ORDER)
    fig,axes=plt.subplots(nr,nc,figsize=(3.0*nc,3.0*nr))
    for ri,(cyt,cl,G) in enumerate(rows_def):
        ci_loc=0 if cyt=="il8" else 3
        dp=PREPRO/f"{G}x{G}"
        with open(dp/"metadata.json") as fmeta: meta_loc=json.load(fmeta)
        cm=float(meta_loc["scaling"]["max"][ci_loc])
        Y_tgt=np.load(dp/"Y_target.npy").astype(np.float64)
        gt_matched=np.maximum((Y_tgt[hour-2,:,:,ci_loc]+1)/2*cm, 0)
        dmax=diff_vmax[cyt]
        for col,mn in enumerate(MODEL_ORDER):
            ax=axes[ri,col]
            if G in all_preds[cyt].get(mn,{}):
                pred=all_preds[cyt][mn][G];diff=np.abs(pred-gt_matched)
                ssr=np.sum((pred-gt_matched)**2);sst=np.sum((gt_matched-gt_matched.mean())**2)
                r2=1-ssr/sst if sst>0 else 0
                im=ax.imshow(diff,cmap=CMAP_DIFF,vmin=0,vmax=dmax,origin="lower")
                ax.text(0.95,0.05,f"$R^2={r2:.3f}$",transform=ax.transAxes,fontsize=16,ha="right",va="bottom",
                       bbox=dict(boxstyle="round,pad=0.3",fc="white",alpha=0.85,ec="gray",lw=0.5))
            else:
                ax.text(0.5,0.5,"N/A",transform=ax.transAxes,fontsize=12,ha="center",va="center",color="gray")
            ax.set_xticks([]);ax.set_yticks([])
            if ri==0: ax.set_title(mn,fontsize=9)
            if col==0: ax.set_ylabel(f"{cl}, ${G}^2$",fontsize=9)
    cbar_ax=fig.add_axes([0.93,0.15,0.012,0.7])
    fig.colorbar(im,cax=cbar_ax,label=r"$\vert$Pred - GT$\vert$")
    fig.tight_layout(rect=[0,0,0.92,1]); sf(fig,f"Fig6b_diff_all_t{hour}")


# measuring pred times
def measure_pred_times():
    """measures prediction times for all models/cytokines/grids where JSON lacks pred_time."""
    import tensorflow as tf; import time
    os.environ["TF_CPP_MIN_LOG_LEVEL"]="3"; warnings.filterwarnings("ignore")
    tf.config.optimizer.set_jit(False)
    for gpu in tf.config.list_physical_devices("GPU"):
        try: tf.config.experimental.set_memory_growth(gpu,True)
        except: pass

    BASE=Path(".").resolve(); PREPRO=BASE/"preprocessed"; MODELS=BASE/"models"
    MODEL_DIRS={"DeepONet":"deeponet_h","PI-DeepONet":"pi_deeponet",
                "U-Net":"unet","STA-LSTM":"sta_lstm","PINN":"pinn"}

    def _cb(x,f):
        x=tf.keras.layers.Conv2D(f,3,padding="same",activation="relu")(x)
        x=tf.keras.layers.BatchNormalization()(x)
        x=tf.keras.layers.Conv2D(f,3,padding="same",activation="relu")(x)
        return tf.keras.layers.BatchNormalization()(x)
    def build_unet(G,bf=32,depth=4,do=0.0):
        inp=tf.keras.Input(shape=(G,G,22));sk=[];x=inp
        for i in range(depth): s=_cb(x,bf*(2**i));sk.append(s);x=tf.keras.layers.MaxPooling2D(2,padding="same")(s)
        x=_cb(x,bf*(2**depth))
        if do>0: x=tf.keras.layers.Dropout(do)(x)
        for i in reversed(range(depth)):
            x=tf.keras.layers.Conv2DTranspose(bf*(2**i),2,strides=2,padding="same",activation="relu")(x);s=sk[i]
            if x.shape[1]!=s.shape[1] or x.shape[2]!=s.shape[2]: x=tf.keras.layers.Resizing(s.shape[1],s.shape[2])(x)
            x=tf.keras.layers.Concatenate()([x,s]);x=_cb(x,bf*(2**i))
        return tf.keras.Model(inp,tf.keras.layers.Conv2D(1,1,padding="same")(x))
    class SpatialAttention(tf.keras.layers.Layer):
        def __init__(s,**kw): super().__init__(**kw);s.attn_conv=tf.keras.layers.Conv2D(1,1,padding="same",activation="sigmoid")
        def call(s,x): return x*s.attn_conv(x)
    class STALSTM(tf.keras.Model):
        def __init__(s,G,filters=64,lstm_units=128):
            super().__init__();s.grid_size=G;s.filters=filters;s.lstm_units=lstm_units;s.latent_size=max(G//4,8)
            s.enc=tf.keras.layers.TimeDistributed(tf.keras.Sequential([tf.keras.layers.Conv2D(filters,3,strides=2,padding="same",activation="relu"),tf.keras.layers.Conv2D(filters,3,strides=2,padding="same",activation="relu")]),name="encoder")
            s.spatial_attn=tf.keras.layers.TimeDistributed(SpatialAttention(),name="spatial_attention")
            s.gap=tf.keras.layers.TimeDistributed(tf.keras.layers.GlobalAveragePooling2D(),name="gap")
            s.lstm=tf.keras.layers.LSTM(lstm_units,return_sequences=False,name="lstm");s.relu=tf.keras.layers.Activation("relu")
            s.fc=tf.keras.layers.Dense(s.latent_size**2*filters,activation="relu");s.reshape_latent=tf.keras.layers.Reshape((s.latent_size,s.latent_size,filters))
            s.deconv1=tf.keras.layers.Conv2DTranspose(filters//2,3,strides=2,padding="same",activation="relu")
            s.deconv2=tf.keras.layers.Conv2DTranspose(filters//4,3,strides=2,padding="same",activation="relu")
            s.out_conv=tf.keras.layers.Conv2D(1,3,padding="same",activation="linear");s.out_resize=tf.keras.layers.Resizing(G,G)
        def call(s,x): h=s.enc(x);h=s.spatial_attn(h);h=s.gap(h);h=s.lstm(h);h=s.relu(h);h=s.fc(h);h=s.reshape_latent(h);return s.out_resize(s.out_conv(s.deconv2(s.deconv1(h))))
    class Branch(tf.keras.layers.Layer):
        def __init__(s,h,p,**kw): super().__init__(**kw);s.fc1=tf.keras.layers.Dense(h,activation="relu");s.fc2=tf.keras.layers.Dense(p)
        def call(s,x,training=False): return s.fc2(s.fc1(x))
    class Trunk(tf.keras.layers.Layer):
        def __init__(s,h,p,**kw):
            super().__init__(**kw);s.U=tf.keras.layers.Dense(h,activation="tanh");s.V=tf.keras.layers.Dense(h,activation="tanh")
            s.W1a=tf.keras.layers.Dense(h,activation="relu");s.W1b=tf.keras.layers.Dense(h)
            s.W2a=tf.keras.layers.Dense(h,activation="relu");s.W2b=tf.keras.layers.Dense(h);s.out=tf.keras.layers.Dense(p)
        def call(s,x): u=s.U(x);v=s.V(x);h=s.W1b(s.W1a(x));h=h*u+(1-h)*v;h=s.W2b(s.W2a(h));h=h*u+(1-h)*v;return s.out(h)
    class DON(tf.keras.Model):
        def __init__(s,h,p): super().__init__();s.branch=Branch(h,p);s.trunk=Trunk(h,p);s.bias=s.add_weight(shape=(1,),initializer="zeros",trainable=True)
        def call(s,inp,training=False): b=s.branch(inp[0],training=training);t=s.trunk(inp[1]);return tf.expand_dims(tf.einsum("bp,bnp->bn",b,t)+s.bias,-1)
    class PIDON(tf.keras.Model):
        def __init__(s,h,p): super().__init__();s.branch=Branch(h,p);s.trunk=Trunk(h,p);s.bias=s.add_weight(shape=(1,),initializer="zeros",trainable=True)
        def call_data(s,xb,xt,training=False): b=s.branch(xb,training=training);t=s.trunk(xt);return tf.expand_dims(tf.einsum("bp,bnp->bn",b,t)+s.bias,-1)
        def call(s,inp,training=False): return s.call_data(inp[0],inp[1],training=training)
    def branch_inp(Xb,Xt,ci):
        N,_,G,_,_=Xb.shape;f0=Xb[:,0,:,:,ci];mask=(Xb[:,0,:,:,6:].max(-1)>0.5).astype(np.float32)
        xs=np.linspace(0,1,G,dtype=np.float32);xx,yy=np.meshgrid(xs,xs,indexing='ij');o=np.zeros((N,7),np.float32)
        for i in range(N):
            f=f0[i];m=mask[i];na=float(np.sum(m))+1e-6
            o[i]=[(float(np.max(f))+1)/2,(float(np.mean(f))+1)/2,float(np.std(f)),float(np.sum(xx*m)/na),float(np.sum(yy*m)/na),na/(G*G),float(Xt[i,0,2])]
        return o
    def trunk_inp(Xb,Xt):
        N,_,G,_,_=Xb.shape;return np.concatenate([Xt[:,:,:2].astype(np.float32),Xb.transpose(0,2,3,1,4).reshape(N,G*G,22).astype(np.float32)],-1)

    _load_pred_time_cache()
    n_measured = 0

    for cyt in ["il8","il10"]:
        ci = 0 if cyt=="il8" else 3
        for grid in [100, 250]:
            for seed in [42]:
                for mname in MO_100:
                    # skip if already have pred_time
                    r = lj(W100[mname], cyt, grid, seed)
                    if not r: continue
                    pt = gpt(r)
                    key = (mname, cyt, grid, seed)
                    if (pt and pt > 0) or key in _PRED_TIME_CACHE: continue

                    d = MODELS / MODEL_DIRS[mname]
                    dp = PREPRO / f"{grid}x{grid}"

                    bp = r["best_params"]
                    print(f"    Measuring {mname} {cyt} {grid} seed {seed}...", end=" ", flush=True)

                    try:
                        tf.keras.backend.clear_session()
                        G2 = grid*grid

                        if mname == "U-Net":
                            X = np.load(dp/"X_unet.npy").astype(np.float32)
                            m = build_unet(grid, bp["base_filters"], bp["depth"], bp.get("dropout",0))
                            w = d/f"weights_{cyt}_{grid}_{seed}.weights.h5"
                            if not w.exists(): print("NO WEIGHTS"); continue
                            m.load_weights(str(w))
                            _ = m.predict(X[:1], verbose=0)  # warmup
                            t0 = time.time(); _ = m.predict(X[80:99], batch_size=2, verbose=0)
                            elapsed = time.time() - t0

                        elif mname == "STA-LSTM":
                            X = np.load(dp/"X_lstm.npy").astype(np.float32)
                            m = STALSTM(grid, bp["filters"], bp["lstm_units"]); _ = m(X[:1])
                            w = d/f"weights_{cyt}_{grid}_{seed}.weights.h5"
                            if not w.exists(): print("NO WEIGHTS"); continue
                            m.load_weights(str(w))
                            _ = m.predict(X[:1], verbose=0)
                            t0 = time.time(); _ = m.predict(X[80:99], batch_size=2, verbose=0)
                            elapsed = time.time() - t0

                        elif mname in ("DeepONet","PI-DeepONet"):
                            Xb = np.load(dp/"X_branch.npy").astype(np.float32)
                            Xt = np.load(dp/"X_trunk.npy").astype(np.float32)
                            xbr = branch_inp(Xb, Xt, ci); xtr = trunk_inp(Xb, Xt)
                            is_pi = mname=="PI-DeepONet"
                            if is_pi: m=PIDON(bp["hidden"],bp["p"]);_=m.call_data(tf.constant(xbr[:1]),tf.constant(xtr[:1,:10,:]))
                            else: m=DON(bp["hidden"],bp["p"]);_=m([tf.constant(xbr[:1]),tf.constant(xtr[:1,:10,:])])
                            w = d/f"weights_{cyt}_{grid}_{seed}.weights.h5"
                            if not w.exists(): print("NO WEIGHTS"); continue
                            m.load_weights(str(w))
                            t0 = time.time()
                            for i in range(80, 99):
                                xb = tf.constant(xbr[i:i+1])
                                for s in range(0, G2, EVAL_CHUNK):
                                    e = min(s+EVAL_CHUNK, G2); xt = tf.constant(xtr[i:i+1, s:e])
                                    if is_pi: _ = m.call_data(xb, xt, training=False)
                                    else: _ = m([xb, xt], training=False)
                            elapsed = time.time() - t0

                        elif mname == "PINN":
                            import deepxde as dde; dde.config.set_default_float("float64"); dde.config.disable_xla_jit()
                            net = dde.maps.FNN([3]+[bp["hidden"]]*bp["n_layers"]+[1],"tanh","Glorot uniform")
                            net.apply_output_transform(lambda x,y:y); net(tf.constant(np.zeros((1,3),np.float64)))
                            wp = str(d/f"weights_{cyt}_{grid}_{seed}-*.weights.h5"); wfs = sorted(glob.glob(wp))
                            if not wfs:
                                wp2 = str(d/f"weights_{cyt}_{grid}_{seed}.weights.h5")
                                if os.path.exists(wp2): wfs = [wp2]
                            if not wfs: print("NO WEIGHTS"); continue
                            net.load_weights(wfs[-1])
                            xs_g = np.linspace(-1,1,grid,np.float64); xx,yy = np.meshgrid(xs_g,xs_g,indexing="ij")
                            xy = np.stack([xx.ravel(),yy.ravel()],1); tn = np.linspace(-1,1,101,np.float64)
                            t0 = time.time()
                            for i in range(80, 99):
                                tt = np.full((G2,1), tn[i+2], np.float64)
                                _ = net(tf.constant(np.hstack([xy, tt])))
                            elapsed = time.time() - t0
                        else:
                            continue

                        _PRED_TIME_CACHE[key] = elapsed
                        n_measured += 1
                        print(f"{elapsed:.2f}s")
                    except Exception as e:
                        print(f"FAILED: {e}")

    if n_measured > 0:
        _save_pred_time_cache()
    print(f"    Measured {n_measured} new pred times")

#  vis
def clean_vis():
    import tensorflow as tf; import matplotlib.colors as mcolors
    os.environ["TF_CPP_MIN_LOG_LEVEL"]="3";warnings.filterwarnings("ignore")
    tf.config.optimizer.set_jit(False)
    for gpu in tf.config.list_physical_devices("GPU"):
        try: tf.config.experimental.set_memory_growth(gpu,True)
        except: pass
    PREPRO=Path(".").resolve()/"preprocessed";MDIR=Path(".").resolve()/"models"/"deeponet_h"
    G=250;HOUR=88;t_idx=HOUR-2;G2=G*G;dp=PREPRO/f"{G}x{G}"
    Y_raw=np.load(dp/"Y_raw_phys.npy");Xb=np.load(dp/"X_branch.npy").astype(np.float32)
    Xt=np.load(dp/"X_trunk.npy").astype(np.float32)
    with open(dp/"metadata.json") as f: meta=json.load(f)
    class Branch(tf.keras.layers.Layer):
        def __init__(s,h,p,**kw): super().__init__(**kw);s.fc1=tf.keras.layers.Dense(h,activation="relu");s.fc2=tf.keras.layers.Dense(p)
        def call(s,x,training=False): return s.fc2(s.fc1(x))
    class Trunk(tf.keras.layers.Layer):
        def __init__(s,h,p,**kw):
            super().__init__(**kw);s.U=tf.keras.layers.Dense(h,activation="tanh");s.V=tf.keras.layers.Dense(h,activation="tanh")
            s.W1a=tf.keras.layers.Dense(h,activation="relu");s.W1b=tf.keras.layers.Dense(h)
            s.W2a=tf.keras.layers.Dense(h,activation="relu");s.W2b=tf.keras.layers.Dense(h);s.out=tf.keras.layers.Dense(p)
        def call(s,x):
            u=s.U(x);v=s.V(x);h=s.W1b(s.W1a(x));h=h*u+(1-h)*v;h=s.W2b(s.W2a(h));h=h*u+(1-h)*v;return s.out(h)
    class DON(tf.keras.Model):
        def __init__(s,h,p): super().__init__();s.branch=Branch(h,p);s.trunk=Trunk(h,p);s.bias=s.add_weight(shape=(1,),initializer="zeros",trainable=True)
        def call(s,inp,training=False): b=s.branch(inp[0],training=training);t=s.trunk(inp[1]);return tf.expand_dims(tf.einsum("bp,bnp->bn",b,t)+s.bias,-1)
    def denorm(s,cm): return (np.asarray(s,np.float64)+1)/2*cm
    def branch_inp(Xb,Xt,ci):
        N,_,Gs,_,_=Xb.shape;f0=Xb[:,0,:,:,ci];mask=(Xb[:,0,:,:,6:].max(-1)>0.5).astype(np.float32)
        xs=np.linspace(0,1,Gs,dtype=np.float32);xx,yy=np.meshgrid(xs,xs,indexing='ij');o=np.zeros((N,7),np.float32)
        for i in range(N):
            f=f0[i];m=mask[i];na=float(np.sum(m))+1e-6
            o[i]=[(float(np.max(f))+1)/2,(float(np.mean(f))+1)/2,float(np.std(f)),float(np.sum(xx*m)/na),float(np.sum(yy*m)/na),na/(Gs*Gs),float(Xt[i,0,2])]
        return o
    def trunk_inp(Xb,Xt):
        N,_,Gs,_,_=Xb.shape;return np.concatenate([Xt[:,:,:2].astype(np.float32),Xb.transpose(0,2,3,1,4).reshape(N,Gs*Gs,22).astype(np.float32)],-1)
    for cyt,cl,ci in [("il8","IL-8",0),("il10","IL-10",3)]:
        print(f"  {cl}...",end=" ",flush=True);tf.keras.backend.clear_session()
        cm=float(meta["scaling"]["max"][ci]);rp=MDIR/f"res_{cyt}_{G}_{SEED}.json"
        if not rp.exists(): print("SKIP");continue
        with open(rp) as f: bp=json.load(f)["best_params"]
        xbr=branch_inp(Xb,Xt,ci);xtr=trunk_inp(Xb,Xt)
        model=DON(bp["hidden"],bp["p"]);_=model([tf.constant(xbr[:1]),tf.constant(xtr[:1,:10,:])])
        wgt=MDIR/f"weights_{cyt}_{G}_{SEED}.weights.h5"
        if not wgt.exists(): print("NO WEIGHTS");continue
        model.load_weights(str(wgt));out=np.zeros((1,G2,1),np.float32);xb_t=tf.constant(xbr[t_idx:t_idx+1])
        for s in range(0,G2,EVAL_CHUNK):
            e=min(s+EVAL_CHUNK,G2);out[0,s:e]=model([xb_t,tf.constant(xtr[t_idx:t_idx+1,s:e])],training=False).numpy()[0]
        pred=np.maximum(denorm(out.reshape(G,G),cm),0);gt=Y_raw[HOUR,:,:,ci]
        vmax=np.percentile(gt[gt>0],99.5) if (gt>0).any() else max(gt.max(),1e-15)
        norm=mcolors.PowerNorm(gamma=0.5,vmin=0,vmax=vmax)
        for data,tag in [(gt,"GT"),(pred,"Pred")]:
            fig,ax=plt.subplots(figsize=(5,5));ax.imshow(data,cmap=CMAP_FIELD,norm=norm,origin="lower");ax.axis("off")
            fig.savefig(FIGDIR/f"clean_{tag}_{cl}_t{HOUR}.png",dpi=300,bbox_inches="tight",pad_inches=0,transparent=True);plt.close(fig)
        fig,(a1,a2)=plt.subplots(1,2,figsize=(10,5))
        a1.imshow(gt,cmap=CMAP_FIELD,norm=norm,origin="lower");a1.axis("off")
        a2.imshow(pred,cmap=CMAP_FIELD,norm=norm,origin="lower");a2.axis("off")
        fig.subplots_adjust(wspace=0.02,left=0,right=1,top=1,bottom=0)
        fig.savefig(FIGDIR/f"clean_GTvsPred_{cl}_t{HOUR}.png",dpi=300,bbox_inches="tight",pad_inches=0.01,transparent=True);plt.close(fig)
        print("OK")


# CSV export
def export_csv():
    _load_pred_time_cache()
    mks=["Global_R2","SSIM","Avg_Dice","Spatial_Correlation","Masked_RMSE","Unmasked_RMSE"]

    # Per-seed metrics (long form)
    rows=[]
    for cyt in ["il8","il10"]:
        for model in MO_100:
            for grid in [100, 250]:
                for seed in SEEDS:
                    r=lj(W100[model],cyt,grid,seed)
                    if not r: continue
                    for hn,hf in [("near",get_near),("far",get_far)]:
                        h=hf(r)
                        if not h: continue
                        pt = _get_pred_time(model, cyt, grid, seed, W100)
                        row={"dataset":"100h","cytokine":cyt,"model":model,"grid":grid,"seed":seed,
                             "horizon":hn,"train_time_s":gtt(r),"pred_time_s":pt}
                        for mk in mks:
                            v=h.get(mk);row[mk]=v if v is not None and (not isinstance(v,float) or v==v) else None
                        rows.append(row)
    if rows:
        outpath=FIGDIR/"metrics_100h.csv"
        with open(outpath,"w",newline="") as f:
            w=csv.DictWriter(f,fieldnames=list(rows[0].keys()));w.writeheader();w.writerows(rows)
        print(f"    -> {outpath} ({len(rows)} rows)")

    # Mean across seeds (one row per (cyt, model, grid, horizon))
    agg_rows = []
    for cyt in ["il8","il10"]:
        for model in MO_100:
            for grid in [100, 250]:
                tt_m, n_tt = gtt_mean(W100[model], cyt, grid)
                pt_m, n_pt = gpt_mean(model, cyt, grid, W100)
                abm_t = abm_time(grid)
                speedup = (abm_t / pt_m) if (abm_t is not None and pt_m and pt_m > 0) else None
                for hn, hf in [("near",get_near),("far",get_far)]:
                    h = hf(lj(W100[model], cyt, grid, 42))
                    if not h: continue
                    row = {"dataset":"100h","cytokine":cyt,"model":model,"grid":grid,
                           "horizon":hn,"metrics_seed":42,
                           "train_time_s_mean":tt_m,"train_time_n_seeds":n_tt,
                           "pred_time_s_mean":pt_m,"pred_time_n_seeds":n_pt,
                           "abm_time_s":abm_t,"inference_speedup_vs_abm":speedup}
                    for mk in mks:
                        v = h.get(mk)
                        row[mk] = v if v is not None and (not isinstance(v,float) or v==v) else None
                    agg_rows.append(row)
    if agg_rows:
        outpath=FIGDIR/"metrics_100h_means.csv"
        with open(outpath,"w",newline="") as f:
            w=csv.DictWriter(f,fieldnames=list(agg_rows[0].keys()));w.writeheader();w.writerows(agg_rows)
        print(f"    -> {outpath} ({len(agg_rows)} rows)")

    # Standalone budgets CSV: one row per (model, cytokine, grid) with all costs side-by-side
    budget_rows = []
    for cyt in ["il8","il10"]:
        for model in MO_100:
            for grid in [100, 250]:
                tt_m, n_tt = gtt_mean(W100[model], cyt, grid)
                pt_m, n_pt = gpt_mean(model, cyt, grid, W100)
                abm_t = abm_time(grid)
                speedup = (abm_t / pt_m) if (abm_t is not None and pt_m and pt_m > 0) else None
                budget_rows.append({
                    "dataset": "100h", "cytokine": cyt, "model": model, "grid": grid,
                    "abm_time_s": abm_t,
                    "train_time_s_mean": tt_m, "train_time_n_seeds": n_tt,
                    "pred_time_s_mean": pt_m, "pred_time_n_seeds": n_pt,
                    "inference_speedup_vs_abm": speedup,
                })
    if budget_rows:
        outpath = FIGDIR/"computational_budgets.csv"
        with open(outpath, "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=list(budget_rows[0].keys()))
            w.writeheader(); w.writerows(budget_rows)
        print(f"    -> {outpath} ({len(budget_rows)} rows)")


# LATEX TABLE GENERATION
def _fv(v):
    if v is None or (isinstance(v,float) and v!=v): return "--"
    if v<0: return f"${v:.4f}$"
    return f"{v:.4f}"
def _fr(v):
    if v is None or v==0: return "--"
    exp=int(np.floor(np.log10(abs(v))));m=v/10**exp;return f"${m:.1f}\\text{{e-}}{abs(exp)}$"
def _ft(v):
    if v is None: return "--"
    return "$<$1s" if v<1 else f"{v:.0f}s"
def _gm(r):
    if not r: return {}
    return {"R2":r.get("Global_R2"),"SSIM":r.get("SSIM"),"Dice":r.get("Avg_Dice"),"Corr":r.get("Spatial_Correlation"),"RMSE":r.get("Masked_RMSE")}
def _bb(rows,col):
    vals=[r.get(col) for r in rows if r.get(col) is not None and isinstance(r.get(col),(int,float)) and r.get(col)==r.get(col)]
    if not vals: return
    best=min(vals) if col=="RMSE" else max(vals)
    for r in rows:
        v=r.get(col)
        if v is not None and isinstance(v,(int,float)) and v==v and abs(v-best)<1e-12: r[f"{col}_bold"]=True
def gen_results_table(models,wdirs,grids,cyt,cl,dlabel,seed=42,label=None):
    L=[f"% Auto-generated - {cl}, {dlabel}","\\begin{table*}[!htbp]",
       f"\\caption{{{cl}, {dlabel} (metrics from seed {seed}; train/pred times are means across seeds {SEEDS}). Best per column in bold.}}"]
    if label: L.append(f"\\label{{{label}}}")
    L+=["\\centering\\footnotesize","\\setlength{\\tabcolsep}{3.5pt}","\\begin{tabular}{l ccccc ccccc cc}","\\toprule"]
    for gi,grid in enumerate(grids):
        pfx=f"$\\mathbf{{{grid}^2}}$ " if len(grids)>1 else ""
        L+=[f"& \\multicolumn{{5}}{{c}}{{{pfx}\\textbf{{Near}}}} & \\multicolumn{{5}}{{c}}{{{pfx}\\textbf{{Far}}}} & \\multicolumn{{2}}{{c}}{{\\textbf{{Time (mean)}}}}\\\\",
            "\\cmidrule(lr){2-6}\\cmidrule(lr){7-11}\\cmidrule(lr){12-13}",
            "& $R^2$ & SSIM & Dice & Corr & RMSE & $R^2$ & SSIM & Dice & Corr & RMSE & Tr & Pr\\\\","\\midrule"]
        nr=[_gm(get_near(lj(wdirs[m],cyt,grid,seed))) for m in models]
        fr=[_gm(get_far(lj(wdirs[m],cyt,grid,seed))) for m in models]
        for c in ["R2","SSIM","Dice","Corr","RMSE"]: _bb(nr,c);_bb(fr,c)
        for mi,model in enumerate(models):
            cols=[model]
            for metrics in [nr[mi],fr[mi]]:
                for c in ["R2","SSIM","Dice","Corr","RMSE"]:
                    v=metrics.get(c);s=_fr(v) if c=="RMSE" else _fv(v)
                    if metrics.get(f"{c}_bold") and s!="--": s=f"\\bf{s}"
                    cols.append(s)
            tt_m, _ = gtt_mean(wdirs[model], cyt, grid)
            pt_m, _ = gpt_mean(model, cyt, grid, wdirs)
            cols+=[_ft(tt_m), _ft(pt_m)]
            L.append(" & ".join(cols)+"\\\\")
        if gi<len(grids)-1: L.append("\\midrule")
    L+=["\\bottomrule","\\multicolumn{13}{l}{\\scriptsize RMSE = Masked RMSE. Corr = Spatial Correlation. -- = undefined.}","\\end{tabular}","\\end{table*}"]
    return "\n".join(L)
def gen_seed_table(models,wdirs,grids,cyt,cl,seeds=[1,42,100],label=None):
    L=[f"% Auto-generated - Seed variability {cl}","\\begin{table}[!htbp]",
       f"\\caption{{Seed variability for {cl} (s.d.\\ across seeds {', '.join(map(str,seeds))}; near horizon).}}"]
    if label: L.append(f"\\label{{{label}}}")
    L+=["\\centering\\footnotesize","\\begin{tabular}{l"+" cccc"*len(grids)+"}","\\toprule"]
    L.append("& "+" & ".join([f"\\multicolumn{{4}}{{c}}{{\\textbf{{${g}^2$}}}}" for g in grids])+"\\\\")
    L.append(" ".join([f"\\cmidrule(lr){{{2+i*4}-{5+i*4}}}" for i in range(len(grids))]))
    L.append("& "+" & ".join(["$R^2$ & SSIM & Dice & Corr"]*len(grids))+"\\\\");L.append("\\midrule")
    mks=["Global_R2","SSIM","Avg_Dice","Spatial_Correlation"]
    for model in models:
        row=[model]
        for grid in grids:
            for mk in mks:
                vals=[]
                for seed in seeds:
                    r=lj(wdirs[model],cyt,grid,seed);v=get_near(r).get(mk)
                    if v is not None and isinstance(v,(int,float)) and v==v: vals.append(v)
                row.append(f"{np.std(vals,ddof=0):.4f}" if len(vals)>=2 else "--")
        L.append(" & ".join(row)+"\\\\")
    L+=["\\bottomrule",f"\\multicolumn{{{1+4*len(grids)}}}{{l}}{{\\scriptsize Masked RMSE omitted.}}","\\end{tabular}","\\end{table}"]
    return "\n".join(L)

def _fmt_abm(v):
    if v is None: return "--"
    return f"{v:.1f}\\,s"

def _fmt_speedup(v):
    if v is None: return "--"
    if v >= 1e4: return f"{v/1000:.1f}k$\\times$"
    if v >= 1:   return f"{v:.0f}$\\times$"
    return f"{v:.2f}$\\times$"

def gen_budgets_table(label=None):
    """Computational budgets table: ABM cost, mean train cost, mean pred cost, speedup.
    Aggregates across cytokines (mean) so each (model, grid) is a single row."""
    L=["% Auto-generated - Computational budgets","\\begin{table*}[!htbp]",
       "\\caption{Measured wall-clock budgets for one full 100\\,h simulation. ABM column gives the CompuCell3D / CPM cost of generating one trajectory at each grid resolution. Training and prediction columns are means across cytokines (IL-8, IL-10) and seeds " + ", ".join(map(str,SEEDS)) + ". Inference speed-up is computed as ABM time divided by mean prediction time at the same grid resolution.}"]
    if label: L.append(f"\\label{{{label}}}")
    L+=["\\centering\\footnotesize","\\setlength{\\tabcolsep}{4pt}",
        "\\begin{tabular}{l c c c c c}","\\toprule",
        "\\textbf{Grid} & \\textbf{Model} & \\textbf{ABM (CPM)} & \\textbf{Train (mean)} & \\textbf{Pred (mean)} & \\textbf{Inference speed-up} \\\\",
        "\\midrule"]
    for grid in [100, 250]:
        abm_t = abm_time(grid)
        for mi, model in enumerate(MO_100):
            tt_vals = []; pt_vals = []
            for cyt in ["il8","il10"]:
                tt_m, _ = gtt_mean(W100[model], cyt, grid)
                pt_m, _ = gpt_mean(model, cyt, grid, W100)
                if tt_m is not None: tt_vals.append(tt_m)
                if pt_m is not None: pt_vals.append(pt_m)
            tt_avg = float(np.mean(tt_vals)) if tt_vals else None
            pt_avg = float(np.mean(pt_vals)) if pt_vals else None
            speedup = (abm_t / pt_avg) if (abm_t is not None and pt_avg and pt_avg > 0) else None
            grid_cell = f"$\\mathbf{{{grid}^2}}$" if mi == 0 else ""
            L.append(f"{grid_cell} & {model} & {_fmt_abm(abm_t)} & {_ft(tt_avg)} & {_ft(pt_avg)} & {_fmt_speedup(speedup)} \\\\")
        if grid != 250: L.append("\\midrule")
    L+=["\\bottomrule","\\end{tabular}","\\end{table*}"]
    return "\n".join(L)

def gen_tables():
    _load_pred_time_cache()
    out=["% AUTO-GENERATED LATEX TABLES",""]
    out+=[gen_results_table(MO_100,W100,[100,250],"il8","IL-8","100h",label="tab:100h_il8"),""]
    out+=[gen_results_table(MO_100,W100,[100,250],"il10","IL-10","100h",label="tab:100h_il10"),""]
    out+=["% -- COMPUTATIONAL BUDGETS --",""]
    out+=[gen_budgets_table(label="tab:budgets"),""]
    out+=["% -- SEED STABILITY --",""]
    out+=[gen_seed_table(MO_100,W100,[100,250],"il8","IL-8",label="tab:seed_il8"),""]
    out+=[gen_seed_table(MO_100,W100,[100,250],"il10","IL-10",label="tab:seed_il10"),""]
    outpath=FIGDIR/"tables_auto.tex"
    with open(outpath,"w") as f: f.write("\n".join(out))
    print(f"    -> {outpath}")


FIGS = {
    1:("Concentrations + sparsity (EDA)",fig1),
    2:("R2 bars (RQ1)",fig2),
    "2b":("Pareto + Speedup (RQ1)",fig2b),
    3:("Multi-metric + Scalability + Temporal (RQ3)",fig3),
    6:("Spatial maps GT/Pred/|Diff| (RQ2, needs TF)",fig6),
}

if __name__=="__main__":
    ap=argparse.ArgumentParser()
    ap.add_argument("--fig",nargs="*",default=None,help="Figure numbers: 1 2 2b 3 6. Default: all.")
    ap.add_argument("--clean",action="store_true",help="Clean DeepONet GT/Pred for graphical abstract (needs TF)")
    args=ap.parse_args()
    FIGDIR.mkdir(parents=True,exist_ok=True); setup()

    if args.clean:
        print("Clean DeepONet vis (graphical abstract)...\n"); clean_vis()
    else:
        print("Measuring prediction times")
        measure_pred_times()

        print("\nCSV export"); export_csv()
        print("\nLaTeX tables"); gen_tables()

        # Figures
        if args.fig:
            figs=[]
            for f in args.fig:
                try: figs.append(int(f))
                except ValueError: figs.append(f)
        else:
            figs=[1,2,"2b",3,6]
        print(f"\nGenerating {len(figs)} figure(s) \n")
        for n in figs:
            if n not in FIGS: print(f"Unknown: {n}"); continue
            nm,fn=FIGS[n]; print(f"  Fig {n}: {nm}")
            try: fn()
            except Exception as e:
                import traceback;print(f"FAILED: {e}");traceback.print_exc()
        print(f"\nDone -> {FIGDIR}/")
