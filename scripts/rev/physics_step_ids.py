"""ABM cytokine update re-implemented from the raw lattice output, with cell identities.

Order of operations in one ABM step (every 10,000 MCS, model file endothelialSteppables.py):
  1. the lattice site of each cell's centre of mass is marked for the cell types that enter the
     cytokine equations, using the cell types as they are at that moment;
  2. one implicit solver step is taken;
  3. cells past their life span are deleted and cell-type transitions (e.g. M1 -> M2) are applied;
  4. the lattice (cell types, cell ids, cytokine fields) is written.
The stored cell types of frame t are therefore those AFTER the transitions of step t, whereas the
solver at step t used the types stored in frame t-1, at the positions of frame t. This script
reconstructs that from the CellId field:

  ids      source cells = cells whose type in frame t-1 is the secreting type, placed at their
           centre-of-mass site in frame t (at their frame t-1 site if they were deleted in step t)
  lattice  all lattice sites whose stored type in frame t is the secreting type (no identities)
  persistence  previous frame

Usage: python physics_step_ids.py --runs-root <folder with one sub-folder per run> [--runs name ...]
Output: outputs/<REV_OUT>/physics_step_ids.txt and .csv
"""
import argparse, csv, json, os, re, sys, warnings
from pathlib import Path
import numpy as np
import scipy.sparse as sp
from scipy.sparse.linalg import splu
import pyvista as pv
import physics_step as PS
import common as C
from paths import OUTPUTS
warnings.filterwarnings("ignore")

EC, NA, NDN, M2 = 1, 5, 6, 9
AREA = lambda G: PS.TRUE_SIZE ** 2 / G ** 2
#          D (cm2/s)  mu (1/h)  sources {type: k}             uptake {type: theta}
PAR = {"il8": (2.09e-6, 0.2, {EC: 234e-5, NDN: 1.46e-5}, {NA: 3.024e-5}),
       "il10": (1.45e-8, 0.5, {M2: 45e-5}, {})}


def read_frames(d, G):
    files = sorted([f for f in os.listdir(d) if f.endswith(".vtk")], key=lambda x: int("".join(filter(str.isdigit, x)) or 0))
    out = []
    for f in files:
        m = pv.read(str(Path(d) / f)); g = lambda k: np.asarray(m.point_data[k]).reshape(G, G, order="F")
        ids, typ = g("CellId").astype(np.int64), g("CellType").astype(np.int64)
        xs, ys = np.meshgrid(np.arange(G), np.arange(G), indexing="ij")
        sel = ids > 0; u, inv = np.unique(ids[sel], return_inverse=True); n = np.bincount(inv)
        cx = np.minimum((np.bincount(inv, xs[sel]) / n).astype(int), G - 1); cy = np.minimum((np.bincount(inv, ys[sel]) / n).astype(int), G - 1)
        ty = np.zeros(len(u), int); ty[inv] = typ[sel]
        out.append({"il8": g("il8").astype(np.float64), "il10": g("il10").astype(np.float64), "type": typ,
                    "cells": {int(c): (int(t), int(x), int(y)) for c, t, x, y in zip(u, ty, cx, cy)}})
    return out


def presence_ids(prev, cur, T, G):
    p = np.zeros((G, G))
    if prev is None:                                           # first step: types as created
        for c, (t, x, y) in cur["cells"].items():
            if t == T and T == EC: p[x, y] = 1.0
        return p
    for c, (t, x, y) in prev["cells"].items():
        if t == T:
            if c in cur["cells"]: _, x, y = cur["cells"][c]
            p[x, y] = 1.0
    return p


def evaluate(frames, G, b, cyt):
    D, mu, src, upt = PAR[cyt]; a = AREA(G); D = D * PS.S_MCS / a; mu = mu * PS.H_MCS
    src = {T: k * a * PS.H_MCS for T, k in src.items()}; upt = {T: k * a * PS.H_MCS for T, k in upt.items()}
    K = PS.operator(G, D, b); n = G * G; I = sp.identity(n)
    base = splu((I * (1.0 + mu) + K).tocsc())

    def step(prev_u, pres, th):
        rhs = prev_u.ravel(order="F") + sum(k * pres[T].ravel(order="F") for T, k in src.items())
        lu = splu((I * (1.0 + mu) + sp.diags(sum(k * th[T].ravel(order="F") for T, k in upt.items())) + K).tocsc()) if upt else base
        return lu.solve(rhs).reshape(G, G, order="F")

    T = len(frames); u = np.stack([f[cyt] for f in frames]); types = set(src) | set(upt)
    P_ids = [{t_: presence_ids(frames[t - 1] if t else None, frames[t], t_, G) for t_ in types} for t in range(T)]
    P_lat = [{t_: (frames[t]["type"] == t_).astype(float) for t_ in types} for t in range(T)]
    zero = np.zeros((G, G))
    ids = np.stack([step(u[t - 1] if t else zero, P_ids[t], P_ids[t]) for t in range(T)])
    lat = np.stack([step(u[t - 1] if t else zero, P_lat[t], P_lat[t]) for t in range(T)])
    closed = [step(zero, P_ids[0], P_ids[0])]
    for t in range(1, T): closed.append(step(closed[-1], P_ids[t], P_ids[t]))
    closed = np.stack(closed)
    n_src = float(np.mean([sum(P_ids[t][t_].sum() for t_ in src) for t in range(1, T)]))
    n_lat = float(np.mean([sum(P_lat[t][t_].sum() for t_ in src) for t in range(1, T)]))
    out = {}
    for w, sl in (("t82-100", slice(82, T)), ("t1-100", slice(1, T))):
        if sl.start >= T: continue
        truth, pers = u[sl], u[sl.start - 1:T - 1]; mse = lambda p: float(np.mean((truth - p) ** 2)); mp = max(mse(pers), 1e-300)
        out[w] = {"R2_persistence": C._r2(truth, pers), "R2_ids": C._r2(truth, ids[sl]), "skill_ids": 1 - mse(ids[sl]) / mp,
                  "R2_lattice": C._r2(truth, lat[sl]), "skill_lattice": 1 - mse(lat[sl]) / mp, "R2_closed_ids": C._r2(truth, closed[sl]),
                  "max_rel_dev_ids": float(np.max(np.abs(truth - ids[sl])) / max(truth.max(), 1e-300)),
                  "source_sites_ids": n_src, "source_sites_lattice": n_lat}
    return out


if __name__ == "__main__":
    ap = argparse.ArgumentParser(); ap.add_argument("--runs-root", required=True); ap.add_argument("--runs", nargs="*")
    a = ap.parse_args(); root = Path(a.runs_root).expanduser()
    DST = OUTPUTS / os.environ.get("REV_OUT", "rev"); DST.mkdir(parents=True, exist_ok=True)
    runs = sorted(d.name for d in root.iterdir() if (d / "LatticeData").is_dir() and (not a.runs or d.name in a.runs))
    rows, lines = [], ["ABM CYTOKINE UPDATE FROM THE RAW LATTICE OUTPUT (R2 against the stored fields)",
                       "ids: source cells identified by CellId with their type before the transitions of the step; lattice: stored types of the same frame", ""]
    hdr = f"{'run':38s}{'cyt':>5s}{'window':>9s}{'R2 pers.':>10s}{'R2 ids':>10s}{'skill':>8s}{'R2 lattice':>11s}{'skill':>8s}{'R2 closed':>10s}{'max dev':>9s}{'src ids':>8s}{'src lat':>8s}"
    lines.append(hdr); print(hdr, flush=True)
    for r in runs:
        G = int(re.search(r"_G(\d+)_s\d+$", r).group(1))
        pf = root / r / "abm_provenance.json"; b = json.load(open(pf)).get("boundaryat", G // 10) if pf.exists() else G // 10
        frames = read_frames(root / r / "LatticeData", G)
        for cyt in ("il8", "il10"):
            for w, o in evaluate(frames, G, b, cyt).items():
                rows.append({"run": r, "grid": G, "cytokine": cyt, "window": w, **{k: float(f"{v:.6g}") for k, v in o.items()}})
                s = f"{r:38s}{cyt:>5s}{w:>9s}{o['R2_persistence']:10.5f}{o['R2_ids']:10.6f}{o['skill_ids']:8.4f}{o['R2_lattice']:11.5f}{o['skill_lattice']:8.3f}{o['R2_closed_ids']:10.5f}{o['max_rel_dev_ids']:9.1e}{o['source_sites_ids']:8.1f}{o['source_sites_lattice']:8.1f}"
                lines.append(s); print(s, flush=True)
    with open(DST / "physics_step_ids.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys())); w.writeheader(); w.writerows(rows)
    (DST / "physics_step_ids.txt").write_text("\n".join(lines) + "\n"); print(f"\nwritten to {DST / 'physics_step_ids.txt'}")
