"""Re-implementation of the ABM cytokine update.

The ABM advances the cytokine fields once per hourly frame by one implicit (backward-Euler, dt = 1)
finite-volume step of
    du/dt = D lap(u) - mu u - theta * NA * u + sum_c k_c * cell_c ,
with zero-value constraints on four boundary segments of length `boundaryat` and no-flux elsewhere
(Simulation/test.py and variablevals.py of the ABM). The cell-presence fields come from the cell
positions at the time of the update, so the next frame is a deterministic linear function of the
previous frame and the new cell positions. This script evaluates that update:

  exact      previous ABM frame + cell masks at the target time t      (what the simulator does)
  lagged     previous ABM frame + cell masks at t-1                    (no knowledge of the new positions)
  closed     started from u = 0 and iterated with the ABM cell masks   (whole trajectory)

against the stored ABM frames, with persistence as reference. NDN neutrophils (second IL-8 source)
are not among the stored masks and are ignored (0-4 lattice sites).
Output: outputs/<REV_OUT>/physics_step.txt and .csv
"""
import csv, json, sys, warnings
import numpy as np
import scipy.sparse as sp
from scipy.sparse.linalg import splu
import common as C
from paths import OUTPUTS
warnings.filterwarnings("ignore")

TRUE_SIZE, S_MCS, H_MCS = 5.0, 60.0, 1.0 / 60.0
#            D (cm2/s)  mu (1/h)  sources {mask channel: k}           uptake {mask channel: theta}
PARAMS = {"il8":  (2.09e-6, 0.2, {0: 234e-5},                         {2: 3.024e-5}),      # EC source, activated-neutrophil uptake
          "il10": (1.45e-8, 0.5, {4: 45e-5},                          {})}                 # M2 source
MASKS = ["EC", "NN", "NA", "M1", "M2"]


def coefficients(cyt, G):
    area = TRUE_SIZE ** 2 / G ** 2
    D, mu, src, upt = PARAMS[cyt]
    return D * S_MCS / area, mu * H_MCS, {c: k * area * H_MCS for c, k in src.items()}, {c: k * area * H_MCS for c, k in upt.items()}


def operator(G, D, b):
    """(-D * Laplacian) with no-flux boundaries and zero-value faces on four boundary segments.
    Unknown index = x + y * G (array[x, y] flattened in Fortran order)."""
    idx = np.arange(G * G).reshape(G, G, order="F")
    rows, cols, vals = [], [], []; diag = np.zeros(G * G)
    for a, c in ((idx[:-1, :], idx[1:, :]), (idx[:, :-1], idx[:, 1:])):
        a, c = a.ravel(), c.ravel()
        rows += [a, c]; cols += [c, a]; vals += [-D * np.ones(a.size)] * 2
        np.add.at(diag, a, D); np.add.at(diag, c, D)
    b = int(b)
    for cells in (idx[:b, 0], idx[G - b:, G - 1], idx[G - 1, G - b:], idx[0, :b]):     # bottom-left, top-right, right-top, left-bottom
        np.add.at(diag, cells, 2.0 * D)                                                # face value 0 at half a cell from the centre
    return sp.coo_matrix((np.concatenate(vals), (np.concatenate(rows), np.concatenate(cols))), shape=(G * G, G * G)).tocsr() + sp.diags(diag)


def frames_and_masks(grid, run):
    dp = C.run_dir(grid, run)
    raw = np.load(dp / "Y_raw_phys.npy").astype(np.float64)                    # (101, G, G, 6)
    Xu = np.load(dp / "X_unet.npy", mmap_mode="r")
    M = np.concatenate([np.asarray(Xu[0][..., 6:11])[None], np.asarray(Xu[0][..., 17:22])[None], np.load(dp / "Y_masks_spatial.npy")]).astype(np.float64)
    meta = json.load(open(dp / "metadata.json"))
    b = (meta.get("abm_provenance") or {}).get("boundaryat", grid // 10)
    return raw, M, b


def evaluate(grid, run, cyt):
    raw, M, b = frames_and_masks(grid, run); u = raw[..., C.CYTS.index(cyt)]
    D, mu, src, upt = coefficients(cyt, grid); K = operator(grid, D, b); n = grid * grid
    base = splu((sp.identity(n) * (1.0 + mu) + K).tocsc()) if not upt else None

    def step(prev, masks):
        rhs = prev.ravel(order="F") + sum(k * masks[..., c].ravel(order="F") for c, k in src.items())
        if upt:
            th = sum(k * masks[..., c].ravel(order="F") for c, k in upt.items())
            lu = splu((sp.identity(n) * (1.0 + mu) + sp.diags(th) + K).tocsc())
        else:
            lu = base
        return lu.solve(rhs).reshape(grid, grid, order="F")

    T = u.shape[0]
    exact = np.stack([step(u[t - 1], M[t]) for t in range(1, T)])
    lagged = np.stack([step(u[t - 1], M[t - 1]) for t in range(1, T)])
    closed = [step(np.zeros((grid, grid)), M[0])]
    for t in range(1, T): closed.append(step(closed[-1], M[t]))
    closed = np.stack(closed)[1:]
    truth, pers = u[1:], u[:-1]
    out = {}
    for wname, sl in (("t82-100", slice(81, 100)), ("t1-100", slice(0, 100))):
        mse = lambda p: float(np.mean((truth[sl] - p[sl]) ** 2)); mp = mse(pers)
        out[wname] = {"R2_persistence": C._r2(truth[sl], pers[sl]),
                      **{f"R2_{k}": C._r2(truth[sl], p[sl]) for k, p in (("exact", exact), ("lagged", lagged), ("closed", closed))},
                      **{f"skill_{k}": 1.0 - mse(p) / max(mp, 1e-300) for k, p in (("exact", exact), ("lagged", lagged))},
                      "max_rel_dev_exact": float(np.max(np.abs(truth[sl] - exact[sl])) / max(truth[sl].max(), 1e-300))}
    return out


if __name__ == "__main__":
    DST = OUTPUTS / (C.os.environ.get("REV_OUT") or "rev_abm"); DST.mkdir(parents=True, exist_ok=True)
    import re
    grids = sorted({int(re.search(r"_G(\d+)_s\d+$", d.name).group(1)) for d in C.DATA_ROOT.iterdir() if d.is_dir() and re.search(r"_G(\d+)_s\d+$", d.name)})
    rows, lines = [], ["ABM CYTOKINE UPDATE RE-IMPLEMENTED (one implicit step per frame); R2 against the stored ABM frames",
                       "exact: masks at the target time; lagged: masks at t-1; closed: iterated from u = 0 with the ABM masks", ""]
    for grid in grids:
        runs = {C.train_run(grid): C.TRAIN_EXP, **{r: e for r, e in C.test_runs(grid).items() if "-ownscale" not in r}}
        for cyt in ("il8", "il10"):
            lines.append(f"=== {cyt.upper()} {grid}x{grid} ===")
            lines.append(f"{'run':40s}{'window':>9s}{'R2 pers.':>10s}{'R2 exact':>10s}{'skill':>9s}{'R2 lagged':>11s}{'skill':>9s}{'R2 closed':>11s}{'max dev':>9s}")
            for run, exp in runs.items():
                res = evaluate(grid, run, cyt)
                for w, r in res.items():
                    rows.append({"grid": grid, "cytokine": cyt, "run": run, "experiment": exp, "window": w, **{k: float(f"{v:.6g}") for k, v in r.items()}})
                    lines.append(f"{run:40s}{w:>9s}{r['R2_persistence']:10.5f}{r['R2_exact']:10.6f}{r['skill_exact']:9.4f}{r['R2_lagged']:11.5f}{r['skill_lagged']:9.3f}{r['R2_closed']:11.5f}{r['max_rel_dev_exact']:9.1e}")
                print(lines[-2] + "\n" + lines[-1], flush=True)
            lines.append("")
    with open(DST / "physics_step.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys())); w.writeheader(); w.writerows(rows)
    (DST / "physics_step.txt").write_text("\n".join(lines) + "\n"); print(f"\nwritten to {DST / 'physics_step.txt'}")
