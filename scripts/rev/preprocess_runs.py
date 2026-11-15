"""Preprocess the revision ABM runs (one folder per run with LatticeData/Step_*.vtk, 101 hourly frames).

For every grid, the training run (default E3_il10_from_m2_G<grid>_s1) defines the scaling:
kurtosis-adaptive percentile clip per cytokine, then linear map to [-1, 1], exactly as in the
original preprocessing. All other runs of that grid are scaled with the SAME constants, so models
trained on the training run can be applied to them with frozen weights.

Cell-type masks use the CompuCell3D type IDs of the model: endothelial = 1, resting neutrophil = 2,
activated neutrophil = 5, M1 macrophage = 8, M2 macrophage = 9.

Runs of an experiment whose concentration scale differs from the training run by more than 25 %
(e.g. a different number of secreting cells) are written a second time as
<experiment>-ownscale_G<grid>_s<seed>, scaled with the constants of that experiment's first seed,
so that transfer can be evaluated both in the training units and in the experiment's own units.
Tiny negative solver round-off values in the raw fields are set to zero.

Output per run (<out>/<run_id>/): X_unet (99,G,G,22), Y_target (99,G,G,6), Y_masks_spatial (99,G,G,5),
Y_raw_phys (101,G,G,6), X_trunk (99,G*G,3), metadata.json; for the training run also Y_masks_pinn, Y_ic.
"""
import argparse, json, os, re, sys
from pathlib import Path
import numpy as np
import pyvista as pv

CYTOKINE_NAMES = ["il8", "il1", "il6", "il10", "tnf", "tgf"]
CELL_TYPE_IDS = {"EC": 1, "NN": 2, "NA": 5, "M1": 8, "M2": 9}
N_T, WINDOW = 101, 2


def clip_percentile(ch):
    flat = ch.ravel().astype(np.float64)
    if flat.std() < 1e-30: return 100.0
    k = float(np.mean(((flat - flat.mean()) / flat.std()) ** 4)) - 3.0
    return 100.0 if k < 20 else 99.5 if k < 100 else 99.0 if k < 300 else 98.5 if k < 600 else 98.0


def read_run(path, G):
    files = sorted([f for f in os.listdir(path / "LatticeData") if f.endswith(".vtk")], key=lambda x: int("".join(filter(str.isdigit, x)) or 0))
    if len(files) != N_T:
        raise RuntimeError(f"{path.name}: expected {N_T} VTK frames, found {len(files)}")
    raw = np.zeros((N_T, G, G, 6), np.float32); masks = np.zeros((N_T, G, G, 5), np.float32); types = {}
    for i, f in enumerate(files):
        mesh = pv.read(str(path / "LatticeData" / f))
        for j, ck in enumerate(CYTOKINE_NAMES):
            raw[i, ..., j] = np.asarray(mesh.point_data[ck]).reshape(G, G, order="F")
        ct = np.asarray(mesh.point_data["CellType"]).reshape(G, G, order="F")
        for j, cid in enumerate(CELL_TYPE_IDS.values()):
            masks[i, ..., j] = ct == cid
        if i == N_T - 1:
            u, n = np.unique(ct.astype(int), return_counts=True); types = {int(a): int(b) for a, b in zip(u, n)}
    return np.maximum(raw, 0.0), masks, types


def scaling_of(raw):
    pcts = [clip_percentile(raw[..., j]) for j in range(6)]
    cmax = [float(np.percentile(raw[..., j], pcts[j])) for j in range(6)]
    cmax = [m if m > 0 else float(raw[..., j].max()) for j, m in enumerate(cmax)]
    return [0.0] * 6, cmax, pcts


def write_run(out, run, G, raw, masks, cmin, cmax, pcts, source, types, prov, is_train):
    d = out / run; d.mkdir(parents=True, exist_ok=True)
    scaled = np.empty_like(raw)
    for j in range(6):
        scaled[..., j] = (np.clip(raw[..., j], cmin[j], cmax[j]) - cmin[j]) / (cmax[j] - cmin[j]) * 2.0 - 1.0 if cmax[j] > cmin[j] else -1.0
    n = N_T - WINDOW
    seq = np.concatenate([np.stack([scaled[i:i + WINDOW] for i in range(n)]), np.stack([masks[i:i + WINDOW] for i in range(n)])], -1)
    np.save(d / "X_unet.npy", seq.transpose(0, 2, 3, 1, 4).reshape(n, G, G, WINDOW * 11).astype(np.float32))
    np.save(d / "Y_target.npy", scaled[WINDOW:].astype(np.float32))
    np.save(d / "Y_masks_spatial.npy", masks[WINDOW:].astype(np.float32))
    np.save(d / "Y_raw_phys.npy", raw)
    xs = np.linspace(-1.0, 1.0, G, dtype=np.float32); xx, yy = np.meshgrid(xs, xs, indexing="ij")
    Xt = np.zeros((n, G * G, 3), np.float32); Xt[:, :, 0] = xx.ravel(); Xt[:, :, 1] = yy.ravel()
    Xt[:, :, 2] = np.linspace(-1.0, 1.0, N_T, dtype=np.float32)[WINDOW:, None]
    np.save(d / "X_trunk.npy", Xt)
    if is_train:
        np.save(d / "Y_masks_pinn.npy", masks[WINDOW:].reshape(n, G * G, 5).astype(np.float32)); np.save(d / "Y_ic.npy", scaled[0].astype(np.float32))
    clipped = [float(np.mean(raw[..., j] > cmax[j])) for j in range(6)]
    meta = {"run": run, "grid": G, "n_timesteps": N_T, "n_samples": n, "window": WINDOW, "cytokines": CYTOKINE_NAMES,
            "cell_types": list(CELL_TYPE_IDS), "cell_type_ids": CELL_TYPE_IDS, "cell_type_pixels_last_frame": types,
            "scaling": {"method": "adaptive_percentile_clip_linear_neg1_to_1", "source_run": source, "clip_percentile": pcts,
                        "min": [float(x) for x in cmin], "max": [float(x) for x in cmax], "fraction_of_values_above_max": clipped,
                        "raw_max": [float(raw[..., j].max()) for j in range(6)], "denorm": "u_phys = (u_scaled + 1) / 2 * max[j]"},
            "abm_provenance": prov}
    json.dump(meta, open(d / "metadata.json", "w"), indent=2)
    print(f"  {run:42s} clip pct {dict(zip(CYTOKINE_NAMES, pcts))}  above c_max: il8 {clipped[0]:.4f} il10 {clipped[3]:.4f}  cell types (last frame): {types}", flush=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--runs-root", required=True); ap.add_argument("--out", required=True)
    ap.add_argument("--train-exp", default="E3_il10_from_m2"); ap.add_argument("--train-seed", type=int, default=1)
    a = ap.parse_args()
    root, out = Path(a.runs_root).expanduser(), Path(a.out).expanduser(); out.mkdir(parents=True, exist_ok=True)
    runs = sorted(d.name for d in root.iterdir() if (d / "LatticeData").is_dir())
    grids = sorted({int(re.search(r"_G(\d+)_s\d+$", r).group(1)) for r in runs})
    print(f"{len(runs)} runs, grids {grids}", flush=True)
    for G in grids:
        train = f"{a.train_exp}_G{G}_s{a.train_seed}"
        if train not in runs: sys.exit(f"training run {train} not found in {root}")
        order = [train] + [r for r in runs if re.search(rf"_G{G}_s\d+$", r) and r != train]
        cmin = cmax = pcts = None
        own = {}                                   # experiment -> its own scaling, if it differs from the training scale
        for r in order:
            exp = re.sub(rf"_G{G}_s\d+$", "", r)
            done = (out / r / "metadata.json").exists()
            if done and r != train and exp not in own and not any(x.startswith(exp + "_G") for x in order[order.index(r) + 1:]):
                print(f"  {r}: already done, skipped", flush=True); continue
            raw, masks, types = read_run(root / r, G)
            if r == train:
                cmin, cmax, pcts = scaling_of(raw)
            pf = root / r / "abm_provenance.json"
            prov = json.load(open(pf)) if pf.exists() else None
            if not done or r == train:
                write_run(out, r, G, raw, masks, cmin, cmax, pcts, train, types, prov, r == train)
            if exp != a.train_exp:
                if exp not in own:
                    omin, omax, opct = scaling_of(raw)
                    ratio = max(max(omax[j] / cmax[j], cmax[j] / omax[j]) for j in (0, 3))
                    own[exp] = (omin, omax, opct, r) if ratio > 1.25 else None
                    print(f"    {exp}: c_max relative to training run  il8 x{omax[0] / cmax[0]:.2f}  il10 x{omax[3] / cmax[3]:.2f}"
                          + ("  -> additional own-scale copies" if own[exp] else ""), flush=True)
                if own[exp]:
                    omin, omax, opct, src = own[exp]
                    write_run(out, r.replace(exp, exp + "-ownscale"), G, raw, masks, omin, omax, opct, src, types, prov, False)
    print("Done.")


if __name__ == "__main__":
    main()
