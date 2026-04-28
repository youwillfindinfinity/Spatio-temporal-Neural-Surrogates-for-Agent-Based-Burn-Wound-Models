import os
import json
from pathlib import Path

import numpy as np
import pyvista as pv

BASE_LATTICE_DIR = Path("./ICCS2026_v2/scan_iteration_0/combi_clean/LatticeData")
BASE_OUT_DIR     = Path("./preprocessed_200h")
BASE_OUT_DIR.mkdir(parents=True, exist_ok=True)

CYTOKINE_NAMES  = ["il8", "il1", "il6", "il10", "tnf", "tgf"]
CELL_TYPE_NAMES = ["EC", "NN", "NA", "M1", "M2"]
CELL_TYPE_IDS   = {"EC": 1, "NN": 2, "NA": 3, "M1": 4, "M2": 5}

N_TIMESTEPS  = 201
WINDOW       = 2


def adaptive_clip_percentile(channel: np.ndarray) -> float:
    flat = channel.flatten().astype(np.float64)
    if len(flat) < 4 or flat.std() < 1e-30:
        return 100.0
    mu   = flat.mean()
    sig  = flat.std()
    kurt = float(np.mean(((flat - mu) / sig) ** 4)) - 3.0
    if   kurt <  20: return 100.0
    elif kurt < 100: return  99.5
    elif kurt < 300: return  99.0
    elif kurt < 600: return  98.5
    else:            return  98.0


def scale_channel(channel: np.ndarray, pct: float):
    c_min = float(channel.min())
    c_max = float(np.percentile(channel, pct))
    if c_max <= c_min:
        c_max = float(channel.max())
    if c_max <= c_min:
        return np.full_like(channel, -1.0, dtype=np.float32), c_min, c_min
    scaled = (np.clip(channel, c_min, c_max) - c_min) / (c_max - c_min) * 2.0 - 1.0
    return scaled.astype(np.float32), c_min, c_max


def process():
    data_path = BASE_LATTICE_DIR

    vtk_files = sorted(
        [f for f in os.listdir(data_path) if f.endswith(".vtk")],
        key=lambda x: int("".join(filter(str.isdigit, x)) or 0)
    )[:N_TIMESTEPS]

    if not vtk_files:
        print(f"No .vtk files found in {data_path}")
        return

    print(f"Found {len(vtk_files)} VTK files")

    mesh0 = pv.read(str(data_path / vtk_files[0]))
    n_pts = mesh0.n_points
    G = int(round(n_pts ** 0.5))
    assert G * G == n_pts, f"Grid not square: {n_pts} points"
    print(f"Grid: {G}x{G}")

    out_path = BASE_OUT_DIR / f"{G}x{G}"
    out_path.mkdir(parents=True, exist_ok=True)

    T = len(vtk_files)
    raw_cyt = np.zeros((T, G, G, 6), dtype=np.float32)
    masks   = np.zeros((T, G, G, 5), dtype=np.float32)

    for i, fname in enumerate(vtk_files):
        if i % 20 == 0:
            print(f"  Loading {i}/{T}...")
        mesh = pv.read(str(data_path / fname))
        for j, ck in enumerate(CYTOKINE_NAMES):
            raw_cyt[i, :, :, j] = mesh.point_data[ck].reshape(G, G, order="F")
        if "CellType" in mesh.point_data:
            ct = mesh.point_data["CellType"].reshape(G, G, order="F")
            for j, (_, cid) in enumerate(CELL_TYPE_IDS.items()):
                masks[i, :, :, j] = (ct == cid).astype(np.float32)

    np.save(out_path / "Y_raw_phys.npy", raw_cyt)

    scaled    = np.zeros_like(raw_cyt)
    clip_mins = np.zeros(6, dtype=np.float64)
    clip_maxs = np.zeros(6, dtype=np.float64)
    clip_pcts = np.zeros(6, dtype=np.float64)

    for j in range(6):
        pct                = adaptive_clip_percentile(raw_cyt[:, :, :, j])
        s, c_min, c_max    = scale_channel(raw_cyt[:, :, :, j], pct)
        scaled[:, :, :, j] = s
        clip_mins[j]       = c_min
        clip_maxs[j]       = c_max
        clip_pcts[j]       = pct

    n = T - WINDOW

    Y_target = scaled[WINDOW:].astype(np.float32)
    np.save(out_path / "Y_target.npy", Y_target)

    Y_masks_spatial = masks[WINDOW:].astype(np.float32)
    Y_masks_pinn    = Y_masks_spatial.reshape(n, G * G, 5)
    np.save(out_path / "Y_masks_spatial.npy", Y_masks_spatial)
    np.save(out_path / "Y_masks_pinn.npy",    Y_masks_pinn)

    cyto_seq = np.stack(
        [scaled[i : i + WINDOW] for i in range(n)], axis=0
    ).astype(np.float32)

    mask_seq = np.stack(
        [masks[i : i + WINDOW] for i in range(n)], axis=0
    ).astype(np.float32)

    X_combined = np.concatenate([cyto_seq, mask_seq], axis=-1)

    np.save(out_path / "X_lstm.npy",   X_combined)
    np.save(out_path / "X_branch.npy", X_combined)

    xs = np.linspace(-1.0, 1.0, G, dtype=np.float32)
    ys = np.linspace(-1.0, 1.0, G, dtype=np.float32)
    xx, yy      = np.meshgrid(xs, ys, indexing="ij")
    coords_flat = np.stack([xx.ravel(), yy.ravel()], axis=-1)

    t_norm = np.linspace(-1.0, 1.0, T, dtype=np.float32)[WINDOW:]

    X_trunk = np.zeros((n, G * G, 3), dtype=np.float32)
    for i in range(n):
        X_trunk[i, :, :2] = coords_flat
        X_trunk[i, :,  2] = t_norm[i]

    np.save(out_path / "X_trunk.npy",  X_trunk)
    np.save(out_path / "X_colloc.npy", X_trunk)

    Y_ic = scaled[0].astype(np.float32)
    np.save(out_path / "Y_ic.npy", Y_ic)

    bc_mask = np.zeros((G, G), dtype=np.float32)
    bc_mask[0, :]  = 1.0
    bc_mask[-1, :] = 1.0
    bc_mask[:, 0]  = 1.0
    bc_mask[:, -1] = 1.0
    np.save(out_path / "Y_bc_mask.npy", bc_mask)

    X_unet = X_combined.transpose(0, 2, 3, 1, 4)
    X_unet = X_unet.reshape(n, G, G, WINDOW * 11)
    np.save(out_path / "X_unet.npy", X_unet.astype(np.float32))

    meta = {
        "grid":        G,
        "n_timesteps": T,
        "n_samples":   n,
        "window":      WINDOW,
        "cytokines":   CYTOKINE_NAMES,
        "cell_types":  CELL_TYPE_NAMES,
        "dataset":     "200h_iteration_0",
        "scaling": {
            "method":          "adaptive_percentile_clip_linear_neg1_to_1",
            "feature_range":   [-1, 1],
            "clip_percentile": clip_pcts.tolist(),
            "min":             clip_mins.tolist(),
            "max":             clip_maxs.tolist(),
            "denorm":          "u_phys = (u_scaled + 1) / 2 * max[j]",
        },
        "splits": {
            "train": "0:140",
            "val":   "140:160",
            "test_near": "160:180",
            "test_far":  "180:199",
        },
        "files": {
            "Y_raw_phys":      f"({T},G,G,6) raw physical",
            "Y_target":        f"({n},G,G,6) scaled targets",
            "Y_masks_spatial": f"({n},G,G,5) cell-type masks",
            "Y_masks_pinn":    f"({n},G*G,5) masks flattened",
            "X_lstm":          f"({n},2,G,G,11) STA-LSTM",
            "X_branch":        f"({n},2,G,G,11) DeepONet branch",
            "X_trunk":         f"({n},G*G,3) DeepONet trunk",
            "X_colloc":        f"({n},G*G,3) PINN collocation",
            "Y_ic":            "(G,G,6) initial condition",
            "Y_bc_mask":       "(G,G) boundary mask",
            "X_unet":          f"({n},G,G,{WINDOW*11}) U-Net",
        },
    }
    with open(out_path / "metadata.json", "w") as f:
        json.dump(meta, f, indent=4)

    print(f"\n  {G}x{G} | {T} timesteps | {n} samples")
    print(f"  range [{scaled.min():+.3f}, {scaled.max():+.3f}]")
    print(f"  clips: { {CYTOKINE_NAMES[j]: f'{clip_pcts[j]:.1f}' for j in range(6)} }")
    print(f"  Saved to {out_path}")


if __name__ == "__main__":
    print("Preprocessing 200h VTK data...\n")
    process()
    print("\nDone.")