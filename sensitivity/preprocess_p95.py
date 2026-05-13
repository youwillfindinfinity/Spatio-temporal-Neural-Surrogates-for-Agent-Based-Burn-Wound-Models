import os
import re
import json
from pathlib import Path

import numpy as np
import pyvista as pv

BASE_LATTICE_DIR = Path("./LatticeData")
BASE_OUT_DIR     = Path("./preprocessed_p95")
BASE_OUT_DIR.mkdir(parents=True, exist_ok=True)

CYTOKINE_NAMES  = ["il8", "il1", "il6", "il10", "tnf", "tgf"]
CELL_TYPE_NAMES = ["EC", "NN", "NA", "M1", "M2"]
CELL_TYPE_IDS   = {"EC": 1, "NN": 2, "NA": 3, "M1": 4, "M2": 5}

N_TIMESTEPS  = 101
WINDOW       = 2
GPR_MAX_GRID = 100


def fixed_p95_percentile(channel: np.ndarray) -> float:
    """fixed 95th percentile for all cytokines."""
    return 95.0


def scale_channel(channel: np.ndarray, pct: float):
    c_min = float(channel.min())
    c_max = float(np.percentile(channel, pct))

    if c_max <= c_min:
        c_max = float(channel.max())
    if c_max <= c_min:
        return np.full_like(channel, -1.0, dtype=np.float32), c_min, c_min

    scaled = (np.clip(channel, c_min, c_max) - c_min) / (c_max - c_min) * 2.0 - 1.0
    return scaled.astype(np.float32), c_min, c_max


def extract_grid_size(name: str):
    m = re.search(r'(\d+)x(\d+)', name)
    if m:
        return int(m.group(1))
    m = re.search(r'(\d+)', name)
    return int(m.group(1)) if m else None


def process_grid(folder_name: str):
    data_path = BASE_LATTICE_DIR / folder_name
    G         = extract_grid_size(folder_name)

    if G is None:
        print(f"  Skipping '{folder_name}': cannot parse grid size.")
        return

    out_path = BASE_OUT_DIR / f"{G}x{G}"
    out_path.mkdir(parents=True, exist_ok=True)

    vtk_files = sorted(
        [f for f in os.listdir(data_path) if f.endswith(".vtk")],
        key=lambda x: int("".join(filter(str.isdigit, x)) or 0)
    )[:N_TIMESTEPS]

    if not vtk_files:
        print(f"  Skipping '{folder_name}': no .vtk files found.")
        return

    raw_cyt = np.zeros((N_TIMESTEPS, G, G, 6), dtype=np.float32)
    masks   = np.zeros((N_TIMESTEPS, G, G, 5), dtype=np.float32)

    for i, fname in enumerate(vtk_files):
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
        pct                = fixed_p95_percentile(raw_cyt[:, :, :, j])
        s, c_min, c_max    = scale_channel(raw_cyt[:, :, :, j], pct)
        scaled[:, :, :, j] = s
        clip_mins[j]       = c_min
        clip_maxs[j]       = c_max
        clip_pcts[j]       = pct

    n = N_TIMESTEPS - WINDOW

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

    X_unet = X_combined.transpose(0, 2, 3, 1, 4).reshape(n, G, G, 22).astype(np.float32)
    np.save(out_path / "X_unet.npy", X_unet)

    xs = np.linspace(-1.0, 1.0, G, dtype=np.float32)
    ys = np.linspace(-1.0, 1.0, G, dtype=np.float32)
    xx, yy      = np.meshgrid(xs, ys, indexing="ij")
    coords_flat = np.stack([xx.ravel(), yy.ravel()], axis=-1)

    t_norm = np.linspace(-1.0, 1.0, N_TIMESTEPS, dtype=np.float32)[WINDOW:]

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

    if G <= GPR_MAX_GRID:
        cyto_flat = Y_target.reshape(n, -1)
        mask_flat = Y_masks_spatial.reshape(n, -1)
        X_gpr     = np.concatenate([cyto_flat, mask_flat], axis=-1).astype(np.float32)
        np.save(out_path / "X_gpr.npy", X_gpr)
        gpr_note = f"X_gpr {X_gpr.shape}"
    else:
        gpr_note = f"X_gpr skipped (G={G} > {GPR_MAX_GRID})"

    meta = {
        "grid":        G,
        "n_timesteps": N_TIMESTEPS,
        "n_samples":   n,
        "window":      WINDOW,
        "strategy":    "fixed_p95",
        "cytokines":   CYTOKINE_NAMES,
        "cell_types":  CELL_TYPE_NAMES,
        "scaling": {
            "method":          "fixed_p95_clip_linear_neg1_to_1",
            "feature_range":   [-1, 1],
            "clip_percentile": clip_pcts.tolist(),
            "min":             clip_mins.tolist(),
            "max":             clip_maxs.tolist(),
            "denorm":          "u_phys = (u_scaled + 1) / 2 * max[j]",
        },
        "files": {
            "Y_raw_phys":      "(101,G,G,6) raw physical",
            "Y_target":        "(99,G,G,6)  scaled targets",
            "Y_masks_spatial": "(99,G,G,5)  cell-type masks",
            "Y_masks_pinn":    "(99,G*G,5)  flattened masks",
            "X_lstm":          "(99,2,G,G,11) STA-LSTM",
            "X_branch":        "(99,2,G,G,11) DeepONet/PI-DeepONet branch",
            "X_unet":          "(99,G,G,22)   U-Net (flattened look-back)",
            "X_trunk":         "(99,G*G,3)    trunk",
            "X_colloc":        "(99,G*G,3)    PINN collocation",
            "Y_ic":            "(G,G,6)       initial condition",
            "Y_bc_mask":       "(G,G)         boundary mask",
            "X_gpr":           "(99,G*G*11)   GPR (G<=100)",
        },
    }
    with open(out_path / "metadata.json", "w") as f:
        json.dump(meta, f, indent=4)

    print(
        f"  [p95] {G:>4}x{G:<4} | "
        f"range [{scaled.min():+.3f}, {scaled.max():+.3f}] | "
        f"clips: { {CYTOKINE_NAMES[j]: f'{clip_pcts[j]:.1f}' for j in range(6)} } | "
        f"{gpr_note}"
    )


if __name__ == "__main__":
    print("Preprocessing VTK simulation data (FIXED P95)...\n")
    for folder in sorted(os.listdir(BASE_LATTICE_DIR)):
        if (BASE_LATTICE_DIR / folder).is_dir():
            process_grid(folder)
    print("\nDone.")