import sys
import os
import re
import json
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from paths import LATTICE_DATA, PREPROCESSED

import numpy as np
import pyvista as pv

BASE_LATTICE_DIR = LATTICE_DATA
BASE_OUT_DIR     = PREPROCESSED

CYTOKINE_NAMES  = ["il8", "il1", "il6", "il10", "tnf", "tgf"]
CELL_TYPE_NAMES = ["EC", "NN", "NA", "M1", "M2"]
# In endothelial.xml, types 3, 4 and 5 are monocytes, fibroblasts and activated neutrophils;
# scripts/rev/preprocess_runs.py builds the masks from types 1, 2, 5, 8 and 9.
CELL_TYPE_IDS   = {"EC": 1, "NN": 2, "NA": 3, "M1": 4, "M2": 5}

N_TIMESTEPS  = 101
WINDOW       = 2 # look-back window for LSTM/branch inputs
GPR_MAX_GRID = 100     


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

    # raw physical data for PINN/PI-DeepONet, (101, G, G, 6)
    np.save(out_path / "Y_raw_phys.npy", raw_cyt)

    # scaling each cytokine to [-1, 1]
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

    n = N_TIMESTEPS - WINDOW # 99 pred samples

    # Y_target[i] = scaled field at timestep i+2 -> shape: (99,G,G,6)
    Y_target = scaled[WINDOW:].astype(np.float32)
    np.save(out_path / "Y_target.npy", Y_target)

    Y_masks_spatial = masks[WINDOW:].astype(np.float32) # (99, G, G, 5)
    Y_masks_pinn    = Y_masks_spatial.reshape(n, G * G, 5) # (99, G*G, 5)
    np.save(out_path / "Y_masks_spatial.npy", Y_masks_spatial)
    np.save(out_path / "Y_masks_pinn.npy",    Y_masks_pinn)

    #STA-LSTM input and DeepONet/PI-DeepONet branch input -> shape: (99, 2, G, G, 11)
    # cyto_seq[i] = scaled[i], scaled[i+1] (2 frames of 6 cytokines)
    # mask_seq[i] = masks[i],  masks[i+1] (2 frames of 5 cell types)
    # combined[i] = concat([cyto_seq[i], mask_seq[i]], axis=-1)
    
    cyto_seq = np.stack(
        [scaled[i : i + WINDOW] for i in range(n)], axis=0
    ).astype(np.float32) # (99, 2, G, G, 6)

    mask_seq = np.stack(
        [masks[i : i + WINDOW] for i in range(n)], axis=0
    ).astype(np.float32) # (99, 2, G, G, 5)

    X_combined = np.concatenate([cyto_seq, mask_seq], axis=-1) # (99, 2, G, G, 11)

    np.save(out_path / "X_lstm.npy",   X_combined) # STA-LSTM
    np.save(out_path / "X_branch.npy", X_combined) # DeepONet / PI-DeepONet

    # U-Net input: (99, G, G, 22) - 2 frames x 11 channels flattened into channel dim
    X_unet = X_combined.transpose(0, 2, 3, 1, 4).reshape(n, G, G, WINDOW * 11)
    np.save(out_path / "X_unet.npy", X_unet.astype(np.float32))

    # Trunk/collocation coordinates: (x, y, t) in [-1, 1] 
    xs = np.linspace(-1.0, 1.0, G, dtype=np.float32)
    ys = np.linspace(-1.0, 1.0, G, dtype=np.float32)
    xx, yy      = np.meshgrid(xs, ys, indexing="ij")
    coords_flat = np.stack([xx.ravel(), yy.ravel()], axis=-1) # (G*G, 2)

    t_norm = np.linspace(-1.0, 1.0, N_TIMESTEPS, dtype=np.float32)[WINDOW:] # (99,)

    X_trunk = np.zeros((n, G * G, 3), dtype=np.float32)
    for i in range(n):
        X_trunk[i, :, :2] = coords_flat
        X_trunk[i, :,  2] = t_norm[i]

    np.save(out_path / "X_trunk.npy",  X_trunk) # DeepONet/PI-DeepONet trunk
    np.save(out_path / "X_colloc.npy", X_trunk) # PINN collocation (same array)

    # Y_ic: scaled cytokine field at t=0, used to enforce u(x,y,0) = u_0 -> shape: (G, G, 6)
    Y_ic = scaled[0].astype(np.float32)
    np.save(out_path / "Y_ic.npy", Y_ic)

    # Y_bc_mask: boundary points for the Neumann no-flux condition du/dn = 0 (as in the ABM)
    bc_mask = np.zeros((G, G), dtype=np.float32)
    bc_mask[0, :]  = 1.0
    bc_mask[-1, :] = 1.0
    bc_mask[:, 0]  = 1.0
    bc_mask[:, -1] = 1.0
    np.save(out_path / "Y_bc_mask.npy", bc_mask)

    # GPR flat feature vectors -> shape: (99, G*G*11)
    if G <= GPR_MAX_GRID:
        cyto_flat = Y_target.reshape(n, -1) # (99,G*G*6)
        mask_flat = Y_masks_spatial.reshape(n, -1) # (99,G*G*5)
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
        "cytokines":   CYTOKINE_NAMES,
        "cell_types":  CELL_TYPE_NAMES,
        "scaling": {
            "method":          "adaptive_percentile_clip_linear_neg1_to_1",
            "feature_range":   [-1, 1],
            "clip_percentile": clip_pcts.tolist(),
            "min":             clip_mins.tolist(), # all 0.0
            "max":             clip_maxs.tolist(), # per-cytokine clip upper bound
            "denorm":          "u_phys = (u_scaled + 1) / 2 * max[j]",
        },
        "files": {
            "Y_raw_phys":      "(101,G,G,6) raw physical - PINN/PI-DeepONet PDE residuals",
            "Y_target":        "(99,G,G,6)  scaled targets - all models",
            "Y_masks_spatial": "(99,G,G,5)  cell-type masks - eval + loss weighting",
            "Y_masks_pinn":    "(99,G*G,5)  masks flattened - PINN/PI-DeepONet",
            "X_lstm":          "(99,2,G,G,11) cytokine+mask seq - STA-LSTM",
            "X_branch":        "(99,2,G,G,11) cytokine+mask seq - DeepONet/PI-DeepONet branch",
            "X_unet":          "(99,G,G,22)   2-frame flat channels - U-Net",
            "X_trunk":         "(99,G*G,3)  (x,y,t) coords - DeepONet/PI-DeepONet trunk",
            "X_colloc":        "(99,G*G,3)  collocation pts - PINN/PI-DeepONet",
            "Y_ic":            "(G,G,6)     initial condition - PINN",
            "Y_bc_mask":       "(G,G)       boundary mask - PINN Neumann BCs",
            "X_gpr":           "(99,G*G*11) flat features - GPR (G<=100 only)",
        },
    }
    with open(out_path / "metadata.json", "w") as f:
        json.dump(meta, f, indent=4)

    print(
        f"  {G:>4}x{G:<4} | "
        f"range [{scaled.min():+.3f}, {scaled.max():+.3f}] | "
        f"clips: { {CYTOKINE_NAMES[j]: f'{clip_pcts[j]:.1f}' for j in range(6)} } | "
        f"{gpr_note}"
    )

if __name__ == "__main__":
    print("Preprocessing VTK simulation data...\n")
    for folder in sorted(os.listdir(BASE_LATTICE_DIR)):
        if (BASE_LATTICE_DIR / folder).is_dir():
            process_grid(folder)
    print("\nDone.")