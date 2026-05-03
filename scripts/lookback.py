"""
U-Net look-back window sensitivity sweep.
"""

import os, json, argparse, random, time, re
from pathlib import Path

import numpy as np
import pyvista as pv
import tensorflow as tf
import optuna
from sklearn.metrics import r2_score
from scipy.stats import pearsonr
from skimage.metrics import structural_similarity as ssim

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"
optuna.logging.set_verbosity(optuna.logging.WARNING)

N_TRIALS    = 20
TUNE_EPOCHS = 20
FULL_EPOCHS = 200
N_TIMESTEPS = 101

CYTOKINE_NAMES  = ["il8", "il1", "il6", "il10", "tnf", "tgf"]
CELL_TYPE_NAMES = ["EC", "NN", "NA", "M1", "M2"]
CELL_TYPE_IDS   = {"EC": 1, "NN": 2, "NA": 3, "M1": 4, "M2": 5}

BASE_LATTICE_DIR = Path("./LatticeData")
OUT_DIR = Path("./sensitivity_lookback")
OUT_DIR.mkdir(parents=True, exist_ok=True)


def set_seed(seed: int):
    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed); np.random.seed(seed); tf.random.set_seed(seed)


# Preprocessing (matches preprocessing.py exactly, with window as parameter)
def adaptive_clip_percentile(channel: np.ndarray) -> float:
    flat = channel.flatten().astype(np.float64)
    if len(flat) < 4 or flat.std() < 1e-30:
        return 100.0
    mu = flat.mean(); sig = flat.std()
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


def find_lattice_folder(grid: int) -> Path:
    candidates = []
    for folder in sorted(os.listdir(BASE_LATTICE_DIR)):
        path = BASE_LATTICE_DIR / folder
        if not path.is_dir():
            continue
        m = re.search(r'(\d+)x(\d+)', folder)
        if m and int(m.group(1)) == grid:
            candidates.append(path)
        else:
            m2 = re.search(r'(\d+)', folder)
            if m2 and int(m2.group(1)) == grid:
                candidates.append(path)
    if not candidates:
        raise FileNotFoundError(
            f"No LatticeData folder found for grid={grid}. "
            f"Looked in {BASE_LATTICE_DIR}"
        )
    return candidates[0]


def preprocess_for_window(grid: int, window: int) -> Path:
    out_path = Path(f"./preprocessed_w{window}/{grid}x{grid}")
    out_path.mkdir(parents=True, exist_ok=True)

    # Skip if already preprocessed
    required_files = ["X_unet.npy", "Y_target.npy", "Y_masks_spatial.npy", "metadata.json"]
    if all((out_path / f).exists() for f in required_files):
        with open(out_path / "metadata.json") as f:
            meta = json.load(f)
        if meta.get("window") == window:
            print(f"  [preprocess] Reusing existing {out_path}")
            return out_path

    print(f"  [preprocess] Building from VTK for window={window}...")
    lattice_path = find_lattice_folder(grid)

    vtk_files = sorted(
        [f for f in os.listdir(lattice_path) if f.endswith(".vtk")],
        key=lambda x: int("".join(filter(str.isdigit, x)) or 0),
    )[:N_TIMESTEPS]
    if not vtk_files:
        raise RuntimeError(f"No VTK files in {lattice_path}")

    raw_cyt = np.zeros((N_TIMESTEPS, grid, grid, 6), dtype=np.float32)
    masks   = np.zeros((N_TIMESTEPS, grid, grid, 5), dtype=np.float32)
    for i, fname in enumerate(vtk_files):
        mesh = pv.read(str(lattice_path / fname))
        for j, ck in enumerate(CYTOKINE_NAMES):
            raw_cyt[i, :, :, j] = mesh.point_data[ck].reshape(grid, grid, order="F")
        if "CellType" in mesh.point_data:
            ct = mesh.point_data["CellType"].reshape(grid, grid, order="F")
            for j, (_, cid) in enumerate(CELL_TYPE_IDS.items()):
                masks[i, :, :, j] = (ct == cid).astype(np.float32)

    # Per-channel clipping + rescale
    scaled = np.zeros_like(raw_cyt)
    clip_mins = np.zeros(6); clip_maxs = np.zeros(6); clip_pcts = np.zeros(6)
    for j in range(6):
        pct = adaptive_clip_percentile(raw_cyt[:, :, :, j])
        s, c_min, c_max = scale_channel(raw_cyt[:, :, :, j], pct)
        scaled[:, :, :, j] = s
        clip_mins[j] = c_min; clip_maxs[j] = c_max; clip_pcts[j] = pct

    n = N_TIMESTEPS - window
    Y_target = scaled[window:].astype(np.float32)
    Y_masks_spatial = masks[window:].astype(np.float32)

    # Stack `window` consecutive frames as input
    cyto_seq = np.stack([scaled[i:i + window] for i in range(n)], axis=0).astype(np.float32)
    mask_seq = np.stack([masks[i:i + window]  for i in range(n)], axis=0).astype(np.float32)
    X_combined = np.concatenate([cyto_seq, mask_seq], axis=-1)  # (n, w, G, G, 11)

    X_unet = X_combined.transpose(0, 2, 3, 1, 4).reshape(n, grid, grid, window * 11)

    np.save(out_path / "Y_target.npy",        Y_target)
    np.save(out_path / "Y_masks_spatial.npy", Y_masks_spatial)
    np.save(out_path / "X_unet.npy",          X_unet.astype(np.float32))
    np.save(out_path / "Y_raw_phys.npy",      raw_cyt)

    meta = {
        "grid": grid, "n_timesteps": N_TIMESTEPS, "n_samples": n, "window": window,
        "cytokines": CYTOKINE_NAMES, "cell_types": CELL_TYPE_NAMES,
        "scaling": {
            "method": "adaptive_percentile_clip_linear_neg1_to_1",
            "feature_range": [-1, 1],
            "clip_percentile": clip_pcts.tolist(),
            "min": clip_mins.tolist(), "max": clip_maxs.tolist(),
            "denorm": "u_phys = (u_scaled + 1) / 2 * max[j]",
        },
    }
    with open(out_path / "metadata.json", "w") as f:
        json.dump(meta, f, indent=4)
    print(f"  [preprocess] Saved {out_path} ({n} samples, {window * 11} channels)")
    return out_path


def _conv_block(x, filters, kernel_size=3):
    x = tf.keras.layers.Conv2D(filters, kernel_size, padding="same", activation="relu")(x)
    x = tf.keras.layers.BatchNormalization()(x)
    x = tf.keras.layers.Conv2D(filters, kernel_size, padding="same", activation="relu")(x)
    x = tf.keras.layers.BatchNormalization()(x)
    return x

def _encoder_block(x, filters):
    skip = _conv_block(x, filters)
    pool = tf.keras.layers.MaxPooling2D(pool_size=2, padding="same")(skip)
    return skip, pool

def _decoder_block(x, skip, filters):
    x = tf.keras.layers.Conv2DTranspose(filters, 2, strides=2, padding="same", activation="relu")(x)
    if x.shape[1] != skip.shape[1] or x.shape[2] != skip.shape[2]:
        x = tf.keras.layers.Resizing(skip.shape[1], skip.shape[2])(x)
    x = tf.keras.layers.Concatenate()([x, skip])
    x = _conv_block(x, filters)
    return x

def build_unet(grid_size: int, in_channels: int,
               base_filters: int = 32, depth: int = 4, dropout: float = 0.0):
    inputs = tf.keras.Input(shape=(grid_size, grid_size, in_channels))
    skips = []; x = inputs
    for i in range(depth):
        skip, x = _encoder_block(x, base_filters * (2 ** i))
        skips.append(skip)
    x = _conv_block(x, base_filters * (2 ** depth))
    if dropout > 0:
        x = tf.keras.layers.Dropout(dropout)(x)
    for i in reversed(range(depth)):
        x = _decoder_block(x, skips[i], base_filters * (2 ** i))
    outputs = tf.keras.layers.Conv2D(1, 1, padding="same", activation="linear")(x)
    return tf.keras.Model(inputs, outputs, name="unet")


def _fisher_z(r):
    r = np.clip(r, -0.9999, 0.9999)
    return 0.5 * np.log((1.0 + r) / (1.0 - r))

def _inv_fisher_z(z):
    return float(np.tanh(z))

def calculate_metrics(y_true, y_pred, masks, clip_max):
    min_t = min(y_true.shape[0], y_pred.shape[0], masks.shape[0])
    y_t = y_true[:min_t]; y_p = np.maximum(y_pred[:min_t], 0.0)
    m_s = np.max(masks[:min_t], axis=-1, keepdims=True)
    sq_diff = np.square(y_t - y_p)
    rmse = float(np.sqrt(np.sum(sq_diff * m_s) / (np.sum(m_s) + 1e-12)))
    r2 = float(r2_score(y_t.flatten(), y_p.flatten()))
    dice_thr = 0.05 * clip_max if clip_max > 0 else 1e-9
    dices = []; z_corrs = []; ssims_v = []
    fixed_dr = float(clip_max) if clip_max > 0 else 1.0
    for t in range(min_t):
        gt = y_t[t, :, :, 0]; pr = y_p[t, :, :, 0]
        gb = (gt > dice_thr).astype(float); pb = (pr > dice_thr).astype(float)
        if np.sum(gb) + np.sum(pb) > 0:
            dices.append((2.0 * np.sum(gb * pb)) / (np.sum(gb) + np.sum(pb) + 1e-12))
        if np.std(gt) > 1e-12 and np.std(pr) > 1e-12:
            r_val = float(pearsonr(gt.flatten(), pr.flatten())[0])
            if np.isfinite(r_val):
                z_corrs.append(_fisher_z(r_val))
        dr = float(np.max(gt) - np.min(gt))
        if dr > 1e-12:
            ssims_v.append(float(ssim(gt, pr, data_range=fixed_dr)))
    return {
        "Global_R2": r2,
        "Masked_RMSE": rmse,
        "Avg_Dice": float(np.mean(dices)) if dices else 0.0,
        "Spatial_Correlation": _inv_fisher_z(float(np.mean(z_corrs))) if z_corrs else 0.0,
        "SSIM": float(np.mean(ssims_v)) if ssims_v else 0.0,
    }


def denormalize(scaled, clip_max):
    return (np.asarray(scaled, dtype=np.float64) + 1.0) / 2.0 * clip_max

def make_objective(X_train, Y_train, X_val, Y_val, grid_size, in_channels, seed):
    def objective(trial):
        set_seed(seed)
        tf.keras.backend.clear_session()
        base_filters = trial.suggest_categorical("base_filters", [16, 32, 64])
        depth        = trial.suggest_categorical("depth",        [3, 4])
        dropout      = trial.suggest_float("dropout", 0.0, 0.3)
        lr           = trial.suggest_float("learning_rate", 1e-5, 1e-3, log=True)
        batch_size   = trial.suggest_categorical("batch_size", [4, 8, 16])
        model = build_unet(grid_size, in_channels, base_filters, depth, dropout)
        model.compile(optimizer=tf.keras.optimizers.Adam(lr), loss="mse")
        history = model.fit(
            X_train, Y_train, validation_data=(X_val, Y_val),
            epochs=TUNE_EPOCHS, batch_size=batch_size, verbose=0,
            callbacks=[tf.keras.callbacks.EarlyStopping(
                monitor="val_loss", patience=5, restore_best_weights=True)],
        )
        return float(min(history.history["val_loss"]))
    return objective


def run_one_window(grid: int, w: int, cytokine: str, seed: int = 42):
    set_seed(seed)
    cyt_idx = CYTOKINE_NAMES.index(cytokine.lower())

    print(f"\n=== {cytokine.upper()}, grid={grid}, w={w} ===")
    data_path = preprocess_for_window(grid, w)

    X = np.load(data_path / "X_unet.npy").astype(np.float32)
    Y = np.load(data_path / "Y_target.npy").astype(np.float32)[..., cyt_idx:cyt_idx + 1]
    M = np.load(data_path / "Y_masks_spatial.npy").astype(np.float32)
    with open(data_path / "metadata.json") as f:
        meta = json.load(f)
    clip_max = float(meta["scaling"]["max"][cyt_idx])
    in_channels = X.shape[-1]
    N = X.shape[0]
    print(f"  Loaded: X={X.shape}, Y={Y.shape}, in_channels={in_channels}")

    # Splits: first 70 train, next 10 val, 20 test 
    n_train, n_val = 70, 10
    if N < n_train + n_val:
        raise RuntimeError(f"Not enough samples for w={w}: N={N} < {n_train + n_val}")
    X_train, Y_train = X[:n_train], Y[:n_train]
    X_val,   Y_val   = X[n_train:n_train + n_val], Y[n_train:n_train + n_val]

    # Optuna search (per window, since input dimensionality changes)
    print(f"  Optuna: {N_TRIALS} trials × {TUNE_EPOCHS} epochs...")
    study = optuna.create_study(
        direction="minimize",
        sampler=optuna.samplers.TPESampler(seed=seed),
        pruner=optuna.pruners.MedianPruner(n_warmup_steps=5),
    )
    study.optimize(
        make_objective(X_train, Y_train, X_val, Y_val, grid, in_channels, seed),
        n_trials=N_TRIALS, show_progress_bar=False,
    )
    best = study.best_params
    optuna_val = float(study.best_value)
    print(f"  Best: {best}  |  val_loss = {optuna_val:.6f}")

    # Final training
    tf.keras.backend.clear_session(); set_seed(seed)
    t_start = time.time()
    model = build_unet(grid, in_channels, best["base_filters"], best["depth"], best["dropout"])
    model.compile(optimizer=tf.keras.optimizers.Adam(best["learning_rate"]), loss="mse")
    print(f"  Final training (max {FULL_EPOCHS} epochs)...")
    model.fit(
        X_train, Y_train, validation_data=(X_val, Y_val),
        epochs=FULL_EPOCHS, batch_size=best["batch_size"], verbose=0,
        callbacks=[
            tf.keras.callbacks.EarlyStopping(
                monitor="val_loss", patience=20, restore_best_weights=True),
            tf.keras.callbacks.ReduceLROnPlateau(
                monitor="val_loss", factor=0.5, patience=10, min_lr=1e-6),
        ],
    )

    # Predict full test set
    Y_p_scaled = model.predict(X, batch_size=2, verbose=0)
    Y_p_phys   = denormalize(Y_p_scaled, clip_max)
    Y_a_phys   = denormalize(Y, clip_max)

    # Test windows: original biological frames t=82..91 (near), 92..100 (far).
    # Sample i corresponds to original frame index t = w + i, so frame t -> sample i = t - w.
    near_lo = max(0, 82 - w);     near_hi = min(N, 91 - w + 1)
    far_lo  = max(0, 92 - w);     far_hi  = min(N, 100 - w + 1)
    train_elapsed = time.time() - t_start
    n_params = int(model.count_params())

    near = calculate_metrics(Y_a_phys[near_lo:near_hi], Y_p_phys[near_lo:near_hi],
                              M[near_lo:near_hi], clip_max)
    far  = calculate_metrics(Y_a_phys[far_lo:far_hi],   Y_p_phys[far_lo:far_hi],
                              M[far_lo:far_hi],  clip_max)

    result = {
        "cytokine": cytokine, "grid": grid, "window": w, "seed": seed,
        "in_channels": in_channels, "n_params": n_params,
        "train_time_seconds": round(train_elapsed, 2),
        "best_params": best, "optuna_best_val_loss": optuna_val,
        "near": near, "far": far,
    }
    out_json = OUT_DIR / f"res_{cytokine}_{grid}_w{w}.json"
    with open(out_json, "w") as f:
        json.dump(result, f, indent=4)
    print(f"  Near R2={near['Global_R2']:.4f}  Far R2={far['Global_R2']:.4f}  "
          f"params={n_params:,}  train={train_elapsed:.0f}s")
    return result


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--grid", type=int, default=100)
    ap.add_argument("--windows", type=int, nargs="+", default=[1, 2, 3, 4, 5])
    ap.add_argument("--cytokines", nargs="+", default=["il8", "il10"])
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    rows = []
    for cyt in args.cytokines:
        for w in args.windows:
            try:
                rows.append(run_one_window(args.grid, w, cyt, seed=args.seed))
            except Exception as e:
                print(f"  FAILED [{cyt}, w={w}]: {e}")
                import traceback; traceback.print_exc()

    csv_path = OUT_DIR / "results.csv"
    with open(csv_path, "w") as f:
        f.write("cytokine,grid,window,n_params,train_time_s,"
                "near_R2,near_SSIM,near_Dice,near_Corr,near_RMSE,"
                "far_R2,far_SSIM,far_Dice,far_Corr,far_RMSE\n")
        for r in rows:
            n, fr = r["near"], r["far"]
            f.write(f"{r['cytokine']},{r['grid']},{r['window']},{r['n_params']},"
                    f"{r['train_time_seconds']},"
                    f"{n['Global_R2']:.4f},{n['SSIM']:.4f},{n['Avg_Dice']:.4f},"
                    f"{n['Spatial_Correlation']:.4f},{n['Masked_RMSE']:.6e},"
                    f"{fr['Global_R2']:.4f},{fr['SSIM']:.4f},{fr['Avg_Dice']:.4f},"
                    f"{fr['Spatial_Correlation']:.4f},{fr['Masked_RMSE']:.6e}\n")
    print(f"\nSummary written to {csv_path}")


if __name__ == "__main__":
    main()
