import sys
import argparse, json, gc, os, random, time
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from paths import RESULTS_ADDITIONAL
import numpy as np
import tensorflow as tf
import optuna
from sklearn.metrics import r2_score
from scipy.stats import pearsonr
from skimage.metrics import structural_similarity as ssim

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"
optuna.logging.set_verbosity(optuna.logging.WARNING)

GRID        = 500
RESULTS_DIR = str(RESULTS_ADDITIONAL / "sta_lstm")
CYT_MAP     = {"il8": 0, "il10": 3}
WINDOW      = 2

N_OPTUNA    = 20
TUNE_EPOCHS = 20
FULL_EPOCHS = 200
PATIENCE    = 20

# 70/10/19 split boundaries over ~198 combined samples
N_TRAIN = 139   # 0: N_TRAIN
N_VAL   = 159   # N_TRAIN: N_VAL
# test    = N_VAL : end (39 samples, all from iter_1)

tf.config.set_visible_devices([], "GPU")  # CPU-only (rome partition)

def set_seed(seed):
    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed); np.random.seed(seed); tf.random.set_seed(seed)


# model 
class SpatialAttention(tf.keras.layers.Layer):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.attn_conv = tf.keras.layers.Conv2D(
            1, kernel_size=1, padding="same", activation="sigmoid"
        )

    def call(self, x):
        return x * self.attn_conv(x)

class STALSTM(tf.keras.Model):
    def __init__(self, grid_size, filters=64, lstm_units=128):
        super().__init__()
        self.grid_size   = grid_size
        self.filters     = filters
        self.lstm_units  = lstm_units
        self.latent_size = max(grid_size // 4, 8)

        self.enc = tf.keras.layers.TimeDistributed(
            tf.keras.Sequential([
                tf.keras.layers.Conv2D(filters, 3, strides=2, padding="same", activation="relu"),
                tf.keras.layers.Conv2D(filters, 3, strides=2, padding="same", activation="relu"),
            ]), name="encoder"
        )
        self.spatial_attn = tf.keras.layers.TimeDistributed(SpatialAttention(), name="spatial_attention")
        self.gap  = tf.keras.layers.TimeDistributed(tf.keras.layers.GlobalAveragePooling2D(), name="gap")
        self.lstm = tf.keras.layers.LSTM(lstm_units, return_sequences=False, name="lstm")
        self.relu = tf.keras.layers.Activation("relu")
        self.fc   = tf.keras.layers.Dense(self.latent_size * self.latent_size * filters, activation="relu")
        self.reshape_latent = tf.keras.layers.Reshape((self.latent_size, self.latent_size, filters))
        self.deconv1  = tf.keras.layers.Conv2DTranspose(filters // 2, 3, strides=2, padding="same", activation="relu")
        self.deconv2  = tf.keras.layers.Conv2DTranspose(filters // 4, 3, strides=2, padding="same", activation="relu")
        self.out_conv = tf.keras.layers.Conv2D(1, 3, padding="same", activation="linear")
        self.out_resize = tf.keras.layers.Resizing(grid_size, grid_size)

    def call(self, x):
        h = self.enc(x)
        h = self.spatial_attn(h)
        h = self.gap(h)
        h = self.lstm(h)
        h = self.relu(h)
        h = self.fc(h)
        h = self.reshape_latent(h)
        h = self.deconv1(h)
        h = self.deconv2(h)
        h = self.out_conv(h)
        return self.out_resize(h)


# metrics 
def _fisher_z(r):
    r = np.clip(r, -0.9999, 0.9999)
    return 0.5 * np.log((1.0 + r) / (1.0 - r))

def calculate_metrics(y_true, y_pred, masks, clip_max):
    """All inputs in physical units. clip_max from iter_0 (train iteration)."""
    min_t = min(y_true.shape[0], y_pred.shape[0], masks.shape[0])
    yt = y_true[:min_t]
    yp = np.maximum(y_pred[:min_t], 0.0)
    ms = np.max(masks[:min_t], axis=-1, keepdims=True)

    sq_diff       = np.square(yt - yp)
    rmse          = float(np.sqrt(np.sum(sq_diff * ms) / (np.sum(ms) + 1e-12)))
    unmasked_rmse = float(np.sqrt(np.mean(sq_diff)))
    r2            = float(r2_score(yt.flatten(), yp.flatten()))

    dice_thr = 0.05 * clip_max if clip_max > 0 else 1e-9
    dices, n_empty, z_corrs, ssims_v, n_ssim_skip = [], 0, [], [], 0
    fixed_dr = float(clip_max) if clip_max > 0 else 1.0

    for t in range(min_t):
        gt = yt[t, :, :, 0]; pr = yp[t, :, :, 0]
        gb = (gt > dice_thr).astype(float); pb = (pr > dice_thr).astype(float)
        if np.sum(gb) + np.sum(pb) == 0:
            n_empty += 1
        else:
            dices.append((2.0 * np.sum(gb * pb)) / (np.sum(gb) + np.sum(pb) + 1e-12))
        if np.std(gt) > 1e-12 and np.std(pr) > 1e-12:
            r_val = float(pearsonr(gt.flatten(), pr.flatten())[0])
            if np.isfinite(r_val):
                z_corrs.append(_fisher_z(r_val))
        if float(np.max(gt) - np.min(gt)) > 1e-12:
            ssims_v.append(float(ssim(gt, pr, data_range=fixed_dr)))
        else:
            n_ssim_skip += 1

    return {
        "Global_R2":           r2,
        "Masked_RMSE":         rmse,
        "Unmasked_RMSE":       unmasked_rmse,
        "Avg_Dice":            float(np.mean(dices)) if dices else 0.0,
        "Dice_Empty_Skipped":  n_empty,
        "Spatial_Correlation": float(np.tanh(np.mean(z_corrs))) if z_corrs else 0.0,
        "SSIM":                float(np.mean(ssims_v)) if ssims_v else 0.0,
        "SSIM_Skipped_Frames": n_ssim_skip,
    }


def denormalize(scaled, clip_max):
    return (np.asarray(scaled, dtype=np.float64) + 1.0) / 2.0 * clip_max


# data loading 
def rescale_x1_cytokines(X1, clip_max_1, clip_max_0):
    """
    Rescale iter_1 cytokine channels (0-5) to iter_0's scale.
    X1: (N, 2, G, G, 11) float32 (mmap slice → copy to float64 first).
    Returns float32 array with channels 0-5 rescaled.
    """
    X = X1.astype(np.float64)
    for j in range(6):
        raw = (X[..., j] + 1.0) / 2.0 * clip_max_1[j]
        if clip_max_0[j] > 0:
            X[..., j] = np.clip(raw / clip_max_0[j] * 2.0 - 1.0, -1.0, 1.0)
        else:
            X[..., j] = -1.0
    return X.astype(np.float32)


def build_combined(iter0_dir, iter1_dir, cyt_idx, clip_max_0, clip_max_1):
    """
    Load and concatenate iter_0 + iter_1 into a single combined dataset.

        Returns:
    X_combined : (N_total, 2, G, G, 11)  float32, all in iter_0 scale
    Y_phys     : (N_total, G, G, 1)      float64, physical units (iter_0 scale)
    M_combined : (N_total, G, G, 5)      float32, cell masks
    """
    # iter_0 - already in iter_0 scale
    X0 = np.load(f"{iter0_dir}/X_lstm.npy")                               # (99,2,G,G,11)
    Yraw0 = np.load(f"{iter0_dir}/Y_raw_phys.npy")[WINDOW:, :, :, cyt_idx]  # (99,G,G)
    M0    = np.load(f"{iter0_dir}/Y_masks_spatial.npy")                   # (99,G,G,5)

    # iter_1 - rescale cytokine channels to iter_0 scale
    X1_raw = np.load(f"{iter1_dir}/X_lstm.npy")                           # (99,2,G,G,11)
    X1     = rescale_x1_cytokines(X1_raw, clip_max_1, clip_max_0)
    Yraw1  = np.load(f"{iter1_dir}/Y_raw_phys.npy")[WINDOW:, :, :, cyt_idx]  # (99,G,G)
    M1     = np.load(f"{iter1_dir}/Y_masks_spatial.npy")                  # (99,G,G,5)

    X_combined = np.concatenate([X0, X1], axis=0)                         # (198,2,G,G,11)
    Yraw_all   = np.concatenate([Yraw0, Yraw1], axis=0)                   # (198,G,G)
    M_combined = np.concatenate([M0,    M1],    axis=0)                   # (198,G,G,5)

    Y_phys = Yraw_all[:, :, :, np.newaxis].astype(np.float64)             # (198,G,G,1)

    return X_combined, Y_phys, M_combined


# training  
def get_batch(X_arr, Y_phys_arr, clip_max, indices):
    """
    X_arr: (N, 2, G, G, 11) float32, already in iter_0 scale
    Y_phys_arr: (N, G, G, 1) float64, physical units
    Returns x (float32) and y_scaled (float32 in iter_0 scale, for MSE loss).
    """
    x = X_arr[indices].astype(np.float32)
    # Scale ground truth to [-1,1] using iter_0's clip_max for the loss
    y_scaled = np.clip(
        Y_phys_arr[indices] / clip_max * 2.0 - 1.0, -1.0, 1.0
    ).astype(np.float32)
    return x, y_scaled


def train_loop(model, X_arr, Y_phys_arr, clip_max, lr, bs, epochs, patience, vl_idx):
    opt     = tf.keras.optimizers.Adam(lr)
    tr_idx  = np.arange(N_TRAIN)
    best_val, stagnant, best_w = 1e9, 0, None

    for epoch in range(epochs):
        np.random.shuffle(tr_idx)
        for s in range(0, N_TRAIN, bs):
            idx      = tr_idx[s:s+bs]
            x_b, y_b = get_batch(X_arr, Y_phys_arr, clip_max, idx)
            with tf.GradientTape() as tape:
                loss = tf.reduce_mean((model(x_b, training=True) - y_b)**2)
            opt.apply_gradients(zip(tape.gradient(loss, model.trainable_variables),
                                    model.trainable_variables))

        x_v, y_v = get_batch(X_arr, Y_phys_arr, clip_max, vl_idx)
        vl = float(tf.reduce_mean((model(x_v, training=False) - y_v)**2).numpy())
        if vl < best_val - 1e-6:
            best_val, stagnant, best_w = vl, 0, model.get_weights()
        else:
            stagnant += 1
            if stagnant >= patience:
                break

    if best_w:
        model.set_weights(best_w)
    return best_val


# main run 
def run(cyt_name, seed, iter0_dir, iter1_dir):
    print(f"\n{'='*60}")
    print(f"STA-LSTM Cross-Iter | cytokine={cyt_name} | seed={seed}")
    print(f"  iter_0: {iter0_dir}")
    print(f"  iter_1: {iter1_dir}")
    print(f"{'='*60}")
    set_seed(seed)
    os.makedirs(RESULTS_DIR, exist_ok=True)

    cyt_idx = CYT_MAP[cyt_name]

    md0 = json.load(open(f"{iter0_dir}/metadata.json"))
    md1 = json.load(open(f"{iter1_dir}/metadata.json"))
    clip_max_0 = np.array(md0["clip_max"], dtype=np.float64)
    clip_max_1 = np.array(md1["clip_max"], dtype=np.float64)
    clip_train = clip_max_0[cyt_idx]   # reference scale throughout

    print(f"\nBuilding combined dataset (~198 samples)...")
    print(f"  iter_0 clip_max[{cyt_name}]: {clip_max_0[cyt_idx]:.4e}")
    print(f"  iter_1 clip_max[{cyt_name}]: {clip_max_1[cyt_idx]:.4e}  "
          f"(ratio {clip_max_1[cyt_idx]/clip_max_0[cyt_idx]:.2f}×)")

    X_all, Y_phys_all, M_all = build_combined(
        iter0_dir, iter1_dir, cyt_idx, clip_max_0, clip_max_1
    )
    N_total = X_all.shape[0]   # should be 198
    vl_idx  = np.arange(N_TRAIN, N_VAL)
    te_idx  = np.arange(N_VAL, N_total)

    print(f"  Total samples: {N_total} | "
          f"train={N_TRAIN} | val={N_VAL-N_TRAIN} | test={N_total-N_VAL}")
    print(f"  Test samples come from iter_1 (indices {N_VAL-99}–{N_total-99} of iter_1)")

    # hyperparameter search 
    def objective(trial):
        set_seed(seed)
        tf.keras.backend.clear_session()
        f  = trial.suggest_categorical("filters",    [32, 64])
        lu = trial.suggest_categorical("lstm_units", [64, 128])
        lr = trial.suggest_float("lr",         1e-5, 1e-3, log=True)
        bs = trial.suggest_categorical("batch_size", [2, 4])

        m  = STALSTM(grid_size=GRID, filters=f, lstm_units=lu)
        vl = train_loop(m, X_all, Y_phys_all, clip_train, lr, bs,
                        TUNE_EPOCHS, patience=5, vl_idx=vl_idx)
        del m; gc.collect(); tf.keras.backend.clear_session()
        return vl

    study = optuna.create_study(direction="minimize",
                                sampler=optuna.samplers.TPESampler(seed=seed),
                                pruner=optuna.pruners.MedianPruner())
    study.optimize(objective, n_trials=N_OPTUNA, show_progress_bar=False)
    best = study.best_params
    print(f"Best params: {best}")

    # final training run 
    set_seed(seed); tf.keras.backend.clear_session()
    model = STALSTM(grid_size=GRID, filters=best["filters"], lstm_units=best["lstm_units"])

    t_start = time.time()
    train_loop(model, X_all, Y_phys_all, clip_train, best["lr"], best["batch_size"],
               FULL_EPOCHS, PATIENCE, vl_idx)
    train_elapsed = time.time() - t_start

    # inference on test set (iter_1 tail) 
    all_preds = []
    for s in range(0, len(te_idx), 4):
        idx   = te_idx[s:s+4]
        x_b   = X_all[idx].astype(np.float32)
        all_preds.append(model(x_b, training=False).numpy())
    Yp_scaled = np.concatenate(all_preds, axis=0) # (N_test, G, G, 1)

    # Denormalise predictions → physical units (iter_0 scale)
    Yp_phys = denormalize(Yp_scaled, clip_train)

    # Ground truth physical (alr loaded)
    Ya_phys = Y_phys_all[te_idx]  # (N_test, G, G, 1)
    M_test  = M_all[te_idx]  # (N_test, G, G, 5)

    n_test = len(te_idx)
    mid    = n_test // 2

    results = {
        "grid":               GRID,
        "seed":               seed,
        "cytokine":           cyt_name,
        "iter0_dir":          iter0_dir,
        "iter1_dir":          iter1_dir,
        "best_params":        best,
        "train_time_seconds": round(train_elapsed, 2),
        "split": {
            "n_total":  N_total,
            "n_train":  N_TRAIN,
            "n_val":    N_VAL - N_TRAIN,
            "n_test":   n_test,
            "note":     "test samples are the last n_test of iter_1",
        },
        "clip_max_iter0":  clip_train,
        "clip_max_iter1":  float(clip_max_1[cyt_idx]),
        "scale_ratio":     round(float(clip_max_1[cyt_idx]) / clip_train, 4),
        "results": {
            "Near_Half":  calculate_metrics(Ya_phys[:mid],  Yp_phys[:mid],  M_test[:mid],  clip_train),
            "Far_Half":   calculate_metrics(Ya_phys[mid:],  Yp_phys[mid:],  M_test[mid:],  clip_train),
            "All_Test":   calculate_metrics(Ya_phys,        Yp_phys,        M_test,        clip_train),
        },
    }
    out_path = f"{RESULTS_DIR}/res_cross_iter_{cyt_name}_500_{seed}.json"
    with open(out_path, "w") as f:
        json.dump(results, f, indent=4)
    print(f"Saved → {out_path}")
    del model; gc.collect()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="STA-LSTM cross-iteration generalization")
    parser.add_argument("--cytokine",   choices=["il8", "il10"], required=True)
    parser.add_argument("--seed",       type=int,                 required=True)
    parser.add_argument("--iter0-dir",  dest="iter0_dir",         required=True,
                        help="Path to scan_iteration_0 preprocessed/500x500")
    parser.add_argument("--iter1-dir",  dest="iter1_dir",         required=True,
                        help="Path to scan_iteration_1 preprocessed/500x500")
    args = parser.parse_args()
    run(args.cytokine, args.seed, args.iter0_dir, args.iter1_dir)