import sys
import argparse, json, gc, os, random, time
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from paths import RESULTS_ADDITIONAL, PREPROCESSED_200H
import numpy as np
import tensorflow as tf
import optuna
from sklearn.metrics import r2_score
from scipy.stats import pearsonr
from skimage.metrics import structural_similarity as ssim

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"
optuna.logging.set_verbosity(optuna.logging.WARNING)

tf.config.optimizer.set_jit(False)

gpus = tf.config.list_physical_devices('GPU')
if gpus:
    for gpu in gpus:
        try:
            tf.config.experimental.set_memory_growth(gpu, True)
        except RuntimeError: pass

GRID         = 500
RESULTS_DIR  = str(RESULTS_ADDITIONAL / "200hrs" / "sta_lstm")
CYT_MAP      = {"il8": 0, "il10": 3}
N_OPTUNA     = 20
TUNE_EPOCHS  = 20
FULL_EPOCHS  = 200
PATIENCE     = 20

def set_seed(seed):
    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed); np.random.seed(seed); tf.random.set_seed(seed)

class SpatialAttention(tf.keras.layers.Layer):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.attn_conv = tf.keras.layers.Conv2D(1, kernel_size=1, padding="same", activation="sigmoid")
    def call(self, x):
        return x * self.attn_conv(x)

class STALSTM(tf.keras.Model):
    def __init__(self, grid_size, filters=64, lstm_units=128):
        super().__init__()
        self.grid_size = grid_size
        self.latent_size = grid_size // 4
        
        self.enc = tf.keras.layers.TimeDistributed(tf.keras.Sequential([
            tf.keras.layers.Conv2D(filters, 3, strides=2, padding="same", activation="relu"),
            tf.keras.layers.Conv2D(filters, 3, strides=2, padding="same", activation="relu"),
        ]))
        self.spatial_attn = tf.keras.layers.TimeDistributed(SpatialAttention())
        self.gap   = tf.keras.layers.TimeDistributed(tf.keras.layers.GlobalAveragePooling2D())
        self.lstm  = tf.keras.layers.LSTM(lstm_units, return_sequences=False)
        self.fc    = tf.keras.layers.Dense(self.latent_size * self.latent_size * (filters // 2), activation="relu")
        self.resh  = tf.keras.layers.Reshape((self.latent_size, self.latent_size, filters // 2))
        self.dec1  = tf.keras.layers.Conv2DTranspose(filters // 2, 3, strides=2, padding="same", activation="relu")
        self.dec2  = tf.keras.layers.Conv2DTranspose(filters // 4, 3, strides=2, padding="same", activation="relu")
        self.out_c = tf.keras.layers.Conv2D(1, 3, padding="same", activation="linear")
        self.out_r = tf.keras.layers.Resizing(grid_size, grid_size)

    def call(self, x, training=False):
        h = self.enc(x)
        h = self.spatial_attn(h)
        h = self.gap(h)
        h = self.lstm(h)
        h = self.fc(h)
        h = self.resh(h)
        h = self.dec1(h)
        h = self.dec2(h)
        h = self.out_c(h)
        return self.out_r(h)

def _fisher_z(r):
    r = np.clip(r, -0.9999, 0.9999)
    return 0.5 * np.log((1.0 + r) / (1.0 - r))

def calculate_metrics(yt, yp, masks, clip_max):
    T = yt.shape[0]
    ms = np.max(masks, axis=-1, keepdims=True)
    sq = np.square(yt - yp)
    rmse = float(np.sqrt(np.sum(sq * ms) / (np.sum(ms) + 1e-12)))
    u_rmse = float(np.sqrt(np.mean(sq)))
    r2 = float(r2_score(yt.flatten(), yp.flatten()))
    per_t_r2 = [float(r2_score(yt[t].flatten(), yp[t].flatten())) if np.std(yt[t]) > 1e-12 else float('nan') for t in range(T)]
    dice_thr = 0.05 * clip_max if clip_max > 0 else 1e-9
    dices, z_corrs, ssims = [], [], []
    for t in range(T):
        gt, pr = yt[t, ..., 0], yp[t, ..., 0]
        gb, pb = (gt > dice_thr).astype(float), (pr > dice_thr).astype(float)
        if np.sum(gb) + np.sum(pb) > 0:
            dices.append(2.0 * np.sum(gb * pb) / (np.sum(gb) + np.sum(pb) + 1e-12))
        if np.std(gt) > 1e-12 and np.std(pr) > 1e-12:
            r_val = float(pearsonr(gt.flatten(), pr.flatten())[0])
            if np.isfinite(r_val): z_corrs.append(_fisher_z(r_val))
        if np.max(gt) - np.min(gt) > 1e-12:
            ssims.append(ssim(gt, pr, data_range=max(clip_max, 1e-12)))
    return {
        "Global_R2": r2, "Per_Timestep_R2": per_t_r2, "Masked_RMSE": rmse, "Unmasked_RMSE": u_rmse,
        "Avg_Dice": float(np.mean(dices)) if dices else 0.0,
        "Spatial_Correlation": float(np.tanh(np.mean(z_corrs))) if z_corrs else 0.0,
        "SSIM": float(np.mean(ssims)) if ssims else 0.0
    }

def train_loop(model, Xm, Yt_arr, cyt_idx, tr_indices, lr, bs, epochs, patience, vl_indices):
    opt = tf.keras.optimizers.Adam(lr)
    @tf.function
    def train_step(xb, yb):
        with tf.GradientTape() as tape:
            loss = tf.reduce_mean(tf.square(yb - model(xb, training=True)))
        opt.apply_gradients(zip(tape.gradient(loss, model.trainable_variables), model.trainable_variables))
        return loss
    @tf.function
    def val_step(xb, yb):
        return tf.reduce_mean(tf.square(yb - model(xb, training=False)))

    xv, yv = Xm[vl_indices].astype(np.float32), Yt_arr[vl_indices, :, :, cyt_idx:cyt_idx+1].astype(np.float32)
    best_val, stagnant, best_w = 1e9, 0, None
    for epoch in range(1, epochs + 1):
        indices = np.array(tr_indices); np.random.shuffle(indices)
        epoch_losses = []
        for s in range(0, len(indices), bs):
            idx = indices[s:s+bs]
            xb, yb = Xm[idx].astype(np.float32), Yt_arr[idx, :, :, cyt_idx:cyt_idx+1].astype(np.float32)
            epoch_losses.append(train_step(xb, yb))
        vl = float(val_step(xv, yv).numpy())
        if vl < best_val:
            best_val = vl; best_w = model.get_weights(); stagnant = 0
        else:
            stagnant += 1
            if stagnant >= patience: break
    if best_w: model.set_weights(best_w)
    return best_val

def run(cyt_name, seed, data_dir):
    print(f"\nSTA-LSTM 500x500 (A100) | cytokine={cyt_name} | seed={seed}")
    set_seed(seed); Path(RESULTS_DIR).mkdir(parents=True, exist_ok=True)
    md = json.load(open(f"{data_dir}/metadata.json"))
    cyt_idx = CYT_MAP[cyt_name.lower()]
    clip = float(md["scaling"]["max"][cyt_idx])
    Xm = np.load(f"{data_dir}/X_lstm.npy", mmap_mode="r")
    Yt = np.load(f"{data_dir}/Y_target.npy")
    M  = np.load(f"{data_dir}/Y_masks_spatial.npy")
    Yraw = np.load(f"{data_dir}/Y_raw_phys.npy")

    def parse_split(s):
        start, end = s.split(':'); return np.arange(int(start), int(end))

    tr_idx, vl_idx = parse_split(md["splits"]["train"]), parse_split(md["splits"]["val"])
    ts_near_idx, ts_far_idx = parse_split(md["splits"]["test_near"]), parse_split(md["splits"]["test_far"])

    if seed == 42:
        def objective(trial):
            tf.keras.backend.clear_session()
            f = trial.suggest_categorical("filters", [32, 64])
            lu = trial.suggest_categorical("lstm_units", [64, 128])
            lr = trial.suggest_float("lr", 1e-5, 1e-3, log=True)
            bs = trial.suggest_categorical("batch_size", [2, 4])
            m = STALSTM(GRID, f, lu)
            return train_loop(m, Xm, Yt, cyt_idx, tr_idx, lr, bs, TUNE_EPOCHS, 5, vl_idx)
        study = optuna.create_study(direction="minimize", sampler=optuna.samplers.TPESampler(seed=42))
        study.optimize(objective, n_trials=N_OPTUNA); best = study.best_params
    else:
        best = json.load(open(f"{RESULTS_DIR}/res_{cyt_name}_500_42.json"))["best_params"]

    tf.keras.backend.clear_session()
    model = STALSTM(GRID, best["filters"], best["lstm_units"])
    t_start = time.time()
    train_loop(model, Xm, Yt, cyt_idx, tr_idx, best["lr"], best["batch_size"], FULL_EPOCHS, PATIENCE, vl_idx)
    train_elapsed = time.time() - t_start

    def evaluate_full(indices):
        preds = []
        for s in range(0, len(indices), 4):
            xb = Xm[indices[s:s+4]].astype(np.float32)
            preds.append(model(xb, training=False).numpy())
        yp_ph = np.clip((np.concatenate(preds, axis=0) + 1.0) / 2.0 * clip, 0, None)
        gt_ph = Yraw[indices + 1, ..., cyt_idx:cyt_idx+1]
        m_sp  = M[indices]
        return calculate_metrics(gt_ph, yp_ph, m_sp, clip)

    print("Evaluating test horizons...", flush=True)
    t_pred_start = time.time()
    res_near, res_far = evaluate_full(ts_near_idx), evaluate_full(ts_far_idx)
    pred_elapsed = time.time() - t_pred_start

    res = {
        "grid": GRID, "seed": seed, "cytokine": cyt_name, "model": "sta_lstm", "best_params": best,
        "train_time_seconds": round(train_elapsed, 2), "pred_time_seconds": round(pred_elapsed, 4),
        "results": {"Near_Horizon": res_near, "Far_Horizon": res_far}
    }
    out_path = f"{RESULTS_DIR}/res_{cyt_name}_500_{seed}.json"
    with open(out_path, "w") as f: json.dump(res, f, indent=4)
    wgt_path = f"{RESULTS_DIR}/weights_{cyt_name}_500_{seed}.weights.h5"
    model.save_weights(wgt_path)
    print(f"  Weights saved: {wgt_path}")
    print(f"DONE: {out_path}")

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--cytokine", required=True); ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--data-dir", default=str(PREPROCESSED_200H / "500x500"))
    args = ap.parse_args(); run(args.cytokine, args.seed, args.data_dir)
