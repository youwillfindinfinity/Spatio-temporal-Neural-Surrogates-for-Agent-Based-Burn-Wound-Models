import sys
import argparse, json, gc, os, time
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
RESULTS_DIR  = str(RESULTS_ADDITIONAL / "200hrs" / "unet")
CYT_MAP      = {"il8": 0, "il10": 3}
FULL_EPOCHS  = 200
TUNE_EPOCHS  = 20
N_OPTUNA     = 20
PATIENCE     = 20

def build_unet(base_filters, depth, dropout_rate):
    inputs = tf.keras.Input(shape=(GRID, GRID, 22))
    skips  = []
    x = inputs
    f = base_filters
    for d in range(depth):
        x = tf.keras.layers.Conv2D(f, 3, padding="same", activation="relu")(x)
        x = tf.keras.layers.BatchNormalization()(x)
        x = tf.keras.layers.Conv2D(f, 3, padding="same", activation="relu")(x)
        x = tf.keras.layers.BatchNormalization()(x)
        skips.append(x)
        x = tf.keras.layers.MaxPool2D(2)(x)
        f *= 2
    x = tf.keras.layers.Conv2D(f, 3, padding="same", activation="relu")(x)
    x = tf.keras.layers.BatchNormalization()(x)
    x = tf.keras.layers.Conv2D(f, 3, padding="same", activation="relu")(x)
    x = tf.keras.layers.BatchNormalization()(x)
    if dropout_rate > 0:
        x = tf.keras.layers.Dropout(dropout_rate)(x)
    for d in reversed(range(depth)):
        f //= 2
        x = tf.keras.layers.Conv2DTranspose(f, 2, strides=2, padding="same")(x)
        s = skips[d]
        x = tf.keras.layers.Lambda(lambda t: tf.concat([t[0], t[1][:, :tf.shape(t[0])[1], :tf.shape(t[0])[2], :]], axis=-1))([x, s])
        x = tf.keras.layers.Conv2D(f, 3, padding="same", activation="relu")(x)
        x = tf.keras.layers.BatchNormalization()(x)
        x = tf.keras.layers.Conv2D(f, 3, padding="same", activation="relu")(x)
        x = tf.keras.layers.BatchNormalization()(x)
    x = tf.keras.layers.Conv2D(1, 1)(x)
    outputs = tf.keras.layers.Resizing(GRID, GRID)(x)
    return tf.keras.Model(inputs=inputs, outputs=outputs)

def do_train(model, opt, Xu_mmap, Yt, cyt_idx, tr_idx, vl_idx, batch_size, epochs, verbose=True):
    @tf.function
    def train_step(x, y):
        with tf.GradientTape() as tape:
            pred = model(x, training=True)
            loss = tf.reduce_mean(tf.square(y - pred))
        grads = tape.gradient(loss, model.trainable_variables)
        opt.apply_gradients(zip(grads, model.trainable_variables))
        return loss

    @tf.function
    def val_step(x, y):
        return tf.reduce_mean(tf.square(y - model(x, training=False)))

    def get_batch(idx):
        raw = Xu_mmap[idx].astype(np.float32)
        x = raw.transpose(0, 2, 3, 1, 4).reshape(len(idx), GRID, GRID, 22) if raw.ndim == 5 else raw
        y = Yt[idx, :, :, cyt_idx:cyt_idx+1].astype(np.float32)
        return x, y

    x_v, y_v = get_batch(vl_idx)
    best_val, stagnant, best_w = 1e9, 0, None
    for epoch in range(1, epochs + 1):
        np.random.shuffle(tr_idx)
        epoch_losses = []
        for s in range(0, len(tr_idx), batch_size):
            x_b, y_b = get_batch(tr_idx[s:s+batch_size])
            epoch_losses.append(train_step(x_b, y_b))
        vl = float(val_step(x_v, y_v).numpy())
        if verbose and epoch % 5 == 0:
            print(f"  Ep {epoch:3d} | Loss: {np.mean(epoch_losses):.6f} | Val: {vl:.6f}", flush=True)
        if vl < best_val:
            best_val, stagnant, best_w = vl, 0, model.get_weights()
        else:
            stagnant += 1
            if stagnant >= PATIENCE: break
    if best_w: model.set_weights(best_w)
    return best_val

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
    
    per_t_r2 = []
    for t in range(T):
        if np.std(yt[t]) > 1e-12:
            per_t_r2.append(float(r2_score(yt[t].flatten(), yp[t].flatten())))
        else: per_t_r2.append(float('nan'))
    
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
        "Global_R2": r2, "Per_Timestep_R2": per_t_r2,
        "Masked_RMSE": rmse, "Unmasked_RMSE": u_rmse,
        "Avg_Dice": float(np.mean(dices)) if dices else 0.0,
        "Spatial_Correlation": float(np.tanh(np.mean(z_corrs))) if z_corrs else 0.0,
        "SSIM": float(np.mean(ssims)) if ssims else 0.0
    }

def run(cyt_name, seed, data_dir):
    print(f"\nUNet 500x500 (A100) | cytokine={cyt_name} | seed={seed}")
    np.random.seed(seed); tf.random.set_seed(seed)
    os.makedirs(RESULTS_DIR, exist_ok=True)

    cyt_idx = CYT_MAP[cyt_name]
    md = json.load(open(f"{data_dir}/metadata.json"))
    clip = float(md["scaling"]["max"][cyt_idx])
    Xu_mmap = np.load(f"{data_dir}/X_unet.npy", mmap_mode="r")
    Yt      = np.load(f"{data_dir}/Y_target.npy")
    Yraw    = np.load(f"{data_dir}/Y_raw_phys.npy")
    masks_spatial = np.load(f"{data_dir}/Y_masks_spatial.npy")

    def parse_split(s):
        start, end = s.split(':')
        return np.arange(int(start), int(end))

    tr_idx = parse_split(md["splits"]["train"])
    vl_idx = parse_split(md["splits"]["val"])
    ts_near_idx = parse_split(md["splits"]["test_near"])
    ts_far_idx  = parse_split(md["splits"]["test_far"])

    if seed == 42:
        def objective(trial):
            tf.keras.backend.clear_session()
            f = trial.suggest_categorical("base_filters", [16, 32])
            d = trial.suggest_categorical("depth", [2, 3])
            dr = trial.suggest_categorical("dropout", [0.0, 0.1])
            lr = trial.suggest_float("lr", 1e-4, 1e-3, log=True)
            bs = trial.suggest_categorical("batch_size", [2, 4])
            m = build_unet(f, d, dr); opt = tf.keras.optimizers.Adam(lr)
            return do_train(m, opt, Xu_mmap, Yt, cyt_idx, tr_idx, vl_idx, bs, TUNE_EPOCHS, verbose=False)
        study = optuna.create_study(direction="minimize", sampler=optuna.samplers.TPESampler(seed=42))
        study.optimize(objective, n_trials=N_OPTUNA); best = study.best_params
    else:
        best = json.load(open(f"{RESULTS_DIR}/res_{cyt_name}_500_42.json"))["best_params"]

    tf.keras.backend.clear_session()
    model = build_unet(best["base_filters"], best["depth"], best["dropout"])
    opt = tf.keras.optimizers.Adam(best["lr"])
    t_start = time.time()
    do_train(model, opt, Xu_mmap, Yt, cyt_idx, tr_idx, vl_idx, best["batch_size"], FULL_EPOCHS)
    train_elapsed = time.time() - t_start

    def evaluate_full(indices):
        preds = []
        for idx in indices:
            raw = Xu_mmap[idx:idx+1].astype(np.float32)
            xi = raw.transpose(0, 2, 3, 1, 4).reshape(1, GRID, GRID, 22) if raw.ndim == 5 else raw
            preds.append(model(xi, training=False).numpy()[0, :, :, 0])
        
        p_ph = np.clip((np.array(preds)[..., np.newaxis] + 1.0) / 2.0 * clip, 0, None)
        gt_ph = Yraw[[min(i+1, Yraw.shape[0]-1) for i in indices], ..., cyt_idx:cyt_idx+1]
        m_sp  = masks_spatial[[min(i, masks_spatial.shape[0]-1) for i in indices]]
        
        return calculate_metrics(gt_ph, p_ph, m_sp, clip)

    print("Final evaluation...", flush=True)
    t_pred_start = time.time()
    near_res = evaluate_full(ts_near_idx)
    far_res = evaluate_full(ts_far_idx)
    pred_elapsed = time.time() - t_pred_start

    res = {
        "grid": 500, "seed": seed, "cytokine": cyt_name, "model": "unet", "best_params": best,
        "train_time_seconds": round(train_elapsed, 2), "pred_time_seconds": round(pred_elapsed, 4),
        "results": {"Near_Horizon": near_res, "Far_Horizon": far_res}
    }
    
    out_path = f"{RESULTS_DIR}/res_{cyt_name}_500_{seed}.json"
    with open(out_path, "w") as f: json.dump(res, f, indent=2)
    wgt_path = f"{RESULTS_DIR}/weights_{cyt_name}_500_{seed}.weights.h5"
    model.save_weights(wgt_path)
    print(f"  Weights saved: {wgt_path}")
    print(f"DONE: {out_path}")

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--cytokine", required=True); ap.add_argument("--seed", type=int, required=True); ap.add_argument("--data-dir", default=str(PREPROCESSED_200H / "500x500"))
    args = ap.parse_args(); run(args.cytokine, args.seed, args.data_dir)
