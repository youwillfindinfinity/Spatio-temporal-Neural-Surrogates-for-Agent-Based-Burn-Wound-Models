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
tf.config.optimizer.set_jit(False)

gpus = tf.config.list_physical_devices('GPU')
for gpu in gpus:
    try:
        tf.config.experimental.set_memory_growth(gpu, True)
    except RuntimeError:
        pass

CYT_NAMES = ["il8", "il1", "il6", "il10", "tnf", "tgf"]
N_OPTUNA = 20
TUNE_EPOCHS = 30
FULL_EPOCHS = 400
PATIENCE = 40
LR_PATIENCE = 15
EVAL_CHUNK = 4096

def set_seed(seed):
    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed); np.random.seed(seed); tf.random.set_seed(seed)

class Branch(tf.keras.layers.Layer):
    def __init__(self, hidden, p, **kw):
        super().__init__(**kw)
        self.fc1 = tf.keras.layers.Dense(hidden, activation="relu")
        self.fc2 = tf.keras.layers.Dense(p, activation="linear")
    def call(self, x, training=False):
        return self.fc2(self.fc1(x))

class Trunk(tf.keras.layers.Layer):
    def __init__(self, hidden, p, **kw):
        super().__init__(**kw)
        self.U = tf.keras.layers.Dense(hidden, activation="tanh")
        self.V = tf.keras.layers.Dense(hidden, activation="tanh")
        self.W1a = tf.keras.layers.Dense(hidden, activation="relu")
        self.W1b = tf.keras.layers.Dense(hidden, activation="linear")
        self.W2a = tf.keras.layers.Dense(hidden, activation="relu")
        self.W2b = tf.keras.layers.Dense(hidden, activation="linear")
        self.out = tf.keras.layers.Dense(p, activation="linear")
    def call(self, x):
        u = self.U(x); v = self.V(x)
        h = self.W1b(self.W1a(x)); h = h * u + (1.0 - h) * v
        h = self.W2b(self.W2a(h)); h = h * u + (1.0 - h) * v
        return self.out(h)

class DeepONet(tf.keras.Model):
    def __init__(self, hidden, p):
        super().__init__()
        self.branch = Branch(hidden, p)
        self.trunk = Trunk(hidden, p)
        self.bias = self.add_weight(shape=(1,), initializer="zeros", trainable=True, name="bias")
    def call(self, inputs, training=False):
        xb, xt = inputs
        b = self.branch(xb, training=training)
        t = self.trunk(xt)
        r = tf.einsum("bp,bnp->bn", b, t) + self.bias
        return tf.expand_dims(r, -1)

def load_data(data_dir, cyt_idx):
    md = json.load(open(f"{data_dir}/metadata.json"))
    clip = float(md["clip_max"][cyt_idx]) if "clip_max" in md else float(md["scaling"]["max"][cyt_idx])
    G = int(md["grid"]); G2 = G * G; N = int(md["n_samples"])
    Xb = np.load(f"{data_dir}/X_branch.npy", mmap_mode="r")
    Xt = np.load(f"{data_dir}/X_trunk.npy")
    Yt = np.load(f"{data_dir}/Y_target.npy")[..., cyt_idx]
    masks = np.load(f"{data_dir}/Y_masks_spatial.npy")
    return Xb, Xt, Yt, masks, clip, G, G2, N

def build_branch_stats(Xb_mmap, Xt, cyt_idx, G, G2, N):
    stats = np.zeros((N, 7), dtype=np.float32)
    xs_g, ys_g = np.linspace(0, 1, G, dtype=np.float32), np.linspace(0, 1, G, dtype=np.float32)
    xx, yy = np.meshgrid(xs_g, ys_g, indexing='ij')
    for i in range(N):
        f0 = Xb_mmap[i, 0, :, :, cyt_idx].astype(np.float32)
        mask = (Xb_mmap[i, 0, :, :, 6:].max(axis=-1) > 0.5).astype(np.float32)
        na = float(np.sum(mask)) + 1e-6
        stats[i, 0] = (float(np.max(f0)) + 1.0) / 2.0
        stats[i, 1] = (float(np.mean(f0)) + 1.0) / 2.0
        stats[i, 2] = float(np.std(f0))
        stats[i, 3] = float(np.sum(xx * mask) / na)
        stats[i, 4] = float(np.sum(yy * mask) / na)
        stats[i, 5] = na / G2
        stats[i, 6] = float(Xt[i, 0, 2])
    return stats

def build_trunk_for_sample(Xb_mmap, Xt, i, G2):
    xy = Xt[i, :, :2]
    sf = Xb_mmap[i].astype(np.float32).reshape(G2, 22)
    return np.concatenate([xy, sf], axis=-1)

def build_dataset(Xbranch, Xb_mmap, Xt, Yf, indices, batch_size, chunk_size, G2, shuffle=True):
    chunks = list(range(0, G2, chunk_size))
    def gen():
        order = np.array(indices)
        if shuffle: np.random.shuffle(order)
        for i in order:
            xb = Xbranch[i]
            trunk_i = build_trunk_for_sample(Xb_mmap, Xt, i, G2)
            y_i = Yf[i]
            for s in chunks:
                e = min(s + chunk_size, G2)
                xt = trunk_i[s:e]
                y = y_i[s:e]
                size = e - s
                if size < chunk_size:
                    pad = chunk_size - size
                    xt = np.concatenate([xt, np.zeros((pad, 24), np.float32)], axis=0)
                    y = np.concatenate([y, np.zeros((pad, 1), np.float32)], axis=0)
                yield (xb, xt, np.array([size], dtype=np.int32)), y
    sig = ((tf.TensorSpec((7,), tf.float32), tf.TensorSpec((chunk_size, 24), tf.float32), tf.TensorSpec((1,), tf.int32)), tf.TensorSpec((chunk_size, 1), tf.float32))
    return tf.data.Dataset.from_generator(gen, output_signature=sig).batch(batch_size).prefetch(tf.data.AUTOTUNE)

def masked_mse(pred, y, sz):
    mask = tf.sequence_mask(tf.squeeze(sz, -1), maxlen=tf.shape(pred)[1], dtype=tf.float32)
    mask = tf.expand_dims(mask, -1)
    return tf.reduce_sum(tf.square(pred - y) * mask) / (tf.reduce_sum(mask) + 1e-8)

def do_train(model, opt, ds_tr, ds_vl, epochs, verbose=True):
    @tf.function
    def train_step(xb, xt, sz, y):
        with tf.GradientTape() as tape:
            loss = masked_mse(model([xb, xt], training=True), y, sz)
        opt.apply_gradients(zip(tape.gradient(loss, model.trainable_variables), model.trainable_variables))
        return loss

    @tf.function
    def val_step(xb, xt, sz, y):
        return masked_mse(model([xb, xt], training=False), y, sz)

    best_val = np.inf; best_w = None; wait = rw = 0
    for ep in range(1, epochs + 1):
        tr = np.mean([train_step(*b[0], b[1]) for b in ds_tr])
        vl = np.mean([val_step(*b[0], b[1]) for b in ds_vl])
        if verbose and ep % 10 == 0:
            print(f"  Epoch {ep:4d}  loss={tr:.5f}  val={vl:.5f}", flush=True)
        if vl < best_val:
            best_val = vl; best_w = model.get_weights(); wait = rw = 0
        else:
            wait += 1; rw += 1
        if rw >= LR_PATIENCE:
            opt.learning_rate.assign(opt.learning_rate * 0.5)
            rw = 0
        if wait >= PATIENCE: break
    if best_w: model.set_weights(best_w)
    return best_val

def predict_full(model, Xbranch, Xb_mmap, Xt, G2, N, chunk=EVAL_CHUNK):
    out = np.zeros((N, G2, 1), np.float32)
    for i in range(N):
        xb = tf.constant(Xbranch[i:i+1])
        trunk_i = build_trunk_for_sample(Xb_mmap, Xt, i, G2).astype(np.float32)
        for s in range(0, G2, chunk):
            e = min(s + chunk, G2)
            xt = tf.constant(trunk_i[np.newaxis, s:e, :])
            out[i, s:e] = model([xb, xt], training=False).numpy()[0]
    return out

def _fisher_z(r):
    r = np.clip(r, -0.9999, 0.9999)
    return 0.5 * np.log((1.0 + r) / (1.0 - r))

def calculate_metrics(y_true, y_pred, masks, clip_max):
    T = min(y_true.shape[0], y_pred.shape[0], masks.shape[0])
    yt = y_true[:T]; yp = np.maximum(y_pred[:T], 0.0)
    ms = np.max(masks[:T], axis=-1, keepdims=True)
    sq = np.square(yt - yp)
    rmse = float(np.sqrt(np.sum(sq * ms) / (np.sum(ms) + 1e-12)))
    unmasked_rmse = float(np.sqrt(np.mean(sq)))
    r2 = float(r2_score(yt.flatten(), yp.flatten()))
    per_t_r2 = [float(r2_score(yt[t].flatten(), yp[t].flatten())) if np.std(yt[t]) > 1e-12 else float('nan') for t in range(T)]
    dice_thr = 0.05 * clip_max if clip_max > 0 else 1e-9
    dices, n_empty, z_corrs, ssims_v = [], 0, [], []
    fixed_dr = float(clip_max) if clip_max > 0 else 1.0
    for t in range(T):
        gt, pr = yt[t, :, :, 0], yp[t, :, :, 0]
        gb, pb = (gt > dice_thr).astype(float), (pr > dice_thr).astype(float)
        if np.sum(gb) + np.sum(pb) == 0: n_empty += 1
        else: dices.append(2.0 * np.sum(gb * pb) / (np.sum(gb) + np.sum(pb) + 1e-12))
        if np.std(gt) > 1e-12 and np.std(pr) > 1e-12:
            r_val = float(pearsonr(gt.flatten(), pr.flatten())[0])
            if np.isfinite(r_val): z_corrs.append(_fisher_z(r_val))
        if np.max(gt) - np.min(gt) > 1e-12: ssims_v.append(float(ssim(gt, pr, data_range=fixed_dr)))
    return {
        "Global_R2": r2, "Per_Timestep_R2": per_t_r2, "Masked_RMSE": rmse, "Unmasked_RMSE": unmasked_rmse,
        "Avg_Dice": float(np.mean(dices)) if dices else 0.0, "Dice_Empty_Skipped": n_empty,
        "Spatial_Correlation": float(np.tanh(np.mean(z_corrs))) if z_corrs else 0.0, "SSIM": float(np.mean(ssims_v)) if ssims_v else 0.0
    }

def run(data_dir, cytokine, seed):
    set_seed(seed)
    cyt_idx = CYT_NAMES.index(cytokine.lower())
    out_dir = RESULTS_ADDITIONAL / "200hrs" / "deeponet_h"
    out_dir.mkdir(parents=True, exist_ok=True)
    Xb_mmap, Xt, Yt, masks, clip, G, G2, N = load_data(data_dir, cyt_idx)
    Xbranch = build_branch_stats(Xb_mmap, Xt, cyt_idx, G, G2, N)
    Yf = Yt.reshape(N, G2, 1).astype(np.float32)
    tr_idx, vl_idx = list(range(140)), list(range(140, 160))
    if seed == 42:
        def objective(trial):
            set_seed(42); tf.keras.backend.clear_session()
            p = trial.suggest_categorical("p", [64, 128]); h = trial.suggest_categorical("hidden", [128, 256])
            lr = trial.suggest_float("lr", 1e-5, 1e-3, log=True); bs = trial.suggest_categorical("batch_size", [4, 8])
            cs = trial.suggest_categorical("chunk_size", [2048, 4096])
            ds_tr = build_dataset(Xbranch, Xb_mmap, Xt, Yf, tr_idx, bs, cs, G2); ds_vl = build_dataset(Xbranch, Xb_mmap, Xt, Yf, vl_idx, bs, cs, G2, shuffle=False)
            m = DeepONet(hidden=h, p=p); opt = tf.keras.optimizers.Adam(lr)
            return do_train(m, opt, ds_tr, ds_vl, TUNE_EPOCHS, verbose=False)
        study = optuna.create_study(direction="minimize", sampler=optuna.samplers.TPESampler(seed=42))
        study.optimize(objective, n_trials=N_OPTUNA); best = study.best_params; optuna_val = float(study.best_value)
    else:
        ref = json.load(open(out_dir / f"res_{cytokine}_{G}_42.json"))
        best = ref["best_params"]; optuna_val = ref["optuna_best_val_loss"]
    tf.keras.backend.clear_session(); set_seed(seed)
    ds_tr = build_dataset(Xbranch, Xb_mmap, Xt, Yf, tr_idx, best["batch_size"], best["chunk_size"], G2)
    ds_vl = build_dataset(Xbranch, Xb_mmap, Xt, Yf, vl_idx, best["batch_size"], best["chunk_size"], G2, shuffle=False)
    model = DeepONet(hidden=best["hidden"], p=best["p"]); opt = tf.keras.optimizers.Adam(best["lr"])
    t_start = time.time(); do_train(model, opt, ds_tr, ds_vl, FULL_EPOCHS); train_elapsed = time.time() - t_start
    t_pred = time.time(); Yp_f = predict_full(model, Xbranch, Xb_mmap, Xt, G2, N); pred_elapsed = time.time() - t_pred
    Y_ph = (Yt.reshape(N, G, G, 1) + 1.0) / 2.0 * clip; Yp_ph = (Yp_f.reshape(N, G, G, 1) + 1.0) / 2.0 * clip
    res = {"grid": G, "seed": seed, "cytokine": cytokine, "best_params": best, "optuna_best_val_loss": optuna_val, "train_time_seconds": round(train_elapsed, 2), "pred_time_seconds": round(pred_elapsed, 2),
           "results": {"Near_Horizon": calculate_metrics(Y_ph[160:180], Yp_ph[160:180], masks[160:180], clip), "Far_Horizon": calculate_metrics(Y_ph[180:199], Yp_ph[180:199], masks[180:199], clip)}}
    json.dump(res, open(out_dir / f"res_{cytokine}_{G}_{seed}.json", "w"), indent=4)
    model.save_weights(str(out_dir / f"weights_{cytokine}_{G}_{seed}.weights.h5"))

if __name__ == "__main__":
    ap = argparse.ArgumentParser(); ap.add_argument("--cytokine", required=True); ap.add_argument("--seed", type=int, default=42); ap.add_argument("--data-dir", default=str(PREPROCESSED_200H / "500x500"))
    args = ap.parse_args()
    data_dir = args.data_dir
    run(data_dir, args.cytokine, args.seed)
