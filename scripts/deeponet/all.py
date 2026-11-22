import sys
import os, json, argparse, random, time
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from paths import PREPROCESSED, RESULTS_MODELS

import numpy as np
import tensorflow as tf
import optuna
from sklearn.metrics import r2_score
from scipy.stats import pearsonr
from skimage.metrics import structural_similarity as ssim

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"
optuna.logging.set_verbosity(optuna.logging.WARNING)

N_TRIALS    = 20
TUNE_EPOCHS = 30
FULL_EPOCHS = 400
EVAL_CHUNK  = 4096  


def set_seed(seed):
    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed); np.random.seed(seed); tf.random.set_seed(seed)

class Branch(tf.keras.layers.Layer):
    def __init__(self, hidden, p, **kw):
        super().__init__(**kw)
        self.fc1 = tf.keras.layers.Dense(hidden, activation="relu")
        self.fc2 = tf.keras.layers.Dense(p,      activation="linear")

    def call(self, x, training=False):
        return self.fc2(self.fc1(x))


class Trunk(tf.keras.layers.Layer):
    def __init__(self, hidden, p, **kw):
        super().__init__(**kw)
        self.U   = tf.keras.layers.Dense(hidden, activation="tanh")
        self.V   = tf.keras.layers.Dense(hidden, activation="tanh")
        self.W1a = tf.keras.layers.Dense(hidden, activation="relu")
        self.W1b = tf.keras.layers.Dense(hidden, activation="linear")
        self.W2a = tf.keras.layers.Dense(hidden, activation="relu")
        self.W2b = tf.keras.layers.Dense(hidden, activation="linear")
        self.out = tf.keras.layers.Dense(p,      activation="linear")

    def call(self, x):
        u = self.U(x); v = self.V(x)
        h = self.W1b(self.W1a(x));  h = h * u + (1.0 - h) * v
        h = self.W2b(self.W2a(h));  h = h * u + (1.0 - h) * v
        return self.out(h)

class DeepONet(tf.keras.Model):
    def __init__(self, hidden, p):
        super().__init__()
        self.branch = Branch(hidden, p)
        self.trunk  = Trunk(hidden, p)
        self.bias   = self.add_weight(shape=(1,), initializer="zeros",
                                      trainable=True, name="bias")

    def call(self, inputs, training=False):
        xb, xt = inputs
        b = self.branch(xb, training=training) # (batch,p)
        t = self.trunk(xt) # (batch, n_pts, p)
        r = tf.einsum("bp,bnp->bn", b, t) + self.bias # (batch, n_pts)
        return tf.expand_dims(r, -1) # (batch, n_pts, 1)

def build_branch_inputs(Xb, Xt, cyt_idx):
    N, _, G, _, _ = Xb.shape
    f0   = Xb[:, 0, :, :, cyt_idx] # (N, G, G) cytokine frame 0
    mask = (Xb[:, 0, :, :, 6:].max(axis=-1) > 0.5).astype(np.float32) # (N,G,G)

    xs = np.linspace(0, 1, G, dtype=np.float32)
    ys = np.linspace(0, 1, G, dtype=np.float32)
    xx, yy = np.meshgrid(xs, ys, indexing='ij')

    out = np.zeros((N, 7), dtype=np.float32)
    for i in range(N):
        f  = f0[i]; m = mask[i]; na = float(np.sum(m)) + 1e-6
        out[i, 0] = (float(np.max(f))  + 1.0) / 2.0 # max_f0   → [0,1]
        out[i, 1] = (float(np.mean(f)) + 1.0) / 2.0 # mean_f0  → [0,1]
        out[i, 2] = float(np.std(f)) # std_f0   ∈ [0,1]
        out[i, 3] = float(np.sum(xx * m) / na) # centroid_x
        out[i, 4] = float(np.sum(yy * m) / na) # centroid_y
        out[i, 5] = na / (G * G) # extent
        out[i, 6] = float(Xt[i, 0, 2]) # t_norm (same for all pts)
    return out


def build_trunk_inputs(Xb, Xt):
    N, _, G, _, C = Xb.shape
    vals = Xb.transpose(0, 2, 3, 1, 4).reshape(N, G*G, 22).astype(np.float32)
    xy   = Xt[:, :, :2].astype(np.float32) # (N, G*G, 2)
    return np.concatenate([xy, vals], axis=-1) # (N, G*G, 24)

def build_dataset(Xbranch, Xtrunk, Yf, batch_size, chunk_size, shuffle=True):
    N, n_pts, _ = Xtrunk.shape
    chunks = list(range(0, n_pts, chunk_size))

    def gen():
        order = np.arange(N)
        if shuffle:
            np.random.shuffle(order)
        for i in order:
            xb = Xbranch[i]                             
            for s in chunks:
                e    = min(s + chunk_size, n_pts)
                size = e - s
                xt = Xtrunk[i, s:e]                   
                y  = Yf[i, s:e]                        
                if size < chunk_size:
                    pad = chunk_size - size
                    xt = np.concatenate([xt, np.zeros((pad, 24), np.float32)], axis=0)
                    y  = np.concatenate([y,  np.zeros((pad, 1),  np.float32)], axis=0)
                sz = np.array([size], dtype=np.int32)
                yield (xb, xt, sz), y

    sig = (
        (tf.TensorSpec((7,),              tf.float32),
         tf.TensorSpec((chunk_size, 24),  tf.float32),
         tf.TensorSpec((1,),              tf.int32)),
        tf.TensorSpec((chunk_size, 1),    tf.float32),
    )
    return (tf.data.Dataset.from_generator(gen, output_signature=sig)
            .batch(batch_size).prefetch(tf.data.AUTOTUNE))

def masked_mse(pred, y, sz):
    idx  = tf.range(tf.shape(pred)[1])[tf.newaxis, :, tf.newaxis]
    mask = tf.cast(idx < tf.cast(sz[:, tf.newaxis, :], tf.int32), tf.float32)
    return tf.reduce_sum(tf.square(pred - y) * mask) / (tf.reduce_sum(mask) + 1e-8)

def train_step(model, opt, xb, xt, sz, y):
    with tf.GradientTape() as tape:
        loss = masked_mse(model([xb, xt], training=True), y, sz)
    opt.apply_gradients(zip(tape.gradient(loss, model.trainable_variables),
                            model.trainable_variables))
    return float(loss)

def val_step(model, xb, xt, sz, y):
    return float(masked_mse(model([xb, xt], training=False), y, sz))

def train_model(model, opt, ds_tr, ds_vl,
                epochs, patience=40, reduce_patience=15, min_lr=1e-7,
                verbose=True):
    best_val = np.inf; best_w = None; wait = rw = 0
    for ep in range(1, epochs + 1):
        tr = np.mean([train_step(model, opt, *b[0], b[1]) for b in ds_tr])
        vl = np.mean([val_step(model, *b[0], b[1]) for b in ds_vl])
        if verbose and ep % 20 == 0:
            print(f"  Epoch {ep:4d}  loss={tr:.5f}  val={vl:.5f}")
        if vl < best_val:
            best_val = vl; best_w = model.get_weights(); wait = rw = 0
        else:
            wait += 1; rw += 1
        if rw >= reduce_patience:
            lr = float(opt.learning_rate)
            new_lr = max(lr * 0.5, min_lr)
            if new_lr != lr:
                opt.learning_rate.assign(new_lr)
                if verbose: print(f"  LR → {new_lr:.2e}")
            rw = 0
        if wait >= patience:
            if verbose: print(f"  Early stop @ epoch {ep}")
            break
    if best_w:
        model.set_weights(best_w)
    return best_val


def predict_full(model, Xbranch, Xtrunk, chunk=EVAL_CHUNK):
    N, n_pts, _ = Xtrunk.shape
    out = np.zeros((N, n_pts, 1), np.float32)
    for i in range(N):
        xb = tf.constant(Xbranch[i:i+1]) # (1, 7)
        for s in range(0, n_pts, chunk):
            e = min(s + chunk, n_pts)
            xt = tf.constant(Xtrunk[i:i+1, s:e]) # (1, e-s, 24)
            out[i, s:e] = model([xb, xt], training=False).numpy()[0]
    return out



def _fisher_z(r):
    r = np.clip(r, -0.9999, 0.9999)
    return 0.5 * np.log((1.0 + r) / (1.0 - r))

def _inv_fisher_z(z):
    return float(np.tanh(z))

def calculate_metrics(y_true, y_pred, masks, clip_max):
    T = min(y_true.shape[0], y_pred.shape[0], masks.shape[0])
    yt = y_true[:T]; yp = np.maximum(y_pred[:T], 0.0)
    ms = np.max(masks[:T], axis=-1, keepdims=True)

    sq_diff = np.square(yt - yp)
    rmse = float(np.sqrt(np.sum(sq_diff * ms) / (np.sum(ms) + 1e-12)))
    unmasked_rmse = float(np.sqrt(np.mean(sq_diff)))
    r2   = float(r2_score(yt.flatten(), yp.flatten()))

    per_t_r2 = []
    for t in range(T):
        gt_f = yt[t].flatten(); pr_f = yp[t].flatten()
        per_t_r2.append(float(r2_score(gt_f, pr_f)) if np.std(gt_f) > 1e-12 else np.nan)

    dice_thr = 0.05 * clip_max if clip_max > 0 else 1e-9
    dices = []; n_empty = 0
    z_corrs = []
    ssims_v = []; n_ssim_skip = 0
    fixed_dr = float(clip_max) if clip_max > 0 else 1.0

    for t in range(T):
        gt = yt[t,:,:,0]; pr = yp[t,:,:,0]
        gb = (gt > dice_thr).astype(float); pb = (pr > dice_thr).astype(float)
        if np.sum(gb) + np.sum(pb) == 0:
            n_empty += 1
        else:
            dices.append((2.0 * np.sum(gb * pb)) / (np.sum(gb) + np.sum(pb) + 1e-12))
        if np.std(gt) > 1e-12 and np.std(pr) > 1e-12:
            r_val = float(pearsonr(gt.flatten(), pr.flatten())[0])
            if np.isfinite(r_val):
                z_corrs.append(_fisher_z(r_val))
        dr = float(np.max(gt) - np.min(gt))
        if dr > 1e-12:
            ssims_v.append(float(ssim(gt, pr, data_range=fixed_dr)))
        else:
            n_ssim_skip += 1

    return {
        "Global_R2":           r2,
        "Per_Timestep_R2":     per_t_r2,
        "Masked_RMSE":         rmse,
        "Unmasked_RMSE":       unmasked_rmse,
        "Avg_Dice":            float(np.mean(dices)) if dices else 0.0,
        "Dice_Empty_Skipped":  n_empty,
        "Spatial_Correlation": _inv_fisher_z(float(np.mean(z_corrs))) if z_corrs else 0.0,
        "SSIM":                float(np.mean(ssims_v)) if ssims_v else 0.0,
        "SSIM_Skipped_Frames": n_ssim_skip,
    }

def denormalize(x, clip_max):
    return (np.asarray(x, np.float64)+1.0)/2.0*clip_max

def make_objective(Xbr_tr, Xtr_tr, Yf_tr,
                   Xbr_vl, Xtr_vl, Yf_vl, seed):
    def objective(trial):
        set_seed(seed)
        tf.keras.backend.clear_session()
        p          = trial.suggest_categorical("p",          [64, 128, 256])
        hidden     = trial.suggest_categorical("hidden",     [128, 256])
        lr         = trial.suggest_float("learning_rate",    1e-5, 1e-3, log=True)
        bs         = trial.suggest_categorical("batch_size", [4, 8])
        chunk_size = trial.suggest_categorical("chunk_size", [2048, 4096])

        ds_tr = build_dataset(Xbr_tr, Xtr_tr, Yf_tr, bs, chunk_size, shuffle=True)
        ds_vl = build_dataset(Xbr_vl, Xtr_vl, Yf_vl, bs, chunk_size, shuffle=False)

        model = DeepONet(hidden=hidden, p=p)
        opt   = tf.keras.optimizers.Adam(lr)
        best  = train_model(model, opt, ds_tr, ds_vl,
                            epochs=TUNE_EPOCHS, patience=8,
                            reduce_patience=5, verbose=False)
        return float(best)
    return objective

def run_pipeline(grid, seed, cytokine):
    set_seed(seed)
    cyt_names = ["il8", "il1", "il6", "il10", "tnf", "tgf"]
    idx = cyt_names.index(cytokine.lower())

    data_path = PREPROCESSED / f"{grid}x{grid}"
    out_dir   = RESULTS_MODELS / "deeponet_h"; out_dir.mkdir(parents=True, exist_ok=True)

    print(f"\n[{cytokine.upper()}] {grid}x{grid} — loading data...")
    Xb = np.load(data_path/"X_branch.npy").astype(np.float32)
    Xt = np.load(data_path/"X_trunk.npy").astype(np.float32)
    Y  = np.load(data_path/"Y_target.npy").astype(np.float32)[..., idx:idx+1]
    M  = np.load(data_path/"Y_masks_spatial.npy").astype(np.float32)

    with open(data_path/"metadata.json") as f:
        meta = json.load(f)
    clip_max = float(meta["scaling"]["max"][idx])

    N = Xb.shape[0]; G2 = Xt.shape[1]; G = int(round(G2**0.5))
    Yf = Y.reshape(N, G2, 1)

    Xbranch = build_branch_inputs(Xb, Xt, idx)  
    Xtrunk  = build_trunk_inputs(Xb, Xt)   

    print(f"  Branch input: (N, 7) scalars  |  Trunk input: (N, {G2}, 24)")
    print(f"  Full grid per epoch: {G2} pts × {N} samples - no subsampling")

    # 70/10/20 split
    Xbr_tr, Xtr_tr, Yf_tr = Xbranch[:70],   Xtrunk[:70],   Yf[:70]
    Xbr_vl, Xtr_vl, Yf_vl = Xbranch[70:80], Xtrunk[70:80], Yf[70:80]

    if seed == 42:
        print(f"Optuna: {N_TRIALS} trials × {TUNE_EPOCHS} epochs...")
        study = optuna.create_study(
            direction="minimize",
            sampler=optuna.samplers.TPESampler(seed=42),
            pruner=optuna.pruners.MedianPruner(n_warmup_steps=5),
        )
        study.optimize(
            make_objective(Xbr_tr, Xtr_tr, Yf_tr, Xbr_vl, Xtr_vl, Yf_vl, 42),
            n_trials=N_TRIALS, show_progress_bar=True, catch=(Exception,),
        )
        best = study.best_params
        optuna_val = float(study.best_value)
        print(f"  Best: {best}  |  val_loss = {optuna_val:.6f}")
    else:
        ref_path = out_dir / f"res_{cytokine}_{grid}_42.json"
        print(f"  Loading HP from {ref_path.name}")
        with open(ref_path) as f:
            ref = json.load(f)
        best = ref["best_params"]
        optuna_val = ref["optuna_best_val_loss"]

    tf.keras.backend.clear_session(); set_seed(seed)

    t_start = time.time()

    ds_tr = build_dataset(Xbr_tr, Xtr_tr, Yf_tr,
                          best["batch_size"], best["chunk_size"], shuffle=True)
    ds_vl = build_dataset(Xbr_vl, Xtr_vl, Yf_vl,
                          best["batch_size"], best["chunk_size"], shuffle=False)

    model = DeepONet(hidden=best["hidden"], p=best["p"])
    opt   = tf.keras.optimizers.Adam(best["learning_rate"])
    print(f"Final training [{cytokine.upper()}] {grid}x{grid}  (max {FULL_EPOCHS} epochs)...")
    train_model(model, opt, ds_tr, ds_vl,
                epochs=FULL_EPOCHS, patience=40, reduce_patience=15, verbose=True)

    Yp_flat = predict_full(model, Xbranch, Xtrunk)
    Yp      = Yp_flat.reshape(N, G, G, 1)
    Y_phys  = denormalize(Y.reshape(N, G, G, 1), clip_max)
    Yp_phys = denormalize(Yp, clip_max)

    suffix  = f"{cytokine}_{grid}_{seed}"
    train_elapsed = time.time() - t_start
    print(f"  Training + prediction time: {train_elapsed:.1f}s")

    results = {
        "grid": grid, "seed": seed, "cytokine": cytokine,
        "best_params":          best,
        "optuna_best_val_loss": optuna_val,
        "train_time_seconds":   round(train_elapsed, 2),
        "results": {
            "Near_Horizon_t82_t91": calculate_metrics(
                Y_phys[80:90], Yp_phys[80:90], M[80:90], clip_max),
            "Far_Horizon_t92_t100": calculate_metrics(
                Y_phys[90:99], Yp_phys[90:99], M[90:99], clip_max),
        },
    }
    with open(out_dir/f"res_{suffix}.json", "w") as f:
        json.dump(results, f, indent=4)
    model.save_weights(out_dir/f"weights_{suffix}.weights.h5")
    print(f"DONE → models/deeponet_h/res_{suffix}.json")

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--grid",     type=int, default=None)
    ap.add_argument("--cytokine", type=str, required=True)
    ap.add_argument("--seed",     type=int, default=42)
    args = ap.parse_args()

    if args.grid:
        run_pipeline(args.grid, args.seed, args.cytokine)
    else:
        for d in sorted(PREPROCESSED.iterdir()):
            if d.is_dir():
                run_pipeline(int(d.name.split("x")[0]), args.seed, args.cytokine)