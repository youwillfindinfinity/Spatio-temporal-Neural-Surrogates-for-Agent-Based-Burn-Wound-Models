import sys
import argparse, json, gc, os, random, time, warnings
warnings.filterwarnings("ignore")
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from paths import PREPROCESSED, RESULTS_ADDITIONAL
import numpy as np
import tensorflow as tf
from sklearn.metrics import r2_score
from scipy.stats import pearsonr
from skimage.metrics import structural_similarity as ssim
import optuna

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"
optuna.logging.set_verbosity(optuna.logging.WARNING)

GRID        = 500
G2          = GRID * GRID
DATA_DIR    = str(PREPROCESSED / "500x500") # overridden by --data-dir
RESULTS_DIR = str(RESULTS_ADDITIONAL / "pi_deeponet")
CYT_MAP     = {"il8": 0, "il10": 3}

TRUE_SIZE   = 5.0
S_MCS       = 60.0
H_MCS       = 1.0 / S_MCS
CYTOKINE_NAMES = ["il8", "il1", "il6", "il10", "tnf", "tgf"]
MASK_E, MASK_NDN, MASK_NA, MASK_M1, MASK_M2 = 0, 1, 2, 3, 4

N_COLLOC         = 1500   
N_BC             = 150   
N_PHYSICS_SAMPLES = 3
EVAL_CHUNK       = 2048   
N_TRIALS    = 20
TUNE_EPOCHS = 30
FULL_EPOCHS = 400

tf.config.set_visible_devices([], "GPU") 


def set_seed(seed):
    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed); np.random.seed(seed); tf.random.set_seed(seed)

def compute_pde_params(G, cyt_idx):
    areaconv   = TRUE_SIZE**2 / G**2
    volumeconv = TRUE_SIZE**2 / G**2
    D_all = np.array([2.09e-6, 3.00e-7, 8.49e-8, 1.45e-8, 4.07e-9, 2.60e-7]) * S_MCS / areaconv
    k_all = np.array([0.200, 0.600, 0.500, 0.500, 0.500*0.225, 0.500/25.0]) * H_MCS
    sec   = np.array([234e-5, 1.46e-5, 3.024e-5, 225e-5, 250e-5,
                      45e-5, 250e-5, 70e-5, 280e-5]) * volumeconv * H_MCS
    return float(D_all[cyt_idx]), float(k_all[cyt_idx]), sec

def build_source_arrays(masks_dynamic, sec, cyt_idx, G, clip_max):
    T   = masks_dynamic.shape[0]
    n   = G * G
    me  = masks_dynamic[:, :, MASK_E]
    mnn = masks_dynamic[:, :, MASK_NDN]
    mna = masks_dynamic[:, :, MASK_NA]
    mm1 = masks_dynamic[:, :, MASK_M1]
    mm2 = masks_dynamic[:, :, MASK_M2]
    z   = np.zeros((T, n), np.float64)
    s1_map = [sec[0]*me, sec[3]*mna, sec[4]*mm1, sec[5]*mm1, sec[6]*mna, sec[8]*mm2]
    s2_map = [sec[1]*mnn, z, z, z, sec[7]*mm1, z]
    e_map  = [sec[2]*mna, z, z, z, z, z]
    scale = 2.0 / (clip_max + 1e-30)
    s1 = s1_map[cyt_idx].reshape(T, n, 1) * scale
    s2 = s2_map[cyt_idx].reshape(T, n, 1) * scale
    e  = e_map[cyt_idx].reshape(T, n, 1)
    return (tf.constant(s1, tf.float32),
            tf.constant(s2, tf.float32),
            tf.constant(e,  tf.float32))

def corner_thresh(G):
    return 2.0 * (G / 10.0) / G - 1.0

class Branch(tf.keras.layers.Layer):
    def __init__(self, hidden, p, input_dim=15, **kw):
        super().__init__(**kw)
        self.fc1 = tf.keras.layers.Dense(hidden, activation="relu")
        self.fc2 = tf.keras.layers.Dense(hidden, activation="relu")
        self.fc3 = tf.keras.layers.Dense(p, activation="linear")

    def call(self, x, training=False):
        return self.fc3(self.fc2(self.fc1(x)))

class Trunk(tf.keras.layers.Layer):
    def __init__(self, hidden, p, **kw):
        super().__init__(**kw)
        self.U   = tf.keras.layers.Dense(hidden, activation="tanh")
        self.V   = tf.keras.layers.Dense(hidden, activation="tanh")
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


class PIDeepONet(tf.keras.Model):
    def __init__(self, hidden, p, branch_dim=15):
        super().__init__()
        self.branch = Branch(hidden, p, input_dim=branch_dim)
        self.trunk  = Trunk(hidden, p)
        self.bias   = self.add_weight(shape=(1,), initializer="zeros", trainable=True, name="bias")

    def call_data(self, xb, xt, training=False):
        b = self.branch(xb, training=training)
        t = self.trunk(xt)
        r = tf.einsum("bp,bnp->bn", b, t) + self.bias
        return tf.expand_dims(r, -1)

    def call_physics_single(self, xb_single, xyt):
        b = self.branch(xb_single, training=True)
        t = self.trunk(xyt)
        r = tf.reduce_sum(b[0] * t, axis=-1, keepdims=True) + self.bias
        return r

    def call(self, inputs, training=False):
        return self.call_data(inputs[0], inputs[1], training=training)


def build_branch_stats(Xb_mmap, Xt, cyt_idx):
    N = Xb_mmap.shape[0]
    out = np.zeros((N, 15), dtype=np.float32)
    xs = np.linspace(0, 1, GRID, dtype=np.float32)
    ys = np.linspace(0, 1, GRID, dtype=np.float32)
    xx, yy = np.meshgrid(xs, ys, indexing='ij')
    print("Computing branch statistics (mmap'd)...")
    for i in range(N):
        ff0 = Xb_mmap[i, 0, :, :, cyt_idx].astype(np.float32)
        ff1 = Xb_mmap[i, 1, :, :, cyt_idx].astype(np.float32)
        masks = Xb_mmap[i, 0, :, :, 6:].astype(np.float32)
        m = (masks.max(axis=-1) > 0.5)
        na = float(np.sum(m)) + 1e-6
        out[i, 0] = (float(np.max(ff0)) + 1.0) / 2.0
        out[i, 1] = (float(np.mean(ff0)) + 1.0) / 2.0
        out[i, 2] = float(np.std(ff0))
        out[i, 3] = (float(np.max(ff1)) + 1.0) / 2.0
        out[i, 4] = (float(np.mean(ff1)) + 1.0) / 2.0
        out[i, 5] = float(np.std(ff1))
        out[i, 6] = float(np.sum(xx * m) / na)
        out[i, 7] = float(np.sum(yy * m) / na)
        out[i, 8] = na / G2
        for ct in range(5):
            out[i, 9 + ct] = float(np.sum(masks[:, :, ct] > 0.5)) / G2
        out[i, 14] = float(Xt[i, 0, 2])
    return out


def build_dataset(Xbranch, Xtrunk, Yf, batch_size, chunk_size, shuffle=True):
    N, n_pts, _ = Xtrunk.shape
    chunks = list(range(0, n_pts, chunk_size))
    d_branch = Xbranch.shape[1]

    def gen():
        order = np.arange(N)
        if shuffle: np.random.shuffle(order)
        for i in order:
            xb = Xbranch[i]
            for s in chunks:
                e = min(s + chunk_size, n_pts); size = e - s
                xt = Xtrunk[i, s:e]; y = Yf[i, s:e]
                if size < chunk_size:
                    pad = chunk_size - size
                    xt = np.concatenate([xt, np.zeros((pad, 3), np.float32)])
                    y  = np.concatenate([y,  np.zeros((pad, 1), np.float32)])
                yield (xb, xt, np.array([size], dtype=np.int32)), y

    sig = (
        (tf.TensorSpec((d_branch,),            tf.float32),
         tf.TensorSpec((chunk_size, 3),        tf.float32),
         tf.TensorSpec((1,),                   tf.int32)),
        tf.TensorSpec((chunk_size, 1),         tf.float32),
    )
    return (tf.data.Dataset.from_generator(gen, output_signature=sig)
            .batch(batch_size).prefetch(tf.data.AUTOTUNE))


def compute_pde_residual(model, xb_single, xyt, D, k, s1_tf, s2_tf, e_tf, G):
    G_f  = tf.constant(float(G), dtype=tf.float32)
    T_m1 = tf.constant(s1_tf.shape[0] - 1, dtype=tf.int32)
    with tf.GradientTape(persistent=True) as tape2:
        tape2.watch(xyt)
        with tf.GradientTape(persistent=True) as tape1:
            tape1.watch(xyt)
            u = model.call_physics_single(xb_single, xyt)
        du = tape1.gradient(u, xyt)
        u_x = du[:, 0:1]; u_y = du[:, 1:2]; u_t = du[:, 2:3]
    u_xx = tape2.gradient(u_x, xyt)[:, 0:1]
    u_yy = tape2.gradient(u_y, xyt)[:, 1:2]
    del tape1, tape2
    ix = tf.cast(tf.clip_by_value(tf.floor((xyt[:, 0:1] + 1.0) / 2.0 * G_f), 0.0, G_f-1.0), tf.int32)
    iy = tf.cast(tf.clip_by_value(tf.floor((xyt[:, 1:2] + 1.0) / 2.0 * G_f), 0.0, G_f-1.0), tf.int32)
    flat_idx = tf.squeeze(ix * tf.cast(G_f, tf.int32) + iy, axis=1)
    t_norm_idx = tf.cast(tf.round((xyt[:, 2:3] + 1.0) / 2.0 * 198.0), tf.int32)
    pinn_idx   = tf.squeeze(tf.clip_by_value(t_norm_idx - 2, 0, T_m1), axis=1)
    gather_idx = tf.stack([pinn_idx, flat_idx], axis=1)
    s1_q = tf.gather_nd(s1_tf, gather_idx)
    s2_q = tf.gather_nd(s2_tf, gather_idx)
    e_q  = tf.gather_nd(e_tf,  gather_idx)
    rhs = D * (u_xx + u_yy) - k * (u + 1.0) + s1_q + s2_q - e_q * (u + 1.0)
    return u_t - rhs


def compute_bc_residual_mixed(model, xb_single, xyt_neu, xyt_dir):
    neu_res = tf.zeros((0,), tf.float32)
    if xyt_neu.shape[0] > 0:
        xyt_n = tf.constant(xyt_neu)
        with tf.GradientTape() as tape:
            tape.watch(xyt_n)
            u_n = model.call_physics_single(xb_single, xyt_n)
        du_n = tape.gradient(u_n, xyt_n)
        x_c = xyt_neu[:, 0]; y_c = xyt_neu[:, 1]
        on_x = tf.cast(tf.abs(tf.abs(x_c) - 1.0) < 0.02, tf.float32)
        on_y = tf.cast(tf.abs(tf.abs(y_c) - 1.0) < 0.02, tf.float32)
        neu_res = du_n[:, 0] * on_x + du_n[:, 1] * on_y
    dir_res = tf.zeros((0,), tf.float32)
    if xyt_dir.shape[0] > 0:
        dir_res = tf.squeeze(model.call_physics_single(xb_single, tf.constant(xyt_dir)), axis=1)
    return neu_res, dir_res


def sample_collocation(N_c, t_min, t_max):
    xy = np.random.uniform(-1.0, 1.0, (N_c, 2)).astype(np.float32)
    t  = np.random.uniform(t_min, t_max, (N_c, 1)).astype(np.float32)
    return np.hstack([xy, t])


def sample_boundary_mixed(N_bc, t_min, t_max, ct):
    pts = []
    n_per_side = N_bc // 4
    for _ in range(n_per_side):
        t = np.random.uniform(t_min, t_max)
        pts.extend([[-1.0, np.random.uniform(-1,1), t],
                    [ 1.0, np.random.uniform(-1,1), t],
                    [np.random.uniform(-1,1), -1.0, t],
                    [np.random.uniform(-1,1),  1.0, t]])
    pts = np.array(pts, dtype=np.float32)[:N_bc]
    x, y = pts[:, 0], pts[:, 1]
    is_dir = (((np.abs(y+1.0)<0.02)&(x<ct)) | ((np.abs(y-1.0)<0.02)&(x>-ct)) |
              ((np.abs(x-1.0)<0.02)&(y>-ct)) | ((np.abs(x+1.0)<0.02)&(y<ct)))
    return pts[~is_dir], pts[is_dir]


def masked_mse(pred, y, sz):
    idx  = tf.range(tf.shape(pred)[1])[tf.newaxis, :, tf.newaxis]
    mask = tf.cast(idx < tf.cast(sz[:, tf.newaxis, :], tf.int32), tf.float32)
    return tf.reduce_sum(tf.square(pred - y) * mask) / (tf.reduce_sum(mask) + 1e-8)


def train_model(model, opt, ds_tr, ds_vl, Xbranch_tr, t_min, t_max, G, ct,
                D, k, s1_tf, s2_tf, e_tf, lambda_pde, lambda_bc,
                epochs, patience, reduce_patience, min_lr, verbose, physics_every):
    best_val = np.inf; best_w = None; wait = rw = 0
    N_tr = Xbranch_tr.shape[0]

    for ep in range(1, epochs + 1):
        for batch in ds_tr:
            (xb, xt, sz), y = batch
            with tf.GradientTape() as tape:
                loss = masked_mse(model.call_data(xb, xt, training=True), y, sz)
            opt.apply_gradients(zip(tape.gradient(loss, model.trainable_variables),
                                    model.trainable_variables))

        if lambda_pde > 0 and ep % physics_every == 0:
            xyt_c            = sample_collocation(N_COLLOC, t_min, t_max)
            xyt_neu, xyt_dir = sample_boundary_mixed(N_BC, t_min, t_max, ct)
            phys_idxs = np.random.choice(N_tr, size=min(N_PHYSICS_SAMPLES, N_tr), replace=False)
            for pidx in phys_idxs:
                xb_s = tf.constant(Xbranch_tr[pidx:pidx+1])
                xyt_c_tf = tf.constant(xyt_c)
                with tf.GradientTape() as tape:
                    res = compute_pde_residual(model, xb_s, xyt_c_tf, D, k, s1_tf, s2_tf, e_tf, G)
                    neu_r, dir_r = compute_bc_residual_mixed(model, xb_s, xyt_neu, xyt_dir)
                    l_pde = lambda_pde * tf.reduce_mean(tf.square(res))
                    l_bc_n = tf.reduce_mean(tf.square(neu_r)) if tf.size(neu_r) > 0 else 0.0
                    l_bc_d = tf.reduce_mean(tf.square(dir_r)) if tf.size(dir_r) > 0 else 0.0
                    total = l_pde + lambda_bc * (l_bc_n + l_bc_d)
                grads = tape.gradient(total, model.trainable_variables)
                opt.apply_gradients([(g,v) for g,v in zip(grads, model.trainable_variables) if g is not None])

        vl_mean = float(np.mean([float(masked_mse(model.call_data(xb, xt, training=False), y, sz))
                                  for (xb, xt, sz), y in ds_vl]))
        if verbose and ep % 20 == 0:
            print(f"  Ep {ep:4d}  val={vl_mean:.5f}")

        if vl_mean < best_val:
            best_val = vl_mean; best_w = model.get_weights(); wait = rw = 0
        else:
            wait += 1; rw += 1
            if rw >= reduce_patience:
                lr = float(opt.learning_rate)
                new_lr = max(lr * 0.5, min_lr)
                if new_lr != lr: opt.learning_rate.assign(new_lr)
                rw = 0
            if wait >= patience:
                if verbose: print(f"  Early stop @ epoch {ep}")
                break

    if best_w: model.set_weights(best_w)
    return best_val


def predict_full(model, Xbranch, Xtrunk, chunk=EVAL_CHUNK):
    N, n_pts, _ = Xtrunk.shape
    out = np.zeros((N, n_pts, 1), np.float32)
    for i in range(N):
        xb = tf.constant(Xbranch[i:i+1])
        for s in range(0, n_pts, chunk):
            e = min(s + chunk, n_pts)
            xt = tf.constant(Xtrunk[i:i+1, s:e])
            out[i, s:e] = model.call_data(xb, xt, training=False).numpy()[0]
    return out


def _fisher_z(r):
    r = np.clip(r, -0.9999, 0.9999)
    return 0.5 * np.log((1.0 + r) / (1.0 - r))

def calculate_metrics(y_true, y_pred, masks, clip_max):
    T = min(y_true.shape[0], y_pred.shape[0], masks.shape[0])
    yt = y_true[:T]; yp = np.maximum(y_pred[:T], 0.0)
    ms = np.max(masks[:T], axis=-1, keepdims=True)
    rmse = float(np.sqrt(np.sum(np.square(yt-yp)*ms) / (np.sum(ms)+1e-12)))
    unmasked_rmse = float(np.sqrt(np.mean(np.square(yt-yp))))
    r2   = float(r2_score(yt.flatten(), yp.flatten()))
    dice_thr = 0.05 * clip_max if clip_max > 0 else 1e-9
    dices=[]; n_empty=0; z_corrs=[]; ssims_v=[]; n_ssim_skip=0
    fixed_dr = float(clip_max) if clip_max > 0 else 1.0
    for t in range(T):
        gt=yt[t,:,:,0]; pr=yp[t,:,:,0]
        gb=(gt>dice_thr).astype(float); pb=(pr>dice_thr).astype(float)
        if np.sum(gb)+np.sum(pb)==0: n_empty+=1
        else: dices.append((2.*np.sum(gb*pb))/(np.sum(gb)+np.sum(pb)+1e-12))
        if np.std(gt)>1e-12 and np.std(pr)>1e-12:
            r_val=float(pearsonr(gt.flatten(),pr.flatten())[0])
            if np.isfinite(r_val): z_corrs.append(_fisher_z(r_val))
        if float(np.max(gt)-np.min(gt))>1e-12: ssims_v.append(float(ssim(gt,pr,data_range=fixed_dr)))
        else: n_ssim_skip+=1
    return {
        "Global_R2":           r2,
        "Masked_RMSE":         rmse,
        "Unmasked_RMSE":       unmasked_rmse,
        "Avg_Dice":            float(np.mean(dices)) if dices else 0.0,
        "Spatial_Correlation": float(np.tanh(np.mean(z_corrs))) if z_corrs else 0.0,
        "SSIM":                float(np.mean(ssims_v)) if ssims_v else 0.0,
    }

def denormalize(x, clip_max):
    return (np.asarray(x, np.float64)+1.0)/2.0*clip_max


def run(cyt_name, seed):
    print(f"\n{'='*60}")
    print(f"PI-DeepONet 500x500 | cytokine={cyt_name} | seed={seed}")
    print(f"{'='*60}")
    set_seed(seed)
    cyt_idx = CYT_MAP[cyt_name]
    os.makedirs(RESULTS_DIR, exist_ok=True)

    md      = json.load(open(f"{DATA_DIR}/metadata.json"))
    clip    = float(md["clip_max"][cyt_idx])

    Xb_mmap = np.load(f"{DATA_DIR}/X_branch.npy", mmap_mode="r")
    Xt      = np.load(f"{DATA_DIR}/X_trunk.npy").astype(np.float32)
    Y       = np.load(f"{DATA_DIR}/Y_target.npy").astype(np.float32)[..., cyt_idx:cyt_idx+1]
    M       = np.load(f"{DATA_DIR}/Y_masks_spatial.npy").astype(np.float32)
    M_pinn  = np.load(f"{DATA_DIR}/Y_masks_pinn.npy").astype(np.float64)

    N = Xb_mmap.shape[0]; Yf = Y.reshape(N, G2, 1)
    Xbranch = build_branch_stats(Xb_mmap, Xt, cyt_idx)   # (N, 15)
    _tr_mean = Xbranch[:140].mean(axis=0, keepdims=True)
    _tr_std  = Xbranch[:140].std(axis=0, keepdims=True) + 1e-8
    Xbranch  = ((Xbranch - _tr_mean) / _tr_std).astype(np.float32)
    Xtrunk   = Xt[:, :, :3]

    D, k, sec = compute_pde_params(GRID, cyt_idx)
    ct = corner_thresh(GRID)
    s1_tf, s2_tf, e_tf = build_source_arrays(M_pinn[:140], sec, cyt_idx, GRID, clip)

    t_norm_all = np.linspace(-1.0, 1.0, 201, dtype=np.float32)
    t_min = float(t_norm_all[2]); t_max = float(t_norm_all[-1])

    Xbr_tr, Xtr_tr, Yf_tr = Xbranch[:140],    Xtrunk[:140],    Yf[:140]
    Xbr_vl, Xtr_vl, Yf_vl = Xbranch[140:160], Xtrunk[140:160], Yf[140:160]
    d_branch = Xbranch.shape[1]

    def make_obj():
        def objective(trial):
            set_seed(seed); tf.keras.backend.clear_session()
            p          = trial.suggest_categorical("p",          [64, 128])
            hidden     = trial.suggest_categorical("hidden",     [128, 256])
            lr         = trial.suggest_float("learning_rate",    1e-5, 1e-3, log=True)
            bs         = trial.suggest_categorical("batch_size", [4, 8])
            chunk_size = trial.suggest_categorical("chunk_size", [1024, 2048])
            lambda_pde = trial.suggest_float("lambda_pde",      1e-4, 1.0,  log=True)
            lambda_bc  = trial.suggest_float("lambda_bc",       1e-4, 0.1,  log=True)
            ds_tr = build_dataset(Xbr_tr, Xtr_tr, Yf_tr, bs, chunk_size, shuffle=True)
            ds_vl = build_dataset(Xbr_vl, Xtr_vl, Yf_vl, bs, chunk_size, shuffle=False)
            model = PIDeepONet(hidden=hidden, p=p, branch_dim=d_branch)
            opt   = tf.keras.optimizers.Adam(lr)
            best  = train_model(model, opt, ds_tr, ds_vl, Xbr_tr, t_min, t_max, GRID, ct,
                                D, k, s1_tf, s2_tf, e_tf,
                                lambda_pde=lambda_pde, lambda_bc=lambda_bc,
                                epochs=TUNE_EPOCHS, patience=8, reduce_patience=5,
                                min_lr=1e-7, verbose=False, physics_every=2)
            return float(best)
        return objective

    study = optuna.create_study(direction="minimize",
                                sampler=optuna.samplers.TPESampler(seed=seed),
                                pruner=optuna.pruners.MedianPruner(n_warmup_steps=5))
    study.optimize(make_obj(), n_trials=N_TRIALS, show_progress_bar=True, catch=(Exception,))
    best = study.best_params
    print(f"  Best: {best}  |  val_loss = {study.best_value:.6f}")

    tf.keras.backend.clear_session(); set_seed(seed)
    ds_tr = build_dataset(Xbr_tr, Xtr_tr, Yf_tr, best["batch_size"], best["chunk_size"], shuffle=True)
    ds_vl = build_dataset(Xbr_vl, Xtr_vl, Yf_vl, best["batch_size"], best["chunk_size"], shuffle=False)
    model = PIDeepONet(hidden=best["hidden"], p=best["p"], branch_dim=d_branch)
    opt   = tf.keras.optimizers.Adam(best["learning_rate"])

    t_start = time.time()
    train_model(model, opt, ds_tr, ds_vl, Xbr_tr, t_min, t_max, GRID, ct,
                D, k, s1_tf, s2_tf, e_tf,
                lambda_pde=best["lambda_pde"], lambda_bc=best["lambda_bc"],
                epochs=FULL_EPOCHS, patience=40, reduce_patience=15,
                min_lr=1e-7, verbose=True, physics_every=2)
    train_elapsed = time.time() - t_start

    t_pred_start = time.time()
    Yp_flat = predict_full(model, Xbranch, Xtrunk)
    pred_elapsed = time.time() - t_pred_start
    Yp      = Yp_flat.reshape(N, GRID, GRID, 1)
    Y_phys  = denormalize(Y.reshape(N, GRID, GRID, 1), clip)
    Yp_phys = denormalize(Yp, clip)

    suffix  = f"{cyt_name}_500_{seed}"
    results = {
        "grid": GRID, "seed": seed, "cytokine": cyt_name,
        "best_params": best,
        "optuna_best_val_loss": float(study.best_value),
        "train_time_seconds":   round(train_elapsed, 2),
        "pred_time_seconds":    round(pred_elapsed, 4),
        "results": {
            "Near_Horizon_t82_t91": calculate_metrics(Y_phys[80:90], Yp_phys[80:90], M[80:90], clip),
            "Far_Horizon_t92_t100": calculate_metrics(Y_phys[90:99], Yp_phys[90:99], M[90:99], clip),
        },
    }
    out_path = f"{RESULTS_DIR}/res_{suffix}.json"
    with open(out_path, "w") as f:
        json.dump(results, f, indent=4)
    print(f"DONE → {out_path}")
    del model; gc.collect()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--cytokine", choices=["il8","il10"], required=True)
    parser.add_argument("--seed",     type=int,               required=True)
    parser.add_argument("--data-dir", default=str(PREPROCESSED / "500x500"),
                        dest="data_dir",
                        help="Path to preprocessed/500x500 directory")
    args = parser.parse_args()
    DATA_DIR = args.data_dir
    run(args.cytokine, args.seed)
