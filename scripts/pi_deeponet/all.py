import sys
import os, json, argparse, random, warnings, gc, time
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
warnings.filterwarnings("ignore")
optuna.logging.set_verbosity(optuna.logging.WARNING)

N_TRIALS    = 20
TUNE_EPOCHS = 30
FULL_EPOCHS = 400
EVAL_CHUNK  = 4096

TRUE_SIZE   = 5.0
S_MCS       = 60.0
H_MCS       = 1.0 / S_MCS
CYTOKINE_NAMES = ["il8", "il1", "il6", "il10", "tnf", "tgf"]
MASK_E, MASK_NDN, MASK_NA, MASK_M1, MASK_M2 = 0, 1, 2, 3, 4

N_COLLOC         = 2000
N_BC             = 200
N_IC             = 500 # IC points sampled per physics step
N_PHYSICS_SAMPLES = 3

def set_seed(seed):
    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed); np.random.seed(seed); tf.random.set_seed(seed)

# PDE params 
def compute_pde_params(G, cyt_idx):
    areaconv   = TRUE_SIZE**2 / G**2
    volumeconv = TRUE_SIZE**2 / G**2
    D_all = np.array([
        2.09e-6, 3.00e-7, 8.49e-8, 1.45e-8, 4.07e-9, 2.60e-7
    ]) * S_MCS / areaconv
    k_all = np.array([
        0.200, 0.600, 0.500, 0.500, 0.500 * 0.225, 0.500 / 25.0
    ]) * H_MCS
    sec = np.array([
        234e-5, 1.46e-5, 3.024e-5, 225e-5, 250e-5,
        45e-5, 250e-5, 70e-5, 280e-5,
    ]) * volumeconv * H_MCS
    return float(D_all[cyt_idx]), float(k_all[cyt_idx]), sec


def build_source_arrays(masks_flat, sec, cyt_idx, G, clip_max):
    n   = G * G
    me  = masks_flat[:, MASK_E]
    mnn = masks_flat[:, MASK_NDN]
    mna = masks_flat[:, MASK_NA]
    mm1 = masks_flat[:, MASK_M1]
    mm2 = masks_flat[:, MASK_M2]
    z   = np.zeros(n, np.float64)

    s1_map = [sec[0]*me, sec[3]*mna, sec[4]*mm1, sec[5]*mm2, sec[6]*mna, sec[8]*mm2]
    s2_map = [sec[1]*mnn, z, z, z, sec[7]*mm1, z]
    e_map  = [sec[2]*mna, z, z, z, z, z]

    scale = 2.0 / (clip_max + 1e-30)
    s1 = s1_map[cyt_idx].reshape(-1, 1) * scale
    s2 = s2_map[cyt_idx].reshape(-1, 1) * scale
    e  = e_map[cyt_idx].reshape(-1, 1)
    return (tf.constant(s1, tf.float32),
            tf.constant(s2, tf.float32),
            tf.constant(e,  tf.float32))


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


class PIDeepONet(tf.keras.Model):
    def __init__(self, hidden, p):
        super().__init__()
        self.branch = Branch(hidden, p)
        self.trunk  = Trunk(hidden, p)
        self.bias   = self.add_weight(shape=(1,), initializer="zeros",
                                      trainable=True, name="bias")
        self._p = p

    def call_data(self, xb, xt, training=False):
        """data forward: xb (batch, 7), xt (batch, n_pts, D_trunk) -> (batch, n_pts, 1)"""
        b = self.branch(xb, training=training)
        t = self.trunk(xt)
        r = tf.einsum("bp,bnp->bn", b, t) + self.bias
        return tf.expand_dims(r, -1)

    def call_physics_single(self, xb_single, xyt):
        """
        Physics forward for single branch input.
        xb: (1, 7), xyt: (N_c, 3) - only (x,y,t) for autodiff.
        Trunk expects 24D input (xy + 22 field values), so we pad with zeros.
        Gradients flow only through the first 3 columns (x, y, t).
        """
        b = self.branch(xb_single, training=True)
        n = tf.shape(xyt)[0]
        padding = tf.zeros([n, 21], dtype=xyt.dtype)  # 3 + 21 = 24
        xyt_padded = tf.concat([xyt, padding], axis=1)  # (N_c, 24)
        t = self.trunk(xyt_padded)
        r = tf.reduce_sum(b[0] * t, axis=-1, keepdims=True) + self.bias
        return r

    def call(self, inputs, training=False):
        return self.call_data(inputs[0], inputs[1], training=training)


# data prep
def build_branch_inputs(Xb, Xt, cyt_idx):
    N, _, G, _, _ = Xb.shape
    f0   = Xb[:, 0, :, :, cyt_idx]
    mask = (Xb[:, 0, :, :, 6:].max(axis=-1) > 0.5).astype(np.float32)

    xs = np.linspace(0, 1, G, dtype=np.float32)
    ys = np.linspace(0, 1, G, dtype=np.float32)
    xx, yy = np.meshgrid(xs, ys, indexing='ij')

    out = np.zeros((N, 7), dtype=np.float32)
    for i in range(N):
        f = f0[i]; m = mask[i]; na = float(np.sum(m)) + 1e-6
        out[i, 0] = (float(np.max(f))  + 1.0) / 2.0
        out[i, 1] = (float(np.mean(f)) + 1.0) / 2.0
        out[i, 2] = float(np.std(f))
        out[i, 3] = float(np.sum(xx * m) / na)
        out[i, 4] = float(np.sum(yy * m) / na)
        out[i, 5] = na / (G * G)
        out[i, 6] = float(Xt[i, 0, 2])
    return out


def build_trunk_inputs(Xb, Xt):
    N, _, G, _, C = Xb.shape
    vals = Xb.transpose(0, 2, 3, 1, 4).reshape(N, G*G, 22).astype(np.float32)
    xy   = Xt[:, :, :2].astype(np.float32)
    return np.concatenate([xy, vals], axis=-1)


def build_dataset(Xbranch, Xtrunk, Yf, batch_size, chunk_size, shuffle=True):
    N, n_pts, D_trunk = Xtrunk.shape
    chunks = list(range(0, n_pts, chunk_size))

    def gen():
        order = np.arange(N)
        if shuffle:
            np.random.shuffle(order)
        for i in order:
            xb = Xbranch[i]
            for s in chunks:
                e = min(s + chunk_size, n_pts); size = e - s
                xt = Xtrunk[i, s:e]; y = Yf[i, s:e]
                if size < chunk_size:
                    pad = chunk_size - size
                    xt = np.concatenate([xt, np.zeros((pad, D_trunk), np.float32)])
                    y  = np.concatenate([y,  np.zeros((pad, 1), np.float32)])
                sz = np.array([size], dtype=np.int32)
                yield (xb, xt, sz), y

    sig = (
        (tf.TensorSpec((7,),               tf.float32),
         tf.TensorSpec((chunk_size, D_trunk), tf.float32),
         tf.TensorSpec((1,),               tf.int32)),
        tf.TensorSpec((chunk_size, 1),     tf.float32),
    )
    return (tf.data.Dataset.from_generator(gen, output_signature=sig)
            .batch(batch_size).prefetch(tf.data.AUTOTUNE))


# physics loss 
def compute_pde_residual(model, xb_single, xyt, D, k, s1_tf, s2_tf, e_tf, G):
    """
    PDE residual: ∂u/∂t = D·Δu - k·u + s1 + s2 - e·u
    Same simplified form as PINN (no +1 offset).
    xyt: (N_c, 3) collocation points in [-1,1]² * [t_min, t_max]
    """
    G_f = tf.constant(float(G), dtype=tf.float32)

    with tf.GradientTape(persistent=True) as tape2:
        tape2.watch(xyt)
        with tf.GradientTape(persistent=True) as tape1:
            tape1.watch(xyt)
            u = model.call_physics_single(xb_single, xyt)
        du = tape1.gradient(u, xyt)
        u_x, u_y, u_t = du[:, 0:1], du[:, 1:2], du[:, 2:3]

    du_x_grad = tape2.gradient(u_x, xyt)
    du_y_grad = tape2.gradient(u_y, xyt)
    u_xx = du_x_grad[:, 0:1]
    u_yy = du_y_grad[:, 1:2]
    del tape1, tape2

    ix = tf.cast(tf.clip_by_value(
        tf.floor((xyt[:, 0:1] + 1.0) / 2.0 * G_f), 0.0, G_f - 1.0), tf.int32)
    iy = tf.cast(tf.clip_by_value(
        tf.floor((xyt[:, 1:2] + 1.0) / 2.0 * G_f), 0.0, G_f - 1.0), tf.int32)
    flat_idx = tf.squeeze(ix * tf.cast(G_f, tf.int32) + iy, axis=1)

    s1_q = tf.gather(s1_tf, flat_idx)
    s2_q = tf.gather(s2_tf, flat_idx)
    e_q  = tf.gather(e_tf,  flat_idx)

    rhs = D * (u_xx + u_yy) - k * u + s1_q + s2_q - e_q * u
    return u_t - rhs


def compute_bc_residual(model, xb_single, xyt_bc):
    """Neumann BC: ∂u/∂n = 0 on all boundaries."""
    if xyt_bc.shape[0] == 0:
        return tf.zeros((0,), tf.float32)
    xyt_b = tf.constant(xyt_bc)
    with tf.GradientTape() as tape:
        tape.watch(xyt_b)
        u = model.call_physics_single(xb_single, xyt_b)
    du = tape.gradient(u, xyt_b)
    x_c, y_c = xyt_bc[:, 0], xyt_bc[:, 1]
    on_x = tf.cast(tf.abs(tf.abs(x_c) - 1.0) < 0.02, tf.float32)
    on_y = tf.cast(tf.abs(tf.abs(y_c) - 1.0) < 0.02, tf.float32)
    return du[:, 0] * on_x + du[:, 1] * on_y


def compute_ic_residual(model, xb_single, xy_ic, y_ic_true, t_min):
    """
    IC loss: u(x, y, t_min) = u_0(x, y) at sampled spatial points.
    xy_ic:(N_ic, 2) spatial coordinates
    y_ic_true:(N_ic, 1) target IC values (scaled)
    """
    t_col = np.full((xy_ic.shape[0], 1), t_min, dtype=np.float32)
    xyt_ic = tf.constant(np.hstack([xy_ic, t_col]))
    u_pred = model.call_physics_single(xb_single, xyt_ic)  # (N_ic, 1)
    return u_pred - tf.constant(y_ic_true, dtype=tf.float32)


def sample_collocation(N_c, t_min, t_max):
    xy = np.random.uniform(-1.0, 1.0, (N_c, 2)).astype(np.float32)
    t  = np.random.uniform(t_min, t_max, (N_c, 1)).astype(np.float32)
    return np.hstack([xy, t])


def sample_boundary(N_bc, t_min, t_max):
    pts = []
    n_per_side = N_bc // 4
    for _ in range(n_per_side):
        t = np.random.uniform(t_min, t_max)
        pts.append([-1.0, np.random.uniform(-1, 1), t])
        pts.append([ 1.0, np.random.uniform(-1, 1), t])
        pts.append([np.random.uniform(-1, 1), -1.0, t])
        pts.append([np.random.uniform(-1, 1),  1.0, t])
    return np.array(pts, dtype=np.float32)[:N_bc]


# training steps 
def masked_mse(pred, y, sz):
    idx  = tf.range(tf.shape(pred)[1])[tf.newaxis, :, tf.newaxis]
    mask = tf.cast(idx < tf.cast(sz[:, tf.newaxis, :], tf.int32), tf.float32)
    return tf.reduce_sum(tf.square(pred - y) * mask) / (tf.reduce_sum(mask) + 1e-8)


def train_step_data(model, opt, xb, xt, sz, y):
    with tf.GradientTape() as tape:
        loss = masked_mse(model.call_data(xb, xt, training=True), y, sz)
    opt.apply_gradients(zip(tape.gradient(loss, model.trainable_variables),
                            model.trainable_variables))
    return float(loss)


def train_step_physics(model, opt, xb_single, xyt_colloc, xyt_bc,
                       xy_ic, y_ic_true, t_min,
                       D, k, s1_tf, s2_tf, e_tf, G,
                       lambda_pde, lambda_bc, lambda_ic):
    xyt_c = tf.constant(xyt_colloc)
    with tf.GradientTape() as tape:
        residual = compute_pde_residual(
            model, xb_single, xyt_c, D, k, s1_tf, s2_tf, e_tf, G)
        loss_pde = lambda_pde * tf.reduce_mean(tf.square(residual))

        bc_res = compute_bc_residual(model, xb_single, xyt_bc)
        loss_bc = lambda_bc * tf.reduce_mean(tf.square(bc_res)) if tf.size(bc_res) > 0 else 0.0

        ic_res = compute_ic_residual(model, xb_single, xy_ic, y_ic_true, t_min)
        loss_ic = lambda_ic * tf.reduce_mean(tf.square(ic_res))

        loss_total = loss_pde + loss_bc + loss_ic

    grads = tape.gradient(loss_total, model.trainable_variables)
    grads_and_vars = [(g, v) for g, v in zip(grads, model.trainable_variables) if g is not None]
    if grads_and_vars:
        opt.apply_gradients(grads_and_vars)
    return float(loss_pde), float(loss_bc), float(loss_ic)


def val_mse(model, ds_vl):
    losses = []
    for batch in ds_vl:
        (xb, xt, sz), y = batch
        pred = model.call_data(xb, xt, training=False)
        losses.append(float(masked_mse(pred, y, sz)))
    return float(np.mean(losses))


def train_model(model, opt, ds_tr, ds_vl,
                Xbranch_tr, t_min, t_max, G,
                D, k, s1_tf, s2_tf, e_tf,
                xy_grid, Y_ic_flat, cyt_idx,
                lambda_pde=0.01, lambda_bc=0.001, lambda_ic=1.0,
                epochs=FULL_EPOCHS, patience=40, reduce_patience=15,
                min_lr=1e-7, verbose=True, physics_every=1):
    best_val = np.inf; best_w = None; wait = rw = 0
    N_tr = Xbranch_tr.shape[0]
    G2 = G * G

    for ep in range(1, epochs + 1):
        # data loss
        data_losses = []
        for batch in ds_tr:
            (xb, xt, sz), y = batch
            dl = train_step_data(model, opt, xb, xt, sz, y)
            data_losses.append(dl)
        tr_data = float(np.mean(data_losses))

        # physics loss (PDE + BC + IC)
        tr_pde = tr_bc = tr_ic = 0.0
        if lambda_pde > 0 and ep % physics_every == 0:
            xyt_c  = sample_collocation(N_COLLOC, t_min, t_max)
            xyt_bc = sample_boundary(N_BC, t_min, t_max)

            # sample IC points: random subset of spatial grid at t_min
            ic_idx = np.random.choice(G2, size=min(N_IC, G2), replace=False)
            xy_ic  = xy_grid[ic_idx].astype(np.float32)
            y_ic   = Y_ic_flat[ic_idx].astype(np.float32)

            n_phys = min(N_PHYSICS_SAMPLES, N_tr)
            phys_idxs = np.random.choice(N_tr, size=n_phys, replace=False)
            pde_losses = []; bc_losses = []; ic_losses = []
            for pidx in phys_idxs:
                xb_s = tf.constant(Xbranch_tr[pidx:pidx+1])
                pde_l, bc_l, ic_l = train_step_physics(
                    model, opt, xb_s, xyt_c, xyt_bc,
                    xy_ic, y_ic, t_min,
                    D, k, s1_tf, s2_tf, e_tf, G,
                    lambda_pde, lambda_bc, lambda_ic)
                pde_losses.append(pde_l); bc_losses.append(bc_l); ic_losses.append(ic_l)
            tr_pde = float(np.mean(pde_losses))
            tr_bc  = float(np.mean(bc_losses))
            tr_ic  = float(np.mean(ic_losses))

        vl_mean = val_mse(model, ds_vl)

        if verbose and ep % 20 == 0:
            print(f"  Ep {ep:4d}  data={tr_data:.5f}  pde={tr_pde:.5f}  "
                  f"bc={tr_bc:.5f}  ic={tr_ic:.5f}  val={vl_mean:.5f}")

        if vl_mean < best_val:
            best_val = vl_mean; best_w = model.get_weights(); wait = rw = 0
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


# eval 
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


# metrics 
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
    r2 = float(r2_score(yt.flatten(), yp.flatten()))

    per_t_r2 = []
    for t in range(T):
        gt_f = yt[t].flatten(); pr_f = yp[t].flatten()
        per_t_r2.append(float(r2_score(gt_f, pr_f)) if np.std(gt_f) > 1e-12 else np.nan)

    dice_thr = 0.05 * clip_max if clip_max > 0 else 1e-9
    dices = []; n_empty = 0; z_corrs = []; ssims_v = []; n_ssim_skip = 0
    fixed_dr = float(clip_max) if clip_max > 0 else 1.0

    for t in range(T):
        gt = yt[t,:,:,0]; pr = yp[t,:,:,0]
        gb = (gt > dice_thr).astype(float); pb = (pr > dice_thr).astype(float)
        if np.sum(gb) + np.sum(pb) == 0:
            n_empty += 1
        else:
            dices.append((2.0*np.sum(gb*pb)) / (np.sum(gb)+np.sum(pb)+1e-12))
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
        "Global_R2": r2, "Per_Timestep_R2": per_t_r2,
        "Masked_RMSE": rmse, "Unmasked_RMSE": unmasked_rmse,
        "Avg_Dice": float(np.mean(dices)) if dices else 0.0,
        "Dice_Empty_Skipped": n_empty,
        "Spatial_Correlation": _inv_fisher_z(float(np.mean(z_corrs))) if z_corrs else 0.0,
        "SSIM": float(np.mean(ssims_v)) if ssims_v else 0.0,
        "SSIM_Skipped_Frames": n_ssim_skip,
    }

def denormalize(x, clip_max):
    return (np.asarray(x, np.float64)+1.0)/2.0*clip_max

# optuna 
def make_objective(Xbr_tr, Xtr_tr, Yf_tr, Xbr_vl, Xtr_vl, Yf_vl,
                   t_min, t_max, G, D, k, s1_tf, s2_tf, e_tf,
                   xy_grid, Y_ic_flat, cyt_idx, seed):
    def objective(trial):
        set_seed(seed)
        tf.keras.backend.clear_session()
        p          = trial.suggest_categorical("p",          [64, 128, 256])
        hidden     = trial.suggest_categorical("hidden",     [128, 256])
        lr         = trial.suggest_float("learning_rate",    1e-5, 1e-3, log=True)
        bs         = trial.suggest_categorical("batch_size", [4, 8])
        chunk_size = trial.suggest_categorical("chunk_size", [2048, 4096])
        lambda_pde = trial.suggest_float("lambda_pde",      1e-4, 1.0, log=True)
        lambda_bc  = trial.suggest_float("lambda_bc",       1e-4, 0.1, log=True)
        lambda_ic  = trial.suggest_float("lambda_ic",       0.1, 10.0, log=True)

        ds_tr = build_dataset(Xbr_tr, Xtr_tr, Yf_tr, bs, chunk_size, shuffle=True)
        ds_vl = build_dataset(Xbr_vl, Xtr_vl, Yf_vl, bs, chunk_size, shuffle=False)

        model = PIDeepONet(hidden=hidden, p=p)
        opt   = tf.keras.optimizers.Adam(lr)
        best  = train_model(
            model, opt, ds_tr, ds_vl,
            Xbr_tr, t_min, t_max, G,
            D, k, s1_tf, s2_tf, e_tf,
            xy_grid, Y_ic_flat, cyt_idx,
            lambda_pde=lambda_pde, lambda_bc=lambda_bc, lambda_ic=lambda_ic,
            epochs=TUNE_EPOCHS, patience=8, reduce_patience=5,
            verbose=False, physics_every=2,
        )
        return float(best)
    return objective


# main pipeline 
def run_pipeline(grid, seed, cytokine):
    set_seed(seed)
    idx = CYTOKINE_NAMES.index(cytokine.lower())

    data_path = PREPROCESSED / f"{grid}x{grid}"
    out_dir   = RESULTS_MODELS / "pi_deeponet"; out_dir.mkdir(parents=True, exist_ok=True)

    print(f"\n[{cytokine.upper()}] {grid}x{grid} — PI-DeepONet — loading data...")
    Xb = np.load(data_path/"X_branch.npy").astype(np.float32)
    Xt = np.load(data_path/"X_trunk.npy").astype(np.float32)
    Y  = np.load(data_path/"Y_target.npy").astype(np.float32)[..., idx:idx+1]
    M  = np.load(data_path/"Y_masks_spatial.npy").astype(np.float32)
    M_pinn = np.load(data_path/"Y_masks_pinn.npy").astype(np.float64)
    Y_ic_full = np.load(data_path/"Y_ic.npy").astype(np.float32)  # (G, G, 6)

    with open(data_path/"metadata.json") as f:
        meta = json.load(f)
    clip_max = float(meta["scaling"]["max"][idx])

    N = Xb.shape[0]; G2 = Xt.shape[1]; G = int(round(G2**0.5))
    Yf = Y.reshape(N, G2, 1)

    Xbranch = build_branch_inputs(Xb, Xt, idx)
    Xtrunk  = build_trunk_inputs(Xb, Xt)

    print(f"  Branch: (N, 7)  |  Trunk: (N, {G2}, {Xtrunk.shape[2]})  |  clip_max: {clip_max:.6f}")

    # PDE 
    D, k, sec = compute_pde_params(grid, idx)
    masks_mean = M_pinn[:70].mean(axis=0)  #(G*G, 5)
    s1_tf, s2_tf, e_tf = build_source_arrays(masks_mean, sec, idx, grid, clip_max)

    t_norm_all = np.linspace(-1.0, 1.0, 101, dtype=np.float32)
    t_min = float(t_norm_all[2])
    t_max = float(t_norm_all[-1])

    # IC data
    xs = np.linspace(-1.0, 1.0, G, dtype=np.float32)
    ys = np.linspace(-1.0, 1.0, G, dtype=np.float32)
    xx, yy = np.meshgrid(xs, ys, indexing="ij")
    xy_grid_ic = np.stack([xx.ravel(), yy.ravel()], axis=1)# (G*G, 2)
    Y_ic_flat  = Y_ic_full[:, :, idx].reshape(G2, 1) #(G*G, 1) scaled

    # 70/10/20
    Xbr_tr, Xtr_tr, Yf_tr = Xbranch[:70],   Xtrunk[:70],   Yf[:70]
    Xbr_vl, Xtr_vl, Yf_vl = Xbranch[70:80], Xtrunk[70:80], Yf[70:80]

    # Optuna
    if seed == 42:
        print(f"Optuna: {N_TRIALS} trials × {TUNE_EPOCHS} epochs...")
        study = optuna.create_study(
            direction="minimize",
            sampler=optuna.samplers.TPESampler(seed=42),
            pruner=optuna.pruners.MedianPruner(n_warmup_steps=5),
        )
        study.optimize(
            make_objective(Xbr_tr, Xtr_tr, Yf_tr, Xbr_vl, Xtr_vl, Yf_vl,
                           t_min, t_max, G, D, k, s1_tf, s2_tf, e_tf,
                           xy_grid_ic, Y_ic_flat, idx, 42),
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

    # final training
    tf.keras.backend.clear_session(); set_seed(seed)
    ds_tr = build_dataset(Xbr_tr, Xtr_tr, Yf_tr,
                          best["batch_size"], best["chunk_size"], shuffle=True)
    ds_vl = build_dataset(Xbr_vl, Xtr_vl, Yf_vl,
                          best["batch_size"], best["chunk_size"], shuffle=False)

    model = PIDeepONet(hidden=best["hidden"], p=best["p"])
    opt   = tf.keras.optimizers.Adam(best["learning_rate"])

    t_start = time.time()
    print(f"Final training (max {FULL_EPOCHS} epochs, "
          f"λ_pde={best['lambda_pde']:.4f}, λ_bc={best['lambda_bc']:.4f}, "
          f"λ_ic={best['lambda_ic']:.4f})...")
    train_model(
        model, opt, ds_tr, ds_vl,
        Xbr_tr, t_min, t_max, G,
        D, k, s1_tf, s2_tf, e_tf,
        xy_grid_ic, Y_ic_flat, idx,
        lambda_pde=best["lambda_pde"], lambda_bc=best["lambda_bc"],
        lambda_ic=best["lambda_ic"],
        epochs=FULL_EPOCHS, patience=40, reduce_patience=15,
        verbose=True, physics_every=1,
    )
    train_elapsed = time.time() - t_start

    # Evaluate
    t_pred = time.time()
    Yp_flat = predict_full(model, Xbranch, Xtrunk)
    pred_elapsed = time.time() - t_pred
    Yp      = Yp_flat.reshape(N, G, G, 1)
    Y_phys  = denormalize(Y.reshape(N, G, G, 1), clip_max)
    Yp_phys = denormalize(Yp, clip_max)

    suffix = f"{cytokine}_{grid}_{seed}"
    print(f"  Train: {train_elapsed:.1f}s | Pred: {pred_elapsed:.1f}s")

    results = {
        "grid": grid, "seed": seed, "cytokine": cytokine,
        "best_params": best,
        "optuna_best_val_loss": optuna_val,
        "train_time_seconds":   round(train_elapsed, 2),
        "pred_time_seconds":    round(pred_elapsed, 2),
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
    print(f"DONE → models/pi_deeponet/res_{suffix}.json")

    for split, m in results["results"].items():
        print(f"  {split}: R²={m['Global_R2']:.4f}  RMSE={m['Masked_RMSE']:.6f}")


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
