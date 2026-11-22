import os
import json
import argparse
import random
import warnings
import gc
import time
import glob
from pathlib import Path

import numpy as np
import tensorflow as tf
import optuna
from sklearn.metrics import r2_score
from scipy.stats import pearsonr
from skimage.metrics import structural_similarity as ssim

warnings.filterwarnings("ignore")
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"
optuna.logging.set_verbosity(optuna.logging.WARNING)

CYTOKINE_NAMES = ["il8", "il1", "il6", "il10", "tnf", "tgf"]
MASK_E, MASK_NDN, MASK_NA, MASK_M1, MASK_M2 = 0, 1, 2, 3, 4

TRUE_SIZE = 5.0
S_MCS     = 60.0
H_MCS     = 1.0 / S_MCS


def set_seed(seed):
    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed); np.random.seed(seed); tf.random.set_seed(seed)

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


def build_source_arrays_np(masks_flat, sec, cyt_idx, G, clip_max):
    n = G * G
    me  = masks_flat[:, MASK_E]
    mnn = masks_flat[:, MASK_NDN]
    mna = masks_flat[:, MASK_NA]
    mm1 = masks_flat[:, MASK_M1]
    mm2 = masks_flat[:, MASK_M2]
    z = np.zeros(n, np.float64)

    # IL-10 uses the M2 mask
    s1_map = [sec[0]*me, sec[3]*mna, sec[4]*mm1, sec[5]*mm2, sec[6]*mna, sec[8]*mm2]
    s2_map = [sec[1]*mnn, z, z, z, sec[7]*mm1, z]
    e_map  = [sec[2]*mna, z, z, z, z, z]

    scale = 2.0 / (clip_max + 1e-30)
    s1 = s1_map[cyt_idx].reshape(-1, 1) * scale
    s2 = s2_map[cyt_idx].reshape(-1, 1) * scale
    e  = e_map[cyt_idx].reshape(-1, 1)
    return s1, s2, e

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
    return (np.asarray(x, np.float64) + 1.0) / 2.0 * clip_max

N_COLLOC_PIDON   = 2000
N_BC_PIDON       = 200
N_IC_PIDON       = 500
N_PHYS_SAMPLES   = 3

class PIDeepONetBranch(tf.keras.layers.Layer):
    def __init__(self, hidden, p, **kw):
        super().__init__(**kw)
        self.fc1 = tf.keras.layers.Dense(hidden, activation="relu")
        self.fc2 = tf.keras.layers.Dense(p, activation="linear")
    def call(self, x, training=False):
        return self.fc2(self.fc1(x))

class PIDeepONetTrunk(tf.keras.layers.Layer):
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

class PIDeepONet(tf.keras.Model):
    def __init__(self, hidden, p):
        super().__init__()
        self.branch = PIDeepONetBranch(hidden, p)
        self.trunk = PIDeepONetTrunk(hidden, p)
        self.bias = self.add_weight(shape=(1,), initializer="zeros",
                                    trainable=True, name="bias")

    def call_data(self, xb, xt, training=False):
        b = self.branch(xb, training=training)
        t = self.trunk(xt)
        r = tf.einsum("bp,bnp->bn", b, t) + self.bias
        return tf.expand_dims(r, -1)

    def call_physics_single(self, xb_single, xyt):
        b = self.branch(xb_single, training=True)
        n = tf.shape(xyt)[0]
        padding = tf.zeros([n, 21], dtype=xyt.dtype)
        xyt_padded = tf.concat([xyt, padding], axis=1)
        t = self.trunk(xyt_padded)
        r = tf.reduce_sum(b[0] * t, axis=-1, keepdims=True) + self.bias
        return r

    def call(self, inputs, training=False):
        return self.call_data(inputs[0], inputs[1], training=training)


def pidon_build_branch_inputs(Xb, Xt, cyt_idx):
    N, _, G, _, _ = Xb.shape
    f0 = Xb[:, 0, :, :, cyt_idx]
    mask = (Xb[:, 0, :, :, 6:].max(axis=-1) > 0.5).astype(np.float32)
    xs = np.linspace(0, 1, G, dtype=np.float32)
    ys = np.linspace(0, 1, G, dtype=np.float32)
    xx, yy = np.meshgrid(xs, ys, indexing='ij')
    out = np.zeros((N, 7), dtype=np.float32)
    for i in range(N):
        f = f0[i]; m = mask[i]; na = float(np.sum(m)) + 1e-6
        out[i, 0] = (float(np.max(f)) + 1.0) / 2.0
        out[i, 1] = (float(np.mean(f)) + 1.0) / 2.0
        out[i, 2] = float(np.std(f))
        out[i, 3] = float(np.sum(xx * m) / na)
        out[i, 4] = float(np.sum(yy * m) / na)
        out[i, 5] = na / (G * G)
        out[i, 6] = float(Xt[i, 0, 2])
    return out

def pidon_build_trunk_inputs(Xb, Xt):
    N, _, G, _, C = Xb.shape
    vals = Xb.transpose(0, 2, 3, 1, 4).reshape(N, G*G, 22).astype(np.float32)
    xy = Xt[:, :, :2].astype(np.float32)
    return np.concatenate([xy, vals], axis=-1)


def pidon_build_dataset(Xbranch, Xtrunk, Yf, batch_size, chunk_size, shuffle=True):
    N, n_pts, D_trunk = Xtrunk.shape
    chunks = list(range(0, n_pts, chunk_size))
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
                    xt = np.concatenate([xt, np.zeros((pad, D_trunk), np.float32)])
                    y  = np.concatenate([y,  np.zeros((pad, 1), np.float32)])
                sz = np.array([size], dtype=np.int32)
                yield (xb, xt, sz), y
    sig = (
        (tf.TensorSpec((7,), tf.float32),
         tf.TensorSpec((chunk_size, D_trunk), tf.float32),
         tf.TensorSpec((1,), tf.int32)),
        tf.TensorSpec((chunk_size, 1), tf.float32),
    )
    return (tf.data.Dataset.from_generator(gen, output_signature=sig)
            .batch(batch_size).prefetch(tf.data.AUTOTUNE))


def pidon_pde_residual(model, xb_single, xyt, D, k, s1_tf, s2_tf, e_tf, G):
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

def pidon_bc_residual(model, xb_single, xyt_bc):
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


def pidon_ic_residual(model, xb_single, xy_ic, y_ic_true, t_min):
    t_col = np.full((xy_ic.shape[0], 1), t_min, dtype=np.float32)
    xyt_ic = tf.constant(np.hstack([xy_ic, t_col]))
    u_pred = model.call_physics_single(xb_single, xyt_ic)
    return u_pred - tf.constant(y_ic_true, dtype=tf.float32)

def pidon_sample_collocation(N_c, t_min, t_max):
    xy = np.random.uniform(-1.0, 1.0, (N_c, 2)).astype(np.float32)
    t  = np.random.uniform(t_min, t_max, (N_c, 1)).astype(np.float32)
    return np.hstack([xy, t])

def pidon_sample_boundary(N_bc, t_min, t_max):
    pts = []
    n_per_side = N_bc // 4
    for _ in range(n_per_side):
        t = np.random.uniform(t_min, t_max)
        pts.append([-1.0, np.random.uniform(-1, 1), t])
        pts.append([ 1.0, np.random.uniform(-1, 1), t])
        pts.append([np.random.uniform(-1, 1), -1.0, t])
        pts.append([np.random.uniform(-1, 1),  1.0, t])
    return np.array(pts, dtype=np.float32)[:N_bc]

def pidon_masked_mse(pred, y, sz):
    idx = tf.range(tf.shape(pred)[1])[tf.newaxis, :, tf.newaxis]
    mask = tf.cast(idx < tf.cast(sz[:, tf.newaxis, :], tf.int32), tf.float32)
    return tf.reduce_sum(tf.square(pred - y) * mask) / (tf.reduce_sum(mask) + 1e-8)

def pidon_train_step_data(model, opt, xb, xt, sz, y):
    with tf.GradientTape() as tape:
        loss = pidon_masked_mse(model.call_data(xb, xt, training=True), y, sz)
    opt.apply_gradients(zip(tape.gradient(loss, model.trainable_variables),
                            model.trainable_variables))
    return float(loss)

def pidon_train_step_physics(model, opt, xb_single, xyt_colloc, xyt_bc,
                              xy_ic, y_ic_true, t_min,
                              D, k, s1_tf, s2_tf, e_tf, G,
                              lambda_pde, lambda_bc, lambda_ic):
    xyt_c = tf.constant(xyt_colloc)
    with tf.GradientTape() as tape:
        residual = pidon_pde_residual(model, xb_single, xyt_c, D, k,
                                       s1_tf, s2_tf, e_tf, G)
        loss_pde = lambda_pde * tf.reduce_mean(tf.square(residual))
        bc_res = pidon_bc_residual(model, xb_single, xyt_bc)
        loss_bc = lambda_bc * tf.reduce_mean(tf.square(bc_res)) if tf.size(bc_res) > 0 else 0.0
        ic_res = pidon_ic_residual(model, xb_single, xy_ic, y_ic_true, t_min)
        loss_ic = lambda_ic * tf.reduce_mean(tf.square(ic_res))
        loss_total = loss_pde + loss_bc + loss_ic
    grads = tape.gradient(loss_total, model.trainable_variables)
    grads_and_vars = [(g, v) for g, v in zip(grads, model.trainable_variables)
                      if g is not None]
    if grads_and_vars:
        opt.apply_gradients(grads_and_vars)
    return float(loss_pde), float(loss_bc), float(loss_ic)


def pidon_val_mse(model, ds_vl):
    losses = []
    for batch in ds_vl:
        (xb, xt, sz), y = batch
        pred = model.call_data(xb, xt, training=False)
        losses.append(float(pidon_masked_mse(pred, y, sz)))
    return float(np.mean(losses))


def pidon_train_model(model, opt, ds_tr, ds_vl,
                       Xbranch_tr, t_min, t_max, G,
                       D, k, s1_tf, s2_tf, e_tf,
                       xy_grid, Y_ic_flat, cyt_idx,
                       lambda_pde, lambda_bc, lambda_ic,
                       epochs, patience=40, reduce_patience=15,
                       min_lr=1e-7, physics_every=1):
    best_val = np.inf; best_w = None; wait = rw = 0
    N_tr = Xbranch_tr.shape[0]
    G2 = G * G
    for ep in range(1, epochs + 1):
        for batch in ds_tr:
            (xb, xt, sz), y = batch
            pidon_train_step_data(model, opt, xb, xt, sz, y)

        if lambda_pde > 0 and ep % physics_every == 0:
            xyt_c  = pidon_sample_collocation(N_COLLOC_PIDON, t_min, t_max)
            xyt_bc = pidon_sample_boundary(N_BC_PIDON, t_min, t_max)
            ic_idx = np.random.choice(G2, size=min(N_IC_PIDON, G2), replace=False)
            xy_ic = xy_grid[ic_idx].astype(np.float32)
            y_ic = Y_ic_flat[ic_idx].astype(np.float32)
            n_phys = min(N_PHYS_SAMPLES, N_tr)
            phys_idxs = np.random.choice(N_tr, size=n_phys, replace=False)
            for pidx in phys_idxs:
                xb_s = tf.constant(Xbranch_tr[pidx:pidx+1])
                pidon_train_step_physics(model, opt, xb_s, xyt_c, xyt_bc,
                                          xy_ic, y_ic, t_min,
                                          D, k, s1_tf, s2_tf, e_tf, G,
                                          lambda_pde, lambda_bc, lambda_ic)

        vl_mean = pidon_val_mse(model, ds_vl)
        if vl_mean < best_val:
            best_val = vl_mean; best_w = model.get_weights(); wait = rw = 0
        else:
            wait += 1; rw += 1
        if rw >= reduce_patience:
            lr = float(opt.learning_rate)
            new_lr = max(lr * 0.5, min_lr)
            if new_lr != lr:
                opt.learning_rate.assign(new_lr)
            rw = 0
        if wait >= patience:
            break
    if best_w:
        model.set_weights(best_w)
    return best_val


def pidon_predict_full(model, Xbranch, Xtrunk, chunk=4096):
    N, n_pts, _ = Xtrunk.shape
    out = np.zeros((N, n_pts, 1), np.float32)
    for i in range(N):
        xb = tf.constant(Xbranch[i:i+1])
        for s in range(0, n_pts, chunk):
            e = min(s + chunk, n_pts)
            xt = tf.constant(Xtrunk[i:i+1, s:e])
            out[i, s:e] = model.call_data(xb, xt, training=False).numpy()[0]
    return out


def train_pi_deeponet(prep_dir, cytokine, seed, n_trials,
                       tune_epochs, full_epochs, out_dir):
    set_seed(seed)
    idx = CYTOKINE_NAMES.index(cytokine.lower())
    out_dir.mkdir(parents=True, exist_ok=True)

    Xb = np.load(prep_dir / "X_branch.npy").astype(np.float32)
    Xt = np.load(prep_dir / "X_trunk.npy").astype(np.float32)
    Y  = np.load(prep_dir / "Y_target.npy").astype(np.float32)[..., idx:idx+1]
    M  = np.load(prep_dir / "Y_masks_spatial.npy").astype(np.float32)
    M_pinn = np.load(prep_dir / "Y_masks_pinn.npy").astype(np.float64)
    Y_ic_full = np.load(prep_dir / "Y_ic.npy").astype(np.float32)

    with open(prep_dir / "metadata.json") as f:
        meta = json.load(f)
    clip_max = float(meta["scaling"]["max"][idx])
    grid = meta["grid"]

    N = Xb.shape[0]; G2 = Xt.shape[1]; G = int(round(G2**0.5))
    Yf = Y.reshape(N, G2, 1)

    Xbranch = pidon_build_branch_inputs(Xb, Xt, idx)
    Xtrunk  = pidon_build_trunk_inputs(Xb, Xt)

    D, k, sec = compute_pde_params(grid, idx)
    masks_mean = M_pinn[:70].mean(axis=0)
    s1_np, s2_np, e_np = build_source_arrays_np(masks_mean, sec, idx, grid, clip_max)
    s1_tf = tf.constant(s1_np, tf.float32)
    s2_tf = tf.constant(s2_np, tf.float32)
    e_tf  = tf.constant(e_np,  tf.float32)

    t_norm_all = np.linspace(-1.0, 1.0, 101, dtype=np.float32)
    t_min = float(t_norm_all[2])
    t_max = float(t_norm_all[-1])

    xs = np.linspace(-1.0, 1.0, G, dtype=np.float32)
    ys = np.linspace(-1.0, 1.0, G, dtype=np.float32)
    xx, yy = np.meshgrid(xs, ys, indexing="ij")
    xy_grid_ic = np.stack([xx.ravel(), yy.ravel()], axis=1)
    Y_ic_flat = Y_ic_full[:, :, idx].reshape(G2, 1)

    Xbr_tr, Xtr_tr, Yf_tr = Xbranch[:70],   Xtrunk[:70],   Yf[:70]
    Xbr_vl, Xtr_vl, Yf_vl = Xbranch[70:80], Xtrunk[70:80], Yf[70:80]

    def make_obj():
        def obj(trial):
            set_seed(seed); tf.keras.backend.clear_session()
            p          = trial.suggest_categorical("p", [64, 128, 256])
            hidden     = trial.suggest_categorical("hidden", [128, 256])
            lr         = trial.suggest_float("learning_rate", 1e-5, 1e-3, log=True)
            bs         = trial.suggest_categorical("batch_size", [4, 8])
            chunk_size = trial.suggest_categorical("chunk_size", [2048, 4096])
            lambda_pde = trial.suggest_float("lambda_pde", 1e-4, 1.0, log=True)
            lambda_bc  = trial.suggest_float("lambda_bc",  1e-4, 0.1, log=True)
            lambda_ic  = trial.suggest_float("lambda_ic",  0.1, 10.0, log=True)
            ds_tr = pidon_build_dataset(Xbr_tr, Xtr_tr, Yf_tr, bs, chunk_size, shuffle=True)
            ds_vl = pidon_build_dataset(Xbr_vl, Xtr_vl, Yf_vl, bs, chunk_size, shuffle=False)
            m = PIDeepONet(hidden=hidden, p=p)
            opt = tf.keras.optimizers.Adam(lr)
            best = pidon_train_model(
                m, opt, ds_tr, ds_vl,
                Xbr_tr, t_min, t_max, G, D, k, s1_tf, s2_tf, e_tf,
                xy_grid_ic, Y_ic_flat, idx,
                lambda_pde=lambda_pde, lambda_bc=lambda_bc, lambda_ic=lambda_ic,
                epochs=tune_epochs, patience=8, reduce_patience=5,
                physics_every=2)
            return float(best)
        return obj

    print(f"\n  [PI-DeepONet | {cytokine} | grid {grid}] Optuna {n_trials} trials...")
    study = optuna.create_study(direction="minimize",
        sampler=optuna.samplers.TPESampler(seed=seed),
        pruner=optuna.pruners.MedianPruner(n_warmup_steps=5))
    study.optimize(make_obj(), n_trials=n_trials, show_progress_bar=False,
                   catch=(Exception,))
    best = study.best_params
    optuna_val = float(study.best_value)
    print(f"    Best: {best}  | val_loss={optuna_val:.6f}")

    tf.keras.backend.clear_session(); set_seed(seed)
    t_start = time.time()

    ds_tr = pidon_build_dataset(Xbr_tr, Xtr_tr, Yf_tr,
                                 best["batch_size"], best["chunk_size"], shuffle=True)
    ds_vl = pidon_build_dataset(Xbr_vl, Xtr_vl, Yf_vl,
                                 best["batch_size"], best["chunk_size"], shuffle=False)
    model = PIDeepONet(hidden=best["hidden"], p=best["p"])
    opt = tf.keras.optimizers.Adam(best["learning_rate"])
    pidon_train_model(
        model, opt, ds_tr, ds_vl,
        Xbr_tr, t_min, t_max, G, D, k, s1_tf, s2_tf, e_tf,
        xy_grid_ic, Y_ic_flat, idx,
        lambda_pde=best["lambda_pde"], lambda_bc=best["lambda_bc"],
        lambda_ic=best["lambda_ic"],
        epochs=full_epochs, patience=40, reduce_patience=15,
        physics_every=1,
    )

    t_pred = time.time()
    Yp_flat = pidon_predict_full(model, Xbranch, Xtrunk)
    pred_elapsed = time.time() - t_pred
    Yp = Yp_flat.reshape(N, G, G, 1)
    Y_phys  = denormalize(Y.reshape(N, G, G, 1), clip_max)
    Yp_phys = denormalize(Yp, clip_max)
    train_elapsed = time.time() - t_start - pred_elapsed

    suffix = f"{cytokine}_{grid}_{seed}"
    results = {
        "grid": grid, "seed": seed, "cytokine": cytokine, "model": "pi_deeponet",
        "best_params": best, "optuna_best_val_loss": optuna_val,
        "train_time_seconds": round(train_elapsed, 2),
        "pred_time_seconds":  round(pred_elapsed, 2),
        "results": {
            "Near_Horizon_t82_t91": calculate_metrics(
                Y_phys[80:90], Yp_phys[80:90], M[80:90], clip_max),
            "Far_Horizon_t92_t100": calculate_metrics(
                Y_phys[90:99], Yp_phys[90:99], M[90:99], clip_max),
        },
    }
    with open(out_dir / f"res_{suffix}.json", "w") as f:
        json.dump(results, f, indent=4)
    model.save_weights(out_dir / f"weights_{suffix}.weights.h5")
    print(f"    DONE: near R²={results['results']['Near_Horizon_t82_t91']['Global_R2']:.4f}  "
          f"far R²={results['results']['Far_Horizon_t92_t100']['Global_R2']:.4f}  "
          f"({train_elapsed:.0f}s)")

def train_pinn(prep_dir, cytokine, seed, n_trials,
                tune_iters, full_iters, warmup_iters, lbfgs_iters,
                out_dir):
    # deepxde is imported here so float64 is only set when the PINN runs
    import deepxde as dde

    dde.config.set_default_float("float64")
    dde.config.disable_xla_jit()
    gpus = tf.config.list_physical_devices('GPU')
    for gpu in gpus:
        try:
            tf.config.experimental.set_memory_growth(gpu, True)
        except RuntimeError:
            pass

    set_seed(seed)
    dde.config.set_random_seed(seed)

    cyt_idx = CYTOKINE_NAMES.index(cytokine.lower())
    out_dir.mkdir(parents=True, exist_ok=True)

    Y_tgt = np.load(prep_dir / "Y_target.npy").astype(np.float64)
    M_spat = np.load(prep_dir / "Y_masks_spatial.npy").astype(np.float64)
    M_pinn = np.load(prep_dir / "Y_masks_pinn.npy").astype(np.float64)
    Y_ic_full = np.load(prep_dir / "Y_ic.npy").astype(np.float64)

    with open(prep_dir / "metadata.json") as f:
        meta = json.load(f)
    clip_max = float(np.array(meta["scaling"]["max"])[cyt_idx])
    grid = meta["grid"]

    G2 = grid * grid
    N = Y_tgt.shape[0]

    D, k, sec = compute_pde_params(grid, cyt_idx)
    masks_mean = M_pinn[:70].mean(axis=0)
    s1_np, s2_np, e_np = build_source_arrays_np(masks_mean, sec, cyt_idx,
                                                  grid, clip_max)
    s1_tf = tf.constant(s1_np, tf.float64)
    s2_tf = tf.constant(s2_np, tf.float64)
    e_tf  = tf.constant(e_np,  tf.float64)

    def make_pde():
        G_tf = tf.constant(grid, dtype=tf.float64)
        D_tf = tf.constant([[D]], dtype=tf.float64)
        k_tf = tf.constant([[k]], dtype=tf.float64)
        def pde(x, y):
            d2x = dde.grad.hessian(y, x, i=0, j=0)
            d2y = dde.grad.hessian(y, x, i=1, j=1)
            ut = dde.grad.jacobian(y, x, i=0, j=2)
            G_dyn = tf.cast(G_tf, tf.float64)
            ix = tf.cast(tf.clip_by_value(
                tf.floor((x[:, 0:1] + 1.0) / 2.0 * G_dyn),
                0.0, G_dyn - 1.0), tf.int32)
            iy = tf.cast(tf.clip_by_value(
                tf.floor((x[:, 1:2] + 1.0) / 2.0 * G_dyn),
                0.0, G_dyn - 1.0), tf.int32)
            flat_idx = tf.squeeze(ix * tf.cast(G_tf, tf.int32) + iy, axis=1)
            s1_q = tf.gather(s1_tf, flat_idx)
            s2_q = tf.gather(s2_tf, flat_idx)
            e_q  = tf.gather(e_tf,  flat_idx)
            rhs = D_tf * (d2x + d2y) - k_tf * y + s1_q + s2_q - e_q * y
            return ut - rhs
        return pde

    pde_fn = make_pde()

    t_norm_all = np.linspace(-1.0, 1.0, 101, dtype=np.float64)
    t_min_n = float(t_norm_all[2])
    t_max_n = float(t_norm_all[-1])

    geom = dde.geometry.Rectangle([-1.0, -1.0], [1.0, 1.0])
    timedomain = dde.geometry.TimeDomain(t_min_n, t_max_n)
    geomtime = dde.geometry.GeometryXTime(geom, timedomain)

    def bc_fn(x, on_boundary):
        return on_boundary

    xs = np.linspace(-1.0, 1.0, grid, dtype=np.float64)
    ys = np.linspace(-1.0, 1.0, grid, dtype=np.float64)
    xx, yy = np.meshgrid(xs, ys, indexing="ij")
    xy_grid = np.stack([xx.ravel(), yy.ravel()], axis=1)

    tt0 = np.full((G2, 1), t_min_n, dtype=np.float64)
    X_ic = np.hstack([xy_grid, tt0])
    Y_ic_obs = Y_ic_full[:, :, cyt_idx].reshape(G2, 1).astype(np.float64)

    rng = np.random.default_rng(seed)
    val_indices = list(range(70, 80))
    train_indices = list(range(70))

    OBS_PTS_PER_T = min(G2, 2500)
    subsample = OBS_PTS_PER_T < G2

    X_obs_list = []; Y_obs_list = []
    for i in train_indices:
        abs_t = i + 2
        if subsample:
            idx_spatial = rng.choice(G2, size=OBS_PTS_PER_T, replace=False)
            xy_sub = xy_grid[idx_spatial]
            y_sub = Y_tgt[i, :, :, cyt_idx].reshape(G2, 1)[idx_spatial]
        else:
            xy_sub = xy_grid
            y_sub = Y_tgt[i, :, :, cyt_idx].reshape(G2, 1)
        t_obs = np.full((OBS_PTS_PER_T, 1), t_norm_all[abs_t], np.float64)
        X_obs_list.append(np.hstack([xy_sub, t_obs]))
        Y_obs_list.append(y_sub)

    X_obs_tr = np.vstack(X_obs_list)
    Y_obs_tr = np.vstack(Y_obs_list)

    def build_dde_model(hidden, n_layers, lr,
                         lambda_pde, lambda_ic, lambda_bc,
                         num_domain, num_boundary):
        ic_bc = dde.PointSetBC(X_ic, Y_ic_obs, component=0)
        neu_bc = dde.NeumannBC(geomtime,
                                lambda x: np.zeros((len(x), 1), np.float64),
                                bc_fn, component=0)
        obs_bc = dde.PointSetBC(X_obs_tr, Y_obs_tr, component=0)
        data = dde.data.TimePDE(
            geomtime, pde_fn,
            ic_bcs=[ic_bc, neu_bc, obs_bc],
            num_domain=num_domain,
            num_boundary=num_boundary,
            num_initial=0,
            train_distribution="uniform",
        )
        net = dde.maps.FNN(
            [3] + [hidden] * n_layers + [1],
            "tanh", "Glorot uniform"
        )
        net.apply_output_transform(lambda x, y: y)
        loss_weights = [lambda_pde, lambda_ic, lambda_bc, 1.0]
        model = dde.Model(data, net)
        model.compile("adam", lr=lr, loss_weights=loss_weights)
        return model

    def t_to_norm(t_idx):
        return float(t_norm_all[t_idx])

    def make_obj():
        def obj(trial):
            set_seed(seed); dde.config.set_random_seed(seed)
            tf.keras.backend.clear_session(); gc.collect()
            hidden     = trial.suggest_categorical("hidden", [50, 64, 128])
            n_layers   = trial.suggest_categorical("n_layers", [3, 4])
            lr         = trial.suggest_float("learning_rate", 1e-4, 5e-3, log=True)
            lambda_pde = trial.suggest_float("lambda_pde", 1e-3, 1.0, log=True)
            lambda_ic  = trial.suggest_float("lambda_ic",  0.1, 10.0, log=True)
            lambda_bc  = trial.suggest_float("lambda_bc",  1e-3, 1.0, log=True)
            try:
                m = build_dde_model(hidden, n_layers, lr,
                                     lambda_pde, lambda_ic, lambda_bc,
                                     num_domain=1000, num_boundary=200)
                m.train(iterations=tune_iters, display_every=tune_iters + 1)
                mses = []
                for vi in val_indices:
                    abs_t = vi + 2
                    tt = np.full((G2, 1), t_to_norm(abs_t), np.float64)
                    X_q = np.hstack([xy_grid, tt])
                    Yp = m.predict(X_q)
                    Yt = Y_tgt[vi, :, :, cyt_idx].reshape(-1, 1)
                    mses.append(float(np.mean((Yt - Yp) ** 2)))
                del m; gc.collect()
                return float(np.mean(mses))
            except Exception as e:
                gc.collect()
                raise optuna.TrialPruned(str(e))
        return obj

    print(f"\n  [PINN | {cytokine} | grid {grid}] Optuna {n_trials} trials × {tune_iters} iters...")
    study = optuna.create_study(direction="minimize",
        sampler=optuna.samplers.TPESampler(seed=seed),
        pruner=optuna.pruners.MedianPruner(n_warmup_steps=3))
    study.optimize(make_obj(), n_trials=n_trials, show_progress_bar=False,
                   catch=(Exception,))
    best = study.best_params
    optuna_val = float(study.best_value)
    print(f"    Best: {best}  | val_loss={optuna_val:.6f}")

    tf.keras.backend.clear_session(); set_seed(seed)
    dde.config.set_random_seed(seed)
    t_start = time.time()

    print(f"    Phase 1: warmup ({warmup_iters} Adam, λ_pde=0)...")
    model = build_dde_model(
        best["hidden"], best["n_layers"], best["learning_rate"],
        lambda_pde=0.0, lambda_ic=best["lambda_ic"], lambda_bc=best["lambda_bc"],
        num_domain=5000, num_boundary=1000)
    model.train(iterations=warmup_iters, display_every=warmup_iters + 1)

    physics_iters = full_iters - warmup_iters
    print(f"    Phase 2: physics ({physics_iters} Adam, λ_pde={best['lambda_pde']:.4f})...")
    model.compile("adam", lr=best["learning_rate"],
                   loss_weights=[best["lambda_pde"], best["lambda_ic"],
                                 best["lambda_bc"], 1.0])
    ckpt_path = str(out_dir / f"ckpt_{cytokine}_{grid}_{seed}")
    checker = dde.callbacks.ModelCheckpoint(ckpt_path, save_better_only=True, period=100)
    model.train(iterations=physics_iters, display_every=physics_iters + 1, callbacks=[checker])

    print(f"    Phase 3: L-BFGS ({lbfgs_iters} iters)...")
    dde.optimizers.set_LBFGS_options(maxiter=lbfgs_iters)
    model.compile("L-BFGS-B",
                   loss_weights=[best["lambda_pde"], best["lambda_ic"],
                                 best["lambda_bc"], 1.0])
    model.train(display_every=lbfgs_iters + 1, callbacks=[checker])

    best_step = model.train_state.best_step
    ckpt_files = sorted(glob.glob(ckpt_path + "*"))
    restore_path = None
    for suffix_try in [f"-{best_step}.ckpt", f"-{best_step}", ""]:
        candidate = ckpt_path + suffix_try
        if any(f.startswith(candidate) and (".index" in f or ".data-" in f)
               for f in ckpt_files):
            restore_path = candidate
            break
    if restore_path:
        model.restore(restore_path, verbose=0)

    t_pred = time.time()
    Yp_all = np.zeros((N, grid, grid, 1), dtype=np.float64)
    for i in range(N):
        abs_t = i + 2
        tt = np.full((G2, 1), t_to_norm(abs_t), np.float64)
        X_q = np.hstack([xy_grid, tt])
        Yp_all[i, :, :, 0] = model.predict(X_q).reshape(grid, grid)
    pred_elapsed = time.time() - t_pred
    train_elapsed = time.time() - t_start - pred_elapsed

    Y_phys = denormalize(Y_tgt[..., cyt_idx:cyt_idx + 1], clip_max)
    Yp_phys = denormalize(Yp_all, clip_max)

    suffix = f"{cytokine}_{grid}_{seed}"
    results = {
        "grid": grid, "seed": seed, "cytokine": cytokine, "model": "pinn",
        "best_params": best, "optuna_best_val_loss": optuna_val,
        "train_time_seconds": round(train_elapsed, 2),
        "pred_time_seconds":  round(pred_elapsed, 2),
        "results": {
            "Near_Horizon_t82_t91": calculate_metrics(
                Y_phys[80:90], Yp_phys[80:90], M_spat[80:90], clip_max),
            "Far_Horizon_t92_t100": calculate_metrics(
                Y_phys[90:99], Yp_phys[90:99], M_spat[90:99], clip_max),
        },
    }
    with open(out_dir / f"res_{suffix}.json", "w") as f:
        json.dump(results, f, indent=4)
    model.save(str(out_dir / f"weights_{suffix}"))
    print(f"    DONE: near R²={results['results']['Near_Horizon_t82_t91']['Global_R2']:.4f}  "
          f"far R²={results['results']['Far_Horizon_t92_t100']['Global_R2']:.4f}  "
          f"({train_elapsed:.0f}s)")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--prep-dir", type=str, required=True)
    ap.add_argument("--out-base", type=str, required=True)
    ap.add_argument("--seed",     type=int, default=42)
    ap.add_argument("--n-trials", type=int, default=20)
    ap.add_argument("--cytokines", type=str, default="il8,il10")
    ap.add_argument("--models", type=str, default="pi_deeponet,pinn",
                    help="Run both, or only pi_deeponet, or only pinn")
    # PI-DeepONet schedule
    ap.add_argument("--pidon-tune-epochs", type=int, default=30)
    ap.add_argument("--pidon-full-epochs", type=int, default=400)
    # PINN schedule
    ap.add_argument("--pinn-tune-iters",   type=int, default=2000)
    ap.add_argument("--pinn-full-iters",   type=int, default=15000)
    ap.add_argument("--pinn-warmup-iters", type=int, default=5000)
    ap.add_argument("--pinn-lbfgs-iters",  type=int, default=5000)
    args = ap.parse_args()

    prep_dir = Path(args.prep_dir)
    out_base = Path(args.out_base)
    models = [m.strip() for m in args.models.split(",")]
    cytokines = [c.strip() for c in args.cytokines.split(",")]

    if not prep_dir.exists():
        raise RuntimeError(f"Prep dir not found: {prep_dir}")

    with open(prep_dir / "metadata.json") as f:
        meta = json.load(f)
    grid = meta["grid"]

    print(f"\n{'='*70}")
    print(f"  PHYSICS MODELS TRAINING")
    print(f"  Prep dir:   {prep_dir}")
    print(f"  Strategy:   {meta.get('strategy', 'unknown')}")
    print(f"  Out base:   {out_base}")
    print(f"  Grid:       {grid}  |  Seed: {args.seed}  |  Trials: {args.n_trials}")
    print(f"  Models:     {models}")
    print(f"  Cytokines:  {cytokines}")
    print(f"{'='*70}")

    if "pi_deeponet" in models:
        out_dir = out_base / "pi_deeponet"
        for cyt in cytokines:
            res_file = out_dir / f"res_{cyt}_{grid}_{args.seed}.json"
            if res_file.exists():
                print(f"\n  [pi_deeponet | {cyt}] Already exists, skipping.")
                continue
            train_pi_deeponet(prep_dir, cyt, args.seed, args.n_trials,
                              args.pidon_tune_epochs, args.pidon_full_epochs,
                              out_dir)

    if "pinn" in models:
        out_dir = out_base / "pinn"
        for cyt in cytokines:
            res_file = out_dir / f"res_{cyt}_{grid}_{args.seed}.json"
            if res_file.exists():
                print(f"\n  [pinn | {cyt}] Already exists, skipping.")
                continue
            train_pinn(prep_dir, cyt, args.seed, args.n_trials,
                       args.pinn_tune_iters, args.pinn_full_iters,
                       args.pinn_warmup_iters, args.pinn_lbfgs_iters,
                       out_dir)

if __name__ == "__main__":
    main()