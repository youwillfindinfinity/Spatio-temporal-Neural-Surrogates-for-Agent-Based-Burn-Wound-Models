import os
import json
import argparse
import random
import time
from pathlib import Path

import numpy as np
import tensorflow as tf
import optuna
from sklearn.metrics import r2_score
from scipy.stats import pearsonr
from skimage.metrics import structural_similarity as ssim

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"
optuna.logging.set_verbosity(optuna.logging.WARNING)

CYTOKINE_NAMES = ["il8", "il1", "il6", "il10", "tnf", "tgf"]


def set_seed(seed: int):
    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    tf.random.set_seed(seed)


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
    dices = []; n_empty = 0
    z_corrs = []; ssims_v = []; n_ssim_skip = 0
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


def _unet_conv_block(x, filters, kernel_size=3):
    x = tf.keras.layers.Conv2D(filters, kernel_size, padding="same", activation="relu")(x)
    x = tf.keras.layers.BatchNormalization()(x)
    x = tf.keras.layers.Conv2D(filters, kernel_size, padding="same", activation="relu")(x)
    x = tf.keras.layers.BatchNormalization()(x)
    return x

def _unet_encoder_block(x, filters):
    skip = _unet_conv_block(x, filters)
    pool = tf.keras.layers.MaxPooling2D(pool_size=2, padding="same")(skip)
    return skip, pool

def _unet_decoder_block(x, skip, filters):
    x = tf.keras.layers.Conv2DTranspose(filters, 2, strides=2, padding="same", activation="relu")(x)
    if x.shape[1] != skip.shape[1] or x.shape[2] != skip.shape[2]:
        x = tf.keras.layers.Resizing(skip.shape[1], skip.shape[2])(x)
    x = tf.keras.layers.Concatenate()([x, skip])
    x = _unet_conv_block(x, filters)
    return x

def build_unet(grid_size, in_channels=22, out_channels=1,
               base_filters=32, depth=4, dropout=0.0):
    inputs = tf.keras.Input(shape=(grid_size, grid_size, in_channels))
    skips = []
    x = inputs
    for i in range(depth):
        f = base_filters * (2 ** i)
        skip, x = _unet_encoder_block(x, f)
        skips.append(skip)
    x = _unet_conv_block(x, base_filters * (2 ** depth))
    if dropout > 0:
        x = tf.keras.layers.Dropout(dropout)(x)
    for i in reversed(range(depth)):
        f = base_filters * (2 ** i)
        x = _unet_decoder_block(x, skips[i], f)
    outputs = tf.keras.layers.Conv2D(out_channels, 1, padding="same", activation="linear")(x)
    return tf.keras.Model(inputs, outputs, name="unet")

def train_unet(prep_dir, cytokine, seed, n_trials, tune_epochs, full_epochs, out_dir):
    set_seed(seed)
    idx = CYTOKINE_NAMES.index(cytokine.lower())
    out_dir.mkdir(parents=True, exist_ok=True)

    X = np.load(prep_dir / "X_unet.npy").astype(np.float32)
    Y = np.load(prep_dir / "Y_target.npy").astype(np.float32)[..., idx:idx+1]
    M = np.load(prep_dir / "Y_masks_spatial.npy").astype(np.float32)

    with open(prep_dir / "metadata.json") as f:
        meta = json.load(f)
    clip_max = float(meta["scaling"]["max"][idx])
    grid = meta["grid"]

    in_channels = X.shape[-1]
    X_train, Y_train = X[:70],   Y[:70]
    X_val,   Y_val   = X[70:80], Y[70:80]

    def make_obj():
        def obj(trial):
            set_seed(seed); tf.keras.backend.clear_session()
            base_filters = trial.suggest_categorical("base_filters", [16, 32, 64])
            depth        = trial.suggest_categorical("depth", [3, 4])
            dropout      = trial.suggest_float("dropout", 0.0, 0.3)
            lr           = trial.suggest_float("learning_rate", 1e-5, 1e-3, log=True)
            batch_size   = trial.suggest_categorical("batch_size", [4, 8, 16])
            m = build_unet(grid, in_channels, base_filters=base_filters,
                           depth=depth, dropout=dropout)
            m.compile(optimizer=tf.keras.optimizers.Adam(lr), loss="mse")
            h = m.fit(X_train, Y_train, validation_data=(X_val, Y_val),
                      epochs=tune_epochs, batch_size=batch_size, verbose=0,
                      callbacks=[tf.keras.callbacks.EarlyStopping(
                          monitor="val_loss", patience=5, restore_best_weights=True)])
            return float(min(h.history["val_loss"]))
        return obj

    print(f"\n  [U-Net | {cytokine} | grid {grid}] Optuna {n_trials} trials...")
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

    model = build_unet(grid, in_channels, base_filters=best["base_filters"],
                       depth=best["depth"], dropout=best["dropout"])
    model.compile(optimizer=tf.keras.optimizers.Adam(best["learning_rate"]), loss="mse")
    model.fit(X_train, Y_train, validation_data=(X_val, Y_val),
              epochs=full_epochs, batch_size=best["batch_size"], verbose=0,
              callbacks=[
                  tf.keras.callbacks.EarlyStopping(monitor="val_loss",
                      patience=20, restore_best_weights=True),
                  tf.keras.callbacks.ReduceLROnPlateau(monitor="val_loss",
                      factor=0.5, patience=10, min_lr=1e-6),
              ])

    Yp_scaled = model.predict(X, batch_size=2, verbose=0)
    Yp_phys = denormalize(Yp_scaled, clip_max)
    Ya_phys = denormalize(Y, clip_max)
    train_elapsed = time.time() - t_start

    suffix = f"{cytokine}_{grid}_{seed}"
    results = {
        "grid": grid, "seed": seed, "cytokine": cytokine, "model": "unet",
        "best_params": best, "optuna_best_val_loss": optuna_val,
        "train_time_seconds": round(train_elapsed, 2),
        "results": {
            "Near_Horizon_t82_t91": calculate_metrics(
                Ya_phys[80:90], Yp_phys[80:90], M[80:90], clip_max),
            "Far_Horizon_t92_t100": calculate_metrics(
                Ya_phys[90:99], Yp_phys[90:99], M[90:99], clip_max),
        },
    }
    with open(out_dir / f"res_{suffix}.json", "w") as f:
        json.dump(results, f, indent=4)
    model.save_weights(out_dir / f"weights_{suffix}.weights.h5")
    print(f"    DONE: near R²={results['results']['Near_Horizon_t82_t91']['Global_R2']:.4f}  "
          f"far R²={results['results']['Far_Horizon_t92_t100']['Global_R2']:.4f}  "
          f"({train_elapsed:.0f}s)")


class SpatialAttention(tf.keras.layers.Layer):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.attn_conv = tf.keras.layers.Conv2D(1, 1, padding="same", activation="sigmoid")
    def call(self, x):
        return x * self.attn_conv(x)

class STALSTM(tf.keras.Model):
    def __init__(self, grid_size, filters=64, lstm_units=128):
        super().__init__()
        self.grid_size = grid_size
        self.filters = filters
        self.lstm_units = lstm_units
        self.latent_size = max(grid_size // 4, 8)
        self.enc = tf.keras.layers.TimeDistributed(
            tf.keras.Sequential([
                tf.keras.layers.Conv2D(filters, 3, strides=2, padding="same", activation="relu"),
                tf.keras.layers.Conv2D(filters, 3, strides=2, padding="same", activation="relu"),
            ]))
        self.spatial_attn = tf.keras.layers.TimeDistributed(SpatialAttention())
        self.gap = tf.keras.layers.TimeDistributed(tf.keras.layers.GlobalAveragePooling2D())
        self.lstm = tf.keras.layers.LSTM(lstm_units, return_sequences=False)
        self.relu = tf.keras.layers.Activation("relu")
        self.fc = tf.keras.layers.Dense(self.latent_size * self.latent_size * filters, activation="relu")
        self.reshape_latent = tf.keras.layers.Reshape((self.latent_size, self.latent_size, filters))
        self.deconv1 = tf.keras.layers.Conv2DTranspose(filters // 2, 3, strides=2, padding="same", activation="relu")
        self.deconv2 = tf.keras.layers.Conv2DTranspose(filters // 4, 3, strides=2, padding="same", activation="relu")
        self.out_conv = tf.keras.layers.Conv2D(1, 3, padding="same", activation="linear")
        self.out_resize = tf.keras.layers.Resizing(grid_size, grid_size)

    def call(self, x):
        h = self.enc(x); h = self.spatial_attn(h); h = self.gap(h)
        h = self.lstm(h); h = self.relu(h); h = self.fc(h); h = self.reshape_latent(h)
        h = self.deconv1(h); h = self.deconv2(h); h = self.out_conv(h)
        return self.out_resize(h)

def train_stalstm(prep_dir, cytokine, seed, n_trials, tune_epochs, full_epochs, out_dir):
    set_seed(seed)
    idx = CYTOKINE_NAMES.index(cytokine.lower())
    out_dir.mkdir(parents=True, exist_ok=True)

    X = np.load(prep_dir / "X_lstm.npy").astype(np.float32)
    Y = np.load(prep_dir / "Y_target.npy").astype(np.float32)[..., idx:idx+1]
    M = np.load(prep_dir / "Y_masks_spatial.npy").astype(np.float32)

    with open(prep_dir / "metadata.json") as f:
        meta = json.load(f)
    clip_max = float(meta["scaling"]["max"][idx])
    grid = meta["grid"]

    X_train, Y_train = X[:70],   Y[:70]
    X_val,   Y_val   = X[70:80], Y[70:80]

    def make_obj():
        def obj(trial):
            set_seed(seed); tf.keras.backend.clear_session()
            filters    = trial.suggest_categorical("filters", [32, 64])
            lstm_units = trial.suggest_categorical("lstm_units", [64, 128, 256])
            lr         = trial.suggest_float("learning_rate", 1e-5, 1e-3, log=True)
            batch_size = trial.suggest_categorical("batch_size", [4, 8, 16])
            m = STALSTM(grid, filters=filters, lstm_units=lstm_units)
            m.compile(optimizer=tf.keras.optimizers.Adam(lr), loss="mse")
            h = m.fit(X_train, Y_train, validation_data=(X_val, Y_val),
                      epochs=tune_epochs, batch_size=batch_size, verbose=0,
                      callbacks=[tf.keras.callbacks.EarlyStopping(
                          monitor="val_loss", patience=5, restore_best_weights=True)])
            return float(min(h.history["val_loss"]))
        return obj

    print(f"\n  [STA-LSTM | {cytokine} | grid {grid}] Optuna {n_trials} trials...")
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

    model = STALSTM(grid, filters=best["filters"], lstm_units=best["lstm_units"])
    model.compile(optimizer=tf.keras.optimizers.Adam(best["learning_rate"]), loss="mse")
    model.fit(X_train, Y_train, validation_data=(X_val, Y_val),
              epochs=full_epochs, batch_size=best["batch_size"], verbose=0,
              callbacks=[
                  tf.keras.callbacks.EarlyStopping(monitor="val_loss",
                      patience=20, restore_best_weights=True),
                  tf.keras.callbacks.ReduceLROnPlateau(monitor="val_loss",
                      factor=0.5, patience=10, min_lr=1e-6),
              ])

    Yp_scaled = model.predict(X, batch_size=2, verbose=0)
    Yp_phys = denormalize(Yp_scaled, clip_max)
    Ya_phys = denormalize(Y, clip_max)
    train_elapsed = time.time() - t_start

    suffix = f"{cytokine}_{grid}_{seed}"
    results = {
        "grid": grid, "seed": seed, "cytokine": cytokine, "model": "sta_lstm",
        "best_params": best, "optuna_best_val_loss": optuna_val,
        "train_time_seconds": round(train_elapsed, 2),
        "results": {
            "Near_Horizon_t82_t91": calculate_metrics(
                Ya_phys[80:90], Yp_phys[80:90], M[80:90], clip_max),
            "Far_Horizon_t92_t100": calculate_metrics(
                Ya_phys[90:99], Yp_phys[90:99], M[90:99], clip_max),
        },
    }
    with open(out_dir / f"res_{suffix}.json", "w") as f:
        json.dump(results, f, indent=4)
    model.save_weights(out_dir / f"weights_{suffix}.weights.h5")
    print(f"    DONE: near R²={results['results']['Near_Horizon_t82_t91']['Global_R2']:.4f}  "
          f"far R²={results['results']['Far_Horizon_t92_t100']['Global_R2']:.4f}  "
          f"({train_elapsed:.0f}s)")


class DeepONetBranch(tf.keras.layers.Layer):
    def __init__(self, hidden, p, **kw):
        super().__init__(**kw)
        self.fc1 = tf.keras.layers.Dense(hidden, activation="relu")
        self.fc2 = tf.keras.layers.Dense(p, activation="linear")
    def call(self, x, training=False):
        return self.fc2(self.fc1(x))

class DeepONetTrunk(tf.keras.layers.Layer):
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
        self.branch = DeepONetBranch(hidden, p)
        self.trunk  = DeepONetTrunk(hidden, p)
        self.bias   = self.add_weight(shape=(1,), initializer="zeros",
                                      trainable=True, name="bias")
    def call(self, inputs, training=False):
        xb, xt = inputs
        b = self.branch(xb, training=training)
        t = self.trunk(xt)
        r = tf.einsum("bp,bnp->bn", b, t) + self.bias
        return tf.expand_dims(r, -1)

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
    xy = Xt[:, :, :2].astype(np.float32)
    return np.concatenate([xy, vals], axis=-1)

def build_deeponet_dataset(Xbranch, Xtrunk, Yf, batch_size, chunk_size, shuffle=True):
    N, n_pts, _ = Xtrunk.shape
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
                    xt = np.concatenate([xt, np.zeros((pad, 24), np.float32)], axis=0)
                    y  = np.concatenate([y,  np.zeros((pad, 1), np.float32)], axis=0)
                sz = np.array([size], dtype=np.int32)
                yield (xb, xt, sz), y
    sig = (
        (tf.TensorSpec((7,), tf.float32),
         tf.TensorSpec((chunk_size, 24), tf.float32),
         tf.TensorSpec((1,), tf.int32)),
        tf.TensorSpec((chunk_size, 1), tf.float32),
    )
    return (tf.data.Dataset.from_generator(gen, output_signature=sig)
            .batch(batch_size).prefetch(tf.data.AUTOTUNE))


def deeponet_masked_mse(pred, y, sz):
    idx = tf.range(tf.shape(pred)[1])[tf.newaxis, :, tf.newaxis]
    mask = tf.cast(idx < tf.cast(sz[:, tf.newaxis, :], tf.int32), tf.float32)
    return tf.reduce_sum(tf.square(pred - y) * mask) / (tf.reduce_sum(mask) + 1e-8)


def deeponet_train_step(model, opt, xb, xt, sz, y):
    with tf.GradientTape() as tape:
        loss = deeponet_masked_mse(model([xb, xt], training=True), y, sz)
    opt.apply_gradients(zip(tape.gradient(loss, model.trainable_variables),
                            model.trainable_variables))
    return float(loss)


def deeponet_val_step(model, xb, xt, sz, y):
    return float(deeponet_masked_mse(model([xb, xt], training=False), y, sz))


def train_deeponet_inner(model, opt, ds_tr, ds_vl, epochs, patience=40,
                         reduce_patience=15, min_lr=1e-7):
    best_val = np.inf; best_w = None; wait = rw = 0
    for ep in range(1, epochs + 1):
        tr = np.mean([deeponet_train_step(model, opt, *b[0], b[1]) for b in ds_tr])
        vl = np.mean([deeponet_val_step(model, *b[0], b[1]) for b in ds_vl])
        if vl < best_val:
            best_val = vl; best_w = model.get_weights(); wait = rw = 0
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


def deeponet_predict_full(model, Xbranch, Xtrunk, chunk=4096):
    N, n_pts, _ = Xtrunk.shape
    out = np.zeros((N, n_pts, 1), np.float32)
    for i in range(N):
        xb = tf.constant(Xbranch[i:i+1])
        for s in range(0, n_pts, chunk):
            e = min(s + chunk, n_pts)
            xt = tf.constant(Xtrunk[i:i+1, s:e])
            out[i, s:e] = model([xb, xt], training=False).numpy()[0]
    return out

def train_deeponet(prep_dir, cytokine, seed, n_trials, tune_epochs, full_epochs, out_dir):
    set_seed(seed)
    idx = CYTOKINE_NAMES.index(cytokine.lower())
    out_dir.mkdir(parents=True, exist_ok=True)

    Xb = np.load(prep_dir / "X_branch.npy").astype(np.float32)
    Xt = np.load(prep_dir / "X_trunk.npy").astype(np.float32)
    Y  = np.load(prep_dir / "Y_target.npy").astype(np.float32)[..., idx:idx+1]
    M  = np.load(prep_dir / "Y_masks_spatial.npy").astype(np.float32)

    with open(prep_dir / "metadata.json") as f:
        meta = json.load(f)
    clip_max = float(meta["scaling"]["max"][idx])
    grid = meta["grid"]

    N = Xb.shape[0]; G2 = Xt.shape[1]; G = int(round(G2**0.5))
    Yf = Y.reshape(N, G2, 1)

    Xbranch = build_branch_inputs(Xb, Xt, idx)
    Xtrunk  = build_trunk_inputs(Xb, Xt)

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
            ds_tr = build_deeponet_dataset(Xbr_tr, Xtr_tr, Yf_tr, bs, chunk_size, shuffle=True)
            ds_vl = build_deeponet_dataset(Xbr_vl, Xtr_vl, Yf_vl, bs, chunk_size, shuffle=False)
            m = DeepONet(hidden=hidden, p=p)
            opt = tf.keras.optimizers.Adam(lr)
            best = train_deeponet_inner(m, opt, ds_tr, ds_vl, epochs=tune_epochs,
                                        patience=8, reduce_patience=5)
            return float(best)
        return obj

    print(f"\n  [DeepONet | {cytokine} | grid {grid}] Optuna {n_trials} trials...")
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

    ds_tr = build_deeponet_dataset(Xbr_tr, Xtr_tr, Yf_tr,
                                   best["batch_size"], best["chunk_size"], shuffle=True)
    ds_vl = build_deeponet_dataset(Xbr_vl, Xtr_vl, Yf_vl,
                                   best["batch_size"], best["chunk_size"], shuffle=False)
    model = DeepONet(hidden=best["hidden"], p=best["p"])
    opt = tf.keras.optimizers.Adam(best["learning_rate"])
    train_deeponet_inner(model, opt, ds_tr, ds_vl, epochs=full_epochs,
                         patience=40, reduce_patience=15)

    Yp_flat = deeponet_predict_full(model, Xbranch, Xtrunk)
    Yp = Yp_flat.reshape(N, G, G, 1)
    Y_phys  = denormalize(Y.reshape(N, G, G, 1), clip_max)
    Yp_phys = denormalize(Yp, clip_max)
    train_elapsed = time.time() - t_start

    suffix = f"{cytokine}_{grid}_{seed}"
    results = {
        "grid": grid, "seed": seed, "cytokine": cytokine, "model": "deeponet",
        "best_params": best, "optuna_best_val_loss": optuna_val,
        "train_time_seconds": round(train_elapsed, 2),
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



def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--prep-dir", type=str, required=True,
                    help="Path to preprocessed/{G}x{G}/ folder")
    ap.add_argument("--out-base", type=str, required=True,
                    help="Base output dir; results go under {out_base}/{model}/")
    ap.add_argument("--seed",     type=int, default=42)
    ap.add_argument("--n-trials", type=int, default=20,
                    help="Optuna trials per model (paper used 20)")
    ap.add_argument("--cytokines", type=str, default="il8,il10")
    ap.add_argument("--models", type=str, default="unet,sta_lstm,deeponet",
                    help="Subset of models to run")
    ap.add_argument("--tune-epochs-fast", type=int, default=20)     # U-Net, STA-LSTM
    ap.add_argument("--tune-epochs-deeponet", type=int, default=30)
    ap.add_argument("--full-epochs-fast", type=int, default=200)
    ap.add_argument("--full-epochs-deeponet", type=int, default=400)
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
    print(f"  DATA MODELS TRAINING")
    print(f"  Prep dir:   {prep_dir}")
    print(f"  Strategy:   {meta.get('strategy', 'unknown')}")
    print(f"  Out base:   {out_base}")
    print(f"  Grid:       {grid}  |  Seed: {args.seed}  |  Trials: {args.n_trials}")
    print(f"  Models:     {models}")
    print(f"  Cytokines:  {cytokines}")
    print(f"{'='*70}")

    for model_name in models:
        out_dir = out_base / model_name
        for cyt in cytokines:
            res_file = out_dir / f"res_{cyt}_{grid}_{args.seed}.json"
            if res_file.exists():
                print(f"\n  [{model_name} | {cyt}] Already exists, skipping.")
                continue
            if model_name == "unet":
                train_unet(prep_dir, cyt, args.seed, args.n_trials,
                           args.tune_epochs_fast, args.full_epochs_fast, out_dir)
            elif model_name == "sta_lstm":
                train_stalstm(prep_dir, cyt, args.seed, args.n_trials,
                              args.tune_epochs_fast, args.full_epochs_fast, out_dir)
            elif model_name == "deeponet":
                train_deeponet(prep_dir, cyt, args.seed, args.n_trials,
                               args.tune_epochs_deeponet, args.full_epochs_deeponet, out_dir)
            else:
                raise ValueError(f"Unknown model: {model_name}")

if __name__ == "__main__":
    main()