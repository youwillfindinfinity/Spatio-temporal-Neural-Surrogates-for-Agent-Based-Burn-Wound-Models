"""U-Net, STA-LSTM, ConvLSTM and FNO runs for the revision.

  --model unet      variants: base, weighted, residual, ntrain35, ntrain52
  --model stalstm   variants: base
  --model convlstm  variants: base, residual          (new baseline, fixed hyperparameters)
  --model fno       variants: base, residual          (new baseline, fixed hyperparameters)
U-Net and STA-LSTM use the hyperparameters stored in results/models (seed-42 Optuna search).
"""
import argparse, importlib.util
import numpy as np
import tensorflow as tf
import common as C

L = tf.keras.layers


def _load(name, rel):
    spec = importlib.util.spec_from_file_location(name, C.ROOT / "scripts" / rel)
    m = importlib.util.module_from_spec(spec); spec.loader.exec_module(m); return m


def add_previous_frame(core, grid, idx, n_in=22):
    """Residual formulation: output = previous frame of the target cytokine + network correction."""
    inp = tf.keras.Input((grid, grid, n_in))
    prev = L.Lambda(lambda z: z[..., 11 + idx:12 + idx])(inp)
    return tf.keras.Model(inp, L.Add()([prev, core(inp)]))


def zero_last_conv(model):
    """Start the residual model exactly at persistence."""
    convs = [l for l in model.layers if isinstance(l, (L.Conv2D, L.Dense))]
    if convs:
        convs[-1].set_weights([np.zeros_like(w) for w in convs[-1].get_weights()])


# ------------------------------------------------------------------ models
def build_unet(grid, hp):
    U = _load("unet_all", "unet/all.py")
    return U.build_unet(grid_size=grid, in_channels=22, base_filters=hp["base_filters"], depth=hp["depth"], dropout=hp["dropout"])


def build_convlstm(grid, filters=32, w=2, n_layers=2, n_extra=0):
    """n_extra: additional static channels (e.g. cell masks at the target time) appended after the recurrent part."""
    inp = tf.keras.Input((grid, grid, 11 * w + n_extra))
    seq = L.Lambda(lambda z: z[..., :11 * w])(inp) if n_extra else inp
    x = L.Reshape((grid, grid, w, 11))(seq)
    x = L.Permute((3, 1, 2, 4))(x)                                   # (w, G, G, 11)
    for k in range(n_layers):
        x = L.ConvLSTM2D(filters, 3, padding="same", return_sequences=k < n_layers - 1)(x)
    if n_extra:
        x = L.Concatenate()([x, L.Lambda(lambda z: z[..., 11 * w:])(inp)])
    x = L.Conv2D(filters, 3, padding="same", activation="relu")(x)
    return tf.keras.Model(inp, L.Conv2D(1, 1, padding="same")(x))


class SpectralConv2D(L.Layer):
    """Fourier layer of Li et al. (2021): linear transform of the lowest Fourier modes."""
    def __init__(self, width, modes, **kw):
        super().__init__(**kw); self.width, self.modes = width, modes

    def build(self, shape):
        s = 1.0 / (self.width * self.width)
        shp = (self.width, self.width, self.modes, self.modes)
        self.w = [self.add_weight(name=f"w{i}", shape=shp, trainable=True,
                                  initializer=tf.keras.initializers.RandomUniform(0.0, s)) for i in range(4)]

    def call(self, x):
        H, W = x.shape[1], x.shape[2]; m = self.modes
        xf = tf.signal.rfft2d(tf.transpose(x, [0, 3, 1, 2]))          # (B, C, H, W//2+1)
        w1 = tf.complex(self.w[0], self.w[1]); w2 = tf.complex(self.w[2], self.w[3])
        top = tf.einsum("bixy,ioxy->boxy", xf[:, :, :m, :m], w1)
        bot = tf.einsum("bixy,ioxy->boxy", xf[:, :, -m:, :m], w2)
        mid = tf.zeros([tf.shape(top)[0], self.width, H - 2 * m, m], tf.complex64)
        of = tf.concat([top, mid, bot], axis=2)
        of = tf.pad(of, [[0, 0], [0, 0], [0, 0], [0, W // 2 + 1 - m]])
        return tf.transpose(tf.signal.irfft2d(of, fft_length=[H, W]), [0, 2, 3, 1])


def build_fno(grid, width=32, modes=12, depth=4, n_in=22):
    inp = tf.keras.Input((grid, grid, n_in))
    lin = np.linspace(-1.0, 1.0, grid, dtype=np.float32)
    xy = np.stack(np.meshgrid(lin, lin, indexing="ij"), -1)[None]     # (1, G, G, 2)
    coords = L.Lambda(lambda z: tf.tile(tf.constant(xy), [tf.shape(z)[0], 1, 1, 1]))(inp)
    x = L.Dense(width)(L.Concatenate()([inp, coords]))
    for k in range(depth):
        y = L.Add()([SpectralConv2D(width, modes)(x), L.Conv2D(width, 1)(x)])
        x = L.Activation("gelu")(y) if k < depth - 1 else y
    x = L.Dense(128, activation="gelu")(x)
    return tf.keras.Model(inp, L.Dense(1)(x))


def build_stalstm(grid, hp):
    S = _load("sta_all", "sta_lstm/all.py")
    return S.STALSTM(grid_size=grid, filters=hp["filters"], lstm_units=hp["lstm_units"])


# ------------------------------------------------------------------ one run
def run(model_name, variant, grid, cytokine, seed):
    if C.result_path(model_name, variant, cytokine, grid, seed).exists():
        print("skip (done):", model_name, variant, cytokine, grid, seed, flush=True); return
    d = C.load(grid, cytokine); idx = d["idx"]
    tf.keras.backend.clear_session(); C.set_seed(seed)
    seq = model_name == "stalstm"
    prep = (lambda X: C.to_seq(X)) if seq else (lambda X: X)
    lr, bs, loss, extra = 1e-3, (4 if grid >= 250 else 8), "mse", {}
    tr0 = 0

    if model_name == "unet":
        hp = C.stored_params("unet", cytokine, grid); lr, bs = hp["learning_rate"], hp["batch_size"]
        net = build_unet(grid, hp)
        if variant == "weighted": loss = C.weighted_mse(); extra["loss"] = "weighted MSE, x10 above 0.05 c_max"
        if variant.startswith("ntrain"): tr0 = 70 - int(variant[6:]); extra["n_train_frames"] = 70 - tr0
        if variant == "residual": zero_last_conv(net); net = add_previous_frame(net, grid, idx)
    elif model_name == "stalstm":
        hp = C.stored_params("sta_lstm", cytokine, grid); lr, bs = hp["learning_rate"], hp["batch_size"]
        net = build_stalstm(grid, hp)
    elif model_name in ("convlstm", "fno"):
        net = build_convlstm(grid) if model_name == "convlstm" else build_fno(grid)
        hp = {"learning_rate": lr, "batch_size": bs, "note": "fixed hyperparameters, no Optuna search"}
        if variant == "residual": zero_last_conv(net); net = add_previous_frame(net, grid, idx)
    else:
        raise ValueError(model_name)
    extra["hyperparameters"] = hp

    X, Y = d["Xu"], d["Y"][..., None]
    t = C.fit_keras(net, prep(X[tr0:70]), Y[tr0:70], prep(X[70:80]), Y[70:80], batch_size=bs, lr=lr, loss=loss)
    extra["n_parameters"] = int(net.count_params())
    predict = lambda Xb, ids: net(prep(Xb), training=False).numpy()[..., 0]
    C.evaluate(model_name, variant, d, seed, predict, train_time=round(t, 1), extra=extra)


VARIANTS = {"unet": ["base", "weighted", "residual", "ntrain35", "ntrain52"], "stalstm": ["base"],
            "convlstm": ["base", "residual"], "fno": ["base", "residual"]}

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True, choices=list(VARIANTS))
    ap.add_argument("--grid", type=int, required=True)
    ap.add_argument("--variants", nargs="*", default=None)
    ap.add_argument("--cytokines", nargs="*", default=["il10", "il8"])
    ap.add_argument("--seeds", nargs="*", type=int, default=[42, 1, 100])
    a = ap.parse_args()
    print("GPU:", tf.config.list_physical_devices("GPU"), flush=True)
    for v in (a.variants or VARIANTS[a.model]):
        for c in a.cytokines:
            for s in a.seeds:
                C.run_safely(run, a.model, v, a.grid, c, s)
