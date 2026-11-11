"""DeepONet re-training and ablations for the revision (hyperparameters from results/models).

variants
  base       architecture of the manuscript (7-scalar branch, gated trunk on x, y + 22 field values)
  no_branch  branch replaced by a constant trainable vector (trunk-only model)
  no_gate    gated trunk replaced by a plain tanh MLP of the same width
  no_fields  trunk receives only (x, y); the 22 point-wise field values are removed
When --grid 100 and variant base, the trained model is additionally evaluated without
re-training on the 250^2 data (zero-shot resolution transfer).
"""
import argparse, importlib.util, os, time
import numpy as np
import tensorflow as tf
import common as C

spec = importlib.util.spec_from_file_location("deeponet_all", C.ROOT / "scripts" / "deeponet" / "all.py")
D = importlib.util.module_from_spec(spec); spec.loader.exec_module(D)


class PlainTrunk(tf.keras.layers.Layer):
    def __init__(self, hidden, p, **kw):
        super().__init__(**kw)
        self.ls = [tf.keras.layers.Dense(hidden, activation="tanh") for _ in range(3)]
        self.out = tf.keras.layers.Dense(p)

    def call(self, x):
        for l in self.ls: x = l(x)
        return self.out(x)


class Net(tf.keras.Model):
    def __init__(self, hidden, p, variant):
        super().__init__()
        self.variant = variant
        self.trunk = PlainTrunk(hidden, p) if variant == "no_gate" else D.Trunk(hidden, p)
        if variant == "no_branch":
            self.const = self.add_weight(shape=(p,), initializer="glorot_uniform", trainable=True, name="const_branch")
        else:
            self.branch = D.Branch(hidden, p)
        self.bias = self.add_weight(shape=(1,), initializer="zeros", trainable=True, name="bias")

    def call(self, inputs, training=False):
        xb, xt = inputs
        t = self.trunk(xt)
        if self.variant == "no_branch":
            r = tf.einsum("p,bnp->bn", self.const, t) + self.bias
        else:
            r = tf.einsum("bp,bnp->bn", self.branch(xb, training=training), t) + self.bias
        return tf.expand_dims(r, -1)


def features(Xu, Xt_rows, idx, variant):
    """Branch (n, 7) and trunk (n, G*G, d) inputs, built exactly as in scripts/deeponet/all.py."""
    Xb5 = C.to_seq(Xu)
    br = D.build_branch_inputs(Xb5, Xt_rows, idx)
    tr = D.build_trunk_inputs(Xb5, Xt_rows)
    if variant == "no_fields": tr = tr[..., :2]
    return br.astype(np.float32), tr.astype(np.float32)


def dataset(br, tr, Yf, batch, chunk, shuffle):
    N, n_pts, d = tr.shape
    def gen():
        order = np.random.permutation(N) if shuffle else np.arange(N)
        for i in order:
            for s in range(0, n_pts, chunk):
                e = min(s + chunk, n_pts); size = e - s
                xt, y = tr[i, s:e], Yf[i, s:e]
                if size < chunk:
                    xt = np.concatenate([xt, np.zeros((chunk - size, d), np.float32)])
                    y = np.concatenate([y, np.zeros((chunk - size, 1), np.float32)])
                yield (br[i], xt, np.array([size], np.int32)), y
    sig = ((tf.TensorSpec((7,), tf.float32), tf.TensorSpec((chunk, d), tf.float32), tf.TensorSpec((1,), tf.int32)),
           tf.TensorSpec((chunk, 1), tf.float32))
    return tf.data.Dataset.from_generator(gen, output_signature=sig).batch(batch).prefetch(tf.data.AUTOTUNE)


def make_predict(model, data, variant, chunk=8192):
    idx = data["idx"]
    def predict(Xb, ids):
        br, tr = features(Xb, data["Xt"][ids], idx, variant)
        n, n_pts, _ = tr.shape; G = int(round(n_pts ** 0.5)); out = np.zeros((n, n_pts), np.float32)
        for k in range(n):
            for s in range(0, n_pts, chunk):
                out[k, s:s + chunk] = model([tf.constant(br[k:k + 1]), tf.constant(tr[k:k + 1, s:s + chunk])], training=False).numpy()[0, :, 0]
        return out.reshape(n, G, G)
    return predict


def run(variant, grid, cytokine, seed):
    if C.result_path("deeponet", variant, cytokine, grid, seed).exists():
        print("skip (done):", variant, cytokine, grid, seed, flush=True); return
    d = C.load(grid, cytokine); idx = d["idx"]
    hp = C.stored_params("deeponet_h", cytokine, grid)
    tf.keras.backend.clear_session(); C.set_seed(seed)
    br, tr = features(d["Xu"], d["Xt"], idx, variant)
    Yf = d["Y"].reshape(99, -1, 1)
    t0 = time.time()
    ds_tr = dataset(br[:70], tr[:70], Yf[:70], hp["batch_size"], hp["chunk_size"], True)
    ds_vl = dataset(br[70:80], tr[70:80], Yf[70:80], hp["batch_size"], hp["chunk_size"], False)
    model = Net(hp["hidden"], hp["p"], variant)
    opt = tf.keras.optimizers.Adam(hp["learning_rate"])
    epochs = int(os.environ.get("REV_MAX_EPOCHS", D.FULL_EPOCHS))
    D.train_model(model, opt, ds_tr, ds_vl, epochs=epochs, patience=40, reduce_patience=15, verbose=False)
    t = time.time() - t0
    extra = {"hyperparameters": hp, "n_parameters": int(model.count_params())}
    C.evaluate("deeponet", variant, d, seed, make_predict(model, d, variant), train_time=round(t, 1), extra=extra)
    zs = int(os.environ.get("REV_ZS_TARGET", 250))
    if variant == "base" and grid == int(os.environ.get("REV_ZS_SOURCE", 100)) and (C.PREPROCESSED / f"{zs}x{zs}" / "X_unet.npy").exists():
        d2 = C.load(zs, cytokine)
        C.evaluate("deeponet", f"zeroshot_{grid}to{zs}", d2, seed, make_predict(model, d2, variant),
                   extra={"note": "trained at 100^2, evaluated at 250^2 without re-training; inputs scaled with the 250^2 c_max"},
                   do_rollout=False)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--grid", type=int, required=True)
    ap.add_argument("--variants", nargs="*", default=["base", "no_branch", "no_gate", "no_fields"])
    ap.add_argument("--cytokines", nargs="*", default=["il10", "il8"])
    ap.add_argument("--seeds", nargs="*", type=int, default=[42])
    a = ap.parse_args()
    print("GPU:", tf.config.list_physical_devices("GPU"), flush=True)
    for v in a.variants:
        for c in a.cytokines:
            for s in a.seeds:
                C.run_safely(run, v, a.grid, c, s)
