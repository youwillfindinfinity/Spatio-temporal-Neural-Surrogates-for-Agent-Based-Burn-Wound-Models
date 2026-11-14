"""Look-back window sweep with fixed architectures (U-Net with the stored hyperparameters,
ConvLSTM with fixed hyperparameters). For every window w the network sees the w preceding
frames (6 cytokines + 5 cell masks each). Training targets are identical for all w
(t = 5..71 h), validation t = 72..81 h, test t = 82..100 h, so only the input changes.
"""
import argparse
import numpy as np
import tensorflow as tf
import common as C
import run_grid_models as G


def full_sequence(grid):
    dp = C.run_dir(grid)
    X = np.load(dp / "X_unet.npy").astype(np.float32)
    Y = np.load(dp / "Y_target.npy").astype(np.float32)
    M = np.load(dp / "Y_masks_spatial.npy").astype(np.float32)
    seq = np.concatenate([X[0, ..., 0:11][None], X[0, ..., 11:22][None], np.concatenate([Y, M], -1)])   # (101, G, G, 11)
    assert np.allclose(seq[2:100], X[1:, ..., 11:22], atol=1e-6), "sequence reconstruction mismatch"
    return seq


def windows(seq, w, targets):
    return np.stack([np.concatenate([seq[t - k] for k in range(w, 0, -1)], -1) for t in targets])


def run(model_name, w, grid, cytokine, seed):
    variant = f"w{w}"; name = f"lookback_{model_name}"
    if C.result_path(name, variant, cytokine, grid, seed).exists():
        print("skip (done):", name, variant, cytokine, grid, seed, flush=True); return
    d = C.load(grid, cytokine); idx = d["idx"]; seq = full_sequence(grid)
    tf.keras.backend.clear_session(); C.set_seed(seed)
    ttr, tvl = list(range(5, 72)), list(range(72, 82))
    Xtr, Xvl = windows(seq, w, ttr), windows(seq, w, tvl)
    Ytr, Yvl = seq[ttr][..., idx:idx + 1], seq[tvl][..., idx:idx + 1]
    if model_name == "unet":
        hp = C.stored_params("unet", cytokine, grid); lr, bs = hp["learning_rate"], hp["batch_size"]
        U = G._load("unet_all", "unet/all.py")
        net = U.build_unet(grid_size=grid, in_channels=11 * w, base_filters=hp["base_filters"], depth=hp["depth"], dropout=hp["dropout"])
    else:
        lr, bs = 1e-3, (4 if grid >= 250 else 8); hp = {"learning_rate": lr, "batch_size": bs, "filters": 32}
        net = G.build_convlstm(grid, w=w)
    t = C.fit_keras(net, Xtr, Ytr, Xvl, Yvl, batch_size=bs, lr=lr)
    predict = lambda Xb, ids: net(windows(seq, w, [i + 2 for i in ids]), training=False).numpy()[..., 0]
    C.evaluate(name, variant, d, seed, predict, train_time=round(t, 1), do_rollout=False, save_pred=False,
               extra={"lookback": w, "hyperparameters": hp, "n_parameters": int(net.count_params()), "n_train_targets": len(ttr)})


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--grid", type=int, required=True)
    ap.add_argument("--models", nargs="*", default=["unet", "convlstm"])
    ap.add_argument("--windows", nargs="*", type=int, default=[1, 2, 3, 4, 5])
    ap.add_argument("--cytokines", nargs="*", default=["il10", "il8"])
    ap.add_argument("--seeds", nargs="*", type=int, default=[42, 1, 100])
    a = ap.parse_args()
    print("GPU:", tf.config.list_physical_devices("GPU"), flush=True)
    for m in a.models:
        for w in a.windows:
            for c in a.cytokines:
                for s in a.seeds:
                    C.run_safely(run, m, w, a.grid, c, s)
