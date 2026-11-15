"""Main experiments of the revised manuscript: every architecture in two forms,
  direct     the network outputs the next frame
  increment  the network outputs a correction that is added to the previous frame
each with its own hyperparameter search under the protocol of the original study
(Optuna TPE, seed 42, 20 trials x 20 epochs, early-stopping patience 5, validation MSE),
followed by final training (<= 200 epochs, patience 20) for seeds 42, 1 and 100.

Architectures: unet, convlstm, fno, stalstm, pointwise (gated MLP applied independently at
every lattice point to the local field and mask values plus coordinates; this is the
DeepONet trunk without the branch network).
Results: results/rev/main_<model>/<form>/ ; selected hyperparameters: hp_<cytokine>_<grid>.json

With --target-masks the five cell-type masks at the target time t are given as additional input
channels (prediction conditional on the cell positions at the time being predicted). These runs are
stored as tmask_<model>. Not available for the STA-LSTM.
"""
import argparse, gc, json, os, time
import numpy as np
import optuna
import tensorflow as tf
import common as C
import run_grid_models as G

L = tf.keras.layers
optuna.logging.set_verbosity(optuna.logging.WARNING)
N_TRIALS = int(os.environ.get("REV_TRIALS", 20))
TUNE_EPOCHS = int(os.environ.get("REV_TUNE_EPOCHS", 20))


def space(model, trial):
    hp = {"learning_rate": trial.suggest_float("learning_rate", 1e-5, 1e-3, log=True),
          "batch_size": trial.suggest_categorical("batch_size", [4, 8, 16])}
    if model == "unet":
        hp.update(base_filters=trial.suggest_categorical("base_filters", [16, 32, 64]), depth=trial.suggest_categorical("depth", [3, 4]),
                  dropout=trial.suggest_float("dropout", 0.0, 0.3))
    elif model == "stalstm":
        hp.update(filters=trial.suggest_categorical("filters", [32, 64]), lstm_units=trial.suggest_categorical("lstm_units", [64, 128, 256]))
    elif model == "convlstm":
        hp.update(filters=trial.suggest_categorical("filters", [16, 32, 64]), n_layers=trial.suggest_categorical("n_layers", [1, 2]))
    elif model == "fno":
        hp.update(width=trial.suggest_categorical("width", [16, 32, 64]), modes=trial.suggest_categorical("modes", [8, 12, 16]),
                  depth=trial.suggest_categorical("depth", [3, 4]))
    elif model == "pointwise":
        hp.update(hidden=trial.suggest_categorical("hidden", [128, 256]), p=trial.suggest_categorical("p", [64, 128, 256]))
    return hp


def build(model, form, grid, idx, hp, n_extra=0):
    n_in = 22 + n_extra
    inp = tf.keras.Input((grid, grid, n_in))
    if model == "unet":
        U = G._load("unet_all", "unet/all.py")
        y = U.build_unet(grid_size=grid, in_channels=n_in, base_filters=hp["base_filters"], depth=hp["depth"], dropout=hp["dropout"])(inp)
    elif model == "convlstm":
        y = G.build_convlstm(grid, filters=hp["filters"], n_layers=hp["n_layers"], n_extra=n_extra)(inp)
    elif model == "fno":
        y = G.build_fno(grid, width=hp["width"], modes=hp["modes"], depth=hp["depth"], n_in=n_in)(inp)
    elif model == "stalstm":
        if n_extra: raise ValueError("target-time masks are not implemented for the STA-LSTM")
        S = G._load("sta_all", "sta_lstm/all.py")
        x = L.Permute((3, 1, 2, 4))(L.Reshape((grid, grid, 2, 11))(inp))
        y = S.STALSTM(grid_size=grid, filters=hp["filters"], lstm_units=hp["lstm_units"])(x)
    elif model == "pointwise":
        D = G._load("deeponet_all", "deeponet/all.py")
        lin = np.linspace(-1.0, 1.0, grid, dtype=np.float32)
        xy = np.stack(np.meshgrid(lin, lin, indexing="ij"), -1)[None]
        coords = L.Lambda(lambda z: tf.tile(tf.constant(xy), [tf.shape(z)[0], 1, 1, 1]))(inp)
        y = L.Dense(1)(D.Trunk(hp["hidden"], hp["p"])(L.Concatenate()([coords, inp])))
    else:
        raise ValueError(model)
    if form == "increment":
        scale = L.Dense(1, use_bias=False, kernel_initializer="zeros", name="increment_gate")   # starts exactly at persistence
        y = L.Add()([L.Lambda(lambda z: z[..., 11 + idx:12 + idx])(inp), scale(y)])
    return tf.keras.Model(inp, y)


def inputs(d, tmask):
    """Model input: the two look-back frames, optionally followed by the cell masks at the target time."""
    return np.concatenate([d["Xu"], d["M"]], -1) if tmask else d["Xu"]


def tune(model, form, d, tmask=False):
    X, Y, idx, grid = inputs(d, tmask), d["Y"][..., None], d["idx"], d["grid"]

    def objective(trial):
        tf.keras.backend.clear_session(); C.set_seed(42)
        hp = space(model, trial)
        try:
            net = build(model, form, grid, idx, hp, 5 if tmask else 0)
            net.compile(optimizer=tf.keras.optimizers.Adam(hp["learning_rate"]), loss="mse")
            h = net.fit(X[:70], Y[:70], validation_data=(X[70:80], Y[70:80]), epochs=TUNE_EPOCHS, batch_size=hp["batch_size"], verbose=0,
                        callbacks=[tf.keras.callbacks.EarlyStopping(monitor="val_loss", patience=5, restore_best_weights=True)])
            v = float(np.min(h.history["val_loss"]))
        except (tf.errors.ResourceExhaustedError, MemoryError):
            # configuration does not fit on the GPU: free the memory and tell the sampler it is a bad region
            net = None; tf.keras.backend.clear_session(); gc.collect()
            print(f"  trial {trial.number}: out of GPU memory for {hp} - scored as failed", flush=True)
            return 1e6
        return v if np.isfinite(v) else 1e6

    study = optuna.create_study(direction="minimize", sampler=optuna.samplers.TPESampler(seed=42))
    t0 = time.time(); study.optimize(objective, n_trials=N_TRIALS, catch=(Exception,))
    return dict(study.best_params), float(study.best_value), time.time() - t0


def run(model, form, grid, cytokine, seed, tmask=False):
    name = f"{'tmask' if tmask else 'main'}_{model}"
    if C.result_path(name, form, cytokine, grid, seed).exists():
        print("skip (done):", name, form, cytokine, grid, seed, flush=True); return
    d = C.load(grid, cytokine)
    hp_file = C.result_path(name, form, cytokine, grid, 0).with_name(f"hp_{cytokine}_{grid}.json")
    if hp_file.exists():
        info = json.load(open(hp_file))
    else:
        hp, val, secs = tune(model, form, d, tmask)
        info = {"best_params": hp, "best_val_loss": val, "tuning_seconds": round(secs, 1), "n_trials": N_TRIALS, "tune_epochs": TUNE_EPOCHS}
        json.dump(info, open(hp_file, "w"), indent=1)
        print(f"[{name}/{form} {cytokine} {grid}] tuned in {secs:.0f}s: {hp}", flush=True)
    hp = info["best_params"]
    tf.keras.backend.clear_session(); C.set_seed(seed)
    net = build(model, form, grid, d["idx"], hp, 5 if tmask else 0)
    X, Y = inputs(d, tmask), d["Y"][..., None]
    t = C.fit_keras(net, X[:70], Y[:70], X[70:80], Y[70:80], batch_size=hp["batch_size"], lr=hp["learning_rate"])
    if tmask:
        def predict(Xb, ids, data):          # masks at the target time are taken from the run being predicted
            return net(np.concatenate([Xb, data["M"][ids]], -1), training=False).numpy()[..., 0]
        predict.needs_data = True
    else:
        predict = lambda Xb, ids: net(Xb, training=False).numpy()[..., 0]
    C.evaluate(name, form, d, seed, predict, train_time=round(t, 1), transfer=True,
               extra={"target_time_masks": bool(tmask), "hyperparameters": hp, "tuning": {k: v for k, v in info.items() if k != "best_params"}, "n_parameters": int(net.count_params())})
    net.save_weights(str(C.result_path(name, form, cytokine, grid, seed).with_name(f"weights_{cytokine}_{grid}_{seed}.weights.h5")))


MODELS = ["unet", "convlstm", "fno", "stalstm", "pointwise"]
if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True, choices=MODELS)
    ap.add_argument("--grid", type=int, required=True)
    ap.add_argument("--forms", nargs="*", default=["increment", "direct"])
    ap.add_argument("--cytokines", nargs="*", default=["il10", "il8"])
    ap.add_argument("--seeds", nargs="*", type=int, default=[42, 1, 100])
    ap.add_argument("--target-masks", action="store_true")
    a = ap.parse_args()
    print("GPU:", tf.config.list_physical_devices("GPU"), flush=True)
    for f in a.forms:
        for c in a.cytokines:
            for s in a.seeds:
                C.run_safely(run, a.model, f, a.grid, c, s, a.target_masks)
