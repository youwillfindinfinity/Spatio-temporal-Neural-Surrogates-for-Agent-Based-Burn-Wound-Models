"""Effect of the amount of training data (several ABM trajectories instead of one).
U-Net in increment and direct form (hyperparameters from run_main.py), trained on
the first 70 targets of E3 seeds {1}, {1,2} or {1,2,3}; validation on seed 1 (t = 72-81 h);
frozen-weight evaluation on all other runs (seeds 4 and 5 are never used for training)."""
import argparse, json
import numpy as np
import tensorflow as tf
import common as C
import run_main as M


def run(model, form, grid, cytokine, seed, n_traj):
    name, variant = f"multitraj_{model}", f"{form}_{n_traj}traj"
    if C.result_path(name, variant, cytokine, grid, seed).exists():
        print("skip (done):", name, variant, cytokine, grid, seed, flush=True); return
    d = C.load(grid, cytokine)
    hp_file = C.OUT / f"main_{model}" / form / f"hp_{cytokine}_{grid}.json"
    hp = json.load(open(hp_file))["best_params"]
    Xs, Ys = [d["Xu"][:70]], [d["Y"][:70, ..., None]]
    for s in range(2, n_traj + 1):
        e = C.load(grid, cytokine, C.train_run(grid, s)); Xs.append(e["Xu"][:70]); Ys.append(e["Y"][:70, ..., None])
    X, Y = np.concatenate(Xs), np.concatenate(Ys)
    tf.keras.backend.clear_session(); C.set_seed(seed)
    net = M.build(model, form, grid, d["idx"], hp)
    t = C.fit_keras(net, X, Y, d["Xu"][70:80], d["Y"][70:80, ..., None], batch_size=hp["batch_size"], lr=hp["learning_rate"])
    predict = lambda Xb, ids: net(Xb, training=False).numpy()[..., 0]
    C.evaluate(name, variant, d, seed, predict, train_time=round(t, 1), transfer=True, save_pred=False,
               extra={"hyperparameters": hp, "n_train_trajectories": n_traj, "n_train_samples": int(X.shape[0]),
                      "train_runs": [C.train_run(grid, s) for s in range(1, n_traj + 1)]})


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--grid", type=int, required=True)
    ap.add_argument("--models", nargs="*", default=["unet"])
    ap.add_argument("--forms", nargs="*", default=["increment", "direct"])
    ap.add_argument("--cytokines", nargs="*", default=["il10", "il8"])
    ap.add_argument("--seeds", nargs="*", type=int, default=[42, 1, 100])
    a = ap.parse_args()
    print("GPU:", tf.config.list_physical_devices("GPU"), flush=True)
    for m in a.models:
        for f in a.forms:
            for n in (1, 2, 3):
                for c in a.cytokines:
                    for s in a.seeds:
                        C.run_safely(run, m, f, a.grid, c, s, n)
