import argparse, json, gc, os, sys, time, warnings
warnings.filterwarnings("ignore")
import numpy as np
import tensorflow as tf
import deepxde as dde
import optuna

dde.config.set_default_float("float32")
optuna.logging.set_verbosity(optuna.logging.WARNING)

GRID         = 500
G2           = GRID * GRID
DATA_DIR     = "preprocessed/500x500" # overridden by --data-dir
RESULTS_DIR  = "models/pinn"
CYT_MAP      = {"il8": 0, "il10": 3}
CYT_DIFFUSE  = {0: 2.09e-6, 3: 1.45e-8} # m²/s, scaled to normalised domain
CYT_DECAY    = {0: 0.200,   3: 0.500}

TRAIN_SL     = slice(0, 140)
N_DOMAIN     = 3000 # reduced from 5000 for VRAM safety
N_BOUNDARY   = 500 # reduced from 1000
N_OBS_PER_T  = 100
WARMUP       = 5000
PHYSICS_ITER = 10000
LBFGS_ITER   = 5000
N_OPTUNA     = 20
TUNE_ITERS   = 2000

tf.config.set_visible_devices([], "GPU")  # CPU-only (rome partition)

def load_pinn_data(cyt_idx):
    md    = json.load(open(f"{DATA_DIR}/metadata.json"))
    clip  = float(md["clip_max"][cyt_idx])
    Yt    = np.load(f"{DATA_DIR}/Y_target.npy")[..., cyt_idx] # (99,G,G)
    Yraw  = np.load(f"{DATA_DIR}/Y_raw_phys.npy")[1:, ..., cyt_idx] # (99,G,G)
    Xc    = np.load(f"{DATA_DIR}/X_colloc.npy") # (99,G²,3)
    masks = np.load(f"{DATA_DIR}/Y_masks_spatial.npy") # (99,G,G,5)
    Ym    = np.load(f"{DATA_DIR}/Y_masks_pinn.npy") # (99,G²,5)
    bc_m  = np.load(f"{DATA_DIR}/Y_bc_mask.npy") # (G,G)

    print("Building source term tensors...")
    Xb = np.load(f"{DATA_DIR}/X_branch.npy", mmap_mode="r") # (99,2,G,G,11)
    # channels 6-10 in last dim are the 5 cell-type masks (EC,NN,NA,M1,M2)
    ec_rate  = {0: 234e-5, 3: 0.0}
    ndn_rate = {0: 1.46e-5, 3: 0.0}
    na_rate  = {0: 0.0, 3: 0.0}
    m1_rate  = {0: 0.0, 3: 45e-5}
    uptake   = {0: 3.024e-5, 3: 0.0} # NA uptake for IL-8

    n_train = 140
    s1 = np.zeros((n_train, G2, 1), dtype=np.float32)
    s2 = np.zeros((n_train, G2, 1), dtype=np.float32)
    e  = np.zeros((n_train, G2, 1), dtype=np.float32)
    for i in range(n_train):
        ec  = Xb[i, 1, :, :, 6].astype(np.float32).reshape(G2, 1)
        ndn = Xb[i, 1, :, :, 7].astype(np.float32).reshape(G2, 1)
        na  = Xb[i, 1, :, :, 8].astype(np.float32).reshape(G2, 1)
        m1  = Xb[i, 1, :, :, 9].astype(np.float32).reshape(G2, 1)
        s1[i] = ec_rate[cyt_idx]  * ec + ndn_rate[cyt_idx] * ndn
        s2[i] = m1_rate[cyt_idx]  * m1
        e[i]  = uptake[cyt_idx]   * na
    del Xb; gc.collect()
    return Yt, Yraw, Xc, masks, Ym, bc_m, s1, s2, e, clip


def build_pinn(n_layers, hidden):
    return dde.nn.FNN([3] + [hidden] * n_layers + [1], "tanh", "Glorot normal")


def make_deepxde_problem(Yt_train, Xc, s1, s2, e, bc_m, cyt_idx, clip):
    D = CYT_DIFFUSE[cyt_idx]
    k = CYT_DECAY[cyt_idx]

    obs_pts, obs_vals = [], []
    for ti in range(140):
        idx = np.random.choice(G2, N_OBS_PER_T, replace=False)
        pts = Xc[ti, idx, :]
        val = Yt_train[ti].reshape(G2)[idx, np.newaxis]
        obs_pts.append(pts)
        obs_vals.append(val)
    obs_pts  = np.vstack(obs_pts).astype(np.float32)
    obs_vals = np.vstack(obs_vals).astype(np.float32)

    s1_tf  = tf.constant(s1, dtype=tf.float32)
    s2_tf  = tf.constant(s2, dtype=tf.float32)
    e_tf   = tf.constant(e,  dtype=tf.float32)

    def source_at(x):
        t = x[:, 2:3]
        t_idx = tf.cast(tf.round(t[:, 0] * 139), tf.int32)
        t_idx = tf.clip_by_value(t_idx, 0, 139)
        xi = tf.cast(tf.round((x[:, 0] + 1.0) / 2.0 * (GRID-1)), tf.int32)
        yi = tf.cast(tf.round((x[:, 1] + 1.0) / 2.0 * (GRID-1)), tf.int32)
        xi = tf.clip_by_value(xi, 0, GRID-1)
        yi = tf.clip_by_value(yi, 0, GRID-1)
        flat_idx = yi * GRID + xi
        s1v = tf.gather(tf.gather(s1_tf, t_idx)[:, :, 0], flat_idx, batch_dims=1)
        s2v = tf.gather(tf.gather(s2_tf, t_idx)[:, :, 0], flat_idx, batch_dims=1)
        ev  = tf.gather(tf.gather(e_tf,  t_idx)[:, :, 0], flat_idx, batch_dims=1)
        return s1v + s2v, ev

    geom = dde.geometry.Rectangle([-1, -1], [1, 1])
    timedomain = dde.geometry.TimeDomain(0, 1)
    geomtime   = dde.geometry.GeometryXTime(geom, timedomain)

    def pde(x, u):
        du_t  = dde.grad.jacobian(u, x, i=0, j=2)
        du_xx = dde.grad.hessian(u, x, i=0, j=0)
        du_yy = dde.grad.hessian(u, x, i=1, j=1)
        s, ev = source_at(x)
        u_off = u + 1.0
        return du_t - D*(du_xx + du_yy) + k*u_off - tf.expand_dims(s,1) + tf.expand_dims(ev,1)*u_off

    def bc_func(x, on_boundary):
        corner = 2 * (GRID//10) / GRID - 1.0
        x0, x1 = x[0], x[1]
        return on_boundary and (
            (x1 < -1.0 + 1e-3 and x0 < corner) or
            (x1 >  1.0 - 1e-3 and x0 > -corner) or
            (x0 >  1.0 - 1e-3 and x1 > -corner) or
            (x0 < -1.0 + 1e-3 and x1 < corner))

    bc  = dde.icbc.DirichletBC(geomtime, lambda x: -1.0 * np.ones((len(x),1)), bc_func)
    obs = dde.icbc.PointSetBC(obs_pts, obs_vals, component=0)

    data = dde.data.TimePDE(
        geomtime, pde, [bc, obs],
        num_domain=N_DOMAIN, num_boundary=N_BOUNDARY,
        num_test=2000)
    return data


def run(cyt_name, seed):
    print(f"\n{'='*60}")
    print(f"PINN 500x500 | cytokine={cyt_name} | seed={seed}")
    print(f"{'='*60}")
    np.random.seed(seed)
    tf.random.set_seed(seed)
    os.makedirs(RESULTS_DIR, exist_ok=True)

    cyt_idx = CYT_MAP[cyt_name]
    Yt, Yraw, Xc, masks, Ym, bc_m, s1, s2, e, clip = load_pinn_data(cyt_idx)

    Yt_tr = Yt[:140]
    data  = make_deepxde_problem(Yt_tr, Xc, s1, s2, e, bc_m, cyt_idx, clip)

    def objective(trial):
        n_l  = trial.suggest_categorical("n_layers", [3, 4, 5])
        h    = trial.suggest_categorical("hidden",   [64, 128])
        lr   = trial.suggest_float("lr",         1e-4, 1e-3, log=True)
        lpde = trial.suggest_float("lambda_pde", 1e-4, 1.0,  log=True)
        lbc  = trial.suggest_float("lambda_bc",  1e-4, 0.1,  log=True)

        net  = build_pinn(n_l, h)
        model = dde.Model(data, net)
        model.compile("adam", lr=lr,
                       loss_weights=[lpde, lbc, 1.0])
        model.train(iterations=TUNE_ITERS, display_every=99999)
        loss_hist = model.losshistory.loss_train[-1]
        del model; gc.collect(); tf.keras.backend.clear_session()
        return float(np.sum(loss_hist))

    study = optuna.create_study(direction="minimize",
                                sampler=optuna.samplers.TPESampler(seed=seed),
                                pruner=optuna.pruners.MedianPruner())
    study.optimize(objective, n_trials=N_OPTUNA, show_progress_bar=False)
    best = study.best_params
    print(f"Best params: {best}")

    # Full training
    net   = build_pinn(best["n_layers"], best["hidden"])
    model = dde.Model(data, net)

    # Phase 1: warmup (data only)
    t_train_start = time.time()
    model.compile("adam", lr=best["lr"], loss_weights=[0.0, best["lambda_bc"], 1.0])
    model.train(iterations=WARMUP, display_every=1000)

    # Phase 2: physics
    model.compile("adam", lr=best["lr"],
                  loss_weights=[best["lambda_pde"], best["lambda_bc"], 1.0])
    model.train(iterations=PHYSICS_ITER, display_every=1000)

    # Phase 3: L-BFGS
    dde.optimizers.set_LBFGS_options(maxiter=LBFGS_ITER)
    model.compile("L-BFGS",
                  loss_weights=[best["lambda_pde"], best["lambda_bc"], 1.0])
    model.train()
    train_elapsed = time.time() - t_train_start

    # Eval on test set
    test_idx  = np.arange(160, 199)
    preds     = []
    t_pred_start = time.time()
    for ti in test_idx:
        pts = Xc[ti].astype(np.float32)
        u   = model.predict(pts)
        u_p = (u.reshape(GRID, GRID) + 1.0) / 2.0 * clip
        preds.append(np.clip(u_p, 0, None))
    pred_elapsed = time.time() - t_pred_start
    pred_arr = np.array(preds)
    y_true   = Yraw[160:199]

    from sklearn.metrics import r2_score
    from skimage.metrics import structural_similarity as ssim
    r2 = float(r2_score(y_true.flatten(), pred_arr.flatten()))
    rmse = float(np.sqrt(np.mean((y_true - pred_arr)**2)))
    ssim_v = float(np.mean([ssim(y_true[t], pred_arr[t], data_range=clip)
                             for t in range(39) if y_true[t].std() > 1e-12]))

    metrics = {"Global_R2": r2, "Unmasked_RMSE": rmse, "SSIM": ssim_v,
               "cytokine": cyt_name, "seed": seed, "grid": 500,
               "train_time_seconds": round(train_elapsed, 2),
               "pred_time_seconds":  round(pred_elapsed, 4),
               "best_params": best, "model": "pinn"}
    out_path = f"{RESULTS_DIR}/res_{cyt_name}_500_{seed}.json"
    with open(out_path, "w") as f:
        json.dump(metrics, f, indent=2)
    print(f"Saved → {out_path}")
    del model; gc.collect(); tf.keras.backend.clear_session()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--cytokine", choices=["il8","il10"], required=True)
    parser.add_argument("--seed",     type=int,               required=True)
    parser.add_argument("--data-dir", default="preprocessed/500x500",
                        dest="data_dir",
                        help="Path to preprocessed/500x500 directory")
    args = parser.parse_args()
    DATA_DIR = args.data_dir
    run(args.cytokine, args.seed)
