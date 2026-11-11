"""Shared data loading and evaluation for the revision experiments.

Every model is wrapped as  predict(Xu, idxs) -> scaled prediction (n, G, G)
where Xu has the U-Net input layout (n, G, G, 22): channels 0-10 = frame t-2
(6 cytokines + 5 cell masks), 11-21 = frame t-1, and idxs are the sample indices
(sample i predicts t = i + 2 h). All models are then evaluated identically:
  - the manuscript metrics (near / far window, one step ahead)
  - a persistence baseline and skill = 1 - MSE_model / MSE_persistence
  - region-resolved errors around the cytokine front
  - per-frame spatial correlation and R2
  - autoregressive roll-out from t = 81 h against frozen persistence
"""
import sys, json, time, traceback
from pathlib import Path
import numpy as np
from scipy import ndimage

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT)); sys.path.insert(0, str(ROOT / "scripts"))
from paths import PREPROCESSED, RESULTS, RESULTS_MODELS          # noqa: E402
import metrics as paper_metrics                                    # noqa: E402  (scripts/metrics.py)

CYTS = ["il8", "il1", "il6", "il10", "tnf", "tgf"]
TRAIN, VAL = (0, 70), (70, 80)
START, END = 80, 99
WINDOWS = {"near": (80, 90), "far": (90, 99)}
OUT = RESULTS / "rev"


def set_seed(seed):
    import os, random, tensorflow as tf
    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed); np.random.seed(seed); tf.random.set_seed(seed)


def load(grid, cytokine):
    idx = CYTS.index(cytokine)
    dp = PREPROCESSED / f"{grid}x{grid}"
    Xu = np.load(dp / "X_unet.npy").astype(np.float32)
    Y = np.load(dp / "Y_target.npy").astype(np.float32)[..., idx]
    M = np.load(dp / "Y_masks_spatial.npy").astype(np.float32)
    Xt = np.load(dp / "X_trunk.npy").astype(np.float32)
    cmax = float(json.load(open(dp / "metadata.json"))["scaling"]["max"][idx])
    assert Xu.shape[0] == 99, f"expected 99 samples, got {Xu.shape[0]} (check N_TIMESTEPS = 101)"
    return {"grid": grid, "cytokine": cytokine, "idx": idx, "Xu": Xu, "Y": Y, "M": M, "Xt": Xt, "cmax": cmax}


def to_seq(Xu):
    """(n, G, G, 22) -> (n, 2, G, G, 11), the STA-LSTM / DeepONet-branch layout."""
    n, G = Xu.shape[0], Xu.shape[1]
    return Xu.reshape(n, G, G, 2, 11).transpose(0, 3, 1, 2, 4)


def den(a, cmax):
    return (np.asarray(a, np.float64) + 1.0) / 2.0 * cmax


def stored_params(model_dir, cytokine, grid):
    return json.load(open(RESULTS_MODELS / model_dir / f"res_{cytokine}_{grid}_42.json"))["best_params"]


def result_path(model, variant, cytokine, grid, seed):
    d = OUT / model / variant; d.mkdir(parents=True, exist_ok=True)
    return d / f"res_{cytokine}_{grid}_{seed}.json"


# ---------------------------------------------------------------- evaluation
def _r2(yt, yp):
    ss_tot = float(np.sum((yt - yt.mean()) ** 2))
    return 1.0 - float(np.sum((yt - yp) ** 2)) / ss_tot if ss_tot > 0 else float("nan")


def _corr(a, b):
    a = a.ravel() - a.mean(); b = b.ravel() - b.mean()
    d = float(np.sqrt(np.sum(a * a) * np.sum(b * b)))
    return float(np.sum(a * b) / d) if d > 0 else float("nan")


def _regions(truth_frame, cmax, grid):
    """Front band = +/- w pixels around the 0.05 c_max contour of the true field."""
    w = max(1, int(round(grid / 50)))
    act = truth_frame > 0.05 * cmax
    dil = ndimage.binary_dilation(act, iterations=w)
    ero = ndimage.binary_erosion(act, iterations=w)
    return {"front": dil & ~ero, "interior": ero, "exterior": ~dil}, w


def _region_stats(truth, pred, cmax, grid):
    acc = {k: [0.0, 0] for k in ("front", "interior", "exterior")}
    w = 0
    for t in range(truth.shape[0]):
        reg, w = _regions(truth[t], cmax, grid)
        se = (truth[t] - pred[t]) ** 2
        for k, m in reg.items():
            acc[k][0] += float(se[m].sum()); acc[k][1] += int(m.sum())
    tot_se = sum(v[0] for v in acc.values()); tot_n = sum(v[1] for v in acc.values())
    out = {"band_halfwidth_px": w}
    for k, (se, n) in acc.items():
        out[f"{k}_rmse"] = float(np.sqrt(se / n)) if n else float("nan")
        out[f"{k}_rmse_rel_cmax"] = float(np.sqrt(se / n) / cmax) if n else float("nan")
        out[f"{k}_pixel_share"] = n / tot_n if tot_n else float("nan")
        out[f"{k}_error_share"] = se / tot_se if tot_se > 0 else float("nan")
    return out


def _paper(truth, pred, M, cmax):
    m = paper_metrics.calculate_metrics(truth[..., None], pred[..., None], M, cmax)
    return {k: m[k] for k in ("Global_R2", "SSIM", "Avg_Dice", "Spatial_Correlation", "Masked_RMSE", "Unmasked_RMSE")}


def rollout(predict, Xu, idx, mode):
    prev2 = Xu[START, ..., idx].copy(); prev1 = Xu[START, ..., 11 + idx].copy()
    frozen = Xu[START].copy(); out = []
    for i in range(START, END):
        x = Xu[i].copy()
        if mode == "frozen":
            x[..., 0:6] = frozen[..., 11:17] if i > START else frozen[..., 0:6]
            x[..., 11:17] = frozen[..., 11:17]
        x[..., idx] = prev2; x[..., 11 + idx] = prev1
        y = np.clip(np.asarray(predict(x[None], [i]))[0], -1.0, 1.0)
        out.append(y); prev2, prev1 = prev1, y
    return np.stack(out)


def evaluate(model, variant, data, seed, predict, train_time=None, extra=None, do_rollout=True, save_pred=True):
    """Full evaluation of one trained model; writes a JSON (and test predictions) and returns the dict."""
    grid, cyt, idx, cmax = data["grid"], data["cytokine"], data["idx"], data["cmax"]
    Xu, Y, M = data["Xu"], data["Y"], data["M"]
    ids = list(range(START, END))
    t0 = time.time()
    pred_s = np.concatenate([np.asarray(predict(Xu[i:i + 1], [i])) for i in ids])
    infer_s = time.time() - t0
    truth = den(Y[START:END], cmax)
    pred = np.maximum(den(pred_s, cmax), 0.0)
    pers = den(Xu[START:END, ..., 11 + idx], cmax)
    pers_frozen = np.repeat(pers[:1], END - START, axis=0)

    res = {"model": model, "variant": variant, "grid": grid, "cytokine": cyt, "seed": seed,
           "clip_max": cmax, "train_time_seconds": train_time, "infer_seconds_19_frames": round(infer_s, 3),
           "windows": {}, "per_frame": {}, "rollout": {}}
    if extra: res.update(extra)
    for wname, (a, b) in WINDOWS.items():
        s = slice(a - START, b - START)
        mse_m = float(np.mean((truth[s] - pred[s]) ** 2)); mse_p = float(np.mean((truth[s] - pers[s]) ** 2))
        res["windows"][wname] = {
            "model": _paper(truth[s], pred[s], M[a:b], cmax),
            "persistence": _paper(truth[s], pers[s], M[a:b], cmax),
            "skill_vs_persistence": 1.0 - mse_m / mse_p if mse_p > 0 else float("nan"),
            "regions_model": _region_stats(truth[s], pred[s], cmax, grid),
            "regions_persistence": _region_stats(truth[s], pers[s], cmax, grid),
        }
    res["per_frame"] = {
        "t_hours": [i + 2 for i in ids],
        "R2_model": [_r2(truth[k], pred[k]) for k in range(len(ids))],
        "R2_persistence": [_r2(truth[k], pers[k]) for k in range(len(ids))],
        "corr_model": [_corr(truth[k], pred[k]) for k in range(len(ids))],
        "corr_persistence": [_corr(truth[k], pers[k]) for k in range(len(ids))],
        "RMSE_model": [float(np.sqrt(np.mean((truth[k] - pred[k]) ** 2))) for k in range(len(ids))],
        "RMSE_persistence": [float(np.sqrt(np.mean((truth[k] - pers[k]) ** 2))) for k in range(len(ids))],
    }
    ro = {}
    if do_rollout:
        for mode in ("gt", "frozen"):
            r = np.maximum(den(rollout(predict, Xu, idx, mode), cmax), 0.0)
            ro[mode] = r
            res["rollout"][f"R2_{mode}"] = [_r2(truth[k], r[k]) for k in range(len(ids))]
            res["rollout"][f"RMSE_{mode}"] = [float(np.sqrt(np.mean((truth[k] - r[k]) ** 2))) for k in range(len(ids))]
        res["rollout"]["R2_persist_frozen"] = [_r2(truth[k], pers_frozen[k]) for k in range(len(ids))]
        res["rollout"]["RMSE_persist_frozen"] = [float(np.sqrt(np.mean((truth[k] - pers_frozen[k]) ** 2))) for k in range(len(ids))]

    p = result_path(model, variant, cyt, grid, seed)
    json.dump(res, open(p, "w"), indent=1)
    if save_pred:
        np.savez_compressed(p.with_name(f"pred_{cyt}_{grid}_{seed}.npz"), one_step=pred.astype(np.float32),
                            **{f"rollout_{k}": v.astype(np.float32) for k, v in ro.items()})
    w = res["windows"]["near"]
    print(f"[{model}/{variant} {cyt} {grid} s{seed}] near R2 {w['model']['Global_R2']:.4f} "
          f"(persist {w['persistence']['Global_R2']:.4f}) skill {w['skill_vs_persistence']:+.3f}"
          + (f" | rollout19 R2 gt {res['rollout']['R2_gt'][-1]:.3f} frozen-persist {res['rollout']['R2_persist_frozen'][-1]:.3f}" if do_rollout else "")
          + (f" | train {train_time:.0f}s" if train_time else ""), flush=True)
    return res


def run_safely(fn, *a, **k):
    """Run one configuration; on failure print the traceback and continue with the next."""
    try:
        return fn(*a, **k)
    except Exception:
        print("FAILED:", fn.__name__, a, k, flush=True); traceback.print_exc(); sys.stdout.flush()
        return None


def fit_keras(model, Xtr, Ytr, Xvl, Yvl, batch_size, epochs=200, loss="mse", lr=1e-3):
    import os, tensorflow as tf
    epochs = int(os.environ.get("REV_MAX_EPOCHS", epochs))
    model.compile(optimizer=tf.keras.optimizers.Adam(lr), loss=loss)
    t0 = time.time()
    model.fit(Xtr, Ytr, validation_data=(Xvl, Yvl), epochs=epochs, batch_size=batch_size, verbose=0,
              callbacks=[tf.keras.callbacks.EarlyStopping(monitor="val_loss", patience=20, restore_best_weights=True),
                         tf.keras.callbacks.ReduceLROnPlateau(monitor="val_loss", factor=0.5, patience=10, min_lr=1e-6)])
    return time.time() - t0


def weighted_mse(weight=10.0, thr_scaled=-0.9):
    """MSE with extra weight on points whose true value exceeds 5 % of c_max (-0.9 in scaled units)."""
    import tensorflow as tf
    def loss(yt, yp):
        w = 1.0 + (weight - 1.0) * tf.cast(yt > thr_scaled, tf.float32)
        return tf.reduce_sum(w * tf.square(yt - yp)) / tf.reduce_sum(w)
    return loss
