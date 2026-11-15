"""Autoregressive roll-out of the retrained U-Net over the test period (t = 82..100 h).
The model is started from the last two observed frames (t = 80, 81 h) and then fed its
own predictions for the target cytokine. Cell masks are always taken from the ABM.
Two treatments of the five non-target cytokine channels:
  gt     - taken from the ABM at every step (optimistic)
  frozen - held at their last observed value, t = 81 h (strict)
Baselines: one-step prediction (as in the paper), one-step persistence (previous true
frame) and frozen persistence (frame at t = 81 h held for the whole period)."""
import sys, json, argparse, csv, importlib.util
from pathlib import Path
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from paths import PREPROCESSED, RESULTS, RESULTS_MODELS
import numpy as np

CYTS = ["il8", "il1", "il6", "il10", "tnf", "tgf"]
START, END = 80, 99                     # sample indices; sample i predicts t = i + 2 h


def frame_scores(yt, yp, mask):
    yp = np.maximum(yp, 0.0)
    ss_res = float(np.sum((yt - yp) ** 2)); ss_tot = float(np.sum((yt - yt.mean()) ** 2))
    m = mask > 0.5
    return (1.0 - ss_res / ss_tot if ss_tot > 0 else np.nan,
            float(np.sqrt(np.mean((yt - yp) ** 2))),
            float(np.sqrt(np.mean((yt[m] - yp[m]) ** 2))) if m.any() else np.nan)


def rollout(predict, X, idx, mode):
    """X: (N, G, G, 22) scaled; channels 0-10 = frame t-2, 11-21 = frame t-1."""
    prev2 = X[START, ..., idx].copy()            # target cytokine at t-2 (observed)
    prev1 = X[START, ..., 11 + idx].copy()       # target cytokine at t-1 (observed)
    frozen = X[START].copy()
    out = []
    for i in range(START, END):
        x = X[i].copy()                          # masks (and, in gt mode, other cytokines) from the ABM
        if mode == "frozen":
            x[..., 0:6]   = frozen[..., 11:17] if i > START else frozen[..., 0:6]
            x[..., 11:17] = frozen[..., 11:17]
        x[..., idx] = prev2; x[..., 11 + idx] = prev1
        y = np.clip(predict(x[None])[0, ..., 0], -1.0, 1.0)
        out.append(y); prev2, prev1 = prev1, y
    return np.stack(out)


def main(grid, seed, cytokine):
    import tensorflow as tf
    spec = importlib.util.spec_from_file_location("unet_all", ROOT / "scripts" / "unet" / "all.py")
    U = importlib.util.module_from_spec(spec); spec.loader.exec_module(U)
    idx = CYTS.index(cytokine); tag = f"{cytokine}_{grid}_{seed}"
    dp = PREPROCESSED / f"{grid}x{grid}"
    X = np.load(dp / "X_unet.npy").astype(np.float32)
    Y = np.load(dp / "Y_target.npy").astype(np.float32)[..., idx]
    M = np.load(dp / "Y_masks_spatial.npy").astype(np.float32).max(axis=-1)
    cmax = float(json.load(open(dp / "metadata.json"))["scaling"]["max"][idx])
    best = json.load(open(RESULTS_MODELS / "unet" / f"res_{cytokine}_{grid}_42.json"))["best_params"]
    model = U.build_unet(grid_size=grid, in_channels=22, base_filters=best["base_filters"],
                         depth=best["depth"], dropout=best["dropout"])
    model.load_weights(RESULTS / "repro" / "unet" / f"weights_{tag}.weights.h5")
    predict = lambda x: model(x, training=False).numpy()

    den = lambda a: (np.asarray(a, np.float64) + 1.0) / 2.0 * cmax
    preds = {
        "rollout_gt":     den(rollout(predict, X, idx, "gt")),
        "rollout_frozen": den(rollout(predict, X, idx, "frozen")),
        "one_step":       den(np.concatenate([predict(X[i:i + 1])[..., 0] for i in range(START, END)])),
        "persist_1step":  den(X[START:END, ..., 11 + idx]),
        "persist_frozen": den(np.repeat(X[START:START + 1, ..., 11 + idx], END - START, axis=0)),
    }
    truth = den(Y[START:END])
    out = RESULTS / "repro" / "rollout"; out.mkdir(parents=True, exist_ok=True)
    np.save(out / f"rollout_gt_{tag}.npy", preds["rollout_gt"].astype(np.float32))
    np.save(out / f"rollout_frozen_{tag}.npy", preds["rollout_frozen"].astype(np.float32))

    rows = []
    for k, i in enumerate(range(START, END)):
        row = {"cytokine": cytokine, "grid": grid, "seed": seed, "t_hours": i + 2, "lead_hours": k + 1}
        for name, p in preds.items():
            r2, rmse, mrmse = frame_scores(truth[k], p[k], M[i])
            row[f"R2_{name}"] = round(r2, 5); row[f"RMSE_{name}"] = float(f"{rmse:.4g}"); row[f"mRMSE_{name}"] = float(f"{mrmse:.4g}")
        rows.append(row)
    with open(out / f"rollout_{tag}.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys())); w.writeheader(); w.writerows(rows)

    names = list(preds)
    print(f"\n[{cytokine} {grid} seed {seed}]  per-frame R2")
    print("  lead(h)  t(h) " + "".join(f"{n:>16s}" for n in names))
    for r in rows:
        if r["lead_hours"] in (1, 2, 3, 5, 7, 10, 13, 16, 19):
            print(f"  {r['lead_hours']:6d} {r['t_hours']:5d} " + "".join(f"{r['R2_' + n]:16.4f}" for n in names))
    print("  RMSE (physical units)")
    for r in rows:
        if r["lead_hours"] in (1, 5, 10, 19):
            print(f"  {r['lead_hours']:6d} {r['t_hours']:5d} " + "".join(f"{r['RMSE_' + n]:16.3e}" for n in names))


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--grid", type=int, required=True)
    ap.add_argument("--cytokine", type=str, required=True)
    ap.add_argument("--seed", type=int, default=42)
    a = ap.parse_args()
    main(a.grid, a.seed, a.cytokine.lower())
