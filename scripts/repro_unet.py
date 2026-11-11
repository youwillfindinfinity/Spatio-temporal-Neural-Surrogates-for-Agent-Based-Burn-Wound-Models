"""Retrain the U-Net from the stored hyperparameters, save predictions and weights,
and compare the metrics with the stored result JSON. Also reports a persistence
baseline (prediction = previous frame) on the same windows."""
import sys, json, argparse, time, importlib.util
from pathlib import Path
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from paths import PREPROCESSED, RESULTS, RESULTS_MODELS

import numpy as np
import tensorflow as tf

spec = importlib.util.spec_from_file_location("unet_all", ROOT / "scripts" / "unet" / "all.py")
U = importlib.util.module_from_spec(spec); spec.loader.exec_module(U)

CYTS = ["il8", "il1", "il6", "il10", "tnf", "tgf"]
KEYS = ["Global_R2", "SSIM", "Avg_Dice", "Spatial_Correlation", "Masked_RMSE", "Unmasked_RMSE"]
WINDOWS = {"Near_Horizon_t82_t91": (80, 90), "Far_Horizon_t92_t100": (90, 99)}


def main(grid, seed, cytokine):
    idx = CYTS.index(cytokine)
    dp = PREPROCESSED / f"{grid}x{grid}"
    X = np.load(dp / "X_unet.npy").astype(np.float32)
    Y = np.load(dp / "Y_target.npy").astype(np.float32)[..., idx:idx + 1]
    M = np.load(dp / "Y_masks_spatial.npy").astype(np.float32)
    clip_max = float(json.load(open(dp / "metadata.json"))["scaling"]["max"][idx])
    print(f"[{cytokine} {grid} seed {seed}] samples: {X.shape[0]}  clip_max: {clip_max:.4e}")

    ref_dir = RESULTS_MODELS / "unet"
    best = json.load(open(ref_dir / f"res_{cytokine}_{grid}_42.json"))["best_params"]
    stored = json.load(open(ref_dir / f"res_{cytokine}_{grid}_{seed}.json"))

    tf.keras.backend.clear_session(); U.set_seed(seed)
    t0 = time.time()
    model = U.build_unet(grid_size=grid, in_channels=X.shape[-1], base_filters=best["base_filters"],
                         depth=best["depth"], dropout=best["dropout"])
    model.compile(optimizer=tf.keras.optimizers.Adam(best["learning_rate"]), loss="mse")
    model.fit(X[:70], Y[:70], validation_data=(X[70:80], Y[70:80]), epochs=U.FULL_EPOCHS,
              batch_size=best["batch_size"], verbose=2,
              callbacks=[tf.keras.callbacks.EarlyStopping(monitor="val_loss", patience=20, restore_best_weights=True),
                         tf.keras.callbacks.ReduceLROnPlateau(monitor="val_loss", factor=0.5, patience=10, min_lr=1e-6)])
    train_s = time.time() - t0

    Yp = U.denormalize(model.predict(X, batch_size=2, verbose=0), clip_max)
    Ya = U.denormalize(Y, clip_max)
    Ypers = U.denormalize(X[..., 11 + idx:12 + idx], clip_max)       # previous frame (t-1) of the same cytokine

    out = RESULTS / "repro" / "unet"; out.mkdir(parents=True, exist_ok=True)
    tag = f"{cytokine}_{grid}_{seed}"
    np.save(out / f"pred_{tag}.npy", Yp.astype(np.float32))
    np.save(out / f"truth_{cytokine}_{grid}.npy", Ya.astype(np.float32))
    model.save_weights(out / f"weights_{tag}.weights.h5")

    res = {"train_time_seconds": round(train_s, 1), "clip_max": clip_max, "n_samples": int(X.shape[0]), "windows": {}}
    print(f"\ntraining time: {train_s:.1f} s (stored: {stored['train_time_seconds']} s)\n")
    for w, (a, b) in WINDOWS.items():
        new  = U.calculate_metrics(Ya[a:b], Yp[a:b], M[a:b], clip_max)
        pers = U.calculate_metrics(Ya[a:b], Ypers[a:b], M[a:b], clip_max)
        old  = stored["results"][w]
        res["windows"][w] = {"retrained": new, "stored": {k: old[k] for k in KEYS}, "persistence": pers}
        print(w); print(f"  {'metric':22s}{'stored':>14s}{'retrained':>14s}{'persistence':>14s}")
        for k in KEYS:
            print(f"  {k:22s}{old[k]:14.6g}{new[k]:14.6g}{pers[k]:14.6g}")
        print()
    json.dump(res, open(out / f"repro_{tag}.json", "w"), indent=2)
    print(f"saved to {out}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--grid", type=int, required=True)
    ap.add_argument("--cytokine", type=str, required=True)
    ap.add_argument("--seed", type=int, default=42)
    a = ap.parse_args()
    main(a.grid, a.seed, a.cytokine.lower())
