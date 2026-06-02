#!/usr/bin/env python3

import sys
import os
import json
import argparse
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
try:
    from paths import PREPROCESSED, RESULTS_MODELS  
except Exception:
    PREPROCESSED = Path("./preprocessed")
    RESULTS_MODELS = Path(".")

import numpy as np

os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")
import tensorflow as tf

CYT_NAMES = ["il8", "il1", "il6", "il10", "tnf", "tgf"]
CELL_NAMES = ["EC", "NN", "NA", "M1", "M2"]
CELL_CH_IN_11 = {name: 6 + i for i, name in enumerate(CELL_NAMES)}  
CYT_CH_IN_11 = {name: i for i, name in enumerate(CYT_NAMES)}      


def trunk_indices_for_cell(cell_name):
    m11 = CELL_CH_IN_11[cell_name]  
    return [2 + m11, 2 + 11 + m11]          


def trunk_indices_for_cyt(cyt_name):
    c11 = CYT_CH_IN_11[cyt_name]           
    return [2 + c11, 2 + 11 + c11]


# model definition
class Branch(tf.keras.layers.Layer):
    def __init__(self, hidden, p, **kw):
        super().__init__(**kw)
        self.fc1 = tf.keras.layers.Dense(hidden, activation="relu")
        self.fc2 = tf.keras.layers.Dense(p, activation="linear")

    def call(self, x, training=False):
        return self.fc2(self.fc1(x))


class Trunk(tf.keras.layers.Layer):
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
        self.branch = Branch(hidden, p)
        self.trunk = Trunk(hidden, p)
        self.bias = self.add_weight(shape=(1,), initializer="zeros",
                                    trainable=True, name="bias")

    def call(self, inputs, training=False):
        xb, xt = inputs
        b = self.branch(xb, training=training)
        t = self.trunk(xt)
        r = tf.einsum("bp,bnp->bn", b, t) + self.bias
        return tf.expand_dims(r, -1)


def build_branch_inputs(Xb, Xt, cyt_idx, drop_cell=None):
    N, _, G, _, _ = Xb.shape
    f0 = Xb[:, 0, :, :, cyt_idx]
    mask_stack = Xb[:, 0, :, :, 6:].copy()
    if drop_cell is not None:
        ci = CELL_CH_IN_11[drop_cell] - 6         
        mask_stack[:, :, :, ci] = 0.0
    mask = (mask_stack.max(axis=-1) > 0.5).astype(np.float32)

    xs = np.linspace(0, 1, G, dtype=np.float32)
    ys = np.linspace(0, 1, G, dtype=np.float32)
    xx, yy = np.meshgrid(xs, ys, indexing="ij")

    out = np.zeros((N, 7), dtype=np.float32)
    for i in range(N):
        f = f0[i]; m = mask[i]; na = float(np.sum(m)) + 1e-6
        out[i, 0] = (float(np.max(f)) + 1.0) / 2.0
        out[i, 1] = (float(np.mean(f)) + 1.0) / 2.0
        out[i, 2] = float(np.std(f))
        out[i, 3] = float(np.sum(xx * m) / na)
        out[i, 4] = float(np.sum(yy * m) / na)
        out[i, 5] = na / (G * G)
        out[i, 6] = float(Xt[i, 0, 2])
    return out


def build_trunk_inputs(Xb, Xt, drop_cell=None, ablate_field_cyt=None):
    N, _, G, _, C = Xb.shape
    vals = Xb.transpose(0, 2, 3, 1, 4).reshape(N, G * G, 22).astype(np.float32)
    xy = Xt[:, :, :2].astype(np.float32)
    trunk = np.concatenate([xy, vals], axis=-1) # (N,G*G,24)

    if ablate_field_cyt is not None and drop_cell is not None:
        # voxels where drop_cell is present, per sample, flattened to G*G
        cell_ch = CELL_CH_IN_11[drop_cell]
        # use BOTH look-back frames' presence (union) to define source voxels
        pres0 = Xb[:, 0, :, :, cell_ch] > 0.5
        pres1 = Xb[:, 1, :, :, cell_ch] > 0.5
        pres = (pres0 | pres1).reshape(N, G * G)  
        # normalised "zero concentration" in [-1,1] space is -1 (since
        # u_scaled = 2*u/cmax - 1 and u=0 -> -1). Set field to -1 at those voxels.
        for fi in trunk_indices_for_cyt(ablate_field_cyt):
            col = trunk[:, :, fi]
            col[pres] = -1.0
            trunk[:, :, fi] = col

    if drop_cell is not None:
        for idx in trunk_indices_for_cell(drop_cell):
            trunk[:, :, idx] = 0.0
    return trunk


def predict_full(model, Xbranch, Xtrunk, chunk=4096):
    N, n_pts, _ = Xtrunk.shape
    out = np.zeros((N, n_pts, 1), np.float32)
    for i in range(N):
        xb = tf.constant(Xbranch[i:i + 1])
        for s in range(0, n_pts, chunk):
            e = min(s + chunk, n_pts)
            xt = tf.constant(Xtrunk[i:i + 1, s:e])
            out[i, s:e] = model([xb, xt], training=False).numpy()[0]
    return out


def denorm(x, cmax):
    return (np.asarray(x, np.float64) + 1.0) / 2.0 * cmax

# causal-sensitivity statistics
def m2_voxel_mask(Xb, frame=0):
    ch = CELL_CH_IN_11["M2"]
    return Xb[:, frame, :, :, ch] > 0.5

def sensitivity_stats(base_phys, ablt_phys, target_voxels, active_ref):
    diff = ablt_phys - base_phys # (N,G,G,1)
    diff2 = diff[..., 0]
    tv = target_voxels
    if tv.sum() == 0:
        return dict(n_voxels=0, abs_shift=np.nan, rel_shift=np.nan,
                    signed_shift=np.nan)
    abs_shift = float(np.mean(np.abs(diff2[tv])))
    signed_shift = float(np.mean(diff2[tv]))
    rel = abs_shift / (active_ref + 1e-30)
    return dict(n_voxels=int(tv.sum()), abs_shift=abs_shift,
                rel_shift=float(rel), signed_shift=signed_shift)

def _find_deeponet_dir():
    here = Path(__file__).resolve()
    repo_guesses = [here.parents[1], here.parents[2] if len(here.parents) > 2 else here.parents[1], Path.cwd()]
    candidates = []
    try:
        candidates.append(Path(RESULTS_MODELS) / "deeponet_h")
    except Exception:
        pass
    for root in repo_guesses:
        candidates.append(root / "models" / "deeponet_h")
        candidates.append(root / "deeponet_h")
    seen = set(); uniq = []
    for c in candidates:
        cr = c.resolve()
        if cr not in seen:
            seen.add(cr); uniq.append(c)
    for c in uniq:
        if c.is_dir():
            return c
    # nothing found: report what we tried
    tried = "\n  ".join(str(c) for c in uniq)
    raise FileNotFoundError(
        "Could not locate the deeponet_h results directory. Tried:\n  " + tried +
        "\nPass the correct location by editing RESULTS_MODELS in paths.py, "
        "or run from the repo root.")

def _find_preprocessed_dir(grid):
    here = Path(__file__).resolve()
    roots = []
    try:
        roots.append(Path(PREPROCESSED))
    except Exception:
        pass
    for root in [here.parents[1], here.parents[2] if len(here.parents) > 2 else here.parents[1], Path.cwd()]:
        roots.append(root / "preprocessed")
    sub = f"{grid}x{grid}"
    seen = set(); uniq = []
    for r in roots:
        rr = (r / sub)
        key = rr.resolve()
        if key not in seen:
            seen.add(key); uniq.append(rr)
    for c in uniq:
        if c.is_dir():
            return c
    tried = "\n  ".join(str(c) for c in uniq)
    raise FileNotFoundError(
        f"Could not locate preprocessed/{sub}. Tried:\n  " + tried)


def run(grid, seed, control_cell, test_window=(80, 99), ablate_field=False,
        jacobian=False):
    cyt = "il10"
    idx = CYT_NAMES.index(cyt)
    dp = _find_preprocessed_dir(grid)
    mdir = _find_deeponet_dir()
    print(f"  preprocessed dir: {dp}")
    print(f"  results dir:      {mdir}")

    Xb = np.load(dp / "X_branch.npy").astype(np.float32)
    Xt = np.load(dp / "X_trunk.npy").astype(np.float32)
    with open(dp / "metadata.json") as f:
        meta = json.load(f)
    cmax = float(meta["scaling"]["max"][idx])

    rp = mdir / f"res_{cyt}_{grid}_{seed}.json"
    with open(rp) as f:
        bp = json.load(f)["best_params"]

    # restrict to held-out test frames so the test is on unseen data
    lo, hi = test_window
    sl = slice(lo, hi)
    Xb_t = Xb[sl]; Xt_t = Xt[sl]

    # load pre-trained weights into an identical graph (NO retraining) 
    tf.keras.backend.clear_session()
    model = DeepONet(hidden=bp["hidden"], p=bp["p"])
    xb0 = build_branch_inputs(Xb_t, Xt_t, idx)[:1]
    xt0 = build_trunk_inputs(Xb_t, Xt_t)[:1, :10, :]
    _ = model([tf.constant(xb0), tf.constant(xt0)])
    wpath = mdir / f"weights_{cyt}_{grid}_{seed}.weights.h5"
    model.load_weights(str(wpath))

    G = grid
    # baseline pred 
    Xbr_base = build_branch_inputs(Xb_t, Xt_t, idx)
    Xtr_base = build_trunk_inputs(Xb_t, Xt_t)
    base = denorm(predict_full(model, Xbr_base, Xtr_base).reshape(-1, G, G, 1), cmax)
    active_ref = float(np.mean(base[base > 0])) if (base > 0).any() else 1e-30

    m2_vox = m2_voxel_mask(Xb_t, frame=0) # (Nt,G,G)
    ctrl_vox = Xb_t[:, 0, :, :, CELL_CH_IN_11[control_cell]] > 0.5

    report = {"grid": grid, "seed": seed, "cytokine": cyt,
              "control_cell": control_cell, "test_window": [lo, hi],
              "active_region_mean_phys": active_ref}

    # (A) TRUNK ablation
    Xtr_noM2 = build_trunk_inputs(Xb_t, Xt_t, drop_cell="M2",
                                  ablate_field_cyt=("il10" if ablate_field else None))
    pred_noM2 = denorm(predict_full(model, Xbr_base, Xtr_noM2).reshape(-1, G, G, 1), cmax)

    Xtr_noCtrl = build_trunk_inputs(Xb_t, Xt_t, drop_cell=control_cell,
                                    ablate_field_cyt=("il10" if ablate_field else None))
    pred_noCtrl = denorm(predict_full(model, Xbr_base, Xtr_noCtrl).reshape(-1, G, G, 1), cmax)

    trunk = {
        "ablate_field": bool(ablate_field),
        "M2_on_M2voxels": sensitivity_stats(base, pred_noM2, m2_vox, active_ref),
        f"{control_cell}_on_{control_cell}voxels":
            sensitivity_stats(base, pred_noCtrl, ctrl_vox, active_ref),
        # cross-check: removing the control should barely move M2 voxels
        f"{control_cell}_on_M2voxels":
            sensitivity_stats(base, pred_noCtrl, m2_vox, active_ref),
    }
    # boundary (Dice support) shrinkage when M2 removed
    thr = 0.05 * cmax
    supp_base = (base[..., 0] > thr).sum()
    supp_noM2 = (pred_noM2[..., 0] > thr).sum()
    trunk["support_voxels_baseline"] = int(supp_base)
    trunk["support_voxels_noM2"] = int(supp_noM2)
    trunk["support_fraction_remaining"] = float(supp_noM2 / (supp_base + 1e-9))
    report["trunk_ablation"] = trunk

    # (B) BRANCH ablation 
    Xbr_noM2 = build_branch_inputs(Xb_t, Xt_t, idx, drop_cell="M2")
    pred_b_noM2 = denorm(predict_full(model, Xbr_noM2, Xtr_base).reshape(-1, G, G, 1), cmax)
    # how much did the 7-scalar branch vector even change?
    branch_vec_delta = float(np.mean(np.abs(Xbr_noM2 - Xbr_base)))
    report["branch_ablation"] = {
        "branch_input_mean_abs_delta": branch_vec_delta,
        "M2_on_M2voxels": sensitivity_stats(base, pred_b_noM2, m2_vox, active_ref),
    }

    # (C) JACOBIAN sensitivity -- purely-mathematical, no thresholding.
    #   d(u_hat) / d(M2-mask channel), evaluated at the baseline input, summed
    #   over the two look-back M2 trunk channels. Reports the mean over M2
    #   voxels vs the control's voxels. This avoids the binary-ablation
    #   redundancy issue: even if zeroing the mask does little, a non-zero
    #   gradient shows the network's local reliance on the channel.
    if jacobian:
        m2_idx = trunk_indices_for_cell("M2") 
        ctrl_idx = trunk_indices_for_cell(control_cell)
        il10_idx = trunk_indices_for_cyt("il10")      

        def channel_jacobian(trunk_np, chan_indices, target_voxels_flat):
            """Mean |d u_hat / d x_chan| over target voxels, summed over the
            given channel indices, averaged over test frames. Uses one
            backward pass per frame on the trained graph (no weight update)."""
            Nt = trunk_np.shape[0]
            acc = []
            for i in range(Nt):
                xb = tf.constant(Xbr_base[i:i + 1])
                vals = []
                # chunk to bound memory at 250^2
                npts = trunk_np.shape[1]
                grad_full = np.zeros((npts,), np.float64)
                for s in range(0, npts, 4096):
                    e = min(s + 4096, npts)
                    xt = tf.Variable(trunk_np[i:i + 1, s:e])
                    with tf.GradientTape() as tape:
                        out = model([xb, xt], training=False) # (1,e-s,1)
                        scalar = tf.reduce_sum(out)
                    g = tape.gradient(scalar, xt).numpy()[0] # (e-s,24)
                    grad_full[s:e] = np.sum(np.abs(g[:, chan_indices]), axis=1)
                tvf = target_voxels_flat[i]
                if tvf.sum() > 0:
                    vals.append(float(np.mean(grad_full[tvf])))
            return float(np.mean(vals)) if vals else np.nan

        m2_vox_flat = m2_vox.reshape(m2_vox.shape[0], -1)
        ctrl_vox_flat = ctrl_vox.reshape(ctrl_vox.shape[0], -1)
        jac = {
            "dM2_on_M2voxels": channel_jacobian(Xtr_base, m2_idx, m2_vox_flat),
            f"d{control_cell}_on_{control_cell}voxels":
                channel_jacobian(Xtr_base, ctrl_idx, ctrl_vox_flat),
            "dIL10field_on_M2voxels":
                channel_jacobian(Xtr_base, il10_idx, m2_vox_flat),
        }
        denom = jac[f"d{control_cell}_on_{control_cell}voxels"] + 1e-30
        jac["M2_to_control_grad_ratio"] = float(jac["dM2_on_M2voxels"] / denom)
        jac["field_to_M2mask_ratio"] = float(
            jac["dIL10field_on_M2voxels"] / (jac["dM2_on_M2voxels"] + 1e-30))
        report["jacobian"] = jac

    # verdict
    s_m2 = trunk["M2_on_M2voxels"]["rel_shift"]
    s_ctrl = trunk[f"{control_cell}_on_{control_cell}voxels"]["rel_shift"]
    ratio = s_m2 / (s_ctrl + 1e-30)
    signed = trunk["M2_on_M2voxels"]["signed_shift"]
    report["verdict"] = {
        "ablate_field": bool(ablate_field),
        "trunk_rel_shift_M2": s_m2,
        f"trunk_rel_shift_{control_cell}": s_ctrl,
        "M2_to_control_ratio": float(ratio),
        "M2_signed_shift_phys": signed,
        "causal_dependence_captured": bool(ratio > 1.5 and signed < 0),
        "interpretation": (
            "M2 removal shifts the IL-10 prediction over M2 voxels markedly more "
            "than removal of the non-source control, and in the expected (downward) "
            "direction -> the trunk has encoded the M2->IL-10 dependence."
            if (ratio > 1.5 and signed < 0) else
            "Sensitivity to M2 is not clearly above the control; see per-cell stats."
        ),
    }

    print(json.dumps(report, indent=2))
    out = mdir / f"m2_droptest_{cyt}_{grid}_{seed}.json"
    with open(out, "w") as f:
        json.dump(report, f, indent=2)
    print(f"\nSaved -> {out}")
    return report


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--grid", type=int, default=None,
                    help="Single grid (e.g. 100 or 250). Omit to run both.")
    ap.add_argument("--grids", type=int, nargs="*", default=[100, 250],
                    help="Grids to run when --grid is not given (default: 100 250).")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--control", type=str, default="EC",
                    choices=["EC", "NN", "NA", "M1"],
                    help="Non-source control cell type for the contrast.")
    ap.add_argument("--ablate-field", action="store_true",
                    help="Also zero the IL-10 look-back field at source-cell "
                         "voxels (closes the autoregressive leak; the proper "
                         "'cell never secreted here' counterfactual).")
    ap.add_argument("--jacobian", action="store_true",
                    help="Also compute d(IL-10 pred)/d(M2-mask) gradient "
                         "sensitivity (purely-mathematical test, no thresholding).")
    args = ap.parse_args()

    grids = [args.grid] if args.grid is not None else args.grids
    reports = {}
    for g in grids:
        tag = []
        if args.ablate_field:
            tag.append("field-ablated")
        if args.jacobian:
            tag.append("+jacobian")
        tagstr = (" [" + ", ".join(tag) + "]") if tag else ""
        print(f"\n{'='*60}\n  M2 drop-test  |  grid {g}^2  seed {args.seed}{tagstr}\n{'='*60}")
        reports[g] = run(g, args.seed, args.control,
                         ablate_field=args.ablate_field, jacobian=args.jacobian)

    if len(grids) > 1:
        ftag = " (field-ablated)" if args.ablate_field else ""
        print(f"\n{'='*60}\n  CROSS-RESOLUTION SUMMARY (IL-10, M2 vs {args.control}){ftag}\n{'='*60}")
        print(f"  {'grid':>6} | {'rel_shift(M2)':>14} | {'rel_shift(ctrl)':>15} | "
              f"{'ratio':>7} | {'signed(M2)':>12} | captured")
        for g in grids:
            v = reports[g]["verdict"]
            print(f"  {g:>4}^2 | {v['trunk_rel_shift_M2']:>14.4f} | "
                  f"{v[f'trunk_rel_shift_{args.control}']:>15.4f} | "
                  f"{v['M2_to_control_ratio']:>7.2f} | "
                  f"{v['M2_signed_shift_phys']:>12.3e} | "
                  f"{v['causal_dependence_captured']}")
        if args.jacobian:
            print(f"\n  {'grid':>6} | {'dM2(M2vox)':>12} | {'dCtrl(ctrlvox)':>14} | "
                  f"{'grad_ratio':>10} | {'field/M2mask':>12}")
            for g in grids:
                j = reports[g].get("jacobian", {})
                if not j:
                    continue
                print(f"  {g:>4}^2 | {j['dM2_on_M2voxels']:>12.3e} | "
                      f"{j[f'd{args.control}_on_{args.control}voxels']:>14.3e} | "
                      f"{j['M2_to_control_grad_ratio']:>10.2f} | "
                      f"{j['field_to_M2mask_ratio']:>12.1f}")
            print("\n  grad_ratio > 1: network relies more on the M2 mask than on the "
                  "control at\n  M2 voxels. field/M2mask >> 1 quantifies how much the "
                  "IL-10 field itself\n  dominates the M2 mask as the predictive cue "
                  "(the autoregressive route).")
        print("\n  Higher ratio at higher resolution would corroborate the paper's "
              "claim\n  that localised M2-driven IL-10 structure is better resolved "
              "as the grid\n  refines (representational, not data-size, limit).")
