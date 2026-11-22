"""PINN on the ABM runs with a residual that matches the cytokine equation of the ABM.

Network, observation sampling, loss-weight search, three-stage optimisation and metrics are
those of scripts/pinn/all.py, except for:

  1. Residual written in the normalised coordinates of the network. With x, y in [-1, 1] over G
     lattice sites and t in [-1, 1] over 100 hourly frames,
         u_t = 50 * [ D (2/(G-1))^2 (u_xx + u_yy) - k (u + c) + s - theta * NA * (u + c) ],
     where D, k, theta, s are the per-frame coefficients of the ABM (variablevals.py), s is scaled
     by 2 / c_max, and c = 1 for the strict (normalisation-consistent) residual, c = 0 for the
     softened residual.
  2. Source terms from the cell types that enter the ABM equations: endothelial cells (IL-8) and
     M2 macrophages (IL-10); uptake of IL-8 by activated neutrophils. The NDN-neutrophil source of
     IL-8 is omitted (not among the stored masks; 0-4 lattice sites). The source is the
     cell-presence field averaged over the training window.
  3. Boundary condition of the ABM solver: zero concentration (u = -1 in normalised units) on the
     four boundary segments of length `boundaryat`, zero normal derivative elsewhere.

Variants: strict, soft. Results: results/<REV_OUT>_pinn/<variant>/pinn/res_<cytokine>_<grid>_<seed>.json
"""
import argparse, importlib.util, json, os, sys
from pathlib import Path
import numpy as np
ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT)); sys.path.insert(0, str(Path(__file__).resolve().parent))
from paths import RESULTS
import common as C

spec = importlib.util.spec_from_file_location("pinn_all", ROOT / "scripts" / "pinn" / "all.py")
P = importlib.util.module_from_spec(spec); spec.loader.exec_module(P)
tf, dde = P.tf, P.dde
for k in ("N_TRIALS", "TUNE_ITERS", "FULL_ITERS", "WARMUP_ITERS", "LBFGS_ITERS"):      # optional overrides for quick tests
    if os.environ.get("REV_PINN_" + k): setattr(P, k, int(os.environ["REV_PINN_" + k]))

FRAMES_PER_UNIT_T = 50.0          # t in [-1, 1] spans 100 frames
OFFSET = {"strict": 1.0, "soft": 0.0}


def make_pde(variant):
    c = OFFSET[variant]

    def factory(D, k, s1_tf, s2_tf, e_tf, G):
        Dn = tf.constant([[D * (2.0 / (G - 1)) ** 2]], dtype=tf.float64)
        k_tf = tf.constant([[k]], dtype=tf.float64); G_tf = tf.constant(G, dtype=tf.float64)

        def pde(x, y):
            d2x = dde.grad.hessian(y, x, i=0, j=0); d2y = dde.grad.hessian(y, x, i=1, j=1)
            ut = dde.grad.jacobian(y, x, i=0, j=2)
            ix = tf.cast(tf.clip_by_value(tf.floor((x[:, 0:1] + 1.0) / 2.0 * G_tf), 0.0, G_tf - 1.0), tf.int32)
            iy = tf.cast(tf.clip_by_value(tf.floor((x[:, 1:2] + 1.0) / 2.0 * G_tf), 0.0, G_tf - 1.0), tf.int32)
            flat = tf.squeeze(ix * tf.cast(G_tf, tf.int32) + iy, axis=1)
            s_q = tf.gather(s1_tf, flat); e_q = tf.gather(e_tf, flat)
            rhs = Dn * (d2x + d2y) - k_tf * (y + c) + s_q - e_q * (y + c)
            return ut - FRAMES_PER_UNIT_T * rhs
        return pde
    return factory


_orig_sources = P.build_source_terms_1cyt


def sources(masks_flat, sec, cyt_idx, G, clip_max):
    """Mask channels: 0 EC, 1 resting neutrophil, 2 activated neutrophil, 3 M1, 4 M2.
    Channel 1 is not a cytokine source, so the second source term is dropped."""
    s1, s2, e = _orig_sources(masks_flat, sec, cyt_idx, G, clip_max)
    return s1, tf.zeros_like(s2), e


def make_model_builder(G, boundaryat):
    w = 2.0 * boundaryat / G                                   # segment length in normalised units

    def bc_residual(x, y, X):
        gx = dde.grad.jacobian(y, x, i=0, j=0); gy = dde.grad.jacobian(y, x, i=0, j=1)
        xn, yn = x[:, 0:1], x[:, 1:2]; eps = 1e-6
        left, right = tf.abs(xn + 1.0) < eps, tf.abs(xn - 1.0) < eps
        bottom, top = tf.abs(yn + 1.0) < eps, tf.abs(yn - 1.0) < eps
        normal = tf.where(left, -gx, tf.where(right, gx, tf.where(bottom, -gy, gy)))
        fixed = (bottom & (xn < -1.0 + w)) | (top & (xn > 1.0 - w)) | (right & (yn > 1.0 - w)) | (left & (yn < -1.0 + w))
        return tf.where(fixed, y + 1.0, normal)

    def build(G_, hidden, n_layers, lr, pde_fn, geomtime, X_ic, Y_ic_obs, X_obs, Y_obs, bc_fn,
              lambda_pde=0.01, lambda_ic=1.0, lambda_bc=0.1, num_domain=None, num_boundary=None):
        ic_bc = dde.PointSetBC(X_ic, Y_ic_obs, component=0)
        wall = dde.icbc.OperatorBC(geomtime, bc_residual, bc_fn)
        obs_bc = dde.PointSetBC(X_obs, Y_obs, component=0)
        data = dde.data.TimePDE(geomtime, pde_fn, ic_bcs=[ic_bc, wall, obs_bc],
                                num_domain=num_domain or P.N_DOMAIN, num_boundary=num_boundary or P.N_BOUNDARY,
                                num_initial=P.N_INITIAL, train_distribution="uniform")
        net = dde.maps.FNN([3] + [hidden] * n_layers + [1], "tanh", "Glorot uniform")
        model = dde.Model(data, net)
        model.compile("adam", lr=lr, loss_weights=[lambda_pde, lambda_ic, lambda_bc, 1.0])   # [PDE, IC, boundary, observations]
        return model
    return build


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--variant", required=True, choices=list(OFFSET))
    ap.add_argument("--grid", type=int, required=True)
    ap.add_argument("--cytokines", nargs="*", default=["il10", "il8"])
    ap.add_argument("--seeds", nargs="*", type=int, default=[42, 1, 100])
    a = ap.parse_args()
    if C.DATA_ROOT is None: sys.exit("REV_DATA_ROOT must point to the preprocessed ABM runs")
    run = C.run_dir(a.grid)
    meta = json.load(open(run / "metadata.json"))
    b = (meta.get("abm_provenance") or {}).get("boundaryat", a.grid // 10)
    out_root = RESULTS / ((os.environ.get("REV_OUT") or "rev_abm") + "_pinn") / a.variant
    links = out_root / "_data"; links.mkdir(parents=True, exist_ok=True)      # the pipeline expects <root>/<G>x<G>
    link = links / f"{a.grid}x{a.grid}"
    if not link.exists(): os.symlink(run, link)
    P.PREPROCESSED, P.RESULTS_MODELS = links, out_root
    P.make_pde_1cyt, P.build_source_terms_1cyt, P.build_dde_model = make_pde(a.variant), sources, make_model_builder(a.grid, b)
    print("GPU:", tf.config.list_physical_devices("GPU"), "| variant:", a.variant, "| run:", run.name, "| boundaryat:", b, "| backend:", dde.backend.backend_name, flush=True)
    for c in a.cytokines:
        for s in a.seeds:                                   # seed 42 first: it runs the search that the other seeds reuse
            out = out_root / "pinn" / f"res_{c}_{a.grid}_{s}.json"
            if out.exists():
                print("skip (done):", a.variant, c, a.grid, s, flush=True); continue
            try:
                P.run_pipeline(a.grid, s, c)
                r = json.load(open(out)); r.update({"variant": a.variant, "run": run.name, "boundaryat": b,
                                                    "residual": "normalised coordinates; ABM sources and boundary segments"})
                json.dump(r, open(out, "w"), indent=1)
            except Exception:
                import traceback
                print("FAILED:", a.variant, c, a.grid, s, flush=True); traceback.print_exc(); sys.stdout.flush()
