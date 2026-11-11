"""PINN with the strict (normalisation-aware) residual versus the softened residual of the manuscript.

  soft    residual as in scripts/pinn/all.py:   u_t = D lap(u) - k u       + s1 + s2 - e u
  strict  residual with the (u + 1) offset:     u_t = D lap(u) - k (u + 1) + s1 + s2 - e (u + 1)
          (obtained by substituting u_phys = c_max (u + 1) / 2 into the ABM equation)

Everything else (network, data, loss weights search, three-stage optimisation, metrics) is the
unchanged pipeline of scripts/pinn/all.py. Results go to results/rev_pinn/<variant>/pinn/.
"""
import argparse, importlib.util, sys
from pathlib import Path
ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from paths import RESULTS

spec = importlib.util.spec_from_file_location("pinn_all", ROOT / "scripts" / "pinn" / "all.py")
P = importlib.util.module_from_spec(spec); spec.loader.exec_module(P)
tf, dde = P.tf, P.dde


def make_pde_strict(D, k, s1_tf, s2_tf, e_tf, G):
    G_tf = tf.constant(G, dtype=tf.float64)
    D_tf = tf.constant([[D]], dtype=tf.float64)
    k_tf = tf.constant([[k]], dtype=tf.float64)

    def pde(x, y):
        d2x = dde.grad.hessian(y, x, i=0, j=0)
        d2y = dde.grad.hessian(y, x, i=1, j=1)
        ut = dde.grad.jacobian(y, x, i=0, j=2)
        G_dyn = tf.cast(G_tf, tf.float64)
        ix = tf.cast(tf.clip_by_value(tf.floor((x[:, 0:1] + 1.0) / 2.0 * G_dyn), 0.0, G_dyn - 1.0), tf.int32)
        iy = tf.cast(tf.clip_by_value(tf.floor((x[:, 1:2] + 1.0) / 2.0 * G_dyn), 0.0, G_dyn - 1.0), tf.int32)
        flat_idx = tf.squeeze(ix * tf.cast(G_tf, tf.int32) + iy, axis=1)
        s1_q = tf.gather(s1_tf, flat_idx); s2_q = tf.gather(s2_tf, flat_idx); e_q = tf.gather(e_tf, flat_idx)
        rhs = D_tf * (d2x + d2y) - k_tf * (y + 1.0) + s1_q + s2_q - e_q * (y + 1.0)
        return ut - rhs
    return pde


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--variant", required=True, choices=["soft", "strict"])
    ap.add_argument("--grid", type=int, required=True)
    ap.add_argument("--cytokines", nargs="*", default=["il10", "il8"])
    ap.add_argument("--seeds", nargs="*", type=int, default=[42, 1, 100])
    a = ap.parse_args()
    P.RESULTS_MODELS = RESULTS / "rev_pinn" / a.variant
    if a.variant == "strict":
        P.make_pde_1cyt = make_pde_strict
    print("GPU:", tf.config.list_physical_devices("GPU"), "| variant:", a.variant, flush=True)
    for c in a.cytokines:
        for s in a.seeds:                      # seed 42 first: it runs the Optuna search the others reuse
            out = P.RESULTS_MODELS / "pinn" / f"res_{c}_{a.grid}_{s}.json"
            if out.exists():
                print("skip (done):", a.variant, c, a.grid, s, flush=True); continue
            try:
                P.run_pipeline(a.grid, s, c)
            except Exception:
                import traceback
                print("FAILED:", a.variant, c, a.grid, s, flush=True); traceback.print_exc(); sys.stdout.flush()
