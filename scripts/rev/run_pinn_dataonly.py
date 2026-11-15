"""Control for the PINN: identical network, data, observation sampling and three-stage optimisation as
scripts/rev/run_pinn_abm.py, but the weights of the PDE residual and of the boundary condition are set
to zero, so the network (x, y, t) -> u is fitted to the observations and the initial condition only.
Results: results/<REV_OUT>_pinn/dataonly/pinn/res_<cytokine>_<grid>_<seed>.json
"""
import argparse, json, os, sys
import run_pinn_abm as R
P, dde, C = R.P, R.dde, R.C

_compile = dde.Model.compile
def compile_data_only(self, optimizer, lr=None, loss_weights=None, **kw):
    if loss_weights is not None:                      # order: [PDE, IC, boundary, observations]
        loss_weights = [0.0, loss_weights[1], 0.0, loss_weights[3]]
    return _compile(self, optimizer, lr=lr, loss_weights=loss_weights, **kw)
dde.Model.compile = compile_data_only

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--grid", type=int, required=True)
    ap.add_argument("--cytokines", nargs="*", default=["il10", "il8"])
    ap.add_argument("--seeds", nargs="*", type=int, default=[42, 1, 100])
    a = ap.parse_args()
    if C.DATA_ROOT is None: sys.exit("REV_DATA_ROOT must point to the preprocessed revision runs")
    run = C.run_dir(a.grid)
    meta = json.load(open(run / "metadata.json"))
    b = (meta.get("abm_provenance") or {}).get("boundaryat", a.grid // 10)
    out_root = R.RESULTS / (os.environ.get("REV_OUT", "rev") + "_pinn") / "dataonly"
    links = out_root / "_data"; links.mkdir(parents=True, exist_ok=True)
    link = links / f"{a.grid}x{a.grid}"
    if not link.exists(): os.symlink(run, link)
    P.PREPROCESSED, P.RESULTS_MODELS = links, out_root
    P.make_pde_1cyt, P.build_source_terms_1cyt, P.build_dde_model = R.make_pde("strict"), R.sources, R.make_model_builder(a.grid, b)
    print("GPU:", R.tf.config.list_physical_devices("GPU"), "| variant: dataonly | run:", run.name, flush=True)
    for c in a.cytokines:
        for s in a.seeds:
            out = out_root / "pinn" / f"res_{c}_{a.grid}_{s}.json"
            if out.exists():
                print("skip (done): dataonly", c, a.grid, s, flush=True); continue
            try:
                P.run_pipeline(a.grid, s, c)
                r = json.load(open(out)); r.update({"variant": "dataonly", "run": run.name, "loss": "observations and initial condition only"})
                json.dump(r, open(out, "w"), indent=1)
            except Exception:
                import traceback
                print("FAILED: dataonly", c, a.grid, s, flush=True); traceback.print_exc(); sys.stdout.flush()
