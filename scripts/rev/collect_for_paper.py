"""Collect result summaries of the rev_abm runs into outputs/rev_abm/for_paper.txt."""
import json, glob, os, csv, sys, statistics as st
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from paths import ROOT, RESULTS, OUTPUTS
rel = lambda f: os.path.relpath(f, ROOT)                            # paths relative to the repository
TAG = os.environ.get("REV_OUT") or "rev_abm"
R = RESULTS / TAG; O = OUTPUTS / TAG; out = []
RUNS = Path(os.path.expanduser(os.environ.get("REV_RUNS_ROOT") or next((str(p) for p in (ROOT / "burn_wound_abm_simulations/runs", Path("~/burn_wound_abm_simulations/runs").expanduser()) if p.is_dir()), "")))   # raw ABM runs
P = lambda *a: out.append(" ".join(str(x) for x in a))
ms = lambda v: f"{st.mean(v):.4g} +/- {st.stdev(v):.2g}" if len(v) > 1 else (f"{v[0]:.4g}" if v else "-")

P("== MAIN MODELS: near / far window, mean +/- SD over seeds")
for d in sorted(glob.glob(str(R / "*_*/*/"))):
    name, form = Path(d).parent.name, Path(d).name
    if not (name.startswith("main_") or name.startswith("tmask_")): continue
    for cyt in ("il8", "il10"):
        for g in (100, 250):
            rs = [json.load(open(f)) for f in sorted(glob.glob(f"{d}/res_{cyt}_{g}_*.json"))]
            if not rs: continue
            P(f"\n{name} {form} {cyt} {g} (n={len(rs)})  train_s {ms([r['train_time_seconds'] for r in rs if r.get('train_time_seconds') is not None])}"
              f"  infer_s {ms([r['infer_seconds_19_frames'] for r in rs])}")
            for w in ("near", "far"):
                keys = [k for k, v in rs[0]["windows"][w]["model"].items() if isinstance(v, (int, float))]
                P(f"  {w}: skill {ms([r['windows'][w]['skill_vs_persistence'] for r in rs])} | " +
                  " | ".join(f"{k} {ms([r['windows'][w]['model'][k] for r in rs])}" for k in keys))
            P("  persistence near: " + " | ".join(f"{k} {v:.4g}" for k, v in rs[0]["windows"]["near"]["persistence"].items() if isinstance(v, (int, float))))
            tr = {}
            for r in rs:
                for run, t in (r.get("transfer") or {}).items():
                    tr.setdefault(t.get("experiment", "?"), []).append(t["window"]["skill_vs_persistence"])
            if tr: P("  transfer window skill: " + " | ".join(f"{k} {ms(v)}" for k, v in tr.items()))

P("\n== PINN")
for f in sorted(glob.glob(str(RESULTS / f"{TAG}_pinn" / "*" / "pinn" / "res_*.json"))):
    r = json.load(open(f)); P(rel(f), {k: v for k, v in r.items() if isinstance(v, (int, float, str))})

P("\n== HYPERPARAMETERS")
for f in sorted(glob.glob(str(R / "main_*/*/hp_*.json"))): P(rel(f), json.load(open(f)))

P("\n== ACTIVE FRACTION"); P(open(O / "active_fraction.csv").read() if (O / "active_fraction.csv").exists() else "missing")
P("\n== ABM VARIABILITY"); P(open(O / "abm_variability.txt").read() if (O / "abm_variability.txt").exists() else "missing")
P("\n== PROVENANCE (one run per grid)")
for f in sorted(glob.glob(str(RUNS / "E3_*_s1" / "run_provenance.json"))):
    j = json.load(open(f)); P(f"runs/{Path(f).parent.name}/run_provenance.json"); P(json.dumps(j, indent=1)[:2500])
(O / "for_paper.txt").write_text("\n".join(out) + "\n"); print("written", O / "for_paper.txt", len(out), "lines")
