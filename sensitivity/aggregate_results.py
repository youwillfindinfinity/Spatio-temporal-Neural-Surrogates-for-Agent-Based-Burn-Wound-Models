import argparse
import json
import csv
from pathlib import Path


STRATEGIES = ["no_clip", "p95", "adaptive"]
ALL_MODELS = ["unet", "sta_lstm", "deeponet", "pi_deeponet", "pinn"]
CYTOKINES = ["il8", "il10"]

METRIC_KEYS = ["Global_R2", "SSIM", "Avg_Dice", "Spatial_Correlation",
               "Masked_RMSE"]
HORIZONS = ["Near_Horizon_t82_t91", "Far_Horizon_t92_t100"]

PAPER_MODEL_DIRS = {
    "unet":         "unet",
    "sta_lstm":     "sta_lstm",
    "deeponet":     "deeponet_h",  
    "pi_deeponet":  "pi_deeponet",
    "pinn":         "pinn",
}

def find_result_json(base, strategy, model, cyt, grid, seed,
                     adaptive_from_paper_models=False):
    """Find the result JSON for a given (strategy, model, cyt, grid, seed)."""
    default = base / "sensitivity_results" / strategy / model / f"res_{cyt}_{grid}_{seed}.json"
    if default.exists():
        return default

    if strategy == "adaptive" and adaptive_from_paper_models:
        paper_dir = PAPER_MODEL_DIRS.get(model, model)
        paper_path = base / "models" / paper_dir / f"res_{cyt}_{grid}_{seed}.json"
        if paper_path.exists():
            return paper_path
    return None


def collect_all_results(args):
    base = Path(args.base_dir)
    rows = []
    for strategy in STRATEGIES:
        for model in ALL_MODELS:
            for cyt in CYTOKINES:
                f = find_result_json(base, strategy, model, cyt, args.grid,
                                     args.seed, args.adaptive_from_paper_models)
                if f is None:
                    print(f"MISSING: strategy={strategy} model={model} cyt={cyt}")
                    continue
                with open(f) as fp:
                    res = json.load(fp)
                row = {
                    "strategy": strategy,
                    "model":    model,
                    "cytokine": cyt,
                    "grid":     args.grid,
                    "seed":     args.seed,
                    "source_file": str(f.relative_to(base)) if f.is_relative_to(base) else str(f),
                    "train_time_s": res.get("train_time_seconds"),
                    "optuna_best_val_loss": res.get("optuna_best_val_loss"),
                }
                for horizon in HORIZONS:
                    if horizon not in res.get("results", {}):
                        continue
                    h_short = "near" if "Near" in horizon else "far"
                    for mkey in METRIC_KEYS:
                        v = res["results"][horizon].get(mkey)
                        row[f"{h_short}_{mkey}"] = v
                rows.append(row)
    return rows


def write_master_csv(rows, path):
    if not rows:
        print("  No results to write to master CSV.")
        return
    fieldnames = []
    for r in rows:
        for k in r.keys():
            if k not in fieldnames:
                fieldnames.append(k)
    with open(path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        for r in rows:
            w.writerow(r)
    print(f"  Wrote {len(rows)} rows → {path}")


def write_ranking_stability(rows, path):
    grouped = {}
    for r in rows:
        for h in ["near", "far"]:
            r2_col = f"{h}_Global_R2"
            if r.get(r2_col) is None:
                continue
            grouped.setdefault((r["cytokine"], h), {}).setdefault(
                r["strategy"], []).append((r["model"], r[r2_col]))

    out_rows = []
    for (cyt, horizon), per_strategy in sorted(grouped.items()):
        for strategy in STRATEGIES:
            if strategy not in per_strategy:
                continue
            ranked = sorted(per_strategy[strategy], key=lambda x: -x[1])
            row = {"cytokine": cyt, "horizon": horizon, "strategy": strategy}
            for i, (m, r2) in enumerate(ranked, start=1):
                row[f"r{i}_model"] = m
                row[f"r{i}_R2"] = round(r2, 4)
            out_rows.append(row)

    if not out_rows:
        print("No ranking data to write.")
        return

    fieldnames = []
    for r in out_rows:
        for k in r.keys():
            if k not in fieldnames:
                fieldnames.append(k)
    with open(path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        for r in out_rows:
            w.writerow(r)
    print(f"  Wrote {len(out_rows)} ranking rows → {path}")


def write_summary(rows, path, args):
    lines = []
    lines.append("=" * 70)
    lines.append("  CLIPPING SENSITIVITY ANALYSIS SUMMARY")
    lines.append(f"  Grid: {args.grid}²  |  Seed: {args.seed}")
    lines.append("=" * 70)
    lines.append("")

    grouped = {}
    for r in rows:
        for h in ["near", "far"]:
            r2_col = f"{h}_Global_R2"
            if r.get(r2_col) is None:
                continue
            grouped.setdefault((r["cytokine"], h), {}).setdefault(
                r["strategy"], []).append((r["model"], r[r2_col]))

    for (cyt, horizon), per_strategy in sorted(grouped.items()):
        lines.append(f"--- {cyt.upper()}  ({horizon} horizon) ---")
        rankings = {}
        for strategy in STRATEGIES:
            if strategy not in per_strategy:
                continue
            ranked = sorted(per_strategy[strategy], key=lambda x: -x[1])
            rankings[strategy] = [m for (m, _) in ranked]
            r2_str = " > ".join([f"{m}({r2:.4f})" for (m, r2) in ranked])
            lines.append(f"  {strategy:>10s}:  {r2_str}")

        ref = None
        all_same = True
        for s in STRATEGIES:
            if s not in rankings:
                continue
            if ref is None:
                ref = rankings[s]
            elif rankings[s] != ref:
                all_same = False
        lines.append(f"  RANKING STABLE: {'YES' if all_same else 'NO'}")
        lines.append("")

    lines.append("=" * 70)
    lines.append("  R² SHIFT VS. ADAPTIVE (paper's strategy)")
    lines.append("=" * 70)
    lines.append("")

    by_model_cyt = {}
    for r in rows:
        key = (r["model"], r["cytokine"])
        by_model_cyt.setdefault(key, {})[r["strategy"]] = {
            "near": r.get("near_Global_R2"),
            "far":  r.get("far_Global_R2"),
        }
    for (model, cyt), strat_dict in sorted(by_model_cyt.items()):
        if "adaptive" not in strat_dict:
            continue
        ref_near = strat_dict["adaptive"]["near"]
        ref_far = strat_dict["adaptive"]["far"]
        if ref_near is None or ref_far is None:
            continue
        lines.append(f"  {model:>12s} | {cyt:>5s} | "
                      f"adaptive: near={ref_near:.4f} far={ref_far:.4f}")
        for s in ["no_clip", "p95"]:
            if s in strat_dict and strat_dict[s]["near"] is not None:
                dn = strat_dict[s]["near"] - ref_near
                df = strat_dict[s]["far"] - ref_far
                lines.append(f"               {s:>10s}: Δnear={dn:+.4f} Δfar={df:+.4f}")
        lines.append("")

    with open(path, "w") as f:
        f.write("\n".join(lines))
    print(f"  Wrote summary → {path}")
    print()
    print("\n".join(lines))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--base-dir", type=str, default=".",
                    help="Project root (contains sensitivity_results/, models/, etc.)")
    ap.add_argument("--grid", type=int, default=50)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--adaptive-from-paper-models", action="store_true",
                    help="Use existing paper results under ./models/ for the "
                         "'adaptive' strategy (instead of re-running)")
    args = ap.parse_args()

    base = Path(args.base_dir)
    out_dir = base / "sensitivity_results"
    out_dir.mkdir(parents=True, exist_ok=True)

    print(f"Aggregating results (grid={args.grid}, seed={args.seed})...")
    if args.adaptive_from_paper_models:
        print("  Pulling 'adaptive' results from ./models/ (paper's own runs)")
    print()

    rows = collect_all_results(args)
    if not rows:
        print("No results found. Did you run the training scripts?")
        return

    write_master_csv(rows, out_dir / "master_results.csv")
    write_ranking_stability(rows, out_dir / "ranking_stability.csv")
    write_summary(rows, out_dir / "summary.txt", args)

if __name__ == "__main__":
    main()