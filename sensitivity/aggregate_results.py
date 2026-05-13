#!/usr/bin/env python3

import json, csv
from pathlib import Path

ROOT      = Path("./sensitivity_results")
OUT_CSV   = ROOT / "all_metrics.csv"

VARIANTS  = ["no_clip", "p95", "adaptive"]
ARCHS = [
    ("deeponet",     "DeepONet"),
    ("pi_deeponet",  "PI-DeepONet"),
    ("unet",         "U-Net"),
    ("sta_lstm",     "STA-LSTM"),
    ("pinn",         "PINN"),
]
CYTS      = [("il8", "IL-8"), ("il10", "IL-10")]
HORIZONS  = [("Near_Horizon_t82_t91", "near"),
             ("Far_Horizon_t92_t100", "far")]
METRICS   = ["Global_R2", "SSIM", "Avg_Dice",
             "Spatial_Correlation", "Masked_RMSE"]

GRID = 50
SEED = 42


def _fmt(v):
    if v is None:
        return ""
    try:
        f = float(v)
        if f != f: # NaN
            return ""
        return f
    except (TypeError, ValueError):
        return ""

def main():
    rows = []
    missing = []

    for variant in VARIANTS:
        for arch_dir, arch_label in ARCHS:
            for cyt_key, cyt_label in CYTS:
                path = ROOT / variant / arch_dir / f"res_{cyt_key}_{GRID}_{SEED}.json"
                if not path.exists():
                    missing.append(str(path))
                    for _, horizon_label in HORIZONS:
                        rows.append({
                            "cytokine":     cyt_label,
                            "horizon":      horizon_label,
                            "architecture": arch_label,
                            "clipping":     variant,
                            **{m: "" for m in METRICS},
                        })
                    continue

                with open(path) as f:
                    res = json.load(f)
                results = res.get("results", {})

                for horizon_key, horizon_label in HORIZONS:
                    h = results.get(horizon_key, {})
                    rows.append({
                        "cytokine":     cyt_label,
                        "horizon":      horizon_label,
                        "architecture": arch_label,
                        "clipping":     variant,
                        **{m: _fmt(h.get(m)) for m in METRICS},
                    })

    arch_order = {label: i for i, (_, label) in enumerate(ARCHS)}
    clip_order = {v: i for i, v in enumerate(VARIANTS)}
    cyt_order  = {label: i for i, (_, label) in enumerate(CYTS)}
    horizon_order = {"near": 0, "far": 1}

    rows.sort(key=lambda r: (
        cyt_order[r["cytokine"]],
        horizon_order[r["horizon"]],
        arch_order[r["architecture"]],
        clip_order[r["clipping"]],
    ))

    OUT_CSV.parent.mkdir(parents=True, exist_ok=True)
    with open(OUT_CSV, "w", newline="") as f:
        fieldnames = ["cytokine", "horizon", "architecture", "clipping"] + METRICS
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        w.writerows(rows)

    print(f"Wrote {len(rows)} rows -> {OUT_CSV}")
    if missing:
        print(f"\nWARNING: {len(missing)} missing JSON file(s):")
        for m in missing:
            print(f"  - {m}")


if __name__ == "__main__":
    main()
