"""Table: PINN with softened vs strict residual (results/rev_pinn) next to the stored manuscript runs."""
import json, sys
from pathlib import Path
import numpy as np
ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from paths import RESULTS, RESULTS_MODELS, OUTPUTS

import os
TAG = os.environ.get("REV_OUT")
if TAG:      # revision ABM runs, residual consistent with the ABM
    SRC = {"soft": RESULTS / f"{TAG}_pinn" / "soft" / "pinn", "strict": RESULTS / f"{TAG}_pinn" / "strict" / "pinn"}
else:        # original data and code
    SRC = {"manuscript (stored)": RESULTS_MODELS / "pinn", "soft (re-run)": RESULTS / "rev_pinn" / "soft" / "pinn",
           "strict": RESULTS / "rev_pinn" / "strict" / "pinn"}
KEYS = ["Global_R2", "SSIM", "Avg_Dice", "Spatial_Correlation", "Masked_RMSE"]
lines = ["PINN: softened vs strict residual (mean +/- sample SD over available seeds)", ""]
for cyt in ("il10", "il8"):
    for grid in (100, 250):
        lines.append(f"=== {cyt.upper()} {grid}x{grid} ===")
        lines.append(f"{'variant':22s}{'n':>3s}{'window':>7s}" + "".join(f"{k:>24s}" for k in KEYS) + f"{'train s':>10s}")
        for name, d in SRC.items():
            rs = [json.load(open(p)) for p in sorted(d.glob(f"res_{cyt}_{grid}_*.json"))] if d.exists() else []
            if not rs: continue
            for wk, wn in (("Near_Horizon_t82_t91", "near"), ("Far_Horizon_t92_t100", "far")):
                cells = []
                for k in KEYS:
                    v = np.array([r["results"][wk][k] for r in rs], float)
                    cells.append(f"{v.mean():.4g} +/- {v.std(ddof=1) if len(v) > 1 else float('nan'):.2g}")
                tt = np.mean([r.get("train_time_seconds", np.nan) for r in rs])
                lines.append(f"{name:22s}{len(rs):>3d}{wn:>7s}" + "".join(f"{c:>24s}" for c in cells) + f"{tt:10.0f}")
        lines.append("")
out = OUTPUTS / (TAG or "rev"); out.mkdir(parents=True, exist_ok=True)
(out / "summary_pinn.txt").write_text("\n".join(lines) + "\n")
print("\n".join(lines))
