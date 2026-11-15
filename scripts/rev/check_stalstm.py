"""Check of the STA-LSTM increment model (IL-8): are the three seeds the same prediction, and is the
learned correction spatially uniform (the STA-LSTM pools globally before the LSTM)?"""
import sys, numpy as np
import common as C
grid = int(sys.argv[1]) if len(sys.argv) > 1 else 250
d = C.load(grid, "il8"); idx, cmax = d["idx"], d["cmax"]
S, E = C.START, C.END
truth = C.den(d["Y"][S:E], cmax); pers = C.den(d["Xu"][S:E, ..., 11 + idx], cmax)
mse = lambda a, b: float(np.mean((a - b) ** 2))
dtrue = truth - pers
print(f"grid {grid}: frames {S}-{E - 1}, MSE persistence {mse(truth, pers):.3e}")
oracle = pers + dtrue.mean(axis=(1, 2), keepdims=True)
print(f"oracle 'persistence + true domain-mean change per frame': skill {1 - mse(truth, oracle) / mse(truth, pers):+.3f}")
P = {}
for s in (42, 1, 100):
    p = np.load(C.OUT / "main_stalstm" / "increment" / f"pred_il8_{grid}_{s}.npz")["one_step"].astype(np.float64); P[s] = p
    inc = p - pers
    print(f"seed {s:3d}: skill {1 - mse(truth, p) / mse(truth, pers):+.4f} | spatial SD of predicted change / of true change "
          f"{np.mean(inc.std(axis=(1, 2))) / np.mean(dtrue.std(axis=(1, 2))):.3f} | corr(pred change, true change) "
          f"{np.corrcoef(inc.ravel(), dtrue.ravel())[0, 1]:.3f}")
for a, b in ((42, 1), (42, 100)):
    print(f"seed {a} vs {b}: max |difference| / c_max {np.max(np.abs(P[a] - P[b])) / cmax:.2e}")
