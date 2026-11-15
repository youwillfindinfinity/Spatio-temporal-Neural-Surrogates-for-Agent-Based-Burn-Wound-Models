"""R2 of the stored PINN (final weights) in the training, validation and test windows, IL-10 and IL-8.
The network is a plain tanh MLP (x, y, t) -> u, evaluated here with numpy from the saved weights."""
import glob, json, re, sys
import numpy as np, h5py
import common as C

def load_mlp(path):
    ks, bs = {}, {}
    def visit(name, obj):
        if isinstance(obj, h5py.Dataset):
            g = name.rsplit("/vars/", 1)[0] if "/vars/" in name else name.rsplit("/", 1)[0]
            (ks if obj.ndim == 2 else bs)[g] = np.array(obj, dtype=np.float64)
    with h5py.File(path, "r") as f: f.visititems(visit)
    nat = lambda s: [int(t) if t.isdigit() else t for t in re.split(r"(\d+)", s)]
    layers = [(ks[g], bs[g]) for g in sorted(ks, key=nat)]
    first = [l for l in layers if l[0].shape[0] == 3]; last = [l for l in layers if l[0].shape[1] == 1]
    mid = [l for l in layers if l[0].shape[0] != 3 and l[0].shape[1] != 1]
    return first + mid + last

def forward(layers, X):
    h = X
    for i, (W, b) in enumerate(layers):
        h = h @ W + b
        if i < len(layers) - 1: h = np.tanh(h)
    return h[:, 0]

def r2(y, p): return 1 - np.sum((y - p) ** 2) / np.sum((y - y.mean()) ** 2)
def corr(y, p): return float(np.mean([np.corrcoef(a.ravel(), b.ravel())[0, 1] for a, b in zip(y, p) if a.std() > 0 and b.std() > 0]))

WIN = {"train t2-71": slice(0, 70), "val t72-81": slice(70, 80), "near t82-91": slice(80, 90), "far t92-100": slice(90, 99)}
tn = np.linspace(-1, 1, 101)
print(f"{'file':40s}{'orient':>7s} " + " ".join(f"{w:>22s}" for w in WIN) + "   stored near R2")
for variant in ("strict", "soft"):
    for cyt in ("il10", "il8"):
        for G in (100, 250):
            d = C.load(G, cyt); Y = d["Y"].astype(np.float64)
            lin = np.linspace(-1, 1, G); A, B = np.meshgrid(lin, lin, indexing="ij")
            grids = {"ij": np.stack([A.ravel(), B.ravel()], 1)}
            for wf in sorted(glob.glob(str(C.RESULTS / "rev_abm_pinn" / variant / "pinn" / f"weights_{cyt}_{G}_*-*.weights.h5"))):
                seed = re.search(rf"weights_{cyt}_{G}_(\d+)-", wf).group(1)
                stored = json.load(open(C.RESULTS / "rev_abm_pinn" / variant / "pinn" / f"res_{cyt}_{G}_{seed}.json"))["results"]["Near_Horizon_t82_t91"]["Global_R2"]
                L = load_mlp(wf); best = None
                for o, xy in grids.items():
                    P = np.stack([forward(L, np.hstack([xy, np.full((G * G, 1), tn[i + 2])])).reshape(G, G) for i in range(99)])
                    P = np.clip(P, -1, 1); near = r2(Y[80:90], P[80:90])
                    if best is None or abs(near - stored) < abs(best[2] - stored): best = (o, P, near)
                o, P, near = best
                print(f"{variant + ' ' + cyt + ' ' + str(G) + ' s' + seed:40s}{o:>7s} " +
                      " ".join(f"R2 {r2(Y[s], P[s]):+.3f} corr {corr(Y[s], P[s]):+.2f}".rjust(22) for s in WIN.values()) + f"   {stored:+.4f}", flush=True)
