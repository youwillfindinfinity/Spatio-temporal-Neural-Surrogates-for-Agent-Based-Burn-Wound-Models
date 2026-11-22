"""
Generate a ready-to-run CompuCell3D variant of the burn-wound ABM (combi_clean).

The template in abm/template/ is the unmodified 500x500 / 200 h configuration
(ICCS2026_v2/combi_clean). The earlier 50^2, 100^2 and 250^2 datasets of this model were
produced from the same model with the grid-dependent settings changed by hand;
those edited copies were not archived. This script re-creates them
deterministically and records every setting in abm_provenance.json.

Settings reconstructed from the published LatticeData (see README.md):
  - nx = G, <Dimensions> = G x G
  - boundaryat = G // 10
  - <Steps> = hours * 10000 + 1
  - all initial-seeding and periodic top-up counts scaled by G / 500
    (minimum 1), endocount = round(1000 * G / 500)
  - hard-coded 500 / 499 / 250000 lattice bounds replaced by nx-based bounds

Fixes / additions relative to the template:
  - --seed fixes both the Potts RNG (<RandomSeed> in endothelial.xml) and the
    Python-side RNGs (random and numpy.random), which are seeded in the main
    script endothelial.py and again in endothelialSteppables.py on import.
    Note: the steppables do `import random` and then `from numpy import *`,
    which rebinds `random` to numpy.random, so numpy's global RNG is the one
    that drives cell placement and phenotype transitions.
  - --il10-source selects the IL-10-secreting cell type. The template
    multiplies km2il10 by the M1 presence array (cellpresentm1); the parameter
    name and the model description say M2. Default: m2 (fixed). Use m1 to
    reproduce the published datasets.

Usage:
  python make_variant.py --grid 100 --hours 100 --seed 1 --out runs/G100_h100_s1
  python make_variant.py --grid 250 --hours 100 --seed 1 --il10-source m1 --out runs/G250_published_like
"""
import argparse
import hashlib
import json
import re
import shutil
from pathlib import Path

TEMPLATE = Path(__file__).resolve().parent / "template"
_VTK_WRITER = r"""
def _write_lattice_vtk(stp, mcs, out_dir):
    # Legacy ASCII VTK frame in the layout of the published LatticeData
    # (CC3D 4.2.5 output): 10 point-data fields, x fastest, 9 values per line.
    # Used instead of CC3D's own writer, which segfaults in this build
    # (CompuCell3D 4.7 compiled against VTK 9.5, Python vtk 9.6 in the venv).
    nx_ = ny_ = vv.nx
    ct = np.zeros((nx_, ny_), dtype=np.int64); cid = np.zeros_like(ct); clid = np.zeros_like(ct)
    for cell in stp.cell_list:
        for ptd in stp.get_cell_pixel_list(cell):
            ct[ptd.pixel.x, ptd.pixel.y] = cell.type
            cid[ptd.pixel.x, ptd.pixel.y] = cell.id
            clid[ptd.pixel.x, ptd.pixel.y] = cell.clusterId
    # XML DiffusionSolverFE field "cytokine": D = 0 and nothing secretes into it, so it is
    # identically zero (verified in the published LatticeData). Not read from CC3D:
    # slicing that field throws an uncatchable C++ exception in this build.
    cyto = np.zeros((nx_, ny_))
    fields = [("CellType", "char", ct), ("CellId", "long", cid),
              ("ClusterId", "long", clid), ("cytokine", "double", cyto)]
    for name in ("il1", "il10", "il6", "il8", "tgf", "tnf"):
        fields.append((name, "double",
                       np.array(getattr(stp, "scalarField" + name)[:, :, 0], dtype=float)))
    n = nx_ * ny_
    path = os.path.join(out_dir, "Step_%07d.vtk" % mcs)
    with open(path + ".tmp", "w") as fh:
        fh.write("# vtk DataFile Version 5.1\nvtk output\nASCII\nDATASET STRUCTURED_POINTS\n"
                 f"DIMENSIONS {nx_} {ny_} 1\nSPACING 1 1 1\nORIGIN 0 0 0\n"
                 f"POINT_DATA {n}\nFIELD FieldData {len(fields)}\n")
        for name, typ, arr in fields:
            flat = arr.flatten(order="F")
            fmt = "%.11g" if typ == "double" else "%d"
            fh.write(f"{name} 1 {n} {typ}\n")
            for i in range(0, n, 9):
                fh.write(" ".join(fmt % v for v in flat[i:i + 9]) + " \n")
    os.replace(path + ".tmp", path)
"""
REF_GRID = 500          # grid of the archived template
MCS_PER_HOUR = 10000    # relaxationmcs in variablevals.py


def sub_exact(text, old, new, count, fname):
    """Replace `old` with `new`, asserting it occurs exactly `count` times."""
    n = text.count(old)
    if n != count:
        raise RuntimeError(f"{fname}: expected {count} x {old!r}, found {n}")
    return text.replace(old, new)


def scaled(n, grid):
    return max(1, int(round(n * grid / REF_GRID)))


def md5(path):
    return hashlib.md5(Path(path).read_bytes()).hexdigest()


def make_variant(grid, hours, seed, il10_source, out, endo_factor=1.0, boundary_frac=10,
                 python_vtk_writer=True):
    try:
        return _make_variant(grid, hours, seed, il10_source, out, endo_factor, boundary_frac,
                             python_vtk_writer)
    except Exception:
        if Path(out).exists() and (Path(out) / "Simulation").exists():
            shutil.rmtree(out)  # do not leave a half-edited model behind
        raise


def _make_variant(grid, hours, seed, il10_source, out, endo_factor, boundary_frac, python_vtk_writer):
    if grid % boundary_frac:
        raise ValueError(f"grid must be a multiple of boundary_frac={boundary_frac}")
    out = Path(out)
    if out.exists():
        raise FileExistsError(f"{out} already exists")
    shutil.copytree(TEMPLATE, out)
    sim = out / "Simulation"
    boundaryat = grid // boundary_frac
    endocount = scaled(1000 * endo_factor, grid)

    prov = {
        "model": "burn-wound ABM combi_clean (Korkmaz et al. 2024), CompuCell3D 4.2.5 + FiPy",
        "template_md5": {f.name: md5(TEMPLATE / "Simulation" / f.name)
                         for f in sorted((TEMPLATE / "Simulation").glob("*.py"))},
        "grid": grid, "hours": hours, "steps": hours * MCS_PER_HOUR + 1,
        "boundaryat": boundaryat, "boundary_frac": boundary_frac, "endocount": endocount,
        "endo_factor": endo_factor,
        "lattice_writer": ("python legacy-VTK writer in the steppable (published layout); "
                           "CC3D 4.7 VTK output segfaults in this build") if python_vtk_writer
                          else "CompuCell3D lattice output (-f)",
        "cell_count_scale": f"max(1, round(k * {grid} / {REF_GRID}))",
        "seed": seed, "il10_source": il10_source,
        "published_dataset_equivalent": il10_source == "m1" and endo_factor == 1 and boundary_frac == 10,
        "cell_type_ids": {"Medium": 0, "endothelial": 1, "neutrophil": 2, "monocyte": 3,
                          "fibroblast": 4, "neutrophila": 5, "neutrophilndn": 6,
                          "monocyter": 7, "macrophage1": 8, "macrophage2": 9, "myofibroblast": 10},
    }
    prov_json = json.dumps(prov, indent=2)

    # variablevals.py: grid size and Dirichlet-patch width
    p = sim / "variablevals.py"; t = p.read_text()
    t = sub_exact(t, "nx = 500\n", f"nx = {grid}\n", 1, p.name)
    t = sub_exact(t, "boundaryat = 50\n", f"boundaryat = {boundaryat}\n", 1, p.name)
    t = t.rstrip("\n") + (
        f"\n\n# --- added by make_variant.py ---\n"
        f"seed = {seed}\n"
        f"il10_source = {il10_source!r}  # cell type secreting IL-10 in test.py\n"
    )
    p.write_text(t)

    # endothelial.xml: lattice size, run length, Potts RNG seed
    p = sim / "endothelial.xml"; t = p.read_text()
    t = sub_exact(t, '<Dimensions x="500" y="500" z="1"/>',
                  f'<Dimensions x="{grid}" y="{grid}" z="1"/>', 1, p.name)
    t = sub_exact(t, "<Steps>2000001</Steps>",
                  f"<Steps>{hours * MCS_PER_HOUR + 1}</Steps>\n      <RandomSeed>{seed}</RandomSeed>",
                  1, p.name)
    p.write_text(t)

    # endothelial.py (main CC3D script): seed the Python-side RNGs before any model module
    # is imported. The steppables reseed with the same value on import, before the first draw.
    p = sim / "endothelial.py"; t = p.read_text()
    t = sub_exact(t, "from cc3d import CompuCellSetup\n",
                  "from cc3d import CompuCellSetup\n"
                  "# --- added by make_variant.py: random seed ---\n"
                  "# Potts (cell motion): <RandomSeed> in endothelial.xml, same value.\n"
                  "# numpy.random drives placement, lifespans and phenotype switching.\n"
                  "import random\n"
                  "import numpy as np\n"
                  "from variablevals import seed\n"
                  "random.seed(seed)\n"
                  "np.random.seed(seed)\n",
                  1, p.name)
    p.write_text(t)

    # test.py: IL-10 secreting cell type
    p = sim / "test.py"; t = p.read_text()
    t = sub_exact(t, "km2il10*cellpresentm1", f"km2il10*cellpresent{il10_source}", 1, p.name)
    p.write_text(t)

    # endothelialSteppables.py
    p = sim / "endothelialSteppables.py"; t = p.read_text()
    t = sub_exact(t, "endocount = {{endocount}} #{{endocount}}",
                  f"endocount = {endocount}  # = round(1000 * {endo_factor} * {grid} / {REF_GRID})", 1, p.name)
    # seed the Python-side RNG (numpy.random, see module docstring) after the star import
    t = sub_exact(t, "import numpy as np\n",
                  "import numpy as np\n"
                  "import builtins as _builtins\n"
                  "import random as _py_random\n"
                  "_py_random.seed(vv.seed)\n"
                  "np.random.seed(vv.seed)  # `random` here is numpy.random\n"
                  "def _n(k):\n"
                  f"    '''Cell count k (defined for a {REF_GRID}x{REF_GRID} lattice) scaled to the current grid.'''\n"
                  "    # builtins: `from numpy import *` shadows max/round with numpy versions\n"
                  f"    return _builtins.max(1, _builtins.int(_builtins.round(k * vv.nx / {REF_GRID})))\n",
                  1, p.name)
    # `random` must already be numpy.random (star import) and vv imported when we seed
    i_seed = t.index("_py_random.seed(vv.seed)")
    if not (t.index("import variablevals as vv") < i_seed and t.index("from numpy import *") < i_seed):
        raise RuntimeError("seeding block must follow `import variablevals as vv` and `from numpy import *`")
    # seeding and periodic top-up counts
    for k, cnt in [(1000, 2), (100, 3), (25, 1), (10, 2), (3, 1)]:
        t = sub_exact(t, f"for i in range(0,{k}):", f"for i in range(0,_n({k})):", cnt, p.name)
    # lattice bounds hard-coded for 500 x 500
    t = sub_exact(t, "if xCOM>=500:", "if xCOM>=nx:", 2, p.name)
    t = sub_exact(t, "if yCOM>=500:", "if yCOM>=ny:", 2, p.name)
    t = sub_exact(t, "xCOM = 499", "xCOM = nx-1", 2, p.name)
    t = sub_exact(t, "yCOM = 499", "yCOM = ny-1", 2, p.name)
    t = sub_exact(t, "if pos>250000:", "if pos>=nx*ny:", 1, p.name)
    t = sub_exact(t, "if 50<xCOM<450 and 50<yCOM<450:",
                  "if vv.boundaryat<xCOM<nx-vv.boundaryat and vv.boundaryat<yCOM<ny-vv.boundaryat:",
                  1, p.name)
    # write provenance next to LatticeData in the CC3D output folder
    t = sub_exact(t, '        fileHandle,fullFileName=self.open_file_in_simulation_output_folder("datafiles/creatdoc.txt","w")\n',
                  '        fileHandle,fullFileName=self.open_file_in_simulation_output_folder("datafiles/creatdoc.txt","w")\n'
                  '        _prov = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(fullFileName))), "abm_provenance.json")\n'
                  '        with open(_prov, "w") as _fh:\n'
                  '            _fh.write(_PROVENANCE_JSON)\n',
                  1, p.name)
    t = sub_exact(t, "import csv\n", "import csv\n_PROVENANCE_JSON = " + repr(prov_json) + "\n", 1, p.name)
    if python_vtk_writer:
        # module-level writer function, placed after the class-free helper block
        t = sub_exact(t, "class endothelialSteppable(SteppableBasePy):",
                      _VTK_WRITER + "\n\nclass endothelialSteppable(SteppableBasePy):", 1, p.name)
        # create LatticeData/ next to datafiles/ in start()
        t = sub_exact(t, '            _fh.write(_PROVENANCE_JSON)\n',
                      '            _fh.write(_PROVENANCE_JSON)\n'
                      '        self._lattice_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(fullFileName))), "LatticeData")\n'
                      '        os.makedirs(self._lattice_dir, exist_ok=True)\n', 1, p.name)
        # write one frame at the end of every step (= every 10,000 MCS)
        anchor = "            cytowriter.writerow([str(mcs),str(il8_mean),str(il1_mean),str(il6_mean),str(il10_mean),str(tnf_mean),str(tgf_mean),str(il8_std),str(il1_std),str(il6_std),str(il10_std),str(tnf_std),str(tgf_std)])\n"
        t = sub_exact(t, anchor, anchor + "        _write_lattice_vtk(self, mcs, self._lattice_dir)\n", 1, p.name)
    p.write_text(t)

    for f in sim.glob("*.py"):
        compile(f.read_text(), str(f), "exec")
    steps_txt = "\n".join(l for l in (sim / "endothelialSteppables.py").read_text().splitlines()
                          if not l.lstrip().startswith("#"))
    remaining = re.findall(r"\b(?:499|250000)\b|>=500\b|range\(0,(?:1000|100|25|10|3)\)", steps_txt)

    (out / "abm_provenance.json").write_text(prov_json)
    if remaining:
        raise RuntimeError(f"unexpected leftovers in steppables: {remaining}")
    return prov


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--grid", type=int, required=True, help="lattice size G (multiple of 10)")
    ap.add_argument("--hours", type=int, default=100, help="simulated hours (10,000 MCS each)")
    ap.add_argument("--seed", type=int, required=True)
    ap.add_argument("--il10-source", choices=["m1", "m2"], default="m2",
                    help="IL-10 secreting cell type: m2 = fixed (default), m1 = as in the published data")
    ap.add_argument("--endo-factor", type=float, default=1.0,
                    help="multiply the initial endothelial cell count (scenario test)")
    ap.add_argument("--boundary-frac", type=int, default=10,
                    help="boundaryat = grid // boundary_frac (10 = published geometry)")
    ap.add_argument("--cc3d-vtk-writer", action="store_true",
                    help="use CC3D's own lattice output instead of the Python writer")
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    print(json.dumps(make_variant(a.grid, a.hours, a.seed, a.il10_source, a.out,
                                  a.endo_factor, a.boundary_frac, not a.cc3d_vtk_writer), indent=2))
