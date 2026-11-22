"""
Run one CompuCell3D variant headless and record its provenance.

Uses CompuCell3D 4.6.0 (conda package, compucell3d channel, Python 3.10) in
~/miniforge3/envs/cc3d_460; no environment modules. 4.6.0 was chosen because it
reproduces the published cell dynamics (the 4.7.0 builds remove all motile 1-pixel
cells within the first simulated hour); see experiment.json["simulator"].

Writes <out>/run_provenance.json with: model settings (from abm_provenance.json),
seed, command line, software versions, hardware, SLURM context, wall time,
simulation speed (MCS/s, wall seconds per simulated hour, per-frame timings),
peak memory, output checks and checksums of the model files.

Usage:
  python run_abm.py --run-dir <variant dir> --out <output dir>
"""
import argparse
import datetime
import hashlib
import json
import os
import platform
import resource
import socket
import subprocess
import sys
import time
from pathlib import Path

ENV_PY = Path.home() / "miniforge3" / "envs" / "cc3d_460" / "bin" / "python"
MCS_PER_HOUR = 10000


def cc3d_env():
    env = dict(os.environ)
    env.pop("PYTHONPATH", None)
    env["LC_NUMERIC"] = "C.UTF-8"
    env["OMP_NUM_THREADS"] = "4"  # <NumberOfProcessors>4</NumberOfProcessors> in the XML
    return env


def versions(env):
    code = ("import sys, json, importlib.metadata as m\n"
            "out = {'python': sys.version.split()[0]}\n"
            "import cc3d; out['cc3d'] = getattr(cc3d, '__version__', '?')\n"
            "for p in ('numpy', 'scipy', 'fipy'):\n"
            "    try: out[p] = m.version(p)\n"
            "    except Exception as e: out[p] = repr(e)\n"
            "print(json.dumps(out))")
    r = subprocess.run([str(ENV_PY), "-W", "ignore", "-c", code],
                       env=env, capture_output=True, text=True)
    try:
        return json.loads(r.stdout.strip().splitlines()[-1])
    except Exception:
        return {"error": r.stderr[-2000:]}


def hardware():
    info = {"hostname": socket.gethostname(), "platform": platform.platform(),
            "logical_cpus_on_node": os.cpu_count(),
            "cpus_available_to_job": len(os.sched_getaffinity(0))}
    try:
        for line in subprocess.run(["lscpu"], capture_output=True, text=True).stdout.splitlines():
            k, _, v = line.partition(":")
            if k.strip() in ("Model name", "CPU MHz", "CPU max MHz", "Socket(s)",
                             "Core(s) per socket", "Thread(s) per core"):
                info[k.strip()] = v.strip()
        mem = Path("/proc/meminfo").read_text().splitlines()[0]
        info["MemTotal"] = mem.split(":", 1)[1].strip()
    except Exception as e:
        info["error"] = repr(e)
    return info


def md5(path):
    return hashlib.md5(Path(path).read_bytes()).hexdigest()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run-dir", required=True)
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    run_dir, out = Path(a.run_dir).resolve(), Path(a.out).resolve()
    out.mkdir(parents=True, exist_ok=False)
    model = json.loads((run_dir / "abm_provenance.json").read_text())
    cc3d_file = run_dir / "endothelial.cc3d"
    env = cc3d_env()

    cmd = [str(ENV_PY), "-W", "ignore", "-m", "cc3d.run_script",
           "-i", str(cc3d_file), "-o", str(out), "-f", str(MCS_PER_HOUR),
           "--current-dir", str(run_dir)]
    # CC3D 4.6.0 runs the steppables at MCS 0 by default (as in the published data)
    # and writes LatticeData/Step_*.vtk itself every 10,000 MCS (= 1 simulated hour)
    prov = {
        "run_id": run_dir.name,
        "model": model, "seed": model["seed"], "command": cmd,
        "model_file_md5": {str(p.relative_to(run_dir)): md5(p)
                           for p in sorted(run_dir.rglob("*")) if p.is_file()},
        "software": versions(env), "hardware": hardware(),
        "slurm": {k: v for k, v in os.environ.items() if k.startswith("SLURM_")},
        "environment": {k: env.get(k) for k in ("LC_NUMERIC", "OMP_NUM_THREADS")},
        "simulator": "CompuCell3D 4.6.0, conda env " + str(ENV_PY.parents[1]),
    }
    t0 = time.time(); prov["start"] = datetime.datetime.now().isoformat(timespec="seconds")
    with open(out / "cc3d_stdout.log", "w") as so, open(out / "cc3d_stderr.log", "w") as se:
        rc = subprocess.run(cmd, cwd=run_dir, env=env, stdout=so, stderr=se).returncode
    wall = time.time() - t0
    prov["end"] = datetime.datetime.now().isoformat(timespec="seconds")
    prov["exit_code"] = rc
    prov["peak_rss_children_MB"] = resource.getrusage(resource.RUSAGE_CHILDREN).ru_maxrss / 1024

    vtks = sorted(out.rglob("*.vtk"), key=lambda p: p.stat().st_mtime)
    hours = model["hours"]
    frame_times = [p.stat().st_mtime - t0 for p in vtks]
    per_hour = [b - a for a, b in zip(frame_times, frame_times[1:])]
    prov["speed"] = {
        "wall_seconds": round(wall, 2),
        "mcs_total": hours * MCS_PER_HOUR,
        "mcs_per_second": round(hours * MCS_PER_HOUR / wall, 2),
        "wall_seconds_per_simulated_hour": round(wall / hours, 3),
        "simulated_hours_per_wall_hour": round(hours / (wall / 3600), 2),
        "per_frame_wall_seconds": {
            "mean": round(sum(per_hour) / len(per_hour), 3) if per_hour else None,
            "min": round(min(per_hour), 3) if per_hour else None,
            "max": round(max(per_hour), 3) if per_hour else None},
        "note": "wall time includes CC3D start-up and VTK output",
    }
    prov["output"] = {
        "n_vtk": len(vtks), "expected_vtk": hours + 1, "complete": len(vtks) == hours + 1 and rc == 0,
        "lattice_dir": str(vtks[0].parent.relative_to(out)) if vtks else None,
        "bytes": sum(p.stat().st_size for p in vtks),
        "vtk_md5_first_last": {p.name: md5(p) for p in (vtks[:1] + vtks[-1:])} if vtks else {},
    }
    (out / "run_provenance.json").write_text(json.dumps(prov, indent=2))
    print(json.dumps({k: prov[k] for k in ("run_id", "exit_code", "speed", "output")}, indent=2))
    sys.exit(0 if prov["output"]["complete"] else 1)


if __name__ == "__main__":
    main()
