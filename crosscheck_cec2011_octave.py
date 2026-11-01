#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
crosscheck_cec2011_octave.py -- compare benchmarks/cec2011_official.py with
the organisers' own MATLAB code, run unmodified in GNU Octave.

For each of the eight problems it evaluates the same points with both:
  * 300 uniform points in the official box,
  * 100 points near a good solution (a short CMA-ES run on the Python port,
    then Gaussian perturbations of sigma 1e-4 .. 1e-1 in the unit box),
  * the official test point where one exists (P1: the target parameters).
It reports, per problem, the largest relative difference on points where both
are finite, and whether both agree on which points are non-finite. Octave's
complex results (MATLAB semantics for acos/sqrt/log out of range) are reduced
to their real part, as the official C wrapper does with mxGetPr, and counted.

    python crosscheck_cec2011_octave.py --octave <path to octave-cli.exe>
"""
from __future__ import annotations

import argparse
import subprocess
import sys
import tempfile
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
from benchmarks.cec2011_official import PROBLEMS, unit_box   # noqa: E402

MATLAB_DIR = HERE / "benchmarks" / "cec2011_reference" / "src_Linux-C" / "cec2011" / "matlab"

OCTAVE_CALL = {
    1: "f = bench_func(x, 1);",
    2: "f = bench_func(x, 2);",
    5: "f = bench_func(x, 5);",
    6: "f = bench_func(x, 6);",
    7: "f = bench_func(x, 7);",
    10: "f = antennafunccircular(x, [50 120], 180, 0.5);",
    12: "f = messengerfull(x, MESS);",
    13: "f = cassini2(x, CASS);",
}


def points(pid: int, rng) -> np.ndarray:
    import MDAC_EXPERIMENT as M
    name, D, func, (lb, ub) = PROBLEMS[pid]
    U = [rng.random((300, D))]
    ev = M.Evaluator(unit_box(pid), D, max_fes=20000)
    try:
        M.resolve("CMA-ES")(ev, D, (0.0, 1.0), 12345 + pid, pop_size=None)
    except M.BudgetExhausted:
        pass
    best = ev.best_x
    for sig in (1e-4, 1e-3, 1e-2, 1e-1):
        U.append(np.clip(best + rng.normal(0.0, sig, (25, D)), 0.0, 1.0))
    X = np.vstack([lb + u * (ub - lb) for u in U])
    if pid == 1:
        X = np.vstack([X, [1.0, 5.0, -1.5, 4.8, 2.0, 4.9]])
    return X


def run_octave(octave: str, work: Path, pids) -> None:
    lines = [
        "warning('off', 'all');",
        f"addpath('{MATLAB_DIR.as_posix()}');",
        f"S = load('{(MATLAB_DIR / 'messengerfull.mat').as_posix()}'); MESS = S.MGADSMproblem;",
        f"S = load('{(MATLAB_DIR / 'cassini2.mat').as_posix()}'); CASS = S.MGADSMproblem;",
    ]
    for pid in pids:
        inp = (work / f"x_{pid}.txt").as_posix()
        out = (work / f"oct_{pid}.txt").as_posix()
        lines += [
            f"X = dlmread('{inp}');",
            f"fid = fopen('{out}', 'w');",
            "for r = 1:rows(X)",
            "  x = X(r, :);",
            "  try",
            f"    {OCTAVE_CALL[pid]}",
            "    fprintf(fid, '%.17g %.17g\\n', real(f(1)), imag(f(1)));",
            "  catch err",
            "    fprintf(fid, 'ERR 0\\n');",
            "  end",
            "end",
            "fclose(fid);",
        ]
    script = work / "run_official.m"
    script.write_text("\n".join(lines) + "\n", encoding="utf-8")
    subprocess.run([octave, "--no-gui", "--quiet", "--norc", str(script)],
                   check=True, cwd=str(work))


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--octave", required=True)
    ap.add_argument("--pids", type=int, nargs="+", default=sorted(PROBLEMS))
    ap.add_argument("--tol", type=float, default=1e-8)
    a = ap.parse_args()

    rng = np.random.default_rng(20111031)
    with tempfile.TemporaryDirectory() as td:
        work = Path(td)
        py_vals = {}
        for pid in a.pids:
            X = points(pid, rng)
            np.savetxt(work / f"x_{pid}.txt", X, fmt="%.17g", delimiter=",")
            with np.errstate(all="ignore"):
                py_vals[pid] = PROBLEMS[pid][2](X)
        run_octave(a.octave, work, a.pids)

        print(f"{'problem':24s} {'pts':>4s} {'both finite':>11s} {'max rel diff':>13s} "
              f"{'non-finite agree':>17s} {'oct complex':>11s} {'oct error':>9s}  verdict")
        all_ok = True
        for pid in a.pids:
            name = PROBLEMS[pid][0]
            py = py_vals[pid]
            oc, im, err = [], [], []
            for line in (work / f"oct_{pid}.txt").read_text().split("\n"):
                if not line.strip():
                    continue
                re_, ii = line.split()
                if re_ == "ERR":
                    oc.append(np.nan); im.append(0.0); err.append(True)
                else:
                    oc.append(float(re_)); im.append(float(ii)); err.append(False)
            oc, im, err = np.array(oc), np.array(im), np.array(err)
            assert len(oc) == len(py), (pid, len(oc), len(py))
            fo, fp = np.isfinite(oc) & ~err, np.isfinite(py)
            both = fo & fp
            rel = (np.max(np.abs(py[both] - oc[both]) / np.maximum(1.0, np.abs(oc[both])))
                   if both.any() else 0.0)
            agree = int(np.sum(fo == fp))
            ok = rel <= a.tol and agree == len(py)
            all_ok &= ok
            print(f"P{pid:<2d} {name:20s} {len(py):4d} {int(both.sum()):11d} {rel:13.2e} "
                  f"{agree:8d}/{len(py):<8d} {int(np.sum(im != 0)):11d} {int(err.sum()):9d}  "
                  f"{'ok' if ok else 'DIFF'}")
            if not ok:
                bad = np.where(~(fo == fp) | (both & (np.abs(py - oc) > a.tol * np.maximum(1.0, np.abs(oc)))))[0][:5]
                for b in bad:
                    print(f"      point {b}: python {py[b]!r}  octave {oc[b]!r}  imag {im[b]!r}  err {err[b]}")
        print(f"\nmatching problems: {'all' if all_ok else 'NOT all'} (relative tolerance {a.tol:g})")


if __name__ == "__main__":
    main()
