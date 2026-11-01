#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
crosscheck_cec2020rw_octave.py -- compare benchmarks/cec2020rw_official.py
with the organisers' MATLAB code (cec20_func.m), run unmodified in Octave,
on the objective AND every constraint, one point at a time (ps = 1).

Points per problem: 300 uniform in the official box, 100 near a good point
(short CMA-ES run on the port with a provisional penalty, then Gaussian
perturbations of sigma 1e-4 .. 1e-1 in the unit box).

    python crosscheck_cec2020rw_octave.py --octave <octave-cli.exe>
    python crosscheck_cec2020rw_octave.py --scan     # largest feasible |f| per problem
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
from benchmarks import cec2020rw_official as R   # noqa: E402

CODE_DIR = HERE / "benchmarks" / "cec2020rw_reference" / "src" / "Software" / "CEC2020"


def provisional(pid):
    name, D, func, (lb, ub), fstar, nh = R.PROBLEMS[pid]

    def F(U):
        U = np.atleast_2d(U)
        f, G, H = R.evaluate(pid, lb + U * (ub - lb))
        v = R.violation(pid, G, H)
        out = np.where(v == 0, f, 1e12 + v)
        return np.where(np.isfinite(out), out, 1e20)
    return F


def good_points(pid, rng, n_each=25, fes=20000):
    import MDAC_EXPERIMENT as M
    name, D, func, (lb, ub), fstar, nh = R.PROBLEMS[pid]
    ev = M.Evaluator(provisional(pid), D, max_fes=fes)
    try:
        M.resolve("CMA-ES")(ev, D, (0.0, 1.0), 4242 + pid, pop_size=None)
    except M.BudgetExhausted:
        pass
    U = [np.clip(ev.best_x + rng.normal(0, s, (n_each, D)), 0, 1) for s in (1e-4, 1e-3, 1e-2, 1e-1)]
    return np.vstack(U)


def points(pid, rng):
    name, D, func, (lb, ub), fstar, nh = R.PROBLEMS[pid]
    U = np.vstack([rng.random((300, D)), good_points(pid, rng)])
    return lb + U * (ub - lb)


def run_octave(octave, work, pids, ncols):
    lines = ["warning('off', 'all');", "global initial_flag; initial_flag = 0;",
             f"addpath('{CODE_DIR.as_posix()}');"]
    for pid in pids:
        ng, nh = ncols[pid]
        lines += [
            f"X = dlmread('{(work / f'x_{pid}.txt').as_posix()}');",
            f"fid = fopen('{(work / f'oct_{pid}.txt').as_posix()}', 'w');",
            "for r = 1:rows(X)",
            "  try",
            f"    [f, g, h] = cec20_func(X(r, :), {pid});",
            # cec20_func ends with g = g'; h = h'  -> constraints are columns
            f"    v = [f(1), g(1:{ng}, 1)'" + (f", h(1:{nh}, 1)'" if nh else "") + "];",
            "    fprintf(fid, '%.17g ', [real(v), imag(v)]); fprintf(fid, '\\n');",
            "  catch err",
            "    fprintf(fid, 'ERR\\n');",
            "  end",
            "end",
            "fclose(fid);",
        ]
    (work / "run_official.m").write_text("\n".join(lines) + "\n", encoding="utf-8")
    subprocess.run([octave, "--no-gui", "--quiet", "--norc", str(work / "run_official.m")],
                   check=True, cwd=str(work))


def crosscheck(octave, pids, tol):
    rng = np.random.default_rng(20201031)
    with tempfile.TemporaryDirectory() as td:
        work = Path(td)
        py, ncols = {}, {}
        for pid in pids:
            X = points(pid, rng)
            np.savetxt(work / f"x_{pid}.txt", X, fmt="%.17g", delimiter=",")
            f, G, H = R.evaluate(pid, X)
            nh = R.PROBLEMS[pid][5]
            ncols[pid] = (G.shape[1], nh)
            py[pid] = np.column_stack([f, G, H[:, :nh]])
        run_octave(octave, work, pids, ncols)
        print(f"{'problem':26s} {'pts':>4s} {'cols':>4s} {'complex pts':>11s} {'max rel diff (real pts)':>24s} "
              f"{'non-finite agree':>17s} {'feasible agree':>15s}  verdict")
        all_ok = True
        for pid in pids:
            P = py[pid]
            rows = [l.split() for l in (work / f"oct_{pid}.txt").read_text().splitlines() if l.strip()]
            assert len(rows) == len(P)
            k = P.shape[1]
            O = np.full((len(rows), k), np.nan)
            Oi = np.zeros((len(rows), k))
            err = np.zeros(len(rows), bool)
            for i, r in enumerate(rows):
                if r[0] == "ERR":
                    err[i] = True
                    continue
                vals = np.array([float(t) for t in r])
                O[i], Oi[i] = vals[:k], vals[k:2 * k]
            cplx = (np.abs(Oi) > 0).any(axis=1) | err
            real = ~cplx
            fo, fp = np.isfinite(O[real]), np.isfinite(P[real])
            both = fo & fp
            d = np.abs(P[real] - O[real]) / np.maximum(1.0, np.abs(O[real]))
            rel = float(np.max(d[both])) if both.any() else 0.0
            agree = int(np.sum(np.all(fo == fp, axis=1)))
            # feasibility agreement on real points
            nh = R.PROBLEMS[pid][5]
            ng = k - 1 - nh

            def feas(A):
                G, H = A[:, 1:1 + ng], A[:, 1 + ng:]
                return np.all(G <= 0, axis=1) & np.all(np.abs(H) <= R.EQ_TOL, axis=1)
            fa = int(np.sum(feas(P[real]) == feas(O[real])))
            ok = rel <= tol and agree == real.sum() and fa == real.sum()
            all_ok &= ok
            print(f"RC{pid:<2d} {R.PROBLEMS[pid][0]:21s} {len(P):4d} {k:4d} {int(cplx.sum()):11d} {rel:24.2e} "
                  f"{agree:8d}/{int(real.sum()):<8d} {fa:7d}/{int(real.sum()):<7d}  {'ok' if ok else 'DIFF'}")
            if not ok:
                bad = np.where(~np.all(np.isclose(P[real], O[real], rtol=tol, atol=tol, equal_nan=True), axis=1))[0][:3]
                for b in bad:
                    j = int(np.argmax(np.abs(np.nan_to_num(P[real][b] - O[real][b], nan=1e300))))
                    print(f"      point {b} column {j}: python {P[real][b][j]!r} octave {O[real][b][j]!r}")
        print(f"\nmatching problems: {'all' if all_ok else 'NOT all'} (relative tolerance {tol:g}; "
              f"points where Octave returns complex values are counted, not compared)")


def scan(n_uniform=200_000):
    rng = np.random.default_rng(31102020)
    print(f"{'problem':26s} {'feasible uniform':>16s} {'max |f| feasible':>17s} {'C_k':>9s}")
    for pid in sorted(R.PROBLEMS):
        name, D, func, (lb, ub), fstar, nh = R.PROBLEMS[pid]
        mx, nfeas = 0.0, 0
        for chunk in range(0, n_uniform, 20_000):
            m = min(20_000, n_uniform - chunk) if pid not in (27, 33) else min(2_000, n_uniform - chunk)
            if pid in (27, 33) and chunk >= 20_000:
                break
            X = lb + rng.random((m, D)) * (ub - lb)
            f, G, H = R.evaluate(pid, X)
            ok = (R.violation(pid, G, H) == 0) & np.isfinite(f)
            nfeas += int(ok.sum())
            if ok.any():
                mx = max(mx, float(np.max(np.abs(f[ok]))))
        Xg = lb + good_points(pid, rng) * (ub - lb)
        f, G, H = R.evaluate(pid, Xg)
        ok = (R.violation(pid, G, H) == 0) & np.isfinite(f)
        if ok.any():
            mx = max(mx, float(np.max(np.abs(f[ok]))))
        C = 10.0 ** max(3, int(np.ceil(np.log10(max(mx, 1.0)))) + 3)
        print(f"RC{pid:<2d} {name:21s} {nfeas:16d} {mx:17.6g} {C:9.0e}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--octave")
    ap.add_argument("--pids", type=int, nargs="+", default=sorted(R.PROBLEMS))
    ap.add_argument("--tol", type=float, default=1e-8)
    ap.add_argument("--scan", action="store_true")
    a = ap.parse_args()
    if a.scan:
        scan()
    else:
        crosscheck(a.octave, a.pids, a.tol)


if __name__ == "__main__":
    main()
