#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
crosscheck_cec2017_c.py -- compare benchmarks/cec2017_official.py value by
value with the organisers' compiled C code (benchmarks/cec2017_reference_c).

    g++ -O2 -std=gnu++17 -static -w -I benchmarks/cec2017_reference_c \
        benchmarks/cec2017_reference_c/cec17_eval.cpp -o cec17_eval.exe
    python crosscheck_cec2017_c.py --exe cec17_eval.exe

Exit status is 0 only if every function matches at every dimension.
"""
from __future__ import annotations

import argparse
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
from benchmarks import cec2017_official as C  # noqa: E402

TOL = 1e-8                      # |py - c| / max(1, |c|)
DIMS = (10, 30, 50, 100)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--exe", required=True, help="compiled cec17_eval")
    ap.add_argument("--seed", type=int, default=20261031)
    a = ap.parse_args()
    exe = Path(a.exe).resolve()
    rng = np.random.default_rng(a.seed)

    with tempfile.TemporaryDirectory() as tmp:
        work = Path(tmp)
        shutil.copytree(C.DATA_DIR, work / "input_data")   # the C opens input_data/...

        def c_eval(fid: int, dim: int, X: np.ndarray) -> np.ndarray:
            fin, fout = work / "in.bin", work / "out.bin"
            np.ascontiguousarray(X, dtype="<f8").tofile(fin)
            subprocess.run([str(exe), str(fid), str(dim), str(len(X)), str(fin),
                            str(fout)], cwd=work, check=True)
            return np.fromfile(fout, dtype="<f8")

        def points(fid: int, dim: int) -> np.ndarray:
            rows = np.atleast_2d(np.loadtxt(C.DATA_DIR / f"shift_data_{fid}.txt"))[:, :dim]
            o = rows[0]
            P = [rng.uniform(-100, 100, (200, dim)), o[None, :]]
            for s in (1e-6, 1e-3, 1e-1, 1.0, 10.0):
                P.append(np.clip(o + s * rng.standard_normal((20, dim)), -100, 100))
            if fid >= 21:                       # every composition component's optimum
                for r in rows[1:10]:
                    P.append(r[None, :])
                    for s in (1e-3, 1.0):
                        P.append(np.clip(r + s * rng.standard_normal((5, dim)), -100, 100))
            return np.vstack(P)

        worst = {}
        for dim in DIMS:
            specs = {s.id: s for s in C.official_cec2017_suite(dim)}
            for fid in range(1, 31):
                X = points(fid, dim)
                c = c_eval(fid, dim, X)
                p = np.asarray(specs[fid].func(X), float)
                worst[(fid, dim)] = float(np.max(np.abs(p - c) / np.maximum(1.0, np.abs(c))))

    print(f"{'F':>3} " + "".join(f"{'D=' + str(d):>12}" for d in DIMS) + "   verdict")
    bad = []
    for fid in range(1, 31):
        r = [worst[(fid, d)] for d in DIMS]
        ok = max(r) <= TOL
        if not ok:
            bad.append(fid)
        print(f"{fid:>3} " + "".join(f"{v:12.2e}" for v in r) + ("   ok" if ok else "   MISMATCH"))
    print(f"\nmatching at every dimension: {30 - len(bad)}/30 (relative tolerance {TOL:g})")
    if bad:
        print("differ: " + ", ".join(f"F{f}" for f in bad))
    return 0 if not bad else 1


if __name__ == "__main__":
    raise SystemExit(main())
