#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
verify_cec2017_data.py
----------------------
Sanity check for the official CEC2017 integration.

Run before any production benchmark that claims to use the official suite:

    python verify_cec2017_data.py

Checks, for D in {10, 30, 50, 100} (D=100 added 2026-10-31: the final run
used it, but this script had never checked it):

  1. FILES      every shift_data_*.txt / M_*_D*.txt / shuffle_data_*_D*.txt the
                suite needs is present.
  2. SHAPES     each rotation file is a whole number of DxD blocks, each shuffle
                file's first D entries are a permutation of 1..D, each shift
                file has at least D values (and at least one row per component
                for the compositions).
  3. BUILD      all 30 functions construct without raising.
  4. FINITE     each function returns finite values on a random population and
                on the domain corners.
  5. OPTIMUM    f(o_i) == bias_i = 100*i for the ten simple functions F1-F10,
                where o_i is the official shift vector. This is the strongest
                available alignment test: it fails immediately if a shift is
                misread, a rotation is transposed, or a scale factor is wrong.
  6. OPTIMUM    the same check for the hybrids and compositions F11-F30, at
     (all 30)   the first official shift vector. Every component is zero at its
                own optimum, so this cannot catch a wrong component list or
                wrong proportions; it does catch a misread shuffle, a missing
                optimum offset (the hgbat/happycat defect) or a wrong bias.

Exit status is 0 only when every check passes.
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))

from benchmarks import cec2017_official as C

DIMS = (10, 30, 50, 100)
RTOL_OPTIMUM = 1e-6


def main() -> int:
    print("=" * 78)
    print("  OFFICIAL CEC2017 DATA VERIFICATION")
    print("=" * 78)
    print(f"  data directory: {C.DATA_DIR}")

    status = C.data_status(DIMS)
    n_need = len(status["present"]) + len(status["missing"])
    print(f"  files present : {len(status['present'])} / {n_need}")

    if status["missing"]:
        print(f"  files MISSING : {len(status['missing'])}")
        print("\n  first missing entries:")
        for f in status["missing"][:12]:
            print(f"      {f}")
        if len(status["missing"]) > 12:
            print(f"      ... and {len(status['missing']) - 12} more")
        print("\n  RESULT: NOT READY -- official data is not installed.")
        print("  The suite cannot be built, and `cec_mode=\"official\"` will raise.")
        print("  Install the official CEC2017 input_data files into the directory")
        print("  above, then re-run this script.")
        return 1

    failures = []
    for dim in DIMS:
        print("\n" + "-" * 78)
        print(f"  D = {dim}")
        print("-" * 78)
        try:
            specs = C.official_cec2017_suite(dim, acknowledge_unverified=True)
        except Exception as exc:                                  # noqa: BLE001
            print(f"    BUILD FAILED: {type(exc).__name__}: {exc}")
            failures.append(f"D={dim} build")
            continue
        print(f"    built {len(specs)}/30 functions")

        rng = np.random.default_rng(12345)
        lo, hi = C.SEARCH_RANGE
        X = rng.uniform(lo, hi, size=(64, dim))
        corners = np.stack([np.full(dim, lo), np.full(dim, hi), np.zeros(dim)])

        for s in specs:
            try:
                v = np.concatenate([s.func(X), s.func(corners)])
            except Exception as exc:                              # noqa: BLE001
                print(f"    F{s.id:<2} EVAL FAILED: {type(exc).__name__}: {exc}")
                failures.append(f"D={dim} F{s.id} eval")
                continue
            if not np.all(np.isfinite(v)):
                print(f"    F{s.id:<2} produced non-finite values")
                failures.append(f"D={dim} F{s.id} finite")

        # optimum alignment for the simple functions
        bad = []
        for fid in sorted(C._SIMPLE_SPEC):
            o = C.load_shift(fid, dim)
            if fid == 9:
                # The official C's Levy (scale 1, w = 1 + (z-1)/4) has its
                # optimum where z = M(x-o) = 1, i.e. at x = o + M^-1 * 1, not at
                # o. Check the value 900 there instead.
                M9 = C.load_rotation(9, dim)[0]
                o = o + np.linalg.solve(M9, np.ones(dim))
                if np.any(np.abs(o) > 100.0):
                    bad.append((fid, float("nan"), 900.0))
                    continue
            got = float(next(s for s in specs if s.id == fid).func(o[None, :])[0])
            want = 100.0 * fid
            if not np.isclose(got, want, rtol=RTOL_OPTIMUM, atol=1e-6):
                bad.append((fid, got, want))
        if bad:
            print(f"    OPTIMUM MISMATCH on {len(bad)} of 10 simple functions:")
            for fid, got, want in bad:
                print(f"        F{fid:<2} f(o) = {got:.6e}   expected {want:.1f}")
            failures.append(f"D={dim} optimum")
        else:
            print("    optimum check: F1-F10 all evaluate to 100*i at the official shift")

        # the same alignment check for the hybrids and compositions
        bad = []
        for s in specs:
            if s.id <= 10:
                continue
            o = C.load_shift(s.id, dim)
            got = float(s.func(o[None, :])[0])
            if not np.isclose(got, 100.0 * s.id, rtol=RTOL_OPTIMUM, atol=1e-6):
                bad.append((s.id, got))
        if bad:
            print(f"    OPTIMUM MISMATCH on {len(bad)} of 20 hybrid/composition "
                  f"functions:")
            for fid, got in bad:
                print(f"        F{fid:<2} f(o_1) = {got:.6e}   expected "
                      f"{100.0 * fid:.1f}")
            failures.append(f"D={dim} optimum F11-F30")
        else:
            print("    optimum check: F11-F30 all evaluate to 100*i at the first "
                  "official shift")

    print("\n" + "=" * 78)
    if failures:
        print(f"  RESULT: FAILED ({len(failures)} check groups)")
        for f in failures:
            print(f"      {f}")
        return 1
    print("  RESULT: READY -- official CEC2017 suite verified for D = "
          + ", ".join(str(d) for d in DIMS))
    if not C._TABLES_VERIFIED:
        print("\n  NOTE: _TABLES_VERIFIED is still False. The structural checks above")
        print("  pass, but the component recipes / proportions / sigma-lambda-bias")
        print("  tables must still be read against the official technical report")
        print("  before publishing any number from this suite.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
