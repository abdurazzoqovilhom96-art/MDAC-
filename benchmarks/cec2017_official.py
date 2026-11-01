#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
benchmarks/cec2017_official.py
------------------------------
Official IEEE CEC2017 single-objective bound-constrained suite (F1-F30), built
from the **official** shift vectors, rotation matrices and shuffle indices
rather than from locally generated random ones.

Why this module exists
----------------------
`benchmarks/functions.py::cec2017_style_suite` builds a 30-function bridge that
copies CEC2017's *category structure* but uses `np.random.default_rng`-generated
shift/rotation data. It is internally consistent and reproducible, but its
numbers are NOT comparable with any published CEC2017 result, so no ranking
computed on it can be presented as a CEC2017 ranking. This module replaces that
data with the official files.

Required data layout
--------------------
Place the official `input_data/` contents (from the CEC2017 release) in::

    benchmarks/cec2017_data/
        shift_data_1.txt  ...  shift_data_30.txt
        M_1_D10.txt       ...  M_30_D50.txt
        shuffle_data_11_D10.txt ... shuffle_data_20_D50.txt   (hybrids F11-F20)
        shuffle_data_21_D10.txt ... shuffle_data_30_D50.txt   (compositions that
                                                               use a shuffle)

Supported dimensions are the official ones: 2, 10, 20, 30, 50, 100. This project
uses 10 / 30 / 50.

Fail-loud policy
----------------
If the data is absent or malformed this module raises `Cec2017DataMissing` /
`Cec2017DataInvalid`. It NEVER falls back to generated data: a silent fallback
is exactly how a self-built benchmark ends up being reported as CEC2017.

VERIFY BEFORE PUBLICATION
-------------------------
The per-function tables below (`_HYBRID_SPEC`, `_COMPOSITION_SPEC`, `_SCALE`)
encode CEC2017's component recipes, hybrid proportions, and composition
sigma/lambda/bias values. They must be checked line by line against the official
technical report (Awad, Ali, Liang, Qu & Suganthan, "Problem Definitions and
Evaluation Criteria for the CEC 2017 Special Session and Competition on Single
Objective Real-Parameter Numerical Optimization") or against the official C /
MATLAB reference implementation before any result from this module is
published. `official_cec2017_suite()` requires `acknowledge_unverified=True`
until that check has been done and `_TABLES_VERIFIED` is set to True here.
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Dict, List, Sequence, Tuple

import numpy as np

Array = np.ndarray

DATA_DIR = Path(__file__).resolve().parent / "cec2017_data"
OFFICIAL_DIMS = (2, 10, 20, 30, 50, 100)
SEARCH_RANGE: Tuple[float, float] = (-100.0, 100.0)

# VERIFIED 2026-10-08 against the official C reference implementation
# `CEC17_fast_pow/cec17_test_func.cpp` (from the same master.zip as the data)
# and cross-read against the technical report. Eight defects were found and
# fixed; see RESULTS_FINAL.md §8 for the full list and the impact analysis.
#
#   F14   hybrid proportions were [0.2,0.2,0.3,0.3]  -> [0.2,0.2,0.2,0.4]
#   F21   components/lambda were F22's               -> rosenbrock/elliptic/rastrigin
#   F26   wrong components and lambda                -> escaffer6/schwefel/griewank/
#                                                       rosenbrock/rastrigin
#   F27   had 5 components                           -> SIX (escaffer6 was missing)
#   F28   had 5 components and F27's lambda          -> SIX, own lambda
#   F29   built from basic functions                 -> composition of H15/H16/H17
#   F30   built from basic functions                 -> composition of H15/H18/H19
#   hgbat / happycat  were missing their  z -= 1.0  optimum offset
#   (plus: cf mixing now matches cf_cal's exact-hit branch, w_i=1 / w_j=0)
#
# F9 (Levy), HISTORY. Until 2026-10-31 this module deliberately deviated from
#   the released C here: the C `levy_func` uses sh_rate = 1.0 and computes
#   w = 1 + (z-1)/4 with no optimum offset, so its minimum (still 900) lies at
#   o + 1, not at the shift o. The module used scale 5.12/100 with offset +1.0
#   instead. Since 2026-10-31 F9 follows the C ("levy_c17", below), and
#   verify_cec2017_data.py checks the value 900 where the C's minimum is.
#
# Checks that must pass: `verify_cec2017_data.py` (F1-F10) and the
# all-30 necessary-condition check (f(o_1) == 100*fid at D=10/30/50).
_TABLES_VERIFIED = True

# COMPILED CROSS-CHECK, 2026-10-31 (`crosscheck_cec2017_c.py` against the
# unmodified official C in benchmarks/cec2017_reference_c/). The tables above
# are right -- all ten compositions and seven of the ten hybrids match the C to
# <= 4e-11 -- but EIGHT functions in this module are NOT the functions the
# official code computes. The "verified by reading" claim of 2026-10-08 missed
# them. The reference's behaviour, quirks included, was re-implemented and
# matches the compiled C to <= 4e-14 for all seven non-F2 cases.
#
#   F2   exponent i+1 with 0-based i (here: i+2)                      [excluded]
#   F6   C calls schaffer_F7_func, NOT expanded Schaffer F6: sin^2, and it
#        reads the shifted but UNROTATED vector y
#   F7   bi_rastrigin: scale 10/100 (here 600/100 x 10/100), sign flipped where
#        the SHIFT vector is negative, rotation only inside the cosine term
#   F8   the step rounding is applied to a stale buffer that sr_func then
#        overwrites, so the C's F8 is plain shifted-rotated Rastrigin
#   F9   Levy at scale 1.0 with no optimum offset (the deviation noted above)
#   F13  its Lunacek component inherits the F7 differences
#   F14  its Schaffer component is F7-style with sin^2 and reads y[0..n), the
#        start of the shuffled vector, instead of its own segment
#   F20  same Schaffer quirk as F14
#
# FIXED 2026-10-31, same day: all eight are now C-exact (C-EXACT section
# below; F20's first component was also happycat instead of the C's hgbat).
# `crosscheck_cec2017_c.py` reports 30/30 at D=10/30/50/100. The FINAL_*
# cells of these functions were re-run with the C-exact code (run_cfix.ps1)
# and merged; the pre-fix rows are kept in _backup_2026-10-16/.
C_REFERENCE_DIFFERS: Tuple[int, ...] = ()


class Cec2017DataMissing(FileNotFoundError):
    """Official CEC2017 data files are not present."""


class Cec2017DataInvalid(ValueError):
    """A data file was found but has the wrong shape for the requested dim."""


# ============================================================================
# Base functions used by CEC2017 (all vectorized over (N, D) -> (N,))
# ============================================================================
def f_bent_cigar(X: Array) -> Array:
    return X[:, 0] ** 2 + 1e6 * np.sum(X[:, 1:] ** 2, axis=1)


def f_sum_diff_pow(X: Array) -> Array:
    d = np.arange(1, X.shape[1] + 1)
    return np.sum(np.abs(X) ** (d[None, :] + 1), axis=1)


def f_zakharov(X: Array) -> Array:
    s = np.sum(0.5 * np.arange(1, X.shape[1] + 1)[None, :] * X, axis=1)
    return np.sum(X ** 2, axis=1) + s ** 2 + s ** 4


def f_rosenbrock(X: Array) -> Array:
    a, b = X[:, :-1], X[:, 1:]
    return np.sum(100.0 * (b - a ** 2) ** 2 + (a - 1.0) ** 2, axis=1)


def f_rastrigin(X: Array) -> Array:
    return np.sum(X ** 2 - 10.0 * np.cos(2 * np.pi * X) + 10.0, axis=1)


def f_elliptic(X: Array) -> Array:
    d = X.shape[1]
    w = 10.0 ** (6.0 * np.arange(d) / max(1, d - 1))
    return np.sum(w[None, :] * X ** 2, axis=1)


def f_discus(X: Array) -> Array:
    return 1e6 * X[:, 0] ** 2 + np.sum(X[:, 1:] ** 2, axis=1)


def f_ackley(X: Array) -> Array:
    d = X.shape[1]
    s1 = np.sum(X ** 2, axis=1) / d
    s2 = np.sum(np.cos(2 * np.pi * X), axis=1) / d
    return -20.0 * np.exp(-0.2 * np.sqrt(s1)) - np.exp(s2) + 20.0 + np.e


def f_griewank(X: Array) -> Array:
    d = np.sqrt(np.arange(1, X.shape[1] + 1))
    return (np.sum(X ** 2, axis=1) / 4000.0
            - np.prod(np.cos(X / d[None, :]), axis=1) + 1.0)


def f_weierstrass(X: Array, a: float = 0.5, b: float = 3.0, kmax: int = 20) -> Array:
    d = X.shape[1]
    k = np.arange(kmax + 1)
    ak, bk = a ** k, b ** k
    inner = np.sum(ak[None, None, :] * np.cos(2 * np.pi * bk[None, None, :] * (X[:, :, None] + 0.5)), axis=2)
    const = float(np.sum(ak * np.cos(2 * np.pi * bk * 0.5)))
    return np.sum(inner, axis=1) - d * const


def f_modified_schwefel(X: Array) -> Array:
    d = X.shape[1]
    z = X + 420.9687462275036
    g = np.where(
        np.abs(z) <= 500.0,
        z * np.sin(np.sqrt(np.abs(z))),
        np.where(
            z > 500.0,
            (500.0 - np.mod(z, 500.0)) * np.sin(np.sqrt(np.abs(500.0 - np.mod(z, 500.0))))
            - (z - 500.0) ** 2 / (10000.0 * d),
            (np.mod(np.abs(z), 500.0) - 500.0) * np.sin(np.sqrt(np.abs(np.mod(np.abs(z), 500.0) - 500.0)))
            - (z + 500.0) ** 2 / (10000.0 * d),
        ),
    )
    return 418.9828872724338 * d - np.sum(g, axis=1)


def f_katsuura(X: Array) -> Array:
    n, d = X.shape
    j = 2.0 ** np.arange(1, 33)
    total = np.ones(n, dtype=np.float64)
    for i in range(d):
        t = np.abs(j[None, :] * X[:, i:i + 1] - np.round(j[None, :] * X[:, i:i + 1])) / j[None, :]
        total *= (1.0 + (i + 1) * np.sum(t, axis=1)) ** (10.0 / d ** 1.2)
    f = 10.0 / d ** 2
    return f * total - f


def f_happycat(X: Array) -> Array:
    d = X.shape[1]
    s = np.sum(X, axis=1)
    r = np.sum(X ** 2, axis=1)
    return np.abs(r - d) ** 0.25 + (0.5 * r + s) / d + 0.5


def f_hgbat(X: Array) -> Array:
    d = X.shape[1]
    s = np.sum(X, axis=1)
    r = np.sum(X ** 2, axis=1)
    return np.abs(r ** 2 - s ** 2) ** 0.5 + (0.5 * r + s) / d + 0.5


def _griewank_scalar(t: Array) -> Array:
    return t ** 2 / 4000.0 - np.cos(t) + 1.0


def f_expanded_griewank_rosenbrock(X: Array) -> Array:
    Z = np.concatenate([X, X[:, :1]], axis=1)
    a, b = Z[:, :-1], Z[:, 1:]
    ros = 100.0 * (b - a ** 2) ** 2 + (a - 1.0) ** 2
    return np.sum(_griewank_scalar(ros), axis=1)


def _schaffer_f6_pair(x: Array, y: Array) -> Array:
    s = x ** 2 + y ** 2
    return 0.5 + (np.sin(np.sqrt(s)) ** 2 - 0.5) / (1.0 + 0.001 * s) ** 2


def f_expanded_schaffer_f6(X: Array) -> Array:
    Z = np.concatenate([X, X[:, :1]], axis=1)
    return np.sum(_schaffer_f6_pair(Z[:, :-1], Z[:, 1:]), axis=1)


def f_schaffer_f7(X: Array) -> Array:
    d = X.shape[1]
    s = np.sqrt(X[:, :-1] ** 2 + X[:, 1:] ** 2)
    t = s ** 0.5 * (np.sin(50.0 * s ** 0.2) + 1.0)
    return (np.sum(t, axis=1) / (d - 1)) ** 2


def f_non_continuous_rastrigin(X: Array) -> Array:
    Y = np.where(np.abs(X) < 0.5, X, np.round(2.0 * X) / 2.0)
    return f_rastrigin(Y)


def f_levy(X: Array) -> Array:
    w = 1.0 + (X - 1.0) / 4.0
    t1 = np.sin(np.pi * w[:, 0]) ** 2
    wm = w[:, :-1]
    t2 = np.sum((wm - 1.0) ** 2 * (1.0 + 10.0 * np.sin(np.pi * wm + 1.0) ** 2), axis=1)
    t3 = (w[:, -1] - 1.0) ** 2 * (1.0 + np.sin(2 * np.pi * w[:, -1]) ** 2)
    return t1 + t2 + t3


def f_lunacek_bi_rastrigin(X: Array) -> Array:
    """Lunacek bi-Rastrigin. Operates on the already shifted/scaled input."""
    n, d = X.shape
    mu0 = 2.5
    s = 1.0 - 1.0 / (2.0 * np.sqrt(d + 20.0) - 8.2)
    mu1 = -np.sqrt((mu0 ** 2 - 1.0) / s)
    y = 10.0 * X / 100.0
    x1 = 2.0 * np.sign(y) * np.abs(y) + mu0     # sign handling per reference
    t1 = np.sum((x1 - mu0) ** 2, axis=1)
    t2 = d + s * np.sum((x1 - mu1) ** 2, axis=1)
    t3 = 10.0 * (d - np.sum(np.cos(2 * np.pi * (x1 - mu0)), axis=1))
    return np.minimum(t1, t2) + t3


# ----------------------------------------------------------------------------
# C-EXACT versions (2026-10-31). The organisers' C code, which produced every
# published CEC2017 result, computes these functions differently from the
# technical report. Each one below reproduces the C, quirks included, and is
# checked against the compiled C by `crosscheck_cec2017_c.py`.
# ----------------------------------------------------------------------------
def f_sum_diff_pow_c17(X: Array) -> Array:
    """C sum_diff_pow_func: pow(|z_i|, i+1) with 0-based i -> exponents 1..D
    (the report's formula gives 2..D+1)."""
    d = np.arange(1, X.shape[1] + 1)
    return np.sum(np.abs(X) ** d[None, :], axis=1)


def f_schaffer_f7_c17(X: Array) -> Array:
    """C schaffer_F7_func loop body: sqrt(s)*(1 + sin^2(50 s^0.2)), then
    (sum)^2/(n-1)^2. Which vector it reads is decided by the caller: the C
    reads the global buffer `y`, not its argument (see F6, F14, F20)."""
    n = X.shape[1]
    s = np.sqrt(X[:, :-1] ** 2 + X[:, 1:] ** 2)
    t = np.sqrt(s) + np.sqrt(s) * np.sin(50.0 * s ** 0.2) ** 2
    return np.sum(t, axis=1) ** 2 / (n - 1) / (n - 1)


def f_lunacek_c17(Y: Array, os_ref: Array, rot: Array = None) -> Array:
    """C bi_rastrigin_func after its shift. `Y` is x - o (F7) or the hybrid
    segment (F13). Scale 10/100; the sign is flipped where the SHIFT vector
    `os_ref` is negative; the rotation, when given, enters only the cosine."""
    n = Y.shape[1]
    mu0, d = 2.5, 1.0
    s = 1.0 - 1.0 / (2.0 * np.sqrt(n + 20.0) - 8.2)
    mu1 = -np.sqrt((mu0 * mu0 - d) / s)
    z = 2.0 * (0.1 * Y) * np.where(np.asarray(os_ref)[:n] < 0.0, -1.0, 1.0)[None, :]
    t1 = np.sum(z * z, axis=1)
    t2 = s * np.sum((z + mu0 - mu1) ** 2, axis=1) + d * n
    c = z @ rot.T if rot is not None else z
    return np.minimum(t1, t2) + 10.0 * (n - np.sum(np.cos(2.0 * np.pi * c), axis=1))


BASE: Dict[str, Callable[[Array], Array]] = {
    "bent_cigar": f_bent_cigar, "sum_diff_pow": f_sum_diff_pow, "zakharov": f_zakharov,
    "rosenbrock": f_rosenbrock, "rastrigin": f_rastrigin, "elliptic": f_elliptic,
    "discus": f_discus, "ackley": f_ackley, "griewank": f_griewank,
    "weierstrass": f_weierstrass, "schwefel": f_modified_schwefel,
    "katsuura": f_katsuura, "happycat": f_happycat, "hgbat": f_hgbat,
    "grie_rosen": f_expanded_griewank_rosenbrock, "schaffer_f6": f_expanded_schaffer_f6,
    "schaffer_f7": f_schaffer_f7, "noncont_rastrigin": f_non_continuous_rastrigin,
    "levy": f_levy, "lunacek": f_lunacek_bi_rastrigin,
    # C-exact (see above). F8's C is plain Rastrigin: its step rounding is
    # applied to a buffer that sr_func then overwrites. F9's C is Levy at
    # scale 1 with no optimum offset.
    "sum_diff_pow_c17": f_sum_diff_pow_c17,
    "noncont_rastrigin_c17": f_rastrigin,
    "levy_c17": f_levy,
}

# CEC2017 rescales the search domain onto each base function's natural domain
# before evaluating it.
_SCALE: Dict[str, float] = {
    "rastrigin": 5.12 / 100.0, "noncont_rastrigin": 5.12 / 100.0,
    "schwefel": 1000.0 / 100.0, "griewank": 600.0 / 100.0,
    "weierstrass": 0.5 / 100.0, "katsuura": 5.0 / 100.0,
    "happycat": 5.0 / 100.0, "hgbat": 5.0 / 100.0,
    "grie_rosen": 5.0 / 100.0, "rosenbrock": 2.048 / 100.0,
    "lunacek": 600.0 / 100.0, "levy": 5.12 / 100.0,
    "noncont_rastrigin_c17": 5.12 / 100.0,      # levy_c17 / sum_diff_pow_c17: 1.0
}


# Base functions whose optimum does not sit at the origin. CEC2017 adds the
# offset after the shift/scale/rotate so that the shifted optimum lands on the
# official shift vector. Caught by the f(o) == 100*i check in
# `verify_cec2017_data.py`, which fails loudly if this table is wrong.
_OFFSET: Dict[str, float] = {
    # C reference, applied AFTER shift/scale/rotate, inside each basic function:
    "rosenbrock": 1.0,      # rosenbrock_func:  z[i] += 1.0
    "grie_rosen": 1.0,      # grie_rosen_func:  z[i] += 1.0
    "levy": 1.0,            # levy_func:        w = 1 + (z-1)/4   (see note)
    # VERIFIED FIX -- these two were MISSING, so both functions returned their
    # value at the wrong point and were 0.5 instead of 0 at the optimum:
    #     hgbat_func:     z[i] = z[i]-1.0;  //shift to orgin
    #     happycat_func:  z[i] = z[i]-1.0;  //shift to orgin
    # The omission was invisible to verify_cec2017_data.py because that script
    # only checks F1-F10, and hgbat/happycat appear only in F11-F30.
    "hgbat": -1.0,
    "happycat": -1.0,
}


def scale_for(name: str) -> float:
    return _SCALE.get(name, 1.0)


def offset_for(name: str) -> float:
    return _OFFSET.get(name, 0.0)


# ============================================================================
# Official data loading
# ============================================================================
def _require_dir() -> Path:
    if not DATA_DIR.is_dir():
        raise Cec2017DataMissing(
            f"official CEC2017 data directory not found: {DATA_DIR}\n"
            "Download the official CEC2017 'input_data' archive and place its\n"
            "shift_data_*.txt / M_*_D*.txt / shuffle_data_*_D*.txt files there."
        )
    return DATA_DIR


def load_shift(func_id: int, dim: int) -> Array:
    p = _require_dir() / f"shift_data_{func_id}.txt"
    if not p.is_file():
        raise Cec2017DataMissing(f"missing shift file: {p}")
    v = np.loadtxt(p, dtype=np.float64).ravel()
    if v.size < dim:
        raise Cec2017DataInvalid(f"{p.name}: has {v.size} values, need >= {dim}")
    return v[:dim].copy()


def load_rotation(func_id: int, dim: int) -> Array:
    p = _require_dir() / f"M_{func_id}_D{dim}.txt"
    if not p.is_file():
        raise Cec2017DataMissing(f"missing rotation file: {p}")
    M = np.loadtxt(p, dtype=np.float64)
    n_comp = max(1, M.size // (dim * dim))
    if M.size != n_comp * dim * dim:
        raise Cec2017DataInvalid(
            f"{p.name}: {M.size} values is not a whole number of {dim}x{dim} blocks")
    return M.reshape(n_comp, dim, dim).copy()


def load_shuffle(func_id: int, dim: int) -> Array:
    p = _require_dir() / f"shuffle_data_{func_id}_D{dim}.txt"
    if not p.is_file():
        raise Cec2017DataMissing(f"missing shuffle file: {p}")
    idx = np.loadtxt(p, dtype=np.int64).ravel()
    if idx.size % dim != 0:
        raise Cec2017DataInvalid(f"{p.name}: {idx.size} indices is not a multiple of {dim}")
    idx = idx[:dim] - 1                      # official files are 1-based
    if np.sort(idx).tolist() != list(range(dim)):
        raise Cec2017DataInvalid(f"{p.name}: first {dim} indices are not a permutation of 1..{dim}")
    return idx.copy()


def load_shuffle_blocks(func_id: int, dim: int, n: int) -> Array:
    """The first `n` shuffle blocks of `func_id`, each a 0-based permutation.

    F29/F30 are compositions whose COMPONENTS are hybrid functions, so each
    component needs its own permutation. The official files store 10 blocks of
    `dim` 1-based indices; the C reference reads component k's block as
    `&SS[k*nx]` (see `cf09`/`cf10` in `cec17_test_func.cpp`).
    """
    p = _require_dir() / f"shuffle_data_{func_id}_D{dim}.txt"
    if not p.is_file():
        raise Cec2017DataMissing(f"missing shuffle file: {p}")
    idx = np.loadtxt(p, dtype=np.int64).ravel()
    if idx.size < n * dim:
        raise Cec2017DataInvalid(
            f"{p.name}: {idx.size} indices, need >= {n * dim} for {n} blocks")
    out = idx[:n * dim].reshape(n, dim) - 1
    for k in range(n):
        if np.sort(out[k]).tolist() != list(range(dim)):
            raise Cec2017DataInvalid(
                f"{p.name}: block {k} is not a permutation of 1..{dim}")
    return out.copy()


def data_status(dims: Sequence[int] = (10, 30, 50)) -> Dict[str, List[str]]:
    """Report which official files are present and which are missing."""
    present, missing = [], []
    if not DATA_DIR.is_dir():
        return {"present": [], "missing": [f"<entire directory> {DATA_DIR}"]}
    for fid in range(1, 31):
        f = f"shift_data_{fid}.txt"
        (present if (DATA_DIR / f).is_file() else missing).append(f)
        for d in dims:
            f = f"M_{fid}_D{d}.txt"
            (present if (DATA_DIR / f).is_file() else missing).append(f)
            if 11 <= fid <= 30:
                f = f"shuffle_data_{fid}_D{d}.txt"
                (present if (DATA_DIR / f).is_file() else missing).append(f)
    return {"present": present, "missing": missing}


def data_available(dims: Sequence[int] = (10, 30, 50)) -> bool:
    return not data_status(dims)["missing"]


# ============================================================================
# Function construction
# ============================================================================
@dataclass
class OfficialSpec:
    id: int
    name: str
    bounds: Tuple[float, float]
    func: Callable[[Array], Array]
    bias: float


def _sr(X: Array, shift: Array, rot: Array, scale: float, offset: float = 0.0) -> Array:
    """Official shift -> scale -> rotate (-> optimum offset) transform."""
    return ((X - shift[None, :]) * scale) @ rot.T + offset


def _make_simple(fid: int, dim: int, base_name: str) -> OfficialSpec:
    shift = load_shift(fid, dim)
    rot = load_rotation(fid, dim)[0]
    bias = 100.0 * fid

    if base_name == "schaffer_f7_c17_unrotated":
        # C F6: schaffer_F7_func reads `y`, the shifted vector BEFORE rotation
        def fn(X: Array) -> Array:
            return f_schaffer_f7_c17(X - shift[None, :]) + bias
    elif base_name == "lunacek_c17":
        # C F7: no scale in sr_func; bi_rastrigin shifts and scales itself
        def fn(X: Array) -> Array:
            return f_lunacek_c17(X - shift[None, :], shift, rot) + bias
    else:
        base, sc, off = BASE[base_name], scale_for(base_name), offset_for(base_name)

        def fn(X: Array) -> Array:
            return base(_sr(X, shift, rot, sc, off)) + bias

    return OfficialSpec(fid, f"CEC2017_F{fid}_{base_name}", SEARCH_RANGE, fn, bias)


def _hybrid_edges(fid: int, dim: int, props: Sequence[float]) -> Array:
    """Component boundaries, exactly as the C reference computes them.

    `hf01..hf10`:  G_nx[i] = ceil(Gp[i]*nx) for i < N-1, and the last component
    takes the remainder  G_nx[N-1] = nx - sum(previous).
    """
    sizes = [int(np.ceil(p * dim)) for p in props[:-1]]
    sizes.append(dim - int(np.sum(sizes)))
    if sizes[-1] <= 0:
        raise Cec2017DataInvalid(
            f"F{fid} D={dim}: proportions {list(props)} leave no room for the last component")
    return np.cumsum([0] + sizes)


def _hybrid_value(X: Array, parts: Sequence[str], edges: Array,
                  shift: Array, rot: Array, perm: Array) -> Array:
    """A hybrid function's value WITHOUT its 100*fid bias.

    Shared by `_make_hybrid` and by the F29/F30 compositions, whose components
    are hybrid functions (C reference: `cf09`/`cf10` call `hf05/hf06/hf07` and
    `hf05/hf08/hf09`).
    """
    Z = _sr(X, shift, rot, 1.0)[:, perm]
    total = np.zeros(X.shape[0], dtype=np.float64)
    for k, nm in enumerate(parts):
        seg = Z[:, edges[k]:edges[k + 1]]
        if nm == "lunacek_c17":
            # C hf03: bi_rastrigin with s_flag=0 flips signs by Os[0..n),
            # the first n entries of the HYBRID's shift vector; no rotation
            total += f_lunacek_c17(seg, shift)
        elif nm == "schaffer_f7_c17":
            # C hf04/hf10: schaffer_F7_func reads the global y[0..n), i.e.
            # the START of the shuffled vector, not its own segment
            total += f_schaffer_f7_c17(Z[:, :seg.shape[1]])
        else:
            total += BASE[nm](seg * scale_for(nm) + offset_for(nm))
    return total


def _make_hybrid(fid: int, dim: int, parts: Sequence[str], props: Sequence[float]) -> OfficialSpec:
    shift = load_shift(fid, dim)
    rot = load_rotation(fid, dim)[0]
    perm = load_shuffle(fid, dim)
    bias = 100.0 * fid
    edges = _hybrid_edges(fid, dim, props)

    def fn(X: Array) -> Array:
        return _hybrid_value(X, parts, edges, shift, rot, perm) + bias

    return OfficialSpec(fid, f"CEC2017_F{fid}_hybrid_" + "_".join(parts), SEARCH_RANGE, fn, bias)


def _cf_combine(sq: Array, vals: Array, sigmas: Sequence[float], dim: int,
                bias: float) -> Array:
    """The official composition mixing, matching `cf_cal` in the C reference.

        w_i = exp(-sq_i / (2*D*sigma_i^2)) / sqrt(sq_i)

    with ONE special case taken literally from the reference: a point sitting
    exactly on o_i gets w_i = 1 and every other weight 0 --

        if (w[i]!=0) w[i]=pow(1.0/w[i],0.5)*exp(...);
        else { for (j..) w[j]=0; w[i]=1; }

    A large-but-finite 1/eps instead leaves the other components contributing
    ~1e-5 at the optimum, which is why F30 missed f(o_1)==3000 exactly.
    """
    eps = 1e-12
    sig = np.asarray(sigmas, dtype=np.float64)[None, :]
    w = np.exp(-sq / (2.0 * dim * sig ** 2)) / np.sqrt(np.maximum(sq, eps))
    exact = sq <= 0.0
    rows = np.nonzero(np.any(exact, axis=1))[0]
    if rows.size:
        w[rows, :] = 0.0
        w[rows, np.argmax(exact[rows], axis=1)] = 1.0
    wsum = np.sum(w, axis=1, keepdims=True)
    wsum[wsum < eps] = eps
    return np.sum((w / wsum) * vals, axis=1) + bias


def _make_composition(fid: int, dim: int, parts: Sequence[str], sigmas: Sequence[float],
                      lambdas: Sequence[float], biases: Sequence[float]) -> OfficialSpec:
    n = len(parts)
    shift_all = np.loadtxt(_require_dir() / f"shift_data_{fid}.txt", dtype=np.float64)
    shift_all = np.atleast_2d(shift_all)
    if shift_all.shape[0] < n:
        raise Cec2017DataInvalid(
            f"shift_data_{fid}.txt: {shift_all.shape[0]} rows, composition needs {n}")
    shifts = [shift_all[k, :dim].copy() for k in range(n)]
    rots = load_rotation(fid, dim)
    if rots.shape[0] < n:
        raise Cec2017DataInvalid(
            f"M_{fid}_D{dim}.txt: {rots.shape[0]} blocks, composition needs {n}")
    bias = 100.0 * fid
    eps = 1e-12

    def fn(X: Array) -> Array:
        N = X.shape[0]
        sq = np.empty((N, n), dtype=np.float64)
        vals = np.empty((N, n), dtype=np.float64)
        for k, nm in enumerate(parts):
            diff = X - shifts[k][None, :]
            sq[:, k] = np.sum(diff * diff, axis=1)
            Z = (diff * scale_for(nm)) @ rots[k].T + offset_for(nm)
            vals[:, k] = lambdas[k] * BASE[nm](Z) + biases[k]
        return _cf_combine(sq, vals, sigmas, dim, bias)

    return OfficialSpec(fid, f"CEC2017_F{fid}_comp_" + "_".join(parts), SEARCH_RANGE, fn, bias)


def _make_composition_of_hybrids(fid: int, dim: int, hyb_ids: Sequence[int],
                                 sigmas: Sequence[float],
                                 lambdas: Sequence[float],
                                 biases: Sequence[float]) -> OfficialSpec:
    """F29 / F30: a composition whose components are HYBRID functions.

    C reference (`cec17_test_func.cpp`):
        cf09 -> hf05, hf06, hf07     delta = {10,30,50}  bias = {0,100,200}
        cf10 -> hf05, hf08, hf09     delta = {10,30,50}  bias = {0,100,200}
    Component k uses shift row k, rotation block k AND shuffle block k, which
    is why these two functions take `SS` while cf01..cf08 do not.
    """
    n = len(hyb_ids)
    shift_all = np.atleast_2d(np.loadtxt(_require_dir() / f"shift_data_{fid}.txt",
                                         dtype=np.float64))
    if shift_all.shape[0] < n:
        raise Cec2017DataInvalid(
            f"shift_data_{fid}.txt: {shift_all.shape[0]} rows, need {n}")
    shifts = [shift_all[k, :dim].copy() for k in range(n)]
    rots = load_rotation(fid, dim)
    if rots.shape[0] < n:
        raise Cec2017DataInvalid(
            f"M_{fid}_D{dim}.txt: {rots.shape[0]} blocks, need {n}")
    perms = load_shuffle_blocks(fid, dim, n)

    recipes = [_HYBRID_SPEC[h] for h in hyb_ids]
    edges = [_hybrid_edges(h, dim, props) for h, (_, props) in
             zip(hyb_ids, recipes)]
    bias = 100.0 * fid
    names = "_".join(f"H{h}" for h in hyb_ids)

    def fn(X: Array) -> Array:
        N = X.shape[0]
        sq = np.empty((N, n), dtype=np.float64)
        vals = np.empty((N, n), dtype=np.float64)
        for k in range(n):
            diff = X - shifts[k][None, :]
            sq[:, k] = np.sum(diff * diff, axis=1)
            vals[:, k] = lambdas[k] * _hybrid_value(
                X, recipes[k][0], edges[k], shifts[k], rots[k], perms[k]) + biases[k]
        return _cf_combine(sq, vals, sigmas, dim, bias)

    return OfficialSpec(fid, f"CEC2017_F{fid}_comp_{names}", SEARCH_RANGE, fn, bias)


# ============================================================================
# CEC2017 function tables  -- SEE "VERIFY BEFORE PUBLICATION" IN THE DOCSTRING
# ============================================================================
# F2, F6, F7, F8, F9 are C-exact since 2026-10-31. They were "sum_diff_pow",
# "schaffer_f6", "lunacek", "noncont_rastrigin" and "levy" (report formulas),
# which the compiled cross-check showed the official C does not compute.
_SIMPLE_SPEC: Dict[int, str] = {
    1: "bent_cigar", 2: "sum_diff_pow_c17", 3: "zakharov", 4: "rosenbrock",
    5: "rastrigin", 6: "schaffer_f7_c17_unrotated", 7: "lunacek_c17",
    8: "noncont_rastrigin_c17", 9: "levy_c17", 10: "schwefel",
}

_HYBRID_SPEC: Dict[int, Tuple[List[str], List[float]]] = {
    11: (["zakharov", "rosenbrock", "rastrigin"], [0.2, 0.4, 0.4]),
    12: (["elliptic", "schwefel", "bent_cigar"], [0.3, 0.3, 0.4]),
    13: (["bent_cigar", "rosenbrock", "lunacek_c17"], [0.3, 0.3, 0.4]),
    # VERIFIED FIX: was [0.2, 0.2, 0.3, 0.3]. C reference hf04:
    #     double Gp[4]={0.2,0.2,0.2,0.4};
    # and the report's "13) Hybrid Function 4" agrees: p = [0.2,0.2,0.2,0.4].
    14: (["elliptic", "ackley", "schaffer_f7_c17", "rastrigin"], [0.2, 0.2, 0.2, 0.4]),
    15: (["bent_cigar", "hgbat", "rastrigin", "rosenbrock"], [0.2, 0.2, 0.3, 0.3]),
    16: (["schaffer_f6", "hgbat", "rosenbrock", "schwefel"], [0.2, 0.2, 0.3, 0.3]),
    17: (["katsuura", "ackley", "grie_rosen", "schwefel", "rastrigin"], [0.1, 0.2, 0.2, 0.2, 0.3]),
    18: (["elliptic", "ackley", "rastrigin", "hgbat", "discus"], [0.2, 0.2, 0.2, 0.2, 0.2]),
    19: (["bent_cigar", "rastrigin", "grie_rosen", "weierstrass", "schaffer_f6"], [0.2, 0.2, 0.2, 0.2, 0.2]),
    # C hf10 calls hgbat_func first (was "happycat" here), and its Schaffer
    # component has the y[0..n) quirk
    20: (["hgbat", "katsuura", "ackley", "rastrigin", "schwefel", "schaffer_f7_c17"],
         [0.1, 0.1, 0.2, 0.2, 0.2, 0.2]),
}

# VERIFIED against the official C reference `cec17_test_func.cpp` (cf01..cf08).
# The C code expresses lambda as a rescale `fit[i] = A*fit[i]/B`, so the lambda
# below is A/B. The technical report's lambda list for Composition Function 6
# is garbled (it does not match any component order); the C code is taken as
# authoritative there, per the project rule of trusting released source over
# prose. Six of these ten rows were wrong before this check -- see
# RESULTS_FINAL.md 8.
_COMPOSITION_SPEC: Dict[int, Tuple[List[str], List[float], List[float], List[float]]] = {
    # cf01: rosenbrock | ellips*(1e4/1e10) | rastrigin
    21: (["rosenbrock", "elliptic", "rastrigin"], [10, 20, 30],
         [1, 1e-6, 1], [0, 100, 200]),
    # cf02: rastrigin | griewank*(1e3/1e2) | schwefel
    22: (["rastrigin", "griewank", "schwefel"], [10, 20, 30], [1, 10, 1], [0, 100, 200]),
    # cf03: rosenbrock | ackley*(1e3/1e2) | schwefel | rastrigin
    23: (["rosenbrock", "ackley", "schwefel", "rastrigin"], [10, 20, 30, 40],
         [1, 10, 1, 1], [0, 100, 200, 300]),
    # cf04: ackley*10 | ellips*1e-6 | griewank*10 | rastrigin
    24: (["ackley", "elliptic", "griewank", "rastrigin"], [10, 20, 30, 40],
         [10, 1e-6, 10, 1], [0, 100, 200, 300]),
    # cf05: rastrigin*(1e4/1e3) | happycat*1 | ackley*10 | discus*1e-6 | rosenbrock
    25: (["rastrigin", "happycat", "ackley", "discus", "rosenbrock"], [10, 20, 30, 40, 50],
         [10, 1, 10, 1e-6, 1], [0, 100, 200, 300, 400]),
    # cf06: escaffer6*(1e4/2e7) | schwefel | griewank*10 | rosenbrock | rastrigin*10
    26: (["schaffer_f6", "schwefel", "griewank", "rosenbrock", "rastrigin"],
         [10, 20, 20, 30, 40], [5e-4, 1, 10, 1, 10], [0, 100, 200, 300, 400]),
    # cf07: SIX components -- hgbat*10 | rastrigin*10 | schwefel*2.5
    #       | bent_cigar*1e-26 | ellips*1e-6 | escaffer6*5e-4
    27: (["hgbat", "rastrigin", "schwefel", "bent_cigar", "elliptic", "schaffer_f6"],
         [10, 20, 30, 40, 50, 60], [10, 10, 2.5, 1e-26, 1e-6, 5e-4],
         [0, 100, 200, 300, 400, 500]),
    # cf08: SIX components -- ackley*10 | griewank*10 | discus*1e-6
    #       | rosenbrock | happycat*1 | escaffer6*5e-4
    28: (["ackley", "griewank", "discus", "rosenbrock", "happycat", "schaffer_f6"],
         [10, 20, 30, 40, 50, 60], [10, 10, 1e-6, 1, 1, 5e-4],
         [0, 100, 200, 300, 400, 500]),
}

# F29 / F30 are compositions of HYBRID functions, not of basic functions.
# C reference:  cf09 -> hf05, hf06, hf07  |  cf10 -> hf05, hf08, hf09
# with delta = {10,30,50} and bias = {0,100,200} in both. The previous table
# built them from basic functions with delta = {10,20,30}, which is a different
# landscape entirely.
_COMPOSITION_HYBRID_SPEC: Dict[int, Tuple[List[int], List[float], List[float], List[float]]] = {
    29: ([15, 16, 17], [10, 30, 50], [1, 1, 1], [0, 100, 200]),
    30: ([15, 18, 19], [10, 30, 50], [1, 1, 1], [0, 100, 200]),
}


def official_cec2017_suite(dim: int, acknowledge_unverified: bool = False) -> List[OfficialSpec]:
    """Build F1..F30 from the official data files.

    Raises `Cec2017DataMissing` if the data is not installed. Requires
    `acknowledge_unverified=True` until `_TABLES_VERIFIED` is set.
    """
    if dim not in OFFICIAL_DIMS:
        raise ValueError(f"dim={dim} is not an official CEC2017 dimension {OFFICIAL_DIMS}")
    if not _TABLES_VERIFIED and not acknowledge_unverified:
        raise RuntimeError(
            "The CEC2017 component tables in this module have not yet been checked "
            "against the official technical report. Verify them and set "
            "_TABLES_VERIFIED = True, or pass acknowledge_unverified=True for a "
            "non-publication smoke test."
        )
    specs: List[OfficialSpec] = []
    for fid, name in _SIMPLE_SPEC.items():
        specs.append(_make_simple(fid, dim, name))
    for fid, (parts, props) in _HYBRID_SPEC.items():
        specs.append(_make_hybrid(fid, dim, parts, props))
    for fid, (parts, sig, lam, bia) in _COMPOSITION_SPEC.items():
        specs.append(_make_composition(fid, dim, parts, sig, lam, bia))
    for fid, (hyb, sig, lam, bia) in _COMPOSITION_HYBRID_SPEC.items():
        specs.append(_make_composition_of_hybrids(fid, dim, hyb, sig, lam, bia))
    specs.sort(key=lambda s: s.id)
    if len(specs) != 30:
        raise Cec2017DataInvalid(
            f"built {len(specs)} functions, the CEC2017 suite has 30")
    return specs
