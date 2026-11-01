#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
cec2011_official.py -- CEC2011 real-world problems, ported line by line from
the organisers' MATLAB code.

Source: github.com/P-N-Suganthan/CEC-2011--Real_World_Problems (Matlab.rar,
Linux-C.rar, Tech-Rep.pdf; SHA-256 in benchmarks/cec2011_reference/README.md).
The "C" package there is only a wrapper around the MATLAB code compiled with
`mcc`, so the MATLAB files ARE the reference implementation.

Selection rule (fixed before any run): every problem that Table 1 of the
technical report lists as bound-constrained and that has D >= 6.

    P1   FM sound-wave parameter estimation      D=6    bench_func(x,1)
    P2   Lennard-Jones potential, 10 atoms        D=30   bench_func(x,2)
    P5   Tersoff potential, Si(B)                 D=30   bench_func(x,5)
    P6   Tersoff potential, Si(C)                 D=30   bench_func(x,6)
    P7   spread-spectrum radar polyphase code     D=20   bench_func(x,7)
    P10  circular antenna array design            D=12   antennafunccircular
    P12  Messenger spacecraft trajectory          D=26   messengerfull / mga_dsm
    P13  Cassini 2 spacecraft trajectory          D=22   cassini2 / mga_dsm

Left out: P3 and P4 (1-D optimal control; their value depends on MATLAB's
adaptive ode45 at RelTol 0.1, which cannot be reproduced exactly), and P8, P9
and P11.x (constrained; they are not bound-constrained problems).

The official code's quirks are reproduced on purpose, as with CEC2017:
  * Tersoff (P5, P6): cos(theta_ijk) uses rd3^3, not rd3^2.
  * Radar (P7): var = 2D-1, so hsum has 4D-2 entries, half of them empty.
  * Antenna (P10): pi is truncated to 3.141592654; the null directions
    [50, 120] are passed as radians although they look like degrees.
  * MGA-DSM (P12, P13): Kepler's elliptic equation is also used on
    hyperbolic legs; iteration limits and tolerances are kept.

Bounds are those of the technical report (Table 1, 2, 3). For Messenger they
differ from the `bounds` field inside messengerfull.mat (that file holds ESA's
original box: t0 in [1900, 2200], Vinf in [2.5, 5.0]); the report and
getlimit_messenger.m say t0 in [1900, 2300], Vinf in [2.5, 4.05], and the
report defines the competition.

Two deliberate, documented departures from running MATLAB itself:
  * Non-finite objective values are replaced by NONFINITE_PENALTY = 1e20
    (death penalty) in `unit_box`. They do occur in the official code: in
    Tersoff, fcr(i,k) = 0 times an overflowing exp() gives 0*Inf = NaN, for
    most random configurations (measured here: 233/300 points for Si(B),
    298/300 for Si(C); a separate scalar transliteration agrees point for
    point). MATLAB's `min` skips NaN and every comparison with NaN is false,
    so a NaN point can never become the best one; a finite penalty keeps
    that ordering, and unlike +inf it keeps the success-weighted parameter
    updates of the SHADE family finite (inf - inf would be NaN).
  * Where MATLAB would produce a complex intermediate from an acos argument
    that rounding pushed past +-1, the argument is clipped (the real part of
    MATLAB's result). crosscheck_cec2011_octave.py measures any remaining
    difference against the official code.

Every function here takes the REAL-space matrix X (n, D) and returns (n,).
`problem(pid)` also returns the official bounds; MDAC_EXPERIMENT maps the
unit box [0,1]^D affinely onto them (uniform sampling in the unit box is
uniform sampling in the official box, as the competition rules require).
"""
from __future__ import annotations

import math
from typing import Callable, Dict, List, Tuple

import numpy as np

Array = np.ndarray

# ===========================================================================
# P1  FM sound waves                         bench_func.m, case 1
# ===========================================================================
_T101 = np.arange(101, dtype=float)
_THETA = 2.0 * math.pi / 100.0


def p01_fm(X: Array) -> Array:
    t = _T101[None, :]
    th = _THETA
    y_t = X[:, 0:1] * np.sin(X[:, 1:2] * t * th + X[:, 2:3] * np.sin(
        X[:, 3:4] * t * th + X[:, 4:5] * np.sin(X[:, 5:6] * t * th)))
    y_0 = 1.0 * np.sin(5.0 * _T101 * th - 1.5 * np.sin(
        4.8 * _T101 * th + 2.0 * np.sin(4.9 * _T101 * th)))
    d = y_t - y_0[None, :]
    return np.sum(d * d, axis=1)


# ===========================================================================
# P2  Lennard-Jones potential                bench_func.m, case 2
# ===========================================================================
def _atoms(X: Array) -> Array:
    n, D = X.shape
    # MATLAB: reshape(x,3,n)' -> atom j = x(3j-2:3j)
    return X.reshape(n, D // 3, 3)


def _dist(P: Array) -> Array:
    diff = P[:, :, None, :] - P[:, None, :, :]
    sq = diff * diff
    return np.sqrt(sq[..., 0] + sq[..., 1] + sq[..., 2])


def p02_lennard_jones(X: Array) -> Array:
    r = _dist(_atoms(X))
    N = r.shape[1]
    iu = np.triu_indices(N, k=1)
    rr = r[:, iu[0], iu[1]]
    with np.errstate(divide="ignore", invalid="ignore", over="ignore"):
        v = 1.0 / rr ** 12 - 2.0 / rr ** 6
    return np.sum(v, axis=1)


# ===========================================================================
# P5, P6  Tersoff potential                  bench_func.m, cases 5 and 6
# ===========================================================================
_TERSOFF = {
    "B": dict(R1=3.0, R2=0.2, A=3.2647e+3, B=9.5373e+1, l1=3.2394, l2=1.3258,
              l3=1.3258, c=4.8381, d=2.0417, n1=22.956, gama=0.33675, h=0.0),
    "C": dict(R1=2.85, R2=0.15, A=1.8308e+3, B=4.7118e+2, l1=2.4799,
              l2=1.7322, l3=1.7322, c=1.0039e+05, d=1.6218e+01,
              n1=7.8734e-01, gama=1.0999e-06, h=-5.9826e-01),
}


def _tersoff(X: Array, model: str) -> Array:
    p = _TERSOFF[model]
    R1, R2, A, B = p["R1"], p["R2"], p["A"], p["B"]
    l1, l2, l3 = p["l1"], p["l2"], p["l3"]
    c, d, n1, gama, h = p["c"], p["d"], p["n1"], p["gama"], p["h"]
    r = _dist(_atoms(X))                                   # (n, N, N)
    n, N, _ = r.shape
    with np.errstate(over="ignore", invalid="ignore", divide="ignore"):
        fcr = np.where(r < (R1 - R2), 1.0,
                       np.where(r > (R1 + R2), 0.0,
                                0.5 - 0.5 * np.sin(math.pi / 2 * (r - R1) / R2)))
        VR = A * np.exp(-l1 * r)
        VA = B * np.exp(-l2 * r)
        # zeta over (i, j, k): rd1 = r_ik, rd2 = r_ij, rd3 = r_kj
        r_ij = r[:, :, :, None]
        r_ik = r[:, :, None, :]
        r_kj = r[:, None, :, :]          # r[k, j] == r[j, k] exactly
        ctheta = (r_ik ** 2 + r_ij ** 2 - r_kj ** 3) / (2.0 * r_ik * r_ij)
        G = 1.0 + (c ** 2) / (d ** 2) - (c ** 2) / (d ** 2 + (h - ctheta) ** 2)
        term = fcr[:, :, None, :] * G * np.exp(l3 ** 3 * (r_ij - r_ik) ** 3)
        idx = np.arange(N)
        keep = (idx[None, None, :] != idx[:, None, None]) & \
               (idx[None, None, :] != idx[None, :, None])     # k != i, k != j
        zeta = np.zeros((n, N, N))
        for k in range(N):                                     # sequential, as MATLAB
            zeta = zeta + np.where(keep[None, :, :, k], term[..., k], 0.0)
        Bij = (1.0 + (gama * zeta) ** n1) ** (-0.5 / n1)
        Eterm = fcr * (VR - Bij * VA) / 2.0
    offdiag = ~np.eye(N, dtype=bool)
    E = np.zeros((n, N))
    for j in range(N):                                         # sequential over j
        E = E + np.where(offdiag[None, :, j], Eterm[:, :, j], 0.0)
    f = np.zeros(n)
    for i in range(N):
        f = f + E[:, i]
    return f


def p05_tersoff_b(X: Array) -> Array:
    return _tersoff(X, "B")


def p06_tersoff_c(X: Array) -> Array:
    return _tersoff(X, "C")


# ===========================================================================
# P7  Spread-spectrum radar polyphase code   bench_func.m, case 7
# ===========================================================================
def p07_radar(X: Array) -> Array:
    n, d = X.shape
    # S[a][j] = x(a) + ... + x(j), accumulated in MATLAB's order (1-based)
    C: Dict[Tuple[int, int], Array] = {}
    for a in range(1, d + 1):
        s = np.zeros(n)
        for j in range(a, d + 1):
            s = X[:, j - 1] + s
            C[(a, j)] = np.cos(s)
    var = 2 * d - 1
    best = np.full(n, -np.inf)
    for kk in range(1, 2 * var + 1):
        hs = np.zeros(n)
        if kk % 2 != 0:
            i = (kk + 1) // 2
            for j in range(i, d + 1):
                hs = C[(abs(2 * i - j - 1) + 1, j)] + hs
        else:
            i = kk // 2
            for j in range(i + 1, d + 1):
                hs = C[(abs(2 * i - j) + 1, j)] + hs
            hs = hs + 0.5
        best = np.maximum(best, hs)
    return best


# ===========================================================================
# P10  Circular antenna array                antennafunccircular.m,
#      array_factorcir.m (called with null=[50,120], phi_desired=180,
#      distance=0.5, as in the official wrapper mCEC_Function.cpp)
# ===========================================================================
_PI_T = 3.141592654                         # the official code's truncated pi
_NUM1 = 300
_PHI_DEG = np.linspace(0.0, 360.0, _NUM1)
_NULL = (50.0, 120.0)
_PHI_DESIRED = 180.0
_DISTANCE = 0.5


def _array_factor(X: Array, phi: Array) -> Array:
    """array_factorcir for every row of X and every angle in `phi` (rad)."""
    dim = X.shape[1]
    half = dim // 2
    phi = np.asarray(phi, dtype=float)[None, :]
    y = np.zeros((X.shape[0], phi.shape[1]))
    y1 = np.zeros_like(y)
    k = _PI_T / 180.0
    pd = _PHI_DESIRED * (_PI_T / 180.0)

    def shi_of(i1):
        delphi = 2 * _PI_T * (i1 - 1) / dim
        return (np.cos(phi - delphi) - math.cos(pd - delphi)) * dim * _DISTANCE

    for i1 in range(1, half + 1):
        y = y + X[:, i1 - 1:i1] * np.cos(shi_of(i1) + X[:, half + i1 - 1:half + i1] * k)
    for i1 in range(half + 1, dim + 1):
        y = y + X[:, i1 - half - 1:i1 - half] * np.cos(shi_of(i1) - X[:, i1 - 1:i1] * k)
    for i1 in range(1, half + 1):
        y1 = y1 + X[:, i1 - 1:i1] * np.sin(shi_of(i1) + X[:, half + i1 - 1:half + i1] * k)
    for i1 in range(half + 1, dim + 1):
        y1 = y1 + X[:, i1 - half - 1:i1 - half] * np.sin(shi_of(i1) - X[:, i1 - 1:i1] * k)
    return np.sqrt(y * y + y1 * y1)


def p10_antenna(X: Array) -> Array:
    n = X.shape[0]
    num1 = _NUM1
    phi = _PHI_DEG
    yax = _array_factor(X, (_PI_T / 180.0) * phi)            # (n, 300)
    phi_ref = np.argmax(yax, axis=1)                          # first maximum
    maxi = yax[np.arange(n), phi_ref]
    phizero = phi[phi_ref]

    # local maxima, exactly as the official loops (strict inequalities)
    lm = np.zeros_like(yax, dtype=bool)
    lm[:, 0] = (yax[:, 0] > yax[:, num1 - 1]) & (yax[:, 0] > yax[:, 1])
    lm[:, num1 - 1] = (yax[:, num1 - 1] > yax[:, 0]) & (yax[:, num1 - 1] > yax[:, num1 - 2])
    lm[:, 1:num1 - 1] = (yax[:, 1:num1 - 1] > yax[:, 2:]) & (yax[:, 1:num1 - 1] > yax[:, :num1 - 2])
    lobes = np.where(lm, yax, -np.inf)
    lobes.sort(axis=1)
    second = lobes[:, -2]
    count = lm.sum(axis=1)

    # strict local minima, used for the beamwidth between first nulls
    lmin = np.zeros_like(yax, dtype=bool)
    lmin[:, 1:num1 - 1] = (yax[:, 1:num1 - 1] < yax[:, :num1 - 2]) & \
                          (yax[:, 1:num1 - 1] < yax[:, 2:])
    steps = np.arange(1, num1 // 2 + 1)                       # i = 1 .. 150
    rows = np.arange(n)[:, None]
    # upper side: stop when (phi_ref + i) > num1 - 1 (1-based) or a local min
    up = phi_ref[:, None] + steps[None, :]                    # 0-based index
    up_valid = up <= num1 - 2                                  # 1-based <= 299
    up_hit = up_valid & lmin[rows, np.minimum(up, num1 - 1)]
    up_stop = ~up_valid | up_hit
    upper = np.full(n, 180.0)
    any_up = up_stop.any(axis=1)
    first_up = np.argmax(up_stop, axis=1)
    use = any_up & up_hit[np.arange(n), first_up]
    upper[use] = phi[up[np.arange(n), first_up][use]] - phi[phi_ref[use]]
    # lower side: stop when (phi_ref - i) < 2 (1-based) or a local min
    lo = phi_ref[:, None] - steps[None, :]
    lo_valid = lo >= 1                                         # 1-based >= 2
    lo_hit = lo_valid & lmin[rows, np.maximum(lo, 0)]
    lo_stop = ~lo_valid | lo_hit
    lower = np.full(n, 180.0)
    any_lo = lo_stop.any(axis=1)
    first_lo = np.argmax(lo_stop, axis=1)
    use = any_lo & lo_hit[np.arange(n), first_lo]
    lower[use] = phi[phi_ref[use]] - phi[lo[np.arange(n), first_lo][use]]
    bwfn = upper + lower

    with np.errstate(divide="ignore", invalid="ignore"):
        sllreturn = 20.0 * np.log10(second / maxi)
        af_null = _array_factor(X, np.array(_NULL))
        y1 = np.zeros(n)
        for i in range(len(_NULL)):
            y1 = y1 + af_null[:, i] / maxi
    y3 = np.abs(phizero - _PHI_DESIRED)
    y3 = np.where(y3 < 5, 0.0, y3)
    y = np.where(bwfn > 80, np.abs(bwfn - 80), 0.0)
    f = sllreturn + y + y1 + y3
    # fewer than two lobes: MATLAB's sidelobes(2) would throw -> no value
    return np.where(count >= 2, f, np.nan)


# ===========================================================================
# P12, P13  MGA-1DSM trajectories             mga_dsm.m (ESA/ACT)
# ===========================================================================
_MU_PL = (22321.0, 324860.0, 398601.19, 42828.3, 126.7e6, 37.93951970883e6)
_MU_SUN = 1.32712428e+11
_RPL = (2440.0, 6052.0, 6378.0, 3397.0, 71492.0, 60330.0)
_PI = math.pi


def _vett(a, b):
    return (a[1] * b[2] - a[2] * b[1],
            a[2] * b[0] - a[0] * b[2],
            a[0] * b[1] - a[1] * b[0])


def _dot(a, b):
    return a[0] * b[0] + a[1] * b[1] + a[2] * b[2]


def _norm(a):
    return math.sqrt(a[0] * a[0] + a[1] * a[1] + a[2] * a[2])


def _scale(s, a):
    return (s * a[0], s * a[1], s * a[2])


def _div(a, s):
    return (a[0] / s, a[1] / s, a[2] / s)


def _add(a, b):
    return (a[0] + b[0], a[1] + b[1], a[2] + b[2])


def _sub(a, b):
    return (a[0] - b[0], a[1] - b[1], a[2] - b[2])


def _acos(z):
    # real(acos(z)) as MATLAB would give it for |z| slightly above 1
    if z > 1.0:
        return 0.0
    if z < -1.0:
        return _PI
    return math.acos(z)


def _mod(x, y):
    return x - math.floor(x / y) * y


def _M2E(M, e):
    i = 0
    tol = 1e-10
    err = 1.0
    E = M + e * math.cos(M)
    while err > tol and i < 100:
        i += 1
        Enew = E - (E - e * math.sin(E) - M) / (1 - e * math.cos(E))
        err = abs(E - Enew)
        E = Enew
    return E


def _E2M(E, e):
    if e < 1:
        return E - e * math.sin(E)
    return e * math.tan(E) - math.log(math.tan(E / 2 + _PI / 4))


def _ni2E(ni, e):
    if e < 1:
        return 2 * math.atan(math.sqrt((1 - e) / (1 + e)) * math.tan(ni / 2))
    return 2 * math.atan(math.sqrt((e - 1) / (e + 1)) * math.tan(ni / 2))


def _rotmat(omg, omp, i):
    return ((math.cos(omg) * math.cos(omp) - math.sin(omg) * math.sin(omp) * math.cos(i),
             -math.cos(omg) * math.sin(omp) - math.sin(omg) * math.cos(omp) * math.cos(i),
             math.sin(omg) * math.sin(i)),
            (math.sin(omg) * math.cos(omp) + math.cos(omg) * math.sin(omp) * math.cos(i),
             -math.sin(omg) * math.sin(omp) + math.cos(omg) * math.cos(omp) * math.cos(i),
             -math.cos(omg) * math.sin(i)),
            (math.sin(omp) * math.sin(i),
             math.cos(omp) * math.sin(i),
             math.cos(i)))


def _matvec(R, x, y):
    # R * [x; y; 0]
    return (R[0][0] * x + R[0][1] * y + R[0][2] * 0.0,
            R[1][0] * x + R[1][1] * y + R[1][2] * 0.0,
            R[2][0] * x + R[2][1] * y + R[2][2] * 0.0)


def _conversion(E):
    muSUN = 1.327124280000000e+011
    a, e, i, omg, omp, EA = E
    b = a * math.sqrt(1 - e ** 2)
    n = math.sqrt(muSUN / a ** 3)
    xper = a * (math.cos(EA) - e)
    yper = b * math.sin(EA)
    xdotper = -(a * n * math.sin(EA)) / (1 - e * math.cos(EA))
    ydotper = (b * n * math.cos(EA)) / (1 - e * math.cos(EA))
    R = _rotmat(omg, omp, i)
    return _matvec(R, xper, yper), _matvec(R, xdotper, ydotper)


def _pleph_an(mjd2000, planet):
    RAD = _PI / 180
    KM = 1.495978706910000e+008
    T = (mjd2000 + 36525.00) / 36525.00
    TT = T * T
    TTT = T * TT
    if planet == 1:
        E1 = 0.38709860
        E2 = 0.205614210 + 0.000020460 * T - 0.000000030 * TT
        E3 = 7.002880555555555560 + 1.86083333333333333e-3 * T - 1.83333333333333333e-5 * TT
        E4 = 4.71459444444444444e+1 + 1.185208333333333330 * T + 1.73888888888888889e-4 * TT
        E5 = 2.87537527777777778e+1 + 3.70280555555555556e-1 * T + 1.20833333333333333e-4 * TT
        XM = 1.49472515288888889e+5 + 6.38888888888888889e-6 * T
        E6 = 1.02279380555555556e2 + XM * T
    elif planet == 2:
        E1 = 0.72333160
        E2 = 0.006820690 - 0.000047740 * T + 0.0000000910 * TT
        E3 = 3.393630555555555560 + 1.00583333333333333e-3 * T - 9.72222222222222222e-7 * TT
        E4 = 7.57796472222222222e+1 + 8.9985e-1 * T + 4.1e-4 * TT
        E5 = 5.43841861111111111e+1 + 5.08186111111111111e-1 * T - 1.38638888888888889e-3 * TT
        XM = 5.8517803875e+4 + 1.28605555555555556e-3 * T
        E6 = 2.12603219444444444e2 + XM * T
    elif planet == 3:
        E1 = 1.000000230
        E2 = 0.016751040 - 0.000041800 * T - 0.0000001260 * TT
        E3 = 0.00
        E4 = 0.00
        E5 = 1.01220833333333333e+2 + 1.7191750 * T + 4.52777777777777778e-4 * TT + 3.33333333333333333e-6 * TTT
        XM = 3.599904975e+4 - 1.50277777777777778e-4 * T - 3.33333333333333333e-6 * TT
        E6 = 3.58475844444444444e2 + XM * T
    elif planet == 4:
        E1 = 1.5236883990
        E2 = 0.093312900 + 0.0000920640 * T - 0.0000000770 * TT
        E3 = 1.850333333333333330 - 6.75e-4 * T + 1.26111111111111111e-5 * TT
        E4 = 4.87864416666666667e+1 + 7.70991666666666667e-1 * T - 1.38888888888888889e-6 * TT - 5.33333333333333333e-6 * TTT
        E5 = 2.85431761111111111e+2 + 1.069766666666666670 * T + 1.3125e-4 * TT + 4.13888888888888889e-6 * TTT
        XM = 1.91398585e+4 + 1.80805555555555556e-4 * T + 1.19444444444444444e-6 * TT
        E6 = 3.19529425e2 + XM * T
    elif planet == 5:
        E1 = 5.2025610
        E2 = 0.048334750 + 0.000164180 * T - 0.00000046760 * TT - 0.00000000170 * TTT
        E3 = 1.308736111111111110 - 5.69611111111111111e-3 * T + 3.88888888888888889e-6 * TT
        E4 = 9.94433861111111111e+1 + 1.010530 * T + 3.52222222222222222e-4 * TT - 8.51111111111111111e-6 * TTT
        E5 = 2.73277541666666667e+2 + 5.99431666666666667e-1 * T + 7.0405e-4 * TT + 5.07777777777777778e-6 * TTT
        XM = 3.03469202388888889e+3 - 7.21588888888888889e-4 * T + 1.78444444444444444e-6 * TT
        E6 = 2.25328327777777778e2 + XM * T
    elif planet == 6:
        E1 = 9.5547470
        E2 = 0.055892320 - 0.00034550 * T - 0.0000007280 * TT + 0.000000000740 * TTT
        E3 = 2.492519444444444440 - 3.91888888888888889e-3 * T - 1.54888888888888889e-5 * TT + 4.44444444444444444e-8 * TTT
        E4 = 1.12790388888888889e+2 + 8.73195138888888889e-1 * T - 1.52180555555555556e-4 * TT - 5.30555555555555556e-6 * TTT
        E5 = 3.38307772222222222e+2 + 1.085220694444444440 * T + 9.78541666666666667e-4 * TT + 9.91666666666666667e-6 * TTT
        XM = 1.22155146777777778e+3 - 5.01819444444444444e-4 * T - 5.19444444444444444e-6 * TT
        E6 = 1.75466216666666667e2 + XM * T
    else:
        raise ValueError(f"planet {planet} is not used by P12/P13")
    E1 = E1 * KM
    E3, E4, E5, E6 = E3 * RAD, E4 * RAD, E5 * RAD, E6 * RAD
    E6 = _mod(E6, 2 * _PI)
    E6 = _M2E(E6, E2)
    return _conversion((E1, E2, E3, E4, E5, E6))


def _IC2par(r0, v0, mu):
    k = (0.0, 0.0, 1.0)
    h = _vett(r0, v0)
    p = _dot(h, h) / mu
    n = _vett(k, h)
    n = _div(n, _norm(n))
    R0 = _norm(r0)
    evett = _sub(_div(_vett(v0, h), mu), _div(r0, R0))
    e = _dot(evett, evett)
    E1 = p / (1 - e)
    E2 = math.sqrt(e)
    e = E2
    E3 = _acos(h[2] / _norm(h))
    E5 = _acos(_dot(n, evett) / e)
    if evett[2] < 0:
        E5 = 2 * _PI - E5
    E4 = _acos(n[0])
    if n[1] < 0:
        E4 = 2 * _PI - E4
    ni = _acos(_dot(evett, r0) / e / R0)
    if _dot(r0, v0) < 0:
        ni = 2 * _PI - ni
    E6 = _ni2E(ni, e)
    return [E1, E2, E3, E4, E5, E6]


def _par2IC(E, mu):
    a, e, i, omg, omp, EA = E
    if e < 1:
        b = a * math.sqrt(1 - e ** 2)
        n = math.sqrt(mu / a ** 3)
        xper = a * (math.cos(EA) - e)
        yper = b * math.sin(EA)
        xdotper = -(a * n * math.sin(EA)) / (1 - e * math.cos(EA))
        ydotper = (b * n * math.cos(EA)) / (1 - e * math.cos(EA))
    else:
        b = -a * math.sqrt(e ** 2 - 1)
        n = math.sqrt(-mu / a ** 3)
        tq = math.tan(1 / 2 * EA + 1 / 4 * _PI)
        dNdzeta = e * (1 + math.tan(EA) ** 2) - (1 / 2 + 1 / 2 * tq ** 2) / tq
        xper = a / math.cos(EA) - a * e
        yper = b * math.tan(EA)
        xdotper = a * math.tan(EA) / math.cos(EA) * n / dNdzeta
        ydotper = b / math.cos(EA) ** 2 * n / dNdzeta
    R = _rotmat(omg, omp, i)
    return _matvec(R, xper, yper), _matvec(R, xdotper, ydotper)


def _propagateKEP(r0, v0, t, mu):
    h = _vett(r0, v0)
    ih = _div(h, _norm(h))
    rot = abs(abs(ih[2]) - 1) < 1e-3
    if rot:                                   # DD = [1 0 0; 0 0 1; 0 -1 0]
        r0 = (r0[0], r0[2], -r0[1])
        v0 = (v0[0], v0[2], -v0[1])
    E = _IC2par(r0, v0, mu)
    M0 = _E2M(E[5], E[1])
    if E[1] < 1:
        M = M0 + math.sqrt(mu / E[0] ** 3) * t
    else:
        M = M0 + math.sqrt(-mu / E[0] ** 3) * t
    E[5] = _M2E(M, E[1])
    r, v = _par2IC(E, mu)
    if rot:                                   # DD' = [1 0 0; 0 0 -1; 0 1 0]
        r = (r[0], -r[2], r[1])
        v = (v[0], -v[2], v[1])
    return r, v


def _x2tof(x, s, c, lw, N):
    am = s / 2
    a = am / (1 - x ** 2)
    if x < 1:
        beta = 2 * math.asin(math.sqrt((s - c) / 2 / a))
        if lw:
            beta = -beta
        alfa = 2 * math.acos(x)
    else:
        alfa = 2 * math.acosh(x)
        beta = 2 * math.asinh(math.sqrt((s - c) / (-2 * a)))
        if lw:
            beta = -beta
    return _tofabn(a, alfa, beta, N)


def _tofabn(sigma, alfa, beta, N):
    if sigma > 0:
        return sigma * math.sqrt(sigma) * ((alfa - math.sin(alfa)) - (beta - math.sin(beta)) + N * 2 * _PI)
    return -sigma * math.sqrt(-sigma) * ((math.sinh(alfa) - alfa) - (math.sinh(beta) - beta))


def _lambertI(r1, r2, t, mu, lw):
    N = 0
    if t <= 0:
        nan = (math.nan, math.nan, math.nan)
        return nan, nan
    tol = 1e-11
    R = _norm(r1)
    V = math.sqrt(mu / R)
    T = R / V
    r1 = _div(r1, R)
    r2 = _div(r2, R)
    t = t / T
    r2mod = _norm(r2)
    theta = _acos(_dot(r1, r2) / r2mod)
    if lw:
        theta = 2 * _PI - theta
    c = math.sqrt(1 + r2mod ** 2 - 2 * r2mod * math.cos(theta))
    s = (1 + r2mod + c) / 2
    am = s / 2
    lam = math.sqrt(r2mod) * math.cos(theta / 2) / s
    inn1 = -.5233
    inn2 = .5233
    x1 = math.log(1 + inn1)
    x2 = math.log(1 + inn2)
    lt = math.log(t)
    y1 = math.log(_x2tof(inn1, s, c, lw, N)) - lt
    y2 = math.log(_x2tof(inn2, s, c, lw, N)) - lt
    err = 1.0
    it = 0
    xnew = math.nan
    while err > tol and y1 != y2:
        it += 1
        if it > 10000:                        # MATLAB has no cap; never reached in practice
            break
        xnew = (x1 * y2 - y1 * x2) / (y2 - y1)
        ynew = math.log(_x2tof(math.exp(xnew) - 1, s, c, lw, N)) - lt
        x1 = x2
        y1 = y2
        x2 = xnew
        y2 = ynew
        err = abs(x1 - xnew)
    x = math.exp(xnew) - 1
    a = am / (1 - x ** 2)
    if x < 1:
        beta = 2 * math.asin(math.sqrt((s - c) / 2 / a))
        if lw:
            beta = -beta
        alfa = 2 * math.acos(x)
        psi = (alfa - beta) / 2
        eta2 = 2 * a * math.sin(psi) ** 2 / s
        eta = math.sqrt(eta2)
    else:
        beta = 2 * math.asinh(math.sqrt((c - s) / 2 / a))
        if lw:
            beta = -beta
        alfa = 2 * math.acosh(x)
        psi = (alfa - beta) / 2
        eta2 = -2 * a * math.sinh(psi) ** 2 / s
        eta = math.sqrt(eta2)
    p = r2mod / am / eta2 * math.sin(theta / 2) ** 2
    sigma1 = 1 / eta / math.sqrt(am) * (2 * lam * am - (lam + x * eta))
    w = _vett(r1, r2)
    ih = _div(w, math.sqrt(_dot(w, w)))
    if lw:
        ih = _scale(-1.0, ih)
    vr1 = sigma1
    vt1 = math.sqrt(p)
    v1 = _add(_scale(vr1, r1), _scale(vt1, _vett(ih, r1)))
    vt2 = vt1 / r2mod
    vr2 = -vr1 + (vt1 - vt2) / math.tan(theta / 2)
    r2u = _div(r2, r2mod)
    v2 = _add(_div(_scale(vr2, r2), r2mod), _scale(vt2, _vett(ih, r2u)))
    return _scale(V, v1), _scale(V, v2)


def _lw_flag(a, b):
    z = _vett(a, b)[2]
    sg = 1 if z > 0 else (-1 if z < 0 else 0)
    return 0 if sg == 1 else 1


_MGA = {
    # sequence, objective type, rp_target, e_target  (from the official .mat)
    12: ((3, 2, 2, 1, 1, 1, 1), "orbit insertion", 2640.0, 0.704),
    13: ((3, 2, 2, 3, 5, 6), "total DV rndv", None, None),
}


def mga_dsm(t, pid: int) -> float:
    seq, otype, rp_target, e_target = _MGA[pid]
    T1 = lambda k: float(t[k - 1])            # MATLAB 1-based access
    tdep, VINF, udir, vdir = T1(1), T1(2), T1(3), T1(4)
    N = len(seq)
    tof = [T1(i + 4) for i in range(1, N)]
    alpha = [T1(N + i + 3) for i in range(1, N)]
    rp_non_dim = [T1(i + 2 * N + 2) for i in range(1, N - 1)]
    gamma = [T1(3 * N + i) for i in range(1, N - 1)]

    r, v, muvec = [], [], []
    dT = tof + [0.0]
    T = tdep
    for i in range(N):
        ri, vi = _pleph_an(T, seq[i])
        r.append(ri)
        v.append(vi)
        muvec.append(_MU_PL[seq[i] - 1])
        T = T + dT[i]
    rp = [rp_non_dim[i] * _RPL[seq[i + 1] - 1] for i in range(N - 2)]

    # first block
    vtemp = _vett(r[0], v[0])
    iP1 = _div(v[0], _norm(v[0]))
    zP1 = _div(vtemp, _norm(vtemp))
    jP1 = _vett(zP1, iP1)
    theta = 2 * _PI * udir
    phi = math.acos(2 * vdir - 1) - _PI / 2
    inner = _add(_add(_scale(math.cos(theta) * math.cos(phi), iP1),
                      _scale(math.sin(theta) * math.cos(phi), jP1)),
                 _scale(math.sin(phi), zP1))
    vinf = _scale(VINF, inner)
    v_pl_in = [None] * N
    v_pl_in[0] = v[0]
    v_out = _add(v[0], vinf)
    tDSM = alpha[0] * tof[0]
    rd, v_dsm_in = _propagateKEP(r[0], v_out, tDSM * 24 * 60 * 60, _MU_SUN)
    lw = _lw_flag(rd, r[1])
    v_dsm_out, v_pl_in[1] = _lambertI(rd, r[1], tof[0] * (1 - alpha[0]) * 24 * 60 * 60, _MU_SUN, lw)
    DV = [0.0] * N
    DV[0] = _norm(_sub(v_dsm_out, v_dsm_in))

    # intermediate blocks
    for i in range(N - 2):
        v_rel_in = _sub(v_pl_in[i + 1], v[i + 1])
        s = rp[i] / muvec[i + 1]
        e = 1 + ((s * v_rel_in[0]) * v_rel_in[0] + (s * v_rel_in[1]) * v_rel_in[1]
                 + (s * v_rel_in[2]) * v_rel_in[2])
        beta_rot = 2 * math.asin(1 / e)
        ix = _div(v_rel_in, _norm(v_rel_in))
        iy = _vett(ix, _div(v[i + 1], _norm(v[i + 1])))
        iy = _div(iy, _norm(iy))
        iz = _vett(ix, iy)
        iVout = _add(_add(_scale(math.cos(beta_rot), ix),
                          _scale(math.cos(gamma[i]) * math.sin(beta_rot), iy)),
                     _scale(math.sin(gamma[i]) * math.sin(beta_rot), iz))
        v_rel_out = _scale(_norm(v_rel_in), iVout)
        v_out = _add(v[i + 1], v_rel_out)
        tDSM = alpha[i + 1] * tof[i + 1]
        rd, v_dsm_in = _propagateKEP(r[i + 1], v_out, tDSM * 24 * 60 * 60, _MU_SUN)
        lw = _lw_flag(rd, r[i + 2])
        v_dsm_out, v_pl_in[i + 2] = _lambertI(
            rd, r[i + 2], tof[i + 1] * (1 - alpha[i + 1]) * 24 * 60 * 60, _MU_SUN, lw)
        DV[i + 1] = _norm(_sub(v_dsm_out, v_dsm_in))

    # final block
    DVrel = _norm(_sub(v[N - 1], v_pl_in[N - 1]))
    if otype == "orbit insertion":
        DVper = math.sqrt(DVrel ** 2 + 2 * muvec[N - 1] / rp_target)
        DVper2 = math.sqrt(2 * muvec[N - 1] / rp_target - muvec[N - 1] / rp_target * (1 - e_target))
        DVarr = abs(DVper - DVper2)
    else:
        DVarr = DVrel
    DV[N - 1] = DVarr
    DVtot = 0.0
    for dv in DV:
        DVtot = DVtot + dv
    if otype == "total DV rndv":
        return DVtot + VINF
    return DVtot


def _mga_batch(pid: int) -> Callable[[Array], Array]:
    def f(X: Array) -> Array:
        out = np.empty(X.shape[0])
        for k in range(X.shape[0]):
            try:
                out[k] = mga_dsm(X[k], pid)
            except (ValueError, ZeroDivisionError, OverflowError):
                out[k] = np.nan        # MATLAB would go complex/NaN here
        return out
    return f


def p12_messenger(X: Array) -> Array:
    return _mga_batch(12)(X)


def p13_cassini2(X: Array) -> Array:
    return _mga_batch(13)(X)


# ===========================================================================
# Bounds (technical report, Table 1-3) and the problem table
# ===========================================================================
def _molecule_bounds(D: int) -> Tuple[Array, Array]:
    """P2, P5, P6: x1 in [0,4], x2 in [0,4], x3 in [0,pi], and for i >= 4
    x_i in [-4 - floor((i-4)/3)/4, 4 + floor((i-4)/3)/4]."""
    lb = np.empty(D)
    ub = np.empty(D)
    lb[0], ub[0] = 0.0, 4.0
    lb[1], ub[1] = 0.0, 4.0
    lb[2], ub[2] = 0.0, math.pi
    for i in range(4, D + 1):
        w = 4.0 + 0.25 * math.floor((i - 4) / 3)
        lb[i - 1], ub[i - 1] = -w, w
    return lb, ub


_MESSENGER_LB = np.array([1900, 2.5, 0, 0, 100, 100, 100, 100, 100, 100,
                          0.01, 0.01, 0.01, 0.01, 0.01, 0.01,
                          1.1, 1.1, 1.05, 1.05, 1.05] + [-math.pi] * 5, float)
_MESSENGER_UB = np.array([2300, 4.05, 1, 1, 500, 500, 500, 500, 500, 600,
                          0.99, 0.99, 0.99, 0.99, 0.99, 0.99,
                          6, 6, 6, 6, 6] + [math.pi] * 5, float)
_CASSINI_LB = np.array([-1000, 3, 0, 0, 100, 100, 30, 400, 800,
                        0.01, 0.01, 0.01, 0.01, 0.01,
                        1.05, 1.05, 1.15, 1.7] + [-math.pi] * 4, float)
_CASSINI_UB = np.array([0, 5, 1, 1, 400, 500, 300, 1600, 2200,
                        0.9, 0.9, 0.9, 0.9, 0.9,
                        6, 6, 6.5, 291] + [math.pi] * 4, float)

# pid: (short name, D, func, (lb, ub))
PROBLEMS: Dict[int, Tuple[str, int, Callable[[Array], Array], Tuple[Array, Array]]] = {
    1: ("fm_sound_waves", 6, p01_fm, (np.full(6, -6.4), np.full(6, 6.35))),
    2: ("lennard_jones", 30, p02_lennard_jones, _molecule_bounds(30)),
    5: ("tersoff_si_b", 30, p05_tersoff_b, _molecule_bounds(30)),
    6: ("tersoff_si_c", 30, p06_tersoff_c, _molecule_bounds(30)),
    7: ("radar_polyphase", 20, p07_radar, (np.zeros(20), np.full(20, 2 * math.pi))),
    10: ("circular_antenna", 12, p10_antenna,
         (np.array([0.2] * 6 + [-180.0] * 6), np.array([1.0] * 6 + [180.0] * 6))),
    12: ("messenger", 26, p12_messenger, (_MESSENGER_LB, _MESSENGER_UB)),
    13: ("cassini2", 22, p13_cassini2, (_CASSINI_LB, _CASSINI_UB)),
}


def problem(pid: int):
    """(name, D, real-space function, (lb, ub)) of one CEC2011 problem."""
    return PROBLEMS[pid]


NONFINITE_PENALTY = 1e20


def unit_box(pid: int) -> Callable[[Array], Array]:
    """The problem on [0,1]^D: x = lb + u (ub - lb); non-finite -> 1e20."""
    _, D, func, (lb, ub) = PROBLEMS[pid]
    span = ub - lb

    def f(U: Array) -> Array:
        U = np.atleast_2d(U)
        val = func(lb[None, :] + U * span[None, :])
        return np.where(np.isfinite(val), val, NONFINITE_PENALTY)
    return f
