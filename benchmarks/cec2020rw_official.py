#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
cec2020rw_official.py -- CEC2020 real-world constrained suite, mechanical
engineering problems RC15-RC33 (except RC24), ported line by line from the
organisers' MATLAB code `cec20_func.m` and `Cal_par.m`.

Source: github.com/P-N-Suganthan/2020-RW-Constrained-Optimisation, SOFTWARE.zip
(SHA-256 in benchmarks/cec2020rw_reference/README.md). Protocol, including why
RC24 is dropped: PROTOCOL_CEC2020_MECH.md.

`evaluate(pid, X)` returns (f, G, H) exactly as the official function does:
f (n,), G (n, number of inequality columns the official code returns),
H (n, number of declared equality constraints). Quirks of the official code
are kept (e.g. RC21 returns 8 inequality columns although Cal_par says 7;
RC30's wire-diameter table contains 0.0105; RC26 g79/g80 use Np2/Np3).

MATLAB semantics reproduced where they matter:
  * round() rounds half away from zero (numpy rounds half to even);
  * max(..., [], 2) ignores NaN;
  * RC22: `if beta == real(beta)` is applied per point (ps = 1), so a
    point's value never depends on the other points of a batch.

`unit_box(pid)` gives the scalar fitness used by every algorithm
(protocol: feasible -> f, infeasible -> C_k + mean violation, non-finite ->
1e20), on [0,1]^D mapped affinely onto the official bounds.
"""
from __future__ import annotations

import math
from typing import Callable, Dict, Tuple

import numpy as np

Array = np.ndarray
PI = math.pi
NONFINITE_PENALTY = 1e20
EQ_TOL = 1e-4


def mround(x):
    """MATLAB round: half away from zero."""
    x = np.asarray(x, dtype=float)
    return np.sign(x) * np.floor(np.abs(x) + 0.5)


def _cols(*cs):
    return np.column_stack([np.asarray(c, dtype=float) for c in cs])


# --------------------------------------------------------------------------- RC15
def rc15(x):
    x1, x2, x3, x4, x5, x6, x7 = (x[:, i] for i in range(7))
    f = (0.7854 * x1 * x2 ** 2 * (3.3333 * x3 ** 2 + 14.9334 * x3 - 43.0934)
         - 1.508 * x1 * (x6 ** 2 + x7 ** 2)
         + 7.477 * (x6 ** 3 + x7 ** 3) + 0.7854 * (x4 * x6 ** 2 + x5 * x7 ** 2))
    g = _cols(-x1 * x2 ** 2 * x3 + 27,
              -x1 * x2 ** 2 * x3 ** 2 + 397.5,
              -x2 * x6 ** 4 * x3 * x4 ** (-3) + 1.93,
              -x2 * x7 ** 4 * x3 / x5 ** 3 + 1.93,
              10 * x6 ** (-3) * np.sqrt(16.91 * 10 ** 6 + (745 * x4 / (x2 * x3)) ** 2) - 1100,
              10 * x7 ** (-3) * np.sqrt(157.5 * 10 ** 6 + (745 * x5 / (x2 * x3)) ** 2) - 850,
              x2 * x3 - 40,
              -x1 / x2 + 5,
              x1 / x2 - 12,
              1.5 * x6 - x4 + 1.9,
              1.1 * x7 - x5 + 1.9)
    return f, g, None


# --------------------------------------------------------------------------- RC16
def rc16(x):
    x1, x2, x3, x4, x5, x6, x7, x8, x9, x10, x11, x12, x13, x14 = (x[:, i] for i in range(14))
    f = (63098.88 * x2 * x4 * x12 + 5441.5 * x2 ** 2 * x12 + 115055.5 * x2 ** 1.664 * x6
         + 6172.27 * x2 ** 2 * x6 + 63098.88 * x1 * x3 * x11 + 5441.5 * x1 ** 2 * x11
         + 115055.5 * x1 ** 1.664 * x5 + 6172.27 * x1 ** 2 * x5 + 140.53 * x1 * x11 + 281.29 * x3 * x11
         + 70.26 * x1 ** 2 + 281.29 * x1 * x3 + 281.29 * x3 ** 2 + 14437 * x8 ** 1.8812 * x12 ** 0.3424
         * x10 * x14 ** (-1) * x1 ** 2 * x7 * x9 ** (-1) + 20470.2 * x7 ** (2.893) * x11 ** 0.316 * x1 ** 2)
    g = _cols(1.524 * x7 ** (-1) - 1,
              1.524 * x8 ** (-1) - 1,
              0.07789 * x1 - 2 * x7 ** (-1) * x9 - 1,
              7.05305 * x9 ** (-1) * x1 ** 2 * x10 * x8 ** (-1) * x2 ** (-1) * x14 ** (-1) - 1,
              0.0833 / x13 * x14 - 1,
              0.04771 * x10 * x8 ** 1.8812 * x12 ** 0.3424 - 1,
              0.0488 * x9 * x7 ** 1.893 * x11 ** 0.316 - 1,
              0.0099 * x1 / x3 - 1,
              0.0193 * x2 / x4 - 1,
              0.0298 * x1 / x5 - 1,
              47.136 * x2 ** 0.333 / x10 * x12 - 1.333 * x8 * x13 ** 2.1195
              + 62.08 * x13 ** 2.1195 * x8 ** 0.2 / (x12 * x10) - 1,
              0.056 * x2 / x6 - 1,
              2 / x9 - 1,
              2 / x10 - 1,
              x12 / x11 - 1)
    return f, g, None


# --------------------------------------------------------------------------- RC17
def rc17(x):
    x1, x2, x3 = x[:, 0], x[:, 1], x[:, 2]
    f = x1 ** 2 * x2 * (x3 + 2)
    g = _cols(1 - (x2 ** 3 * x3) / (71785 * x1 ** 4),
              (4 * x2 ** 2 - x1 * x2) / (12566 * (x2 * x1 ** 3 - x1 ** 4)) + 1 / (5108 * x1 ** 2) - 1,
              1 - 140.45 * x1 / (x2 ** 2 * x3),
              (x1 + x2) / 1.5 - 1)
    return f, g, None


# --------------------------------------------------------------------------- RC18
def rc18(x):
    x = x.copy()
    x[:, 0] = 0.0625 * mround(x[:, 0])
    x[:, 1] = 0.0625 * mround(x[:, 1])
    x1, x2, x3, x4 = (x[:, i] for i in range(4))
    f = (0.6224 * x1 * x3 * x4 + 1.7781 * x2 * x3 ** 2
         + 3.1661 * x1 ** 2 * x4 + 19.84 * x1 ** 2 * x3)
    g = _cols(-x1 + 0.0193 * x3,
              -x2 + 0.00954 * x3,
              -PI * x3 ** 2 * x4 - 4 / 3 * PI * x3 ** 3 + 1296000,
              x4 - 240)
    return f, g, None


# --------------------------------------------------------------------------- RC19
def rc19(x):
    x1, x2, x3, x4 = (x[:, i] for i in range(4))
    f = 1.10471 * x1 ** 2 * x2 + 0.04811 * x3 * x4 * (14 + x2)
    P = 6000; L = 14; delta_max = 0.25; E = 30 * 1e6; G = 12 * 1e6
    T_max = 13600; sigma_max = 30000
    Pc = 4.013 * E * np.sqrt(x3 ** 2 * x4 ** 6 / 30) / L ** 2 * (1 - x3 / (2 * L) * math.sqrt(E / (4 * G)))
    sigma = 6 * P * L / (x4 * x3 ** 2)
    delta = 6 * P * L ** 3 / (E * x3 ** 2 * x4)
    J = 2 * (math.sqrt(2) * x1 * x2 * (x2 ** 2 / 4 + (x1 + x3) ** 2 / 4))
    R = np.sqrt(x2 ** 2 / 4 + (x1 + x3) ** 2 / 4)
    M = P * (L + x2 / 2)
    ttt = M * R / J
    tt = P / (math.sqrt(2) * x1 * x2)
    t = np.sqrt(tt ** 2 + 2 * tt * ttt * x2 / (2 * R) + ttt ** 2)
    g = _cols(t - T_max, sigma - sigma_max, x1 - x4, delta - delta_max, P - Pc)
    return f, g, None


# --------------------------------------------------------------------------- RC20
def rc20(x):
    x1, x2 = x[:, 0], x[:, 1]
    s2 = math.sqrt(2)
    f = (2 * s2 * x1 + x2) * 100
    g = _cols((s2 * x1 + x2) / (s2 * x1 ** 2 + 2 * x1 * x2) * 2 - 2,
              x2 / (s2 * x1 ** 2 + 2 * x1 * x2) * 2 - 2,
              1 / (s2 * x2 + x1) * 2 - 2)
    return f, g, None


# --------------------------------------------------------------------------- RC21
def rc21(x):
    x1, x2, x3, x4, x5 = (x[:, i] for i in range(5))
    Mf = 3; Ms = 40; Iz = 55; n = 250; Tmax = 15; s = 1.5; delta = 0.5
    Vsrmax = 10; rho = 0.0000078; pmax = 1; mu = 0.6; Lmax = 30; delR = 20
    Rsr = 2 / 3 * (x2 ** 3 - x1 ** 3) / (x2 ** 2 * x1 ** 2)
    Vsr = PI * Rsr * n / 30
    A = PI * (x2 ** 2 - x1 ** 2)
    Prz = x4 / A
    w = PI * n / 30
    Mh = 2 / 3 * mu * x4 * x5 * (x2 ** 3 - x1 ** 3) / (x2 ** 2 - x1 ** 2)
    T = Iz * w / (Mh + Mf)
    f = PI * (x2 ** 2 - x1 ** 2) * x3 * (x5 + 1) * rho
    g = _cols(-x2 + x1 + delR,
              (x5 + 1) * (x3 + delta) - Lmax,
              Prz - pmax,
              Prz * Vsr - pmax * Vsrmax,
              Vsr - Vsrmax,
              T - Tmax,
              s * Ms - Mh,
              -T)
    return f, g, None


# --------------------------------------------------------------------------- RC22
_PIND = np.array([3.0, 4.0, 5.0])
_MIND = np.array([1.75, 2, 2.25, 2.5, 2.75, 3.0])


def rc22(x):
    x = mround(np.abs(x))
    N1, N2, N3, N4, N5, N6 = (x[:, i] for i in range(6))
    p = _PIND[x[:, 6].astype(int) - 1]
    m1 = _MIND[x[:, 7].astype(int) - 1]
    m2 = _MIND[x[:, 8].astype(int) - 1]
    i1 = N6 / N4; i01 = 3.11
    i2 = N6 * (N1 * N3 + N2 * N4) / (N1 * N3 * (N6 - N4)); i02 = 1.84
    iR = -(N2 * N6 / (N1 * N3)); i0R = -3.11
    f = np.fmax(np.fmax(i1 - i01, i2 - i02), iR - i0R)       # MATLAB max skips NaN
    Dmax = 220; dlt22 = 0.5; dlt33 = 0.5; dlt55 = 0.5; dlt35 = 0.5; dlt34 = 0.5; dlt56 = 0.5
    arg = ((N6 - N3) ** 2 + (N4 + N5) ** 2 - (N3 + N5) ** 2) / (2 * (N6 - N3) * (N4 + N5))
    real_beta = np.isfinite(arg) & (np.abs(arg) <= 1)
    beta = np.arccos(np.where(real_beta, arg, 0.0))
    g8 = np.where(real_beta,
                  (N3 + N5 + 2 + dlt35) ** 2 - ((N6 - N3) ** 2 + (N4 + N5) ** 2
                                                 - 2 * (N6 - N3) * (N4 + N5) * np.cos(2 * PI / p - beta)),
                  1e6)
    g = _cols(m2 * (N6 + 2.5) - Dmax,
              m1 * (N1 + N2) + m1 * (N2 + 2) - Dmax,
              m2 * (N4 + N5) + m2 * (N5 + 2) - Dmax,
              np.abs(m1 * (N1 + N2) - m2 * (N6 - N3)) - m1 - m2,
              -((N1 + N2) * np.sin(PI / p) - N2 - 2 - dlt22),
              -((N6 - N3) * np.sin(PI / p) - N3 - 2 - dlt33),
              -((N4 + N5) * np.sin(PI / p) - N5 - 2 - dlt55),
              g8,
              -(N6 - 2 * N3 - N4 - 4 - 2 * dlt34),
              -(N6 - N4 - 2 * N5 - 4 - 2 * dlt56))
    h = _cols(np.fmod(N6 - N4, p))                            # MATLAB rem
    return f, g, h


# --------------------------------------------------------------------------- RC23
def rc23(x):
    d1 = x[:, 0] * 1e-3; d2 = x[:, 1] * 1e-3; d3 = x[:, 2] * 1e-3; d4 = x[:, 3] * 1e-3; w = x[:, 4] * 1e-3
    N = 350; N1 = 750; N2 = 450; N3 = 250; N4 = 150
    rho = 7200; a = 3; mu = 0.35; s = 1.75 * 1e6; t = 8 * 1e-3
    f = rho * w * PI / 4 * (d1 ** 2 * (1 + (N1 / N) ** 2) + d2 ** 2 * (1 + (N2 / N) ** 2)
                            + d3 ** 2 * (1 + (N3 / N) ** 2) + d4 ** 2 * (1 + (N4 / N) ** 2))

    def C(d, Ni):
        return PI * d / 2 * (1 + Ni / N) + (Ni / N - 1) ** 2 * d ** 2 / (4 * a) + 2 * a

    def Rr(d, Ni):
        return np.exp(mu * (PI - 2 * np.arcsin((Ni / N - 1) * d / (2 * a))))

    def Pp(d, Ni):
        return s * t * w * (1 - np.exp(-mu * (PI - 2 * np.arcsin((Ni / N - 1) * d / (2 * a))))) * PI * d * Ni / 60

    C1, C2, C3, C4 = C(d1, N1), C(d2, N2), C(d3, N3), C(d4, N4)
    lim = 0.75 * 745.6998
    g = _cols(-Rr(d1, N1) + 2, -Rr(d2, N2) + 2, -Rr(d3, N3) + 2, -Rr(d4, N4) + 2,
              -Pp(d1, N1) + lim, -Pp(d2, N2) + lim, -Pp(d3, N3) + lim, -Pp(d4, N4) + lim)
    h = _cols(C1 - C2, C1 - C3, C1 - C4)
    return f, g, h


# --------------------------------------------------------------------------- RC25
def rc25(x):
    R, Ro, mu, Q = x[:, 0], x[:, 1], x[:, 2], x[:, 3]
    gamma = 0.0307; C = 0.5; n = -3.55; C1 = 10.04
    Ws = 101000; Pmax = 1000; delTmax = 50; hmin = 0.001
    gg = 386.4; N = 750
    P = (np.log10(np.log10(8.122 * 1e6 * mu + 0.8)) - C1) / n
    delT = 2 * (10 ** P - 560)
    Ef = 9336 * Q * gamma * C * delT
    h = (2 * PI * N / 60) ** 2 * 2 * PI * mu / Ef * (R ** 4 / 4 - Ro ** 4 / 4) - 1e-5
    Po = (6 * mu * Q / (PI * h ** 3)) * np.log(R / Ro)
    W = PI * Po / 2 * (R ** 2 - Ro ** 2) / (np.log(R / Ro) - 1e-5)
    f = (Q * Po / 0.7 + Ef) / 12
    g = _cols(Ws - W,
              Po - Pmax,
              delT - delTmax,
              hmin - h,
              Ro - R,
              gamma / (gg * Po) * (Q / (2 * PI * R * h)) - 0.001,
              W / (PI * (R ** 2 - Ro ** 2) + 1e-5) - 5000)
    return f, g, None


# --------------------------------------------------------------------------- RC26
_PVALUE = np.array([3.175, 5.715, 8.255, 12.7])
_XYVALUE = np.array([12.7, 25.4, 38.1, 50.8, 63.5, 76.2, 88.9, 101.6, 114.3])


def rc26(x):
    x = mround(x)
    Np1, Ng1, Np2, Ng2, Np3, Ng3, Np4, Ng4 = (x[:, i] for i in range(8))
    b1, b2, b3, b4 = (_PVALUE[x[:, i].astype(int) - 1] for i in range(8, 12))
    xp1, xg1, xg2, xg3, xg4, yp1, yg1, yg2, yg3, yg4 = (
        _XYVALUE[x[:, i].astype(int) - 1] for i in range(12, 22))
    c1 = np.sqrt((xg1 - xp1) ** 2 + (yg1 - yp1) ** 2)
    c2 = np.sqrt((xg2 - xp1) ** 2 + (yg2 - yp1) ** 2)
    c3 = np.sqrt((xg3 - xp1) ** 2 + (yg3 - yp1) ** 2)
    c4 = np.sqrt((xg4 - xp1) ** 2 + (yg4 - yp1) ** 2)
    CRmin = 1.4; dmin = 25.4; phi = 20 * PI / 180; W = 55.9; JR = 0.2; Km = 1.6; Ko = 1.5; Lmax = 127
    sigma_H = 3290; sigma_N = 2090; w1 = 5000; wmin = 245; wmax = 255; Cp = 464
    f = PI / 1000 * (b1 * c1 ** 2 * (Np1 ** 2 + Ng1 ** 2) / (Np1 + Ng1) ** 2
                     + b2 * c2 ** 2 * (Np2 ** 2 + Ng2 ** 2) / (Np2 + Ng2) ** 2
                     + b3 * c3 ** 2 * (Np3 ** 2 + Ng3 ** 2) / (Np3 + Ng3) ** 2
                     + b4 * c4 ** 2 * (Np4 ** 2 + Ng4 ** 2) / (Np4 + Ng4) ** 2)
    sN = sigma_N * JR / (0.0167 * W * Ko * Km)
    sH = (sigma_H / Cp) ** 2 * (math.sin(phi) * math.cos(phi)) / (0.0334 * W * Ko * Km)
    g = [None] * 87
    g[1] = (366000 / (PI * w1) + 2 * c1 * Np1 / (Np1 + Ng1)) * ((Np1 + Ng1) ** 2 / (4 * b1 * c1 ** 2 * Np1)) - sN
    g[2] = (366000 * Ng1 / (PI * w1 * Np1) + 2 * c2 * Np2 / (Np2 + Ng2)) * ((Np2 + Ng2) ** 2 / (4 * b2 * c2 ** 2 * Np2)) - sN
    g[3] = (366000 * Ng1 * Ng2 / (PI * w1 * Np1 * Np2) + 2 * c3 * Np3 / (Np3 + Ng3)) * ((Np3 + Ng3) ** 2 / (4 * b3 * c3 ** 2 * Np3)) - sN
    g[4] = (366000 * Ng1 * Ng2 * Ng3 / (PI * w1 * Np1 * Np2 * Np3) + 2 * c4 * Np4 / (Np4 + Ng4)) * ((Np4 + Ng4) ** 2 / (4 * b4 * c4 ** 2 * Np4)) - sN
    g[5] = (366000 / (PI * w1) + 2 * c1 * Np1 / (Np1 + Ng1)) * ((Np1 + Ng1) ** 3 / (4 * b1 * c1 ** 2 * Ng1 * Np1 ** 2)) - sH
    g[6] = (366000 * Ng1 / (PI * w1 * Np1) + 2 * c2 * Np2 / (Np2 + Ng2)) * ((Np2 + Ng2) ** 3 / (4 * b2 * c2 ** 2 * Ng2 * Np2 ** 2)) - sH
    g[7] = (366000 * Ng1 * Ng2 / (PI * w1 * Np1 * Np2) + 2 * c3 * Np3 / (Np3 + Ng3)) * ((Np3 + Ng3) ** 3 / (4 * b3 * c3 ** 2 * Ng3 * Np3 ** 2)) - sH
    g[8] = (366000 * Ng1 * Ng2 * Ng3 / (PI * w1 * Np1 * Np2 * Np3) + 2 * c4 * Np4 / (Np4 + Ng4)) * ((Np4 + Ng4) ** 3 / (4 * b4 * c4 ** 2 * Ng4 * Np4 ** 2)) - sH
    sp = math.sin(phi)

    def cr(Np, Ng):
        return (CRmin * PI * math.cos(phi) - Np * np.sqrt(sp ** 2 / 4 + 1 / Np + (1 / Np) ** 2)
                - Ng * np.sqrt(sp ** 2 / 4 + 1 / Ng + (1 / Ng) ** 2) + sp * (Np + Ng) / 2)
    g[9], g[10], g[11], g[12] = cr(Np1, Ng1), cr(Np2, Ng2), cr(Np3, Ng3), cr(Np4, Ng4)
    pairs = ((Np1, Ng1, c1), (Np2, Ng2, c2), (Np3, Ng3, c3), (Np4, Ng4, c4))
    for k, (Np, Ng, c) in enumerate(pairs):
        g[13 + k] = dmin - 2 * c * Np / (Np + Ng)
        g[17 + k] = dmin - 2 * c * Ng / (Np + Ng)
    g[21] = xp1 + ((Np1 + 2) * c1 / (Np1 + Ng1)) - Lmax
    g[22] = xg2 + ((Np2 + 2) * c2 / (Np2 + Ng2)) - Lmax
    g[23] = xg3 + ((Np3 + 2) * c3 / (Np3 + Ng3)) - Lmax
    g[24] = xg4 + ((Np4 + 2) * c4 / (Np4 + Ng4)) - Lmax
    g[25] = -xp1 + ((Np1 + 2) * c1 / (Np1 + Ng1))
    g[26] = -xg2 + ((Np2 + 2) * c2 / (Np2 + Ng2))
    g[27] = -xg3 + ((Np3 + 2) * c3 / (Np3 + Ng3))
    g[28] = -xg4 + ((Np4 + 2) * c4 / (Np4 + Ng4))
    g[29] = yp1 + ((Np1 + 2) * c1 / (Np1 + Ng1)) - Lmax
    g[30] = yg2 + ((Np2 + 2) * c2 / (Np2 + Ng2)) - Lmax
    g[31] = yg3 + ((Np3 + 2) * c3 / (Np3 + Ng3)) - Lmax
    g[32] = yg4 + ((Np4 + 2) * c4 / (Np4 + Ng4)) - Lmax
    g[33] = -yp1 + ((Np1 + 2) * c1 / (Np1 + Ng1))
    g[34] = -yg2 + ((Np2 + 2) * c2 / (Np2 + Ng2))
    g[35] = -yg3 + ((Np3 + 2) * c3 / (Np3 + Ng3))
    g[36] = -yg4 + ((Np4 + 2) * c4 / (Np4 + Ng4))
    g[37] = xg1 + ((Ng1 + 2) * c1 / (Np1 + Ng1)) - Lmax
    g[38] = xg2 + ((Ng2 + 2) * c2 / (Np2 + Ng2)) - Lmax
    g[39] = xg3 + ((Ng3 + 2) * c3 / (Np3 + Ng3)) - Lmax
    g[40] = xg4 + ((Ng4 + 2) * c4 / (Np4 + Ng4)) - Lmax
    g[41] = -xg1 + ((Ng1 + 2) * c1 / (Np1 + Ng1))
    g[42] = -xg2 + ((Ng2 + 2) * c2 / (Np2 + Ng2))
    g[43] = -xg3 + ((Ng3 + 2) * c3 / (Np3 + Ng3))
    g[44] = -xg4 + ((Ng4 + 2) * c4 / (Np4 + Ng4))
    g[45] = yg1 + ((Ng1 + 2) * c1 / (Np1 + Ng1)) - Lmax
    g[46] = yg2 + ((Ng2 + 2) * c2 / (Np2 + Ng2)) - Lmax
    g[47] = yg3 + ((Ng3 + 2) * c3 / (Np3 + Ng3)) - Lmax
    g[48] = yg4 + ((Ng4 + 2) * c4 / (Np4 + Ng4)) - Lmax
    g[49] = -yg1 + ((Ng1 + 2) * c1 / (Np1 + Ng1))
    g[50] = -yg2 + ((Ng2 + 2) * c2 / (Np2 + Ng2))
    g[51] = -yg3 + ((Ng3 + 2) * c3 / (Np3 + Ng3))
    g[52] = -yg4 + ((Ng4 + 2) * c4 / (Np4 + Ng4))
    bs = (b1, b2, b3, b4)
    cs = (c1, c2, c3, c4)
    Nps = (Np1, Np2, Np3, Np4)
    Ngs = (Ng1, Ng2, Ng3, Ng4)
    for k in range(4):
        b, c, Np, Ng = bs[k], cs[k], Nps[k], Ngs[k]
        g[53 + k] = (0.945 * c - Np - Ng) * (b - 5.715) * (b - 8.255) * (b - 12.70) * (-1)
        g[57 + k] = (0.646 * c - Np - Ng) * (b - 3.175) * (b - 8.255) * (b - 12.70) * (+1)
        g[61 + k] = (0.504 * c - Np - Ng) * (b - 3.175) * (b - 5.715) * (b - 12.70) * (-1)
        g[65 + k] = (0.0 * c - Np - Ng) * (b - 3.175) * (b - 5.715) * (b - 8.255) * (+1)
        g[69 + k] = (-1.812 * c + Np + Ng) * (b - 5.715) * (b - 8.255) * (b - 12.70) * (-1)
        g[73 + k] = (-0.945 * c + Np + Ng) * (b - 3.175) * (b - 8.255) * (b - 12.70) * (+1)
        g[77 + k] = (-0.646 * c + Np + Ng) * (b - 3.175) * (b - 5.715) * (b - 12.70) * (-1)
        g[81 + k] = (-0.504 * c + Np + Ng) * (b - 3.175) * (b - 5.715) * (b - 8.255) * (+1)
    # official quirk: g79 uses Np2 (not Np3) and g80 uses Np3 (not Np4)
    g[79] = (-0.646 * c3 + Np2 + Ng3) * (b3 - 3.175) * (b3 - 5.715) * (b3 - 12.70) * (-1)
    g[80] = (-0.646 * c4 + Np3 + Ng4) * (b4 - 3.175) * (b4 - 5.715) * (b4 - 12.70) * (-1)
    g[85] = wmin - w1 * (Np1 * Np2 * Np3 * Np4) / (Ng1 * Ng2 * Ng3 * Ng4)
    g[86] = -wmax + w1 * (Np1 * Np2 * Np3 * Np4) / (Ng1 * Ng2 * Ng3 * Ng4)
    G = _cols(*g[1:87])
    G[np.isinf(G)] = 1e6                          # g(g==inf)=1e6; g(g==-inf)=1e6
    return f, G, None


# --------------------------------------------------------------------------- RC27
_GC = np.array([[18.288, 18.288, 9.144, 9.144, 0.0, 0.0],
                [9.144, 0.0, 9.144, 0.0, 9.144, 0.0]])
_EL = np.array([[3, 1, 4, 2, 3, 1, 4, 3, 2, 1],
                [5, 3, 6, 4, 4, 2, 5, 6, 3, 4]])


def _truss_weight(A):
    rho = 2770
    W = 0.0
    for i in range(10):
        nd = _EL[:, i] - 1
        xx, yy = _GC[0, nd], _GC[1, nd]
        le = math.sqrt((xx[1] - xx[0]) ** 2 + (yy[1] - yy[0]) ** 2)
        W = W + rho * le * A[i]
    return W


def _truss_freq_constraints(A):
    from scipy.linalg import eigh
    E = 6.98 * 1e10
    rho = 2770
    sdof = 12
    K = np.zeros((sdof, sdof))
    M = np.zeros((sdof, sdof))
    for iel in range(10):
        nd = _EL[:, iel]
        xx, yy = _GC[0, nd - 1], _GC[1, nd - 1]
        le = math.sqrt((xx[1] - xx[0]) ** 2 + (yy[1] - yy[0]) ** 2)
        l_ij = (xx[1] - xx[0]) / le
        m_ij = (yy[1] - yy[0]) / le
        Te = np.array([[l_ij, m_ij, 0, 0], [0, 0, l_ij, m_ij]])
        ke = A[iel] * E / le * np.array([[1.0, -1.0], [-1.0, 1.0]])
        ke = Te.T @ ke @ Te
        me = rho * le * A[iel] * np.array([[2.0, 0, 1, 0], [0, 2, 0, 1], [1, 0, 2, 0], [0, 1, 0, 2]]) / 6
        idx = np.array([2 * nd[0] - 1, 2 * nd[0], 2 * nd[1] - 1, 2 * nd[1]]) - 1
        K[np.ix_(idx, idx)] += ke
        M[np.ix_(idx, idx)] += me
    for i in range(sdof):
        M[i, i] = M[i, i] + 454
    bc = np.array([9, 11, 10, 12]) - 1
    free = np.setdiff1d(np.arange(sdof), bc)
    om2 = np.sort(eigh(K[np.ix_(free, free)], M[np.ix_(free, free)], eigvals_only=True))
    fr = np.sqrt(om2) / 2 / PI
    return np.array([7 / fr[0] - 1, 15 / fr[1] - 1, 20 / fr[2] - 1])


def rc27(x):
    f = np.array([_truss_weight(r) for r in x])
    g = np.array([_truss_freq_constraints(r) for r in x])
    return f, g, None


# --------------------------------------------------------------------------- RC28
def rc28(x):
    Dm, Db, fi, fo = x[:, 0], x[:, 1], x[:, 3], x[:, 4]
    Z = mround(x[:, 2])
    KDmin, KDmax, eps, e, chi = x[:, 5], x[:, 6], x[:, 7], x[:, 8], x[:, 9]
    D = 160; d = 90; Bw = 30
    T = D - d - 2 * Db
    arg = ((((D - d) * 0.5 - 0.75 * T) ** 2 + (0.5 * D - 0.25 * T - Db) ** 2 - (0.5 * d + 0.25 * T) ** 2)
           / (2 * (0.5 * (D - d) - 0.75 * T) * (0.5 * D - 0.25 * T - Db)))
    phi_o = 2 * PI - 2 * np.arccos(arg)
    gamma = Db / Dm
    fc = (37.91 * (1 + (1.04 * ((1 - gamma) / (1 + gamma)) ** 1.72
                        * (fi * (2 * fo - 1) / (fo * (2 * fi - 1))) ** 0.41) ** (10 / 3)) ** (-0.3)
          * (gamma ** 0.3 * (1 - gamma) ** 1.39 / (1 + gamma) ** (1 / 3)) * (2 * fi / (2 * fi - 1)) ** 0.41)
    f = np.where(Db > 25.4, 3.647 * fc * Z ** (2 / 3) * Db ** 1.4, fc * Z ** (2 / 3) * Db ** (1.8))
    g = _cols(Z - 1 - phi_o / (2 * np.arcsin(Db / Dm)),
              KDmin * (D - d) - 2 * Db,
              2 * Db - KDmax * (D - d),
              chi * Bw - Db,
              0.5 * (D + d) - Dm,
              Dm - (0.5 + e) * (D + d),
              eps * Db - 0.5 * (D - Dm - Db),
              0.515 - fi,
              0.515 - fo)
    return f, g, None


# --------------------------------------------------------------------------- RC29
def rc29(x):
    x1, x2, x3, x4 = (x[:, i] for i in range(4))
    f = (8.61 * 1e5 * x1 ** 0.5 * x2 * x3 ** (-2 / 3) * x4 ** (-1 / 2) + 3.69 * 1e4 * x3
         + 7.72 * 1e8 * x1 ** (-1) * x2 ** (0.219) - 765.43 * 1e6 * x1 ** (-1))
    g = _cols(x4 * x2 ** (-2) + x2 ** (-2) - 1)
    return f, g, None


# --------------------------------------------------------------------------- RC30
_DWIRE = np.array([0.009, 0.0095, 0.0104, 0.0118, 0.0128, 0.0132, 0.014,
                   0.015, 0.0162, 0.0173, 0.018, 0.020, 0.023, 0.025,
                   0.028, 0.032, 0.035, 0.041, 0.047, 0.054, 0.063,
                   0.072, 0.080, 0.092, 0.0105, 0.120, 0.135, 0.148,
                   0.162, 0.177, 0.192, 0.207, 0.225, 0.244, 0.263,
                   0.283, 0.307, 0.331, 0.362, 0.394, 0.4375, 0.500])


def rc30(x):
    x1 = mround(x[:, 0])
    x2 = x[:, 1]
    x3 = _DWIRE[mround(x[:, 2]).astype(int) - 1]
    f = (PI ** 2 * x2 * x3 ** 2 * (x1 + 2)) / 4
    cf = (4 * x2 / x3 - 1) / (4 * x2 / x3 - 4) + 0.615 * x3 / x2
    K = (11.5 * 10 ** 6 * x3 ** 4) / (8 * x1 * x2 ** 3)
    lf = 1000 / K + 1.05 * (x1 + 2) * x3
    sigp = 300 / K
    g = _cols((8000 * cf * x2) / (PI * x3 ** 3) - 189000,
              lf - 14,
              0.2 - x3,
              x2 - 3,
              3 - x2 / x3,
              sigp - 6,
              sigp + 700 / K + 1.05 * (x1 + 2) * x3 - lf,
              1.25 - 700 / K)
    return f, g, None


# --------------------------------------------------------------------------- RC31
def rc31(x):
    x1, x2, x3, x4 = (x[:, i] for i in range(4))
    f = (1 / 6.931 - (x1 * x2) / (x3 * x4)) ** 2
    n = x.shape[0]
    return f, np.zeros((n, 1)), np.zeros((n, 1))


# --------------------------------------------------------------------------- RC32
def rc32(x):
    x1, x2, x3, x4, x5 = (x[:, i] for i in range(5))
    f = 5.3578547 * x3 ** 2 + 0.8356891 * x1 * x5 + 37.293239 * x1 - 40792.141
    G1 = 85.334407 + 0.0056858 * x2 * x5 + 0.0006262 * x1 * x4 - 0.0022053 * x3 * x5
    G2 = 80.51249 + 0.0071317 * x2 * x5 + 0.0029955 * x1 * x2 + 0.0021813 * x3 ** 2
    G3 = 9.300961 + 0.0047026 * x3 * x5 + 0.0012547 * x1 * x3 + 0.0019085 * x3 * x4
    g = _cols(G1 - 92, -G1, G2 - 110, -G2 + 90, G3 - 25, -G3 + 20)
    return f, g, None


# --------------------------------------------------------------------------- RC33
def _lk():
    E = 206000000.
    nu = 0.3
    k = [1 / 2 - nu / 6, 1 / 8 + nu / 8, -1 / 4 - nu / 12, -1 / 8 + 3 * nu / 8,
         -1 / 4 + nu / 12, -1 / 8 - nu / 8, nu / 6, 1 / 8 - 3 * nu / 8]
    k = [None] + k
    M = [[1, 2, 3, 4, 5, 6, 7, 8], [2, 1, 8, 7, 6, 5, 4, 3], [3, 8, 1, 6, 7, 4, 5, 2],
         [4, 7, 6, 1, 8, 3, 2, 5], [5, 6, 7, 8, 1, 2, 3, 4], [6, 5, 4, 3, 2, 1, 8, 7],
         [7, 4, 5, 2, 3, 8, 1, 6], [8, 3, 2, 5, 4, 7, 6, 1]]
    return E / (1 - nu ** 2) * np.array([[k[j] for j in row] for row in M])


_KE = _lk()


def _fe(nelx, nely, X, penal):
    nd = 2 * (nelx + 1) * (nely + 1)
    K = np.zeros((nd, nd))
    F = np.zeros(nd)
    U = np.zeros(nd)
    for elx in range(1, nelx + 1):
        for ely in range(1, nely + 1):
            n1 = (nely + 1) * (elx - 1) + ely
            n2 = (nely + 1) * elx + ely
            edof = np.array([2 * n1 - 1, 2 * n1, 2 * n2 - 1, 2 * n2,
                             2 * n2 + 1, 2 * n2 + 2, 2 * n1 + 1, 2 * n1 + 2]) - 1
            K[np.ix_(edof, edof)] += X[ely - 1, elx - 1] ** penal * _KE
    F[nd - 1] = -10000
    fixed = np.arange(0, 2 * (nely + 1))
    free = np.setdiff1d(np.arange(nd), fixed)
    U[free] = np.linalg.solve(K[np.ix_(free, free)], F[free])
    U[fixed] = 0
    return U


def _check(nelx, nely, rmin, x, dc):
    dcn = np.zeros((nely, nelx))
    r = int(math.floor(rmin))
    for i in range(1, nelx + 1):
        for j in range(1, nely + 1):
            s = 0.0
            for k in range(max(i - r, 1), min(i + r, nelx) + 1):
                for l in range(max(j - r, 1), min(j + r, nely) + 1):
                    fac = rmin - math.sqrt((i - k) ** 2 + (j - l) ** 2)
                    s = s + max(0, fac)
                    dcn[j - 1, i - 1] = dcn[j - 1, i - 1] + max(0, fac) * x[l - 1, k - 1] * dc[l - 1, k - 1]
            dcn[j - 1, i - 1] = dcn[j - 1, i - 1] / (x[j - 1, i - 1] * s)
    return dcn


def _topology(xrow):
    nely, nelx, penal = 10, 3, 3
    X = np.array([xrow[0:10], xrow[10:20], xrow[20:30]]).T
    U = _fe(nelx, nely, X, penal)
    c = 0.0
    dc = np.zeros((nely, nelx))
    for ely in range(1, nely + 1):
        for elx in range(1, nelx + 1):
            n1 = (nely + 1) * (elx - 1) + ely
            n2 = (nely + 1) * elx + ely
            Ue = U[np.array([2 * n1 - 1, 2 * n1, 2 * n2 - 1, 2 * n2,
                             2 * n2 + 1, 2 * n2 + 2, 2 * n1 + 1, 2 * n1 + 2]) - 1]
            q = Ue @ _KE @ Ue
            c = c + X[ely - 1, elx - 1] ** penal * q
            dc[ely - 1, elx - 1] = -penal * X[ely - 1, elx - 1] ** (penal - 1) * q
    dcn = _check(nelx, nely, 1.5, X, dc)
    return c, dcn.flatten(order="F")


def rc33(x):
    res = [_topology(r) for r in x]
    return np.array([c for c, _ in res]), np.array([g for _, g in res]), None


# ===========================================================================
# Problem table: D, bounds (Cal_par.m), best-known f* (Problem-Definitions,
# Table 2), declared equality constraints h (Cal_par.m)
# ===========================================================================
def _b(lo, hi):
    return np.array(lo, dtype=float), np.array(hi, dtype=float)


PROBLEMS: Dict[int, Tuple[str, int, Callable, Tuple[Array, Array], float, int]] = {
    15: ("speed_reducer", 7, rc15, _b([2.6, 0.7, 17, 7.3, 7.3, 2.9, 5], [3.6, 0.8, 28, 8.3, 8.3, 3.9, 5.5]), 2.9944244658E+03, 0),
    16: ("refrigeration_system", 14, rc16, _b([0.001] * 14, [5] * 14), 3.2213000814E-02, 0),
    17: ("spring_case1", 3, rc17, _b([0.05, 0.25, 2.00], [2, 1.3, 15.0]), 1.2665232788E-02, 0),
    18: ("pressure_vessel", 4, rc18, _b([0.51, 0.51, 10, 10], [99.49, 99.49, 200, 200]), 5.8853327736E+03, 0),
    19: ("welded_beam", 4, rc19, _b([0.125, 0.1, 0.1, 0.1], [2, 10, 10, 2]), 1.6702177263E+00, 0),
    20: ("three_bar_truss", 2, rc20, _b([0, 0], [1, 1]), 2.6389584338E+02, 0),
    21: ("clutch_brake", 5, rc21, _b([60, 90, 1, 0, 2], [80, 110, 3, 1000, 9]), 2.3524245790E-01, 0),
    22: ("planetary_gear_train", 9, rc22, _b([16.51, 13.51, 13.51, 16.51, 13.51, 47.51, 0.51, 0.51, 0.51],
                                             [96.49, 54.49, 51.49, 46.49, 51.49, 124.49, 3.49, 6.49, 6.49]), 5.2576870748E-01, 1),
    23: ("step_cone_pulley", 5, rc23, _b([0, 0, 0, 0, 0], [60, 60, 90, 90, 90]), 1.6069868725E+01, 3),
    25: ("thrust_bearing", 4, rc25, _b([1, 1, 1e-6, 1], [16, 16, 16 * 1e-6, 16]), 1.6254428092E+03, 0),
    26: ("four_stage_gearbox", 22, rc26, _b([6.51] * 8 + [0.51] * 14, [76.49] * 8 + [4.49] * 4 + [9.49] * 10), 3.5359231973E+01, 0),
    27: ("ten_bar_truss", 10, rc27, _b([0.645e-4] * 10, [50e-4] * 10), 5.2445076066E+02, 0),
    28: ("rolling_bearing", 10, rc28, _b([125, 10.5, 4.51, 0.515, 0.515, 0.4, 0.6, 0.3, 0.02, 0.6],
                                        [150, 31.5, 50.49, 0.6, 0.6, 0.5, 0.7, 0.4, 0.1, 0.85]), 1.4614135715E+04, 0),
    29: ("gas_compressor", 4, rc29, _b([20, 1, 20, 0.1], [50, 10, 50, 60]), 2.9648954173E+06, 0),
    30: ("spring_case2", 3, rc30, _b([0.51, 0.6, 0.51], [70.49, 3, 42.49]), 2.6138840583E+00, 0),
    31: ("gear_train", 4, rc31, _b([12] * 4, [60] * 4), 0.0, 1),
    32: ("himmelblau", 5, rc32, _b([78, 33, 27, 27, 27], [102, 45, 45, 45, 45]), -3.0665538672E+04, 0),
    33: ("topology", 30, rc33, _b([0.001] * 30, [1] * 30), 2.6393464970E+00, 0),
}


def max_fes(D: int) -> int:
    """Official MaxFEs (Guidelines, Eq. 1)."""
    if D <= 10:
        return 100_000
    if D <= 30:
        return 200_000
    if D <= 50:
        return 400_000
    if D <= 150:
        return 800_000
    return 1_000_000


def evaluate(pid: int, X: Array):
    """(f, G, H) of the official function for every row of X (real space)."""
    X = np.atleast_2d(np.asarray(X, dtype=float))
    with np.errstate(all="ignore"):
        f, G, H = PROBLEMS[pid][2](X)
    n = X.shape[0]
    if H is None:
        H = np.zeros((n, 0))
    return np.asarray(f, float), np.asarray(G, float), np.asarray(H, float)


def violation(pid: int, G: Array, H: Array) -> Array:
    """Official mean violation; NaN constraint values make it NaN."""
    nh = PROBLEMS[pid][5]
    Hd = np.abs(H[:, :nh]) if nh else np.zeros((G.shape[0], 0))
    Hv = np.where(Hd > EQ_TOL, Hd, 0.0)
    Gv = np.where(np.isnan(G), np.nan, np.maximum(G, 0.0))
    m = G.shape[1] + nh
    return (np.sum(Gv, axis=1) + np.sum(Hv, axis=1)) / m


# C_k: fixed before the runs (PROTOCOL_CEC2020_MECH.md, amendment 2): the
# power of ten >= 1000 x max(largest feasible |f| measured by
# `crosscheck_cec2020rw_octave.py --scan`, hand-derived bound where needed).
INFEASIBLE_OFFSET: Dict[int, float] = {
    15: 1e7, 16: 1e10, 17: 1e5, 18: 1e9, 19: 1e5, 20: 1e6, 21: 1e4, 22: 1e7,
    23: 1e5, 25: 1e8, 26: 1e7, 27: 1e7, 28: 1e8, 29: 1e11, 30: 1e5, 31: 1e6,
    32: 1e8, 33: 1e12,
}


def unit_box(pid: int) -> Callable[[Array], Array]:
    name, D, func, (lb, ub), fstar, nh = PROBLEMS[pid]
    span = ub - lb
    C = INFEASIBLE_OFFSET[pid]

    def F(U: Array) -> Array:
        U = np.atleast_2d(U)
        out = np.full(U.shape[0], NONFINITE_PENALTY)
        # Points outside the official search range are invalid (CEC rule;
        # the official code cannot even evaluate some of them): they get the
        # same 1e20 penalty as non-finite values. Protocol, amendment 3.
        inside = np.all((U >= 0.0) & (U <= 1.0), axis=1)
        if inside.any():
            f, G, H = evaluate(pid, lb[None, :] + U[inside] * span[None, :])
            v = violation(pid, G, H)
            val = np.where(v == 0, f, C + v)
            out[inside] = np.where(np.isfinite(val), val, NONFINITE_PENALTY)
        return out
    return F


def decode(pid: int, F: float):
    """(feasible, f or violation) from a scalar fitness value."""
    C = INFEASIBLE_OFFSET[pid]
    if not math.isfinite(F) or F >= NONFINITE_PENALTY:
        return False, math.inf
    if F < C:
        return True, F
    return False, F - C
