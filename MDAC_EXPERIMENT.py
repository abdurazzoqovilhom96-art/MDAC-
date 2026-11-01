#!/usr/bin/env python3
# -*- coding: utf-8 -*-
r"""
MDAC_EXPERIMENT.py
==================
Single-file, self-contained implementation + experiment driver for

    MDAC -- Metric-DE with Adaptive Co-allocation

    (The "M" was Momentum in the first version. Momentum was measured as
    harmful in three independent ablations -- A12 0.455 on the shifted
    classics, and twice on the locally generated BR_* suite (numbers withheld: BR_* is not an official benchmark) --
    and is off by default, so naming it would advertise a disabled
    mechanism. See `alg_mdac` for the full record.)

and every baseline, control and statistic required to evaluate it honestly.
No pandas dependency (stdlib `csv` only). Requires: numpy, scipy.

--------------------------------------------------------------------------
PROVENANCE -- what is borrowed, from where, and what is ours
--------------------------------------------------------------------------
This header is the honesty contract. Every mechanism below is attributed.
If a mechanism is taken from an algorithm, that algorithm IS a competitor
in the comparison (see `ALGORITHMS`). No exceptions.

BORROWED (cited, not claimed as contribution)
  [L-SHADE]  Tanabe & Fukunaga, "Improving the Search Performance of SHADE
             Using Linear Population Size Reduction", IEEE CEC 2014.
             -> success-history memories M_CR, M_F   (their Eq. 1-2)
             -> weighted Lehmer mean update          (their Eq. 7-9)
             -> terminal value `bot` for M_CR        (their Alg. 1)
             -> current-to-pbest/1 + external archive(their Eq. 3)
             -> Linear Population Size Reduction     (their Eq. 10)
             -> binomial crossover, midpoint bound repair (their Eq. 4-5)
             -> tuned constants r_Ninit=18, r_arc=2.6, p=0.11, H=6
                (their Table II, "Tuned settings" column)

  [LSPACMA]  Mohamed, Hadi, Fattouh & Jambi, "LSHADE with Semi-Parameter
             Adaptation Hybrid with CMA-ES for Solving CEC 2017 Benchmark
             Problems", IEEE CEC 2017.
             -> FCP co-allocation of two offspring generators over ONE
                shared population, with the allocation probability learned
                from measured improvement:
                    M_fcp <- (1-c) M_fcp + c * Delta
                    Delta  = clip( w_A / (w_A + w_B), 0.2, 0.8 )
                    w_X    = sum over successful i in X of [f(x_i)-f(u_i)]
             -> crossover applied to the CMA-ES offspring

  [LSMA]     Zhang, Hu, Zhu, Han & Huang, "L-SHADE-MA: Enhancing L-SHADE
             with Momentum and Adaptive Decay Coefficient", SSRN 4647987.
             *** NOT PEER REVIEWED -- preprint. The underlying mechanism is
             *** classical: Polyak (1964) heavy-ball; Kingma & Ba (2015) Adam.
             -> momentum-to-pbest/1                  (their Eq. 20)
                    v_i = x_i + F_i (x_pbest - x_i) + F_i * Mom_i
             -> momentum built from a FITNESS-ORDERED difference vector
                                                     (their Eq. 21)
                    Mom_i = beta_i * Hm_i + (1-beta_i)(x_b1 - x_b2),
                            f(x_b1) < f(x_b2)
             -> historical movement                  (their Eq. 22)
                    Hm_i <- (u_i - x_i) on success, retained otherwise
                *** SEE `MOMENTUM_HM_ON_SUCCESS` BELOW: the comparison
                *** operator in their Eq. 22 is NOT recoverable from the
                *** PDF text layer. Both readings are implemented and
                *** decided empirically, not assumed.
             -> adaptive decay coefficient beta via memory M_beta (Eq. 23)

  [AGDESA]   Soonjun & Krityakierne, "GDESA: Gradient Differential
             Evolution-Simulated Annealing Hybrid", IEEE Access 12, 2024.
             -> Metropolis/Boltzmann acceptance       (their Eq. 10)
                    accept if rand < exp(-Delta / T)
             -> per-individual stagnation trigger     (their Alg. 5)
                    S[i] = 1  iff  g - L[i] >= n_L
             -> Fast-SA inverse-time cooling T = T0/(1+g) (removes alpha,beta)
             *** REJECTED from this work: their L-BFGS-B gradient component.
             *** Reason: run to local convergence with numerical gradients it
             *** costs O(D) evaluations per gradient and an unbounded number
             *** of iterations. Under the FES-honest Evaluator used here that
             *** cost is fully charged, and charging it is the whole point of
             *** this framework. It is implemented ONLY as the competitor
             *** `A-GDESA-fd`, with every finite-difference probe charged, so
             *** the cost is visible rather than hidden.

  [CMAES]    Hansen & Ostermeier (2001); Hansen (2016) "The CMA Evolution
             Strategy: A Tutorial".
             -> (mu/mu_w, lambda)-CMA-ES as a standalone competitor
  [1+lCMA]   Igel, Suttorp & Hansen (2006/2007).
             -> (1+lambda)-CMA-ES with rank-one Cholesky update, used as the
                local operator and in the architectural-parity control

OURS (the only things claimed as contribution)
  (C1) Diversity-regularized exploitation targeting.
       The pbest donor in momentum-to-pbest/1 is drawn from the elite pool
       with probability proportional to exp(nu(t) * zscore(d_min)), where
       d_min is distance to the already-exploited set. Costs ZERO objective
       evaluations -- it reweights a draw the generation has already paid
       for. This replaces the earlier candidate-competition implementation
       of the same idea, which cost C*N evaluations per generation and which
       our own ablation could not justify.
  (C2) Generalising [LSPACMA]'s FCP from "which mutation operator" to
       "global explorer vs local metric-learner", and testing that learned
       allocation against fixed / random / terminal allocation at equal
       budget. The allocation RULE is the claim, not the presence of a
       local searcher.
  (C3) Methodology: the FES-honest Evaluator that charges every internal
       probe; the architectural-parity control; the 1e-8 reporting
       convention applied uniformly; and SHIFTED benchmark optima (see
       `BENCHMARK NOTES`) which remove a center-bias artifact we measured
       in our own earlier suite.

--------------------------------------------------------------------------
BENCHMARK NOTES -- read before quoting any number from this file
--------------------------------------------------------------------------
1. `CL_*` : 12 classic landscapes, each SHIFTED by a per-(function,dim)
   random offset so the optimum is NOT at the centre of a symmetric box.
   Our earlier suite left 8 of 12 optima at exactly x=0, which handed a
   measured advantage to centre-contracting operators (GWO/WOA/HHO reached
   exactly 0.0 there). All optima here satisfy f(x*) = 0 by construction,
   so the reported value IS the error f(x)-f(x*).

2. `BR_*` : 30 functions following CEC2017's CATEGORY STRUCTURE (3 unimodal,
   7 multimodal, 10 hybrid, 10 composition; shift + rotation; [-100,100]^D)
   built from LOCALLY generated shifts and rotations.
   *** THIS IS NOT THE OFFICIAL CEC2017 SUITE. ***
   The prefix is `BR_` (BRidge) and never `CEC17_`, precisely so that no
   table produced by this file can be mistaken for official CEC2017 numbers
   or compared against published CEC2017 results. Installing the official
   `input_data` files is tracked as required work, not done here.

3. Error values below `ZERO_THRESHOLD = 1e-8` are reported as 0. This is
   the convention used by BOTH source papers ([L-SHADE] Sec. III,
   [LSPACMA] Sec. III-D) and by the CEC competitions themselves. It is
   applied UNIFORMLY to every algorithm. `--no_threshold` disables it;
   the driver always writes both variants so the artefact is visible.

--------------------------------------------------------------------------
USAGE
--------------------------------------------------------------------------
    python MDAC_EXPERIMENT.py --stage smoke      #   ~20 s   self-test
    python MDAC_EXPERIMENT.py --stage pilot      #   minutes correctness
    python MDAC_EXPERIMENT.py --stage ablation   #   our mechanisms
    python MDAC_EXPERIMENT.py --stage parity     #   architectural control
    python MDAC_EXPERIMENT.py --stage main       #   hours, full comparison
"""
from __future__ import annotations

import argparse
import csv
import json
import math
import multiprocessing as mp
import os
import sys
import time
from collections import defaultdict
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import dataclass, field, replace
from pathlib import Path
from typing import Callable, Dict, List, Optional, Sequence, Tuple

for _v in ("OPENBLAS_NUM_THREADS", "OMP_NUM_THREADS", "MKL_NUM_THREADS",
           "NUMEXPR_NUM_THREADS"):
    os.environ.setdefault(_v, "1")

import numpy as np
from scipy.optimize import minimize
from scipy.stats import (friedmanchisquare, norm, rankdata,
                         studentized_range, wilcoxon)

Array = np.ndarray
Bounds = Tuple[float, float]
VecFunc = Callable[[Array], Array]

ZERO_THRESHOLD = 1e-8
BENCH_SEED = 20260421

# CEC2017 F2 is left out of every statistic. The organisers withdrew it from
# the competition after release because it behaves unstably, especially at
# high D (here: mean errors from 3e4 to 6e10 at D=100), and implementations of
# it disagree. Its runs are still executed and kept in raw_*.csv. Set
# MDAC_KEEP_F2=1 to include it, e.g. for a with/without sensitivity check.
EXCLUDED_FROM_STATS: Tuple[str, ...] = ("CEC2017_F2_",)

# CEC2011 real-world problems (benchmarks/cec2011_official.py). The
# competition's own budget is 150 000 evaluations per run for every problem,
# whatever its D (technical report, "Evaluation Criteria"). Their optima are
# unknown, so the raw objective is reported and the 1e-8 convention is NOT
# applied: it would turn every negative energy (Lennard-Jones, Tersoff,
# antenna side-lobe level in dB) into 0.
CAT_RW = "CEC2011 real-world"
CEC2011_MAX_FES = 150_000
CEC2011_DIMS = [6, 12, 20, 22, 26, 30]
# CEC2020 real-world constrained, mechanical engineering RC15-RC33 except RC24
# (benchmarks/cec2020rw_official.py, PROTOCOL_CEC2020_MECH.md). Constraints
# enter through one fixed scalar fitness, identical for every algorithm; the
# budget is the official MaxFEs for the problem's D; raw values, no 1e-8 rule.
CAT_RWC = "CEC2020 mechanical"
CEC2020_MECH_DIMS = [2, 3, 4, 5, 7, 9, 10, 14, 22, 30]
RAW_CATEGORIES = (CAT_RW, CAT_RWC)


def _max_fes(spec, fes_mult: int, dim: int) -> int:
    if spec.category == CAT_RW:
        return CEC2011_MAX_FES
    if spec.category == CAT_RWC:
        from benchmarks.cec2020rw_official import max_fes
        return max_fes(dim)
    return fes_mult * dim


def _safe_exp(z: float, cap: float = 60.0) -> float:
    """exp with a hard argument clamp.

    Needed because a step-size update of the form exp(c*(||p||/chiN - 1))
    overflows on an ill-conditioned rotated landscape (measured: CL_RotElliptic
    at D=10 raised OverflowError, which aborted the run and left CMA-ES
    spending 25% of MaxFES -- a budget-parity violation, i.e. the exact class
    of defect this framework exists to catch).
    """
    return math.exp(max(-cap, min(cap, z)))


def fitness_scale(fit: Array) -> float:
    """Robust, strictly positive scale of the current fitness spread.

    Used to make the Metropolis acceptance SCALE-INVARIANT. This is a
    deliberate deviation from [AGDESA], which uses an absolute temperature T
    set by the user. An absolute T cannot work across a 42-function suite:
    measured here, the initial spread ranges from ~1e0 (CL_Rastrigin) to
    ~1e10 (CL_RotElliptic), so one T either freezes acceptance everywhere or
    accepts everything everywhere. With `delta / (T * scale)` the acceptance
    probability depends on the RELATIVE worsening, which is the same
    dimensionless-ratio argument that our earlier work needed for its
    candidate score (and which, left out, broke it).
    """
    f = fit[np.isfinite(fit)]
    if f.size < 2:
        return 1.0
    q75, q25 = np.percentile(f, [75, 25])
    s = float(q75 - q25)
    if s <= 1e-300:
        s = float(np.median(np.abs(f - np.median(f)))) * 1.4826
    if s <= 1e-300:
        s = max(1e-300, float(np.abs(np.median(f))))
    return s


# ===========================================================================
# 1. FES-HONEST EVALUATOR
# ===========================================================================
class BudgetExhausted(Exception):
    """Raised the moment an evaluation is requested with zero budget left."""


class Evaluator:
    """Charges EVERY objective call against one shared MaxFES ceiling.

    `f_star` is the known optimum, so `error()` is f(x) - f(x*). Nothing in
    an algorithm may read `f_star`: it is used by the driver only, after the
    run, to report an error value. The guard `_locked` makes a read-after-run
    the only legal access pattern.
    """

    __slots__ = ("func", "dim", "max_fes", "fes", "best_fitness", "best_x",
                 "_f_star", "_trace_fes", "_trace_best", "_stride", "_since",
                 "max_seconds", "_t0", "aux")

    def __init__(self, func: VecFunc, dim: int, max_fes: int,
                 f_star: float = 0.0, record_stride: int = 1,
                 max_seconds: float = None):
        self.func = func
        self.dim = int(dim)
        self.max_fes = int(max_fes)
        # EQUAL-TIME MODE (A1). When set, wall-clock is the binding budget and
        # `max_fes` is left deliberately huge so it never binds. The clock
        # starts here, immediately before the algorithm is entered, so setup
        # done by the driver is not charged to any algorithm.
        self.max_seconds = max_seconds
        self._t0 = time.perf_counter()
        # Domain data for a DOMAIN baseline only (clustering: {"X","k"}).
        # General-purpose optimizers must never read this -- they are
        # black-box by construction, and `aux` exists so that a specialised
        # competitor such as k-means can be given the information it would
        # really have, instead of being handicapped into a straw man.
        self.aux: Dict = {}
        self.fes = 0
        self.best_fitness = float("inf")
        self.best_x: Optional[Array] = None
        self._f_star = float(f_star)
        self._trace_fes: List[int] = [0]
        self._trace_best: List[float] = [float("inf")]
        self._stride = max(1, int(record_stride))
        self._since = 0

    # -- budget ------------------------------------------------------------
    def remaining(self) -> int:
        if self.max_seconds is not None and self.time_frac() >= 1.0:
            return 0
        return max(0, self.max_fes - self.fes)

    def exhausted(self) -> bool:
        if self.max_seconds is not None and self.time_frac() >= 1.0:
            return True
        return self.fes >= self.max_fes

    def time_frac(self) -> float:
        if self.max_seconds is None:
            return 0.0
        return (time.perf_counter() - self._t0) / self.max_seconds

    # -- evaluation --------------------------------------------------------
    def evaluate_batch(self, X: Array) -> Array:
        X = np.asarray(X, dtype=np.float64)
        if X.ndim == 1:
            X = X.reshape(1, -1)
        n = X.shape[0]
        if self.fes >= self.max_fes:
            raise BudgetExhausted()
        # In equal-time mode the clock is checked BEFORE the batch, so no
        # algorithm can overrun by starting one more expensive generation.
        if self.max_seconds is not None and self.time_frac() >= 1.0:
            raise BudgetExhausted()
        room = self.remaining()
        # TIME-OF-CHECK / TIME-OF-USE RACE. The guard above can pass and the
        # clock can then tick past the budget before `remaining()` is read, so
        # `room` becomes 0 and the partial-batch path below slices X[:0] and
        # calls argmin on an empty array. In FES mode the `fes >= max_fes`
        # check makes room == 0 unreachable; in time mode max_fes is 1e9, so
        # it is reachable and it fired 14 times in 38 556 runs of the first
        # equal-time sweep. The hard failure gate caught it and refused to
        # produce ranks, which is exactly what it exists for.
        if room <= 0:
            raise BudgetExhausted()
        if n <= room:
            vals = np.asarray(self.func(X), dtype=np.float64)
            vals = np.where(np.isfinite(vals), vals, np.inf)
            self.fes += n
            self._update_best(X, vals)
            self._since += n
            if self._since >= self._stride or self.fes >= self.max_fes:
                self._record()
                self._since = 0
            return vals
        vals = np.asarray(self.func(X[:room]), dtype=np.float64)
        vals = np.where(np.isfinite(vals), vals, np.inf)
        self.fes += room
        self._update_best(X[:room], vals)
        self._record()
        raise BudgetExhausted()

    def evaluate(self, x: Array) -> float:
        return float(self.evaluate_batch(np.asarray(x).reshape(1, -1))[0])

    def _update_best(self, X: Array, vals: Array) -> None:
        if vals.size == 0:                 # defensive: never argmin on empty
            return
        j = int(np.argmin(vals))
        if vals[j] < self.best_fitness:
            self.best_fitness = float(vals[j])
            self.best_x = X[j].copy()

    def _record(self) -> None:
        self._trace_fes.append(self.fes)
        self._trace_best.append(self.best_fitness)

    # -- reporting (driver only) -------------------------------------------
    def error(self, threshold: bool = True, clamp: bool = True) -> float:
        """f(best) - f_star. `clamp` floors it at 0, which is right whenever
        f_star is the true optimum; it is switched off only for problems
        whose optimum is unknown (CEC2011), where the raw value can be
        negative and must be reported as it is."""
        e = self.best_fitness - self._f_star
        if not math.isfinite(e):
            return float("inf")
        if clamp:
            e = max(0.0, e)
        return 0.0 if (threshold and e < ZERO_THRESHOLD) else e

    def grid_trace(self, n_points: int = 60) -> Tuple[Array, Array]:
        fes = np.asarray(self._trace_fes, dtype=np.float64)
        best = np.asarray(self._trace_best, dtype=np.float64)
        grid = np.linspace(0, self.max_fes, n_points)
        out = np.empty(n_points)
        j, cur = 0, best[0]
        for gi, g in enumerate(grid):
            while j < len(fes) and fes[j] <= g:
                cur = best[j]
                j += 1
            out[gi] = cur
        return grid, out - self._f_star


@dataclass
class AlgResult:
    name: str
    best_fitness: float
    fes_used: int
    best_x: Optional[Array] = None
    diag: Dict[str, float] = field(default_factory=dict)


def _finish(name: str, ev: Evaluator, diag: Optional[Dict] = None) -> AlgResult:
    return AlgResult(name=name, best_fitness=ev.best_fitness, fes_used=ev.fes,
                     best_x=ev.best_x, diag=diag or {})


# ===========================================================================
# 2. BENCHMARK SUITE
# ===========================================================================
def f_sphere(X): return np.sum(X * X, axis=1)


def f_elliptic(X):
    d = X.shape[1]
    w = 10.0 ** (6.0 * np.arange(d) / max(1, d - 1))
    return np.sum(w * X * X, axis=1)


def f_bent_cigar(X): return X[:, 0] ** 2 + 1e6 * np.sum(X[:, 1:] ** 2, axis=1)


def f_discus(X): return 1e6 * X[:, 0] ** 2 + np.sum(X[:, 1:] ** 2, axis=1)


def f_rosenbrock(X):
    return np.sum(100.0 * (X[:, 1:] - X[:, :-1] ** 2) ** 2
                  + (1.0 - X[:, :-1]) ** 2, axis=1)


def f_ackley(X):
    d = X.shape[1]
    s1 = np.sum(X * X, axis=1)
    s2 = np.sum(np.cos(2.0 * np.pi * X), axis=1)
    return -20.0 * np.exp(-0.2 * np.sqrt(s1 / d)) - np.exp(s2 / d) + 20.0 + np.e


def f_griewank(X):
    d = X.shape[1]
    i = np.sqrt(np.arange(1, d + 1))
    return np.sum(X * X, axis=1) / 4000.0 - np.prod(np.cos(X / i), axis=1) + 1.0


def f_rastrigin(X):
    d = X.shape[1]
    return 10.0 * d + np.sum(X * X - 10.0 * np.cos(2.0 * np.pi * X), axis=1)


_SCHWEFEL_ANCHOR = 4.209687462275036e2


def f_schwefel(X):
    d = X.shape[1]
    az = np.abs(X)
    g_in = X * np.sin(np.sqrt(az + 1e-12))
    zw = 500.0 - np.mod(az, 500.0)
    g_out = np.sign(X) * (zw * np.sin(np.sqrt(np.abs(zw)))
                          - (az - 500.0) ** 2 / (10000.0 * d))
    return 418.9829 * d - np.sum(np.where(az <= 500.0, g_in, g_out), axis=1)


def f_zakharov(X):
    d = X.shape[1]
    i = np.arange(1, d + 1, dtype=np.float64)
    s1 = np.sum(X * X, axis=1)
    s2 = np.sum(0.5 * i * X, axis=1)
    return s1 + s2 ** 2 + s2 ** 4


def f_levy(X):
    w = 1.0 + (X - 1.0) / 4.0
    t1 = np.sin(np.pi * w[:, 0]) ** 2
    t3 = (w[:, -1] - 1.0) ** 2 * (1.0 + np.sin(2.0 * np.pi * w[:, -1]) ** 2)
    t2 = np.sum((w[:, :-1] - 1.0) ** 2
                * (1.0 + 10.0 * np.sin(np.pi * w[:, :-1] + 1.0) ** 2), axis=1)
    return t1 + t2 + t3


def f_weierstrass(X, a=0.5, b=3.0, kmax=20):
    d = X.shape[1]
    k = np.arange(kmax + 1)
    ak, bk = a ** k, b ** k
    inner = 2.0 * np.pi * bk[None, None, :] * (X[:, :, None] + 0.5)
    t1 = np.sum(ak[None, None, :] * np.cos(inner), axis=(1, 2))
    t2 = d * np.sum(ak * np.cos(2.0 * np.pi * bk * 0.5))
    return t1 - t2


# base -> (callable, anchor = location of the optimum in each coordinate)
BASE: Dict[str, Tuple[VecFunc, float]] = {
    "sphere": (f_sphere, 0.0), "elliptic": (f_elliptic, 0.0),
    "bent_cigar": (f_bent_cigar, 0.0), "discus": (f_discus, 0.0),
    "rosenbrock": (f_rosenbrock, 1.0), "ackley": (f_ackley, 0.0),
    "griewank": (f_griewank, 0.0), "rastrigin": (f_rastrigin, 0.0),
    "schwefel": (f_schwefel, _SCHWEFEL_ANCHOR), "zakharov": (f_zakharov, 0.0),
    "levy": (f_levy, 1.0), "weierstrass": (f_weierstrass, 0.0),
}


def rotation(dim: int, seed: int) -> Array:
    q, _ = np.linalg.qr(np.random.default_rng(seed).standard_normal((dim, dim)))
    return np.ascontiguousarray(q, dtype=np.float64)


@dataclass
class Spec:
    fid: int
    name: str
    bounds: Bounds
    func: VecFunc
    f_star: float
    category: str


def _shifted(name: str, fid: int, base_name: str, dim: int, bounds: Bounds,
             seed: int, rotate: bool) -> Spec:
    """base(R (x - o) + anchor): optimum at x = o, value 0. Never at the box
    centre -- that artefact is what this construction exists to remove."""
    lo, hi = bounds
    rng = np.random.default_rng(seed)
    span = hi - lo
    off = rng.uniform(lo + 0.15 * span, hi - 0.15 * span, size=dim)
    base, anchor = BASE[base_name]
    rot = rotation(dim, seed + 7919) if rotate else None

    def fn(X: Array) -> Array:
        Z = X - off
        if rot is not None:
            Z = Z @ rot.T
        return base(Z + anchor)

    return Spec(fid, name, bounds, fn, 0.0, "classic-12")


def classic_suite(dim: int, base_seed: int) -> List[Spec]:
    """12 classic landscapes. Every optimum is SHIFTED off the box centre.

    Names state what the function IS. The earlier suite used labels
    ("NoisySphere", "DynamicSphere", "NoisyRastrigin") whose noisy/dynamic
    flags were never read by any code path, and counted sphere twice; both
    defects are fixed here.
    """
    cfg = [
        ("CL_Sphere",       "sphere",     (-100.0, 100.0), False),
        ("CL_Elliptic",     "elliptic",   (-100.0, 100.0), False),
        ("CL_BentCigar",    "bent_cigar", (-100.0, 100.0), False),
        ("CL_Discus",       "discus",     (-100.0, 100.0), False),
        ("CL_RotElliptic",  "elliptic",   (-100.0, 100.0), True),
        ("CL_Rosenbrock",   "rosenbrock", (-100.0, 100.0), False),
        ("CL_Ackley",       "ackley",     (-32.768, 32.768), False),
        ("CL_Griewank",     "griewank",   (-100.0, 100.0), False),
        ("CL_Rastrigin",    "rastrigin",  (-5.12, 5.12), False),
        ("CL_Schwefel",     "schwefel",   (-500.0, 500.0), False),
        ("CL_Zakharov",     "zakharov",   (-100.0, 100.0), False),
        ("CL_Levy",         "levy",       (-100.0, 100.0), False),
    ]
    out = []
    for i, (name, base_name, bounds, rot) in enumerate(cfg):
        out.append(_shifted(name, i, base_name, dim, bounds,
                            base_seed + 1301 * (i + 1), rot))
    return out


_UNI = ["bent_cigar", "zakharov", "elliptic"]
_MULTI = ["rosenbrock", "rastrigin", "ackley", "griewank", "weierstrass",
          "schwefel", "levy"]
_HYB = [["zakharov", "rastrigin", "elliptic"],
        ["ackley", "rosenbrock", "griewank"],
        ["bent_cigar", "weierstrass", "schwefel", "rastrigin"],
        ["griewank", "levy", "elliptic"],
        ["rastrigin", "ackley", "discus", "rosenbrock"],
        ["schwefel", "griewank", "zakharov"],
        ["weierstrass", "levy", "bent_cigar", "ackley"],
        ["elliptic", "rastrigin", "griewank", "schwefel"],
        ["rosenbrock", "zakharov", "weierstrass"],
        ["ackley", "discus", "rastrigin", "levy"]]
_COMP = [["rastrigin", "ackley", "sphere"],
         ["griewank", "rosenbrock", "elliptic"],
         ["schwefel", "rastrigin", "weierstrass", "bent_cigar"],
         ["ackley", "levy", "zakharov"],
         ["elliptic", "griewank", "rastrigin", "discus"],
         ["rosenbrock", "schwefel", "ackley"],
         ["weierstrass", "zakharov", "griewank", "levy"],
         ["bent_cigar", "rastrigin", "ackley", "schwefel", "elliptic"],
         ["levy", "rosenbrock", "discus"],
         ["rastrigin", "weierstrass", "griewank", "ackley", "zakharov"]]


def _bridge_single(fid: int, base_name: str, dim: int, seed: int,
                   cat: str) -> Spec:
    rng = np.random.default_rng(seed)
    off = rng.uniform(-80.0, 80.0, size=dim)
    rot = rotation(dim, seed + 1)
    base, anchor = BASE[base_name]
    bias = 100.0 * fid

    def fn(X: Array) -> Array:
        return base((X - off) @ rot.T + anchor) + bias

    return Spec(fid, f"BR_F{fid:02d}_{base_name}", (-100.0, 100.0), fn,
                bias, cat)


def _bridge_hybrid(fid: int, subs: List[str], dim: int, seed: int) -> Spec:
    rng = np.random.default_rng(seed)
    off = rng.uniform(-80.0, 80.0, size=dim)
    rot = rotation(dim, seed + 1)
    perm = rng.permutation(dim)
    n = len(subs)
    props = rng.dirichlet(np.ones(n) * 4.0)
    sizes = np.maximum(1, (props * dim).astype(int))
    sizes[-1] = max(1, dim - int(sizes[:-1].sum()))
    sizes[-1] += dim - int(sizes.sum())
    edges = np.cumsum(np.concatenate([[0], sizes]))
    fns = [BASE[s] for s in subs]
    wts = rng.uniform(0.6, 1.0, size=n)
    bias = 100.0 * fid

    def fn(X: Array) -> Array:
        Z = ((X - off) @ rot.T)[:, perm]
        tot = np.zeros(X.shape[0])
        for k in range(n):
            part = Z[:, edges[k]:edges[k + 1]]
            if part.shape[1]:
                base_f, anchor = fns[k]
                tot += wts[k] * base_f(part + anchor)
        return tot + bias

    return Spec(fid, f"BR_F{fid:02d}_hyb_{'_'.join(s[:4] for s in subs)}",
                (-100.0, 100.0), fn, bias, "BR hybrid")


def _bridge_comp(fid: int, subs: List[str], dim: int, seed: int) -> Spec:
    rng = np.random.default_rng(seed)
    n = len(subs)
    offs = [rng.uniform(-70.0, 70.0, size=dim) for _ in range(n)]
    rots = [rotation(dim, seed + 100 * (k + 1)) for k in range(n)]
    sig = rng.uniform(10.0, 40.0, size=n)
    lam = rng.uniform(0.5, 2.0, size=n)
    sub_bias = np.arange(n, dtype=np.float64) * 100.0
    fns = [BASE[s] for s in subs]
    bias = 100.0 * fid
    eps = 1e-12

    def fn(X: Array) -> Array:
        N = X.shape[0]
        w = np.zeros((N, n))
        v = np.zeros((N, n))
        for k in range(n):
            d = X - offs[k]
            sq = np.sum(d * d, axis=1)
            w[:, k] = np.exp(-sq / (2.0 * dim * sig[k] ** 2)) / (np.sqrt(sq) + eps)
            base_f, anchor = fns[k]
            v[:, k] = base_f((d @ rots[k].T) / lam[k] + anchor) + sub_bias[k]
        bad = ~np.isfinite(w).all(axis=1) | (w.sum(axis=1) < eps)
        ws = w.sum(axis=1, keepdims=True)
        ws[ws < eps] = eps
        res = np.sum(w / ws * v, axis=1)
        if bad.any():
            near = np.argmin(np.stack(
                [np.sum((X[bad] - offs[k]) ** 2, axis=1) for k in range(n)],
                axis=1), axis=1)
            res[bad] = v[bad, near]
        return res + bias

    return Spec(fid, f"BR_F{fid:02d}_cmp_{'_'.join(s[:4] for s in subs)}",
                (-100.0, 100.0), fn, bias, "BR composition")


def bridge_suite(dim: int, base_seed: int) -> List[Spec]:
    """NOT official CEC2017 -- see BENCHMARK NOTES. Prefix is BR_ by design."""
    out: List[Spec] = []
    fid = 1
    for b in _UNI:
        out.append(_bridge_single(fid, b, dim, base_seed + 17 * fid,
                                  "BR unimodal")); fid += 1
    for b in _MULTI:
        out.append(_bridge_single(fid, b, dim, base_seed + 17 * fid,
                                  "BR multimodal")); fid += 1
    for r in _HYB:
        out.append(_bridge_hybrid(fid, r, dim, base_seed + 17 * fid)); fid += 1
    for r in _COMP:
        out.append(_bridge_comp(fid, r, dim, base_seed + 17 * fid)); fid += 1
    assert fid - 1 == 30
    return out


def _official_category(fid: int) -> str:
    """Official CEC2017 structure (technical report, Table 1-4)."""
    if fid <= 3:
        return "CEC unimodal"
    if fid <= 10:
        return "CEC multimodal"
    if fid <= 20:
        return "CEC hybrid"
    return "CEC composition"


def official_suite(dim: int) -> List[Spec]:
    """The REAL IEEE CEC2017 F1-F30, from the official input_data files.

    Verified by `verify_cec2017_data.py`: 180/180 files present, 30/30
    functions build for D = 10/30/50, and f(o_i) == 100*i exactly for
    F1-F10 at the official shift vectors -- the check that catches a
    misread shift, a transposed rotation or a wrong scale factor.

    Names are prefixed `CEC2017_`, and ONLY this path may use that prefix:
    the locally generated suite is `BR_` so that no table can confuse them.
    """
    from benchmarks.cec2017_official import official_cec2017_suite
    out: List[Spec] = []
    for o in official_cec2017_suite(dim, acknowledge_unverified=True):
        # OfficialSpec carries the optimum as `bias` (= 100*i by the CEC
        # definition); the verifier confirmed f(o_i) == 100*i exactly.
        out.append(Spec(o.id + 200, o.name, o.bounds, o.func,
                        float(o.bias), _official_category(o.id)))
    return out


def clustering_specs(dim: int) -> List[Spec]:
    """(Track B) real-data clustering cells at this dimension, if any.

    `f_star` is 0.0 because the global optimum of a k-means objective is not
    known, so `error()` returns the raw SSE. That is intentional: the 1e-8
    convention can never fire here, so no clustering cell can saturate into an
    information-free tie, unlike 35.5 % of the CEC2017 pairs.
    """
    from benchmarks.clustering import clustering_suite
    return [Spec(900 + i, s.name, (0.0, 1.0), s.func, 0.0, "clustering")
            for i, s in enumerate(clustering_suite(dim))]


def cec2011_specs(dim: int) -> List[Spec]:
    """CEC2011 real-world problems whose fixed dimension equals `dim`.

    Each problem is posed on the unit box [0,1]^D and mapped affinely onto
    its official bounds inside the function, because those bounds differ
    per variable while every algorithm here takes scalar bounds. Uniform
    sampling in the unit box is uniform in the official box. `f_star` is 0:
    the optima are unknown, so the reported value is the raw objective.
    """
    from benchmarks.cec2011_official import PROBLEMS, unit_box
    return [Spec(400 + pid, f"CEC2011_P{pid:02d}_{name}", (0.0, 1.0),
                 unit_box(pid), 0.0, CAT_RW)
            for pid, (name, D, _f, _b) in PROBLEMS.items() if D == dim]


def cec2020mech_specs(dim: int) -> List[Spec]:
    """CEC2020 mechanical-engineering problems whose D equals `dim`, on the
    unit box, with the protocol's scalar fitness (feasible -> f, infeasible
    -> C_k + mean violation). f_star 0: raw values are reported."""
    from benchmarks.cec2020rw_official import PROBLEMS, unit_box
    return [Spec(500 + pid, f"CEC2020_RC{pid:02d}_{p[0]}", (0.0, 1.0),
                 unit_box(pid), 0.0, CAT_RWC)
            for pid, p in PROBLEMS.items() if p[1] == dim]


def build_registry(dim: int, base_seed: int = BENCH_SEED,
                   include_bridge: bool = True,
                   cec: str = "bridge") -> List[Spec]:
    """`cec`: 'bridge' = locally generated BR_* ; 'official' = real CEC2017 ;
    'both' = classics + official (the configuration used for publication) ;
    'clustering' = the real-data clustering suite ONLY (track B)."""
    if cec == "clustering":
        return clustering_specs(dim)
    if cec == "cec2011":
        return cec2011_specs(dim)
    if cec == "cec2020mech":
        return cec2020mech_specs(dim)
    specs = classic_suite(dim, base_seed)
    if cec == "official" or cec == "both":
        specs.extend(official_suite(dim))
        if cec == "both" and include_bridge:
            for s in bridge_suite(dim, base_seed + 500_000):
                specs.append(replace(s, fid=s.fid + 100))
    elif include_bridge:
        for s in bridge_suite(dim, base_seed + 500_000):
            specs.append(replace(s, fid=s.fid + 100))
    return specs


# ===========================================================================
# 3. SHARED DE PRIMITIVES  [L-SHADE]
# ===========================================================================
def init_pop(n: int, dim: int, bounds: Bounds, rng) -> Array:
    lo, hi = bounds
    return rng.uniform(lo, hi, size=(n, dim))


def repair_midpoint(V: Array, X: Array, bounds: Bounds) -> Array:
    """[L-SHADE] Eq. 4: reflect a violating coordinate to the midpoint
    between the bound and the parent. Preserves feasibility without the
    diversity collapse that hard clipping to the bound causes."""
    lo, hi = bounds
    V = V.copy()
    below = V < lo
    above = V > hi
    if below.any():
        V[below] = (lo + X[below]) / 2.0
    if above.any():
        V[above] = (hi + X[above]) / 2.0
    return V


def binomial_crossover(V: Array, X: Array, CR: Array, rng) -> Array:
    """[L-SHADE] Eq. 5. jrand guarantees at least one inherited coordinate."""
    n, d = X.shape
    mask = rng.random((n, d)) <= CR[:, None]
    jr = rng.integers(0, d, size=n)
    mask[np.arange(n), jr] = True
    return np.where(mask, V, X)


def lehmer_weighted(S: Array, dF: Array) -> float:
    """[L-SHADE] Eq. 7-9. mean_WL(S) = sum(w S^2)/sum(w S), w = dF/sum(dF).

    Cauchy-Schwarz with sum(w)=1 gives mean_WL >= sum(w S): the estimate is
    biased UPWARD relative to the weighted arithmetic mean. For F that bias
    is protective -- small F converges fast but prematurely -- which is why
    the Lehmer mean and not a plain mean is the published choice.
    """
    tot = float(dF.sum())
    w = (dF / tot) if tot > 0 else np.full(len(S), 1.0 / max(1, len(S)))
    num = float(np.sum(w * S * S))
    den = float(np.sum(w * S))
    return num / den if den > 1e-300 else float(np.mean(S))


def lpsr(n_init: int, n_min: int, fes: int, max_fes: int) -> int:
    """[L-SHADE] Eq. 10. n_min = 4 is not a tuned constant: current-to-pbest/1
    needs exactly x_i, x_pbest, x_r1, x_r2."""
    return int(round((n_min - n_init) / max_fes * fes + n_init))


class Memory:
    """Success-history memories for CR, F and (for MDAC/LSMA) beta.
    `bot` implements [L-SHADE]'s terminal value: once M_CR is locked, CR=0
    for the rest of the run ("change one parameter at a time"), which the
    source reports as effective on multimodal landscapes."""

    def __init__(self, H: int, keys: Sequence[str], init: float = 0.5):
        self.H = int(H)
        self.keys = list(keys)
        self.M = {k: np.full(self.H, init) for k in self.keys}
        self.bot = {k: np.zeros(self.H, dtype=bool) for k in self.keys}
        self.k = 0

    def draw_index(self, n: int, rng) -> Array:
        return rng.integers(0, self.H, size=n)

    def draw_normal(self, key: str, idx: Array, rng, sd: float = 0.1) -> Array:
        m = self.M[key][idx]
        out = rng.normal(m, sd)
        out = np.clip(out, 0.0, 1.0)
        out[self.bot[key][idx]] = 0.0
        return out

    def draw_cauchy(self, key: str, idx: Array, rng, sd: float = 0.1) -> Array:
        m = self.M[key][idx]
        out = m + sd * rng.standard_cauchy(size=len(idx))
        for _ in range(12):
            bad = out <= 0.0
            if not bad.any():
                break
            out[bad] = m[bad] + sd * rng.standard_cauchy(size=int(bad.sum()))
        return np.clip(out, 1e-12, 1.0)

    def update(self, succ: Dict[str, Array], dF: Array) -> None:
        if len(dF) == 0:
            return
        for key in self.keys:
            S = succ.get(key)
            if S is None or len(S) == 0:
                continue
            if self.bot[key][self.k] or float(np.max(S)) == 0.0:
                if key == "CR":
                    self.bot[key][self.k] = True
                    continue
            self.M[key][self.k] = lehmer_weighted(np.asarray(S), dF)
        self.k = (self.k + 1) % self.H


def pbest_index(fitness: Array, p_rate: float, n: int, rng) -> Array:
    """Uniform draw from the top p_rate fraction -- [L-SHADE]'s rule."""
    npop = len(fitness)
    top = max(2, int(round(p_rate * npop)))
    pool = np.argsort(fitness)[:top]
    return pool[rng.integers(0, top, size=n)]


def pbest_index_diverse(fitness: Array, X: Array, used: Optional[Array],
                        p_rate: float, n: int, nu: float, rng) -> Array:
    """(C1) OURS -- diversity-regularized exploitation targeting.

    Same elite pool as [L-SHADE], but the draw is reweighted by how far each
    elite sits from the set already being exploited:

        d_j   = min_{a in used} ||x_j - a||          (0 if `used` is empty)
        s_j   = (d_j - mean d) / (std d + eps)       per-pool z-score
        P(j) propto exp(nu * s_j)

    nu = 0 recovers [L-SHADE] exactly. The z-score is what makes nu
    dimensionless and therefore transferable across landscapes whose fitness
    and distance scales differ by tens of orders of magnitude -- without it a
    novelty weight calibrated early in a run silently dominates once fitness
    has collapsed. Costs ZERO objective evaluations: both the fitness ranking
    and the positions were already paid for by this generation.
    """
    npop = len(fitness)
    top = max(2, int(round(p_rate * npop)))
    pool = np.argsort(fitness)[:top]
    if nu <= 0.0 or used is None or len(used) == 0 or top < 2:
        return pool[rng.integers(0, top, size=n)]
    d = np.sqrt(((X[pool][:, None, :] - used[None, :, :]) ** 2).sum(axis=2)).min(axis=1)
    sd = float(d.std())
    s = (d - float(d.mean())) / (sd + 1e-12) if sd > 1e-12 else np.zeros(top)
    w = np.exp(np.clip(nu * s, -30.0, 30.0))
    w /= w.sum()
    return pool[rng.choice(top, size=n, p=w)]


# ===========================================================================
# 4. LOCAL OPERATOR -- (1+lambda)-CMA-ES, Cholesky  [1+lCMA]
# ===========================================================================
def cmaes_local(ev: Evaluator, x0: Array, f0: float, bounds: Bounds,
                sigma0: float, seed: int, sigma_min: float = 1e-18,
                restarts: bool = True, restart_scale: float = 0.05) -> None:
    """(1+lambda)-CMA-ES with rank-one Cholesky covariance update.

    Rotation invariant, so it follows a valley that is not axis aligned --
    the failure mode an axis-aligned pattern search cannot repair. lambda
    offspring per iteration (not 1+1) so a whole generation goes through
    `evaluate_batch` in one vectorised call. Constants per Igel, Suttorp &
    Hansen. Consumes the budget it is given; `ev` tracks the best itself.
    """
    lo, hi = bounds
    d = x0.shape[0]
    rng = np.random.default_rng(seed)
    lam = 4 + int(math.floor(3.0 * math.log(d)))
    p_target = 1.0 / (5.0 + math.sqrt(lam) / 2.0)
    damp = 1.0 + d / (2.0 * lam)
    c_p = p_target * lam / (2.0 + p_target * lam)
    c_c = 2.0 / (d + 2.0)
    c_cov = 2.0 / (d ** 2 + 6.0)
    p_thresh = 0.44

    x, f = x0.copy(), float(f0)
    bx, bf = x0.copy(), float(f0)
    C = np.eye(d)
    A = np.eye(d)
    p_c = np.zeros(d)
    p_succ = p_target
    sigma = max(float(sigma0), sigma_min)

    try:
        while True:
            if sigma <= sigma_min:
                if not restarts:
                    return
                sigma = restart_scale * (hi - lo)
                x = np.clip(bx + rng.normal(0.0, sigma, d), lo, hi)
                f = float(ev.evaluate_batch(x.reshape(1, -1))[0])
                C, A, p_c, p_succ = np.eye(d), np.eye(d), np.zeros(d), p_target
                if f < bf:
                    bf, bx = f, x.copy()
                continue
            Z = rng.standard_normal((lam, d))
            Xo = np.clip(x + sigma * (Z @ A.T), lo, hi)
            vals = ev.evaluate_batch(Xo)
            p_succ = (1.0 - c_p) * p_succ + c_p * float(np.mean(vals < f))
            sigma *= _safe_exp((p_succ - p_target) / (damp * (1.0 - p_target)))
            sigma = min(sigma, float(hi - lo))
            if not math.isfinite(sigma) or sigma <= 0.0:
                sigma = sigma_min
            j = int(np.argmin(vals))
            if vals[j] < f:
                z = Z[j]
                x, f = Xo[j].copy(), float(vals[j])
                if f < bf:
                    bf, bx = f, x.copy()
                if p_succ < p_thresh:
                    p_c = (1.0 - c_c) * p_c + math.sqrt(c_c * (2.0 - c_c)) * (A @ z)
                    C = (1.0 - c_cov) * C + c_cov * np.outer(p_c, p_c)
                else:
                    p_c = (1.0 - c_c) * p_c
                    C = (1.0 - c_cov) * C + c_cov * (
                        np.outer(p_c, p_c) + c_c * (2.0 - c_c) * C)
                C = 0.5 * (C + C.T)
                try:
                    A = np.linalg.cholesky(C)
                except np.linalg.LinAlgError:
                    C, A, p_c = np.eye(d), np.eye(d), np.zeros(d)
    except BudgetExhausted:
        return


class CMAState:
    """A proper (mu/mu_w, lambda)-CMA-ES distribution usable as ONE offspring
    generator inside a larger population loop  [CMAES].

    Why this class exists. The first version of this file updated the local
    generator with a rank-one-only hack: an unweighted mean of the successful
    offspring, no weighted recombination, no rank-mu term, no cumulative
    step-size adaptation, and a mean that jumped to the global best every
    generation. The ablation then reported "FCP co-allocation hurts"
    (MDAC-noFCP ranked above MDAC in that pilot) -- but that measurement
    cannot distinguish "co-allocation is a bad idea" from "our local
    generator is bad", and [LSPACMA] reaches 4th place in the same pilot
    using a real CMA-ES. Concluding against co-allocation from a crippled
    operator would have been a false negative, which is as dishonest as a
    false positive.

    Varying lambda. The allocation rule hands this generator k individuals
    per generation and k changes every generation, so the recombination
    weights and every constant derived from mu_eff are recomputed for the
    actual k (cached). This is a standard (mu/mu_w, lambda) update with a
    non-constant lambda, which is also what [LSPACMA] does implicitly when
    FCP reassigns population share.

    y-vectors are recomputed from the ACTUALLY EVALUATED point, so the
    update stays consistent even though [LSPACMA] applies DE crossover to
    the CMA offspring (their Sec. II-D) and the evaluated point is therefore
    no longer a raw sample from this distribution.
    """

    def __init__(self, dim: int, m0: Array, sigma0: float, bounds: Bounds):
        self.N = int(dim)
        self.lo, self.hi = bounds
        self.m = np.asarray(m0, dtype=np.float64).copy()
        self.sigma = float(sigma0)
        self.C = np.eye(self.N)
        self.B = np.eye(self.N)
        self.D = np.ones(self.N)
        self.invsqrtC = np.eye(self.N)
        self.ps = np.zeros(self.N)
        self.pc = np.zeros(self.N)
        self.g = 0
        self.eig_at = 0
        self.chiN = (math.sqrt(self.N)
                     * (1.0 - 1.0 / (4.0 * self.N) + 1.0 / (21.0 * self.N ** 2)))
        self._cache: Dict[int, Tuple] = {}

    def _consts(self, k: int) -> Tuple:
        if k in self._cache:
            return self._cache[k]
        N = self.N
        mu = max(1, k // 2)
        wr = np.log(mu + 0.5) - np.log(np.arange(1, mu + 1))
        w = wr / wr.sum()
        mueff = float(1.0 / np.sum(w ** 2))
        cs = (mueff + 2.0) / (N + mueff + 5.0)
        ds = 1.0 + 2.0 * max(0.0, math.sqrt((mueff - 1.0) / (N + 1.0)) - 1.0) + cs
        cc = (4.0 + mueff / N) / (N + 4.0 + 2.0 * mueff / N)
        c1 = 2.0 / ((N + 1.3) ** 2 + mueff)
        cmu = min(1.0 - c1,
                  2.0 * (mueff - 2.0 + 1.0 / mueff) / ((N + 2.0) ** 2 + mueff))
        out = (mu, w, mueff, cs, ds, cc, c1, cmu)
        self._cache[k] = out
        return out

    def sample(self, k: int, rng) -> Array:
        """k offspring from N(m, sigma^2 C), clipped to the box."""
        Z = rng.standard_normal((k, self.N))
        Y = Z @ (self.B * self.D).T
        return np.clip(self.m + self.sigma * Y, self.lo, self.hi)

    def restart(self, m0: Array, sigma0: float, C0: Optional[Array] = None) -> None:
        """Re-seed the distribution at a new point.

        (A2) Targets the BR hybrid block. A hybrid landscape partitions the
        coordinates into sub-blocks with different structure, so one basin is
        not enough: a sub-distribution that has collapsed into the wrong one
        must be able to move, not merely shrink. Standalone CMA-ES needs IPOP
        restarts to consume its budget at all (measured here), and the same
        argument applies to a co-allocated sub-distribution.
        """
        self.m = np.clip(np.asarray(m0, dtype=np.float64).copy(), self.lo, self.hi)
        self.sigma = float(max(sigma0, 1e-18))
        self.ps = np.zeros(self.N)
        self.pc = np.zeros(self.N)
        self.g = 0
        self.eig_at = 0
        if C0 is None:
            self.C = np.eye(self.N)
            self.B = np.eye(self.N)
            self.D = np.ones(self.N)
            self.invsqrtC = np.eye(self.N)
            return
        Cs = 0.5 * (C0 + C0.T)
        try:
            Dv, Bn = np.linalg.eigh(Cs)
            if (not np.all(np.isfinite(Dv))) or Dv.min() <= 0 \
                    or Dv.max() / Dv.min() > 1e12:
                raise np.linalg.LinAlgError
            self.C, self.B = Cs, Bn
            self.D = np.sqrt(Dv)
            self.invsqrtC = Bn @ np.diag(1.0 / self.D) @ Bn.T
        except np.linalg.LinAlgError:
            self.C = np.eye(self.N)
            self.B = np.eye(self.N)
            self.D = np.ones(self.N)
            self.invsqrtC = np.eye(self.N)

    @property
    def stalled(self) -> bool:
        return self.sigma <= 1e-16

    def tell(self, X: Array, f: Array) -> None:
        """Full CMA update from the evaluated offspring (any k >= 2)."""
        k = X.shape[0]
        if k < 2 or self.sigma <= 0.0 or not math.isfinite(self.sigma):
            return
        mu, w, mueff, cs, ds, cc, c1, cmu = self._consts(k)
        order = np.argsort(f)[:mu]
        Y = (X[order] - self.m) / self.sigma           # from the real point
        if not np.all(np.isfinite(Y)):
            return
        yw = np.sum(w[:, None] * Y, axis=0)
        self.m = np.clip(self.m + self.sigma * yw, self.lo, self.hi)
        self.g += 1
        self.ps = ((1 - cs) * self.ps
                   + math.sqrt(cs * (2 - cs) * mueff) * (self.invsqrtC @ yw))
        hsig = (float(np.sum(self.ps ** 2))
                / (1 - (1 - cs) ** (2 * self.g)) / self.N
                < 2.0 + 4.0 / (self.N + 1.0))
        self.pc = ((1 - cc) * self.pc
                   + (math.sqrt(cc * (2 - cc) * mueff) * yw if hsig else 0.0))
        dh = (0.0 if hsig else 1.0) * cc * (2 - cc)
        rank_mu = np.einsum("i,ij,ik->jk", w, Y, Y)
        self.C = ((1 - c1 - cmu + c1 * dh) * self.C
                  + c1 * np.outer(self.pc, self.pc) + cmu * rank_mu)
        self.sigma *= _safe_exp((cs / ds)
                                * (float(np.linalg.norm(self.ps)) / self.chiN - 1.0))
        if not math.isfinite(self.sigma):
            self.sigma = 1e-18
        self.sigma = float(np.clip(self.sigma, 1e-18, self.hi - self.lo))

        if self.g - self.eig_at > 1.0 / (10.0 * self.N * (c1 + cmu) + 1e-12):
            self.eig_at = self.g
            self.C = 0.5 * (self.C + self.C.T)
            ok = bool(np.all(np.isfinite(self.C)))
            if ok:
                try:
                    Dv, Bn = np.linalg.eigh(self.C)
                    if Dv.max() / max(Dv.min(), 1e-300) > 1e14 or Dv.min() <= 0:
                        ok = False
                    else:
                        self.B = Bn
                        self.D = np.sqrt(Dv)
                        self.invsqrtC = Bn @ np.diag(1.0 / self.D) @ Bn.T
                except np.linalg.LinAlgError:
                    ok = False
            if not ok:
                self.C = np.eye(self.N)
                self.B = np.eye(self.N)
                self.D = np.ones(self.N)
                self.invsqrtC = np.eye(self.N)
                self.pc = np.zeros(self.N)
                self.ps = np.zeros(self.N)


def shrinkage_covariance(points: Array, dim: int) -> Tuple[Array, float]:
    """(A5) Diagonal-shrinkage covariance of the current search state.

    Returns (C0 normalised to unit mean eigenvalue, sigma0). Regularised
    toward the identity by tau = dim/(dim+k), k = sample count: parameter
    free, dimension-agnostic, identical on every landscape, and it costs no
    objective evaluations because the points are already paid for.

    Motivation specific to the BR hybrid block: a hybrid function's
    coordinates split into sub-blocks of unequal scale, and the population's
    per-coordinate spread already encodes that split. Starting the
    sub-distribution from the identity throws it away and makes the operator
    relearn an O(D^2) metric from scratch after every restart.

    NOTE: an equivalent seeding was tested in our earlier work and REVERTED
    (18 better / 27 worse, no stratum significant). That test was inside a
    terminal block entered at t=0.70, where there was no budget left to
    recover from a bad seed. Here the sub-distribution runs for the whole
    run and restarts repeatedly, so the setting is different -- which is why
    this is a flag under test (`cma_seed_cov`) and not a silent default.
    """
    k = points.shape[0]
    if k < 2:
        return np.eye(dim), 0.0
    Xc = points - points.mean(axis=0, keepdims=True)
    S = (Xc.T @ Xc) / (k - 1)
    tr = float(np.trace(S))
    if not np.isfinite(tr) or tr <= 0.0:
        return np.eye(dim), 0.0
    sigma0 = math.sqrt(tr / dim)
    tau = dim / (dim + k)
    C0 = (1.0 - tau) * (S * (dim / tr)) + tau * np.eye(dim)
    return 0.5 * (C0 + C0.T), sigma0


# ===========================================================================
# 5. COMPETITORS
# ===========================================================================
def alg_de(ev, dim, bounds, seed, pop_size=None, F=0.5, CR=0.9):
    """DE/rand/1/bin -- the floor baseline."""
    rng = np.random.default_rng(seed)
    n = pop_size or min(100, 10 * dim)
    X = init_pop(n, dim, bounds, rng)
    try:
        fit = ev.evaluate_batch(X)
        while True:
            idx = np.array([rng.choice(np.delete(np.arange(n), i), 3,
                                       replace=False) for i in range(n)])
            V = X[idx[:, 0]] + F * (X[idx[:, 1]] - X[idx[:, 2]])
            V = repair_midpoint(V, X, bounds)
            U = binomial_crossover(V, X, np.full(n, CR), rng)
            fu = ev.evaluate_batch(U)
            imp = fu <= fit
            X[imp], fit[imp] = U[imp], fu[imp]
    except BudgetExhausted:
        pass
    return _finish("DE", ev)


def alg_pso(ev, dim, bounds, seed, pop_size=None, w=0.72, c1=1.49, c2=1.49):
    """Standard constricted PSO -- the second floor baseline."""
    rng = np.random.default_rng(seed)
    lo, hi = bounds
    n = pop_size or min(100, 10 * dim)
    X = init_pop(n, dim, bounds, rng)
    V = np.zeros_like(X)
    vmax = 0.2 * (hi - lo)
    try:
        fit = ev.evaluate_batch(X)
        P, pf = X.copy(), fit.copy()
        g = X[int(np.argmin(fit))].copy()
        while True:
            r1, r2 = rng.random((n, dim)), rng.random((n, dim))
            V = np.clip(w * V + c1 * r1 * (P - X) + c2 * r2 * (g - X),
                        -vmax, vmax)
            X = np.clip(X + V, lo, hi)
            fit = ev.evaluate_batch(X)
            imp = fit < pf
            P[imp], pf[imp] = X[imp], fit[imp]
            g = P[int(np.argmin(pf))].copy()
    except BudgetExhausted:
        pass
    return _finish("PSO", ev)


def alg_cmaes(ev, dim, bounds, seed, pop_size=None, sigma_frac=0.3):
    """(mu/mu_w, lambda)-CMA-ES  [CMAES].

    THE critical baseline. Our earlier study omitted it while its measured
    advantage came from a CMA-ES local stage -- the single most damaging
    omission a reviewer could find. Implemented per Hansen's tutorial:
    rank-mu + rank-one covariance update, cumulative step-size adaptation,
    lazy eigendecomposition for C^{-1/2}.
    """
    rng = np.random.default_rng(seed)
    lo, hi = bounds
    N = dim
    lam = pop_size or (4 + int(math.floor(3.0 * math.log(N))))
    mu = lam // 2
    wr = np.log(mu + 0.5) - np.log(np.arange(1, mu + 1))
    w = wr / wr.sum()
    mueff = 1.0 / np.sum(w ** 2)
    cs = (mueff + 2.0) / (N + mueff + 5.0)
    ds = 1.0 + 2.0 * max(0.0, math.sqrt((mueff - 1.0) / (N + 1.0)) - 1.0) + cs
    cc = (4.0 + mueff / N) / (N + 4.0 + 2.0 * mueff / N)
    c1 = 2.0 / ((N + 1.3) ** 2 + mueff)
    cmu = min(1.0 - c1, 2.0 * (mueff - 2.0 + 1.0 / mueff) / ((N + 2.0) ** 2 + mueff))
    chiN = math.sqrt(N) * (1.0 - 1.0 / (4.0 * N) + 1.0 / (21.0 * N * N))

    # IPOP-style restarts: on a stall the run restarts with a doubled
    # population from a fresh mean. Without this CMA-ES stops once sigma
    # collapses and leaves budget unspent, which would be a budget-parity
    # violation against every other algorithm here -- and would quietly make
    # the single most important baseline weaker than it is.
    restarts = 0
    try:
        while True:
            m = rng.uniform(lo, hi, size=N)
            sigma = sigma_frac * (hi - lo)
            C = np.eye(N)
            B, Dd = np.eye(N), np.ones(N)
            invsqrtC = np.eye(N)
            ps, pc = np.zeros(N), np.zeros(N)
            g, eig_at = 0, 0
            stall = 0
            f_prev = np.inf

            while True:
                Z = rng.standard_normal((lam, N))
                Y = Z @ (B * Dd).T
                Xo = np.clip(m + sigma * Y, lo, hi)
                vals = ev.evaluate_batch(Xo)
                order = np.argsort(vals)[:mu]
                yw = np.sum(w[:, None] * Y[order], axis=0)
                m = np.clip(m + sigma * yw, lo, hi)
                ps = (1 - cs) * ps + math.sqrt(cs * (2 - cs) * mueff) * (invsqrtC @ yw)
                g += 1
                hsig = (np.sum(ps ** 2) / (1 - (1 - cs) ** (2 * g)) / N
                        < 2.0 + 4.0 / (N + 1.0))
                pc = ((1 - cc) * pc
                      + (math.sqrt(cc * (2 - cc) * mueff) * yw if hsig else 0.0))
                dh = (0.0 if hsig else 1.0) * cc * (2 - cc)
                rank_mu = np.einsum("i,ij,ik->jk", w, Y[order], Y[order])
                C = ((1 - c1 - cmu + c1 * dh) * C + c1 * np.outer(pc, pc)
                     + cmu * rank_mu)
                sigma *= _safe_exp((cs / ds) * (np.linalg.norm(ps) / chiN - 1.0))
                if not math.isfinite(sigma):
                    sigma = 1e-18
                sigma = float(np.clip(sigma, 1e-18, hi - lo))

                if g - eig_at > lam / (10.0 * N * (c1 + cmu) + 1e-12):
                    eig_at = g
                    C = 0.5 * (C + C.T)
                    ok = np.all(np.isfinite(C))
                    if ok:
                        try:
                            Dv, B = np.linalg.eigh(C)
                            if Dv.max() / max(Dv.min(), 1e-300) > 1e14:
                                ok = False
                            else:
                                Dd = np.sqrt(np.maximum(Dv, 1e-30))
                                invsqrtC = B @ np.diag(1.0 / Dd) @ B.T
                        except np.linalg.LinAlgError:
                            ok = False
                    if not ok:
                        C, B, Dd = np.eye(N), np.eye(N), np.ones(N)
                        invsqrtC = np.eye(N)
                        pc, ps = np.zeros(N), np.zeros(N)

                fb = float(vals[order[0]])
                stall = stall + 1 if fb >= f_prev - 1e-300 else 0
                f_prev = min(f_prev, fb)
                if sigma <= 1e-17 or stall > 40 + 10 * N // lam:
                    break               # -> outer loop restarts
            restarts += 1
            lam = min(lam * 2, 4096)    # IPOP
            mu = max(1, lam // 2)
            wr = np.log(mu + 0.5) - np.log(np.arange(1, mu + 1))
            w = wr / wr.sum()
            mueff = 1.0 / np.sum(w ** 2)
    except BudgetExhausted:
        pass
    return _finish("CMA-ES", ev, {"restarts": restarts})


def _shade_core(ev, dim, bounds, seed, name, *, momentum: bool,
                spa: bool, fcp: bool, metropolis: bool,
                diversity_nu: float = 0.0,
                hm_on_success: bool = True,
                mom_archive: bool = False,
                alloc: str = "adaptive",
                cma_restart: bool = False, cma_seed_cov: bool = False,
                cma_stall_gens: int = 12, cma_cr_full: bool = False,
                cma_cr_mode: str = "shared", cr_loc_init: float = 0.9,
                local_gen: str = "cma",
                H: int = 6, r_ninit: float = 18.0, r_arc: float = 2.6,
                p_rate: float = 0.11, n_L: int = 12, T0_frac: float = 0.10,
                fcp_c: float = 0.8, fcp_lo: float = 0.2, fcp_hi: float = 0.8,
                widen_rho: float = 0.85, widen_floor: float = 0.03,
                handover: float = 0.70) -> AlgResult:
    """One parameterised SHADE-family loop. Every competitor in the
    L-SHADE lineage and MDAC itself are configurations of this function, so
    no comparison can be confounded by implementation differences in the
    shared parts -- only by the flags under test.

        momentum      [LSMA]     momentum-to-pbest/1 instead of
                                 current-to-pbest/1
        spa           [LSPACMA]  semi-parameter adaptation of F
        fcp           [LSPACMA]  two offspring generators over one
                                 population, allocation learned from
                                 measured improvement
        metropolis    [AGDESA]   stagnation-gated Boltzmann acceptance
        diversity_nu  (C1) OURS  novelty-reweighted pbest draw
        alloc         (C2) OURS  'adaptive' | 'fixed' | 'random' | 'terminal'
    """
    rng = np.random.default_rng(seed)
    lo, hi = bounds
    max_fes = ev.max_fes
    n_init = max(8, int(round(r_ninit * dim)))
    n_min = 4
    n = n_init

    keys = ["CR", "F"] + (["beta"] if momentum else [])
    mem = Memory(H, keys)
    # (A7) OURS -- a SECOND crossover-rate memory, owned by the metric
    # generator alone.
    #
    # A6 forced CR=1 for the metric generator to protect rotational
    # invariance. Measured on the official CEC2017 suite that is right where
    # the landscape is fully rotated and wrong where it is separable, and the
    # two effects are both large:
    #     CEC2017 F5 Rastrigin (ROTATED)      A12 0.914  -- we win big
    #     CL_Rastrigin         (axis-aligned) A12 0.030  -- we lose big
    #     CL_Schwefel          (axis-aligned) A12 0.002
    #     CEC hybrid (partially separable)    A12 0.424, D=10 0.238
    # Same base function, opposite outcome, the only difference being
    # rotation. A fixed rule therefore cannot be correct for both.
    #
    # So CR for the metric generator stops being a constant and becomes a
    # learned quantity, adapted by exactly the mechanism [L-SHADE] already
    # validates: draw from a memory, keep what succeeded, update by the
    # weighted Lehmer mean. Initialised HIGH (0.9) because the rotated case
    # is where this generator earns its place; it falls on its own where
    # coordinate-wise moves do better. Costs no objective evaluations and
    # adds one parameter (the initial value), not a schedule.
    mem_loc = Memory(H, ["CR"], init=cr_loc_init)
    X = init_pop(n, dim, bounds, rng)

    try:
        fit = ev.evaluate_batch(X)
    except BudgetExhausted:
        return _finish(name, ev)

    # [L-SHADE] external archive, held in a PREALLOCATED array instead of a
    # Python list that was re-stacked with `np.vstack` every generation.
    #
    # Contents and insertion ORDER are identical to the list version, so the
    # RNG stream and therefore the search are unchanged. VERIFIED: 8 of 8
    # replayed cells reproduce the recorded `error_raw` exactly.
    #
    # HONEST NOTE ON THE SPEEDUP: a cProfile run attributed ~21 % of runtime
    # to `vstack`/`atleast_2d` here, and single-shot before/after timings
    # suggested ~1.17x. A careful repeated measurement (7 seeds, quiet
    # machine) then gave an F5/D=50 ratio to jSO of 2.530 +/- 0.161 -- the
    # noise swamps the claimed gain, and the single-shot numbers (2.36 before,
    # 2.02 after) both sit inside it. **The speedup is NOT established.**
    # The change is kept because it is bit-identical and strictly does less
    # work, not because a benefit was measured. No speedup is claimed from it
    # anywhere in the paper.
    arch_cap = max(1, int(round(r_arc * n_init)))
    arch = np.empty((arch_cap, dim), dtype=np.float64)
    arch_n = 0
    Hm = np.zeros_like(X)                 # [LSMA] historical movement
    last_imp = np.zeros(n, dtype=np.int64)   # [AGDESA] L[i]
    gen = 0
    pi_global = 0.5                       # [LSPACMA] M_fcp
    dom_sign, dom_run = 0, 0              # (A1) confidence-widening state
    cma_last_imp = 0
    n_cma_restart = 0
    n_local_acc = 0
    n_metro_acc = 0

    # Local offspring generator: a REAL (mu/mu_w, lambda)-CMA-ES, seeded at
    # the current incumbent with a step size scaled to the population radius.
    spread0 = float(np.sqrt(np.mean(np.sum((X - X.mean(axis=0)) ** 2, axis=1))))
    cma = CMAState(dim, X[int(np.argmin(fit))],
                   max(0.3 * spread0 / max(1.0, math.sqrt(dim)), 1e-12),
                   bounds)

    try:
        while True:
            if ev.exhausted():
                break
            # ---- terminal allocation: hand the tail to the local operator
            if alloc == "terminal" and ev.fes >= handover * max_fes:
                bx = ev.best_x if ev.best_x is not None else X[int(np.argmin(fit))]
                spread = float(np.sqrt(np.mean(np.sum(
                    (X - X.mean(axis=0)) ** 2, axis=1)))) if len(X) > 1 else 1e-3
                cmaes_local(ev, bx, ev.best_fitness, bounds,
                            sigma0=max(spread, 1e-18), seed=seed, restarts=True)
                break

            gen += 1
            # ---- LPSR  [L-SHADE] Eq. 10
            n_t = max(n_min, lpsr(n_init, n_min, ev.fes, max_fes))
            if n_t < n:
                keep = np.argsort(fit)[:n_t]
                X, fit, Hm, last_imp = X[keep], fit[keep], Hm[keep], last_imp[keep]
                n = n_t
                cap = max(1, int(round(r_arc * n)))
                if arch_n > cap:
                    sel = rng.choice(arch_n, cap, replace=False)
                    arch[:cap] = arch[sel]          # fancy index copies first
                    arch_n = cap
            if n < n_min:
                break

            t = ev.fes / max_fes
            idx = mem.draw_index(n, rng)
            CR = mem.draw_normal("CR", idx, rng)
            if spa and t < 0.5:
                # [LSPACMA] first half: adapt CR only, draw F uniformly.
                F = 0.45 + 0.1 * rng.random(n)
            else:
                F = mem.draw_cauchy("F", idx, rng)
            beta = (mem.draw_normal("beta", idx, rng) if momentum
                    else np.zeros(n))

            # ---- operator assignment  [LSPACMA] FCP / (C2) alloc rule
            if fcp and alloc != "terminal":
                if alloc == "fixed":
                    pi_use = 0.5
                elif alloc == "random":
                    pi_use = float(rng.uniform(fcp_lo, fcp_hi))
                elif alloc == "dimaware":
                    # (A8) the floor binds from generation 1, not only once
                    # improvement evidence has started updating pi_global.
                    pi_use = max(pi_global,
                                 0.65 * max(0.0, 30.0 - dim) / 20.0)
                else:
                    pi_use = pi_global
                use_local = rng.random(n) >= pi_use
            else:
                use_local = np.zeros(n, dtype=bool)
            # guarantee the CMA update is well defined when it is used at all
            if use_local.sum() == 1:
                use_local[int(np.argmax(use_local))] = False

            # ---- donor selection: (C1) diversity-regularized when nu>0
            used = arch[:arch_n] if arch_n else None
            pb = pbest_index_diverse(fit, X, used, p_rate, n,
                                     diversity_nu * (1.0 - t), rng)
            pool = np.concatenate((X, used), axis=0) if used is not None else X

            r1 = rng.integers(0, n, size=n)
            r2 = rng.integers(0, pool.shape[0], size=n)

            if momentum:
                # [LSMA] Eq. 21: ORDER the pair so the difference vector is a
                # descent estimate. For a pair close together,
                #   f(x_b1) < f(x_b2)  =>  <grad f, x_b1 - x_b2> < 0,
                # so the conditioned difference has positive projection on
                # -grad f. Unordered (plain DE/rand) it has zero mean and
                # therefore carries no descent information at all. It costs
                # nothing: the fitnesses were already paid for.
                a = rng.integers(0, n, size=n)
                b = rng.integers(0, n, size=n)
                swap = fit[a] > fit[b]
                b1 = np.where(swap, b, a)
                b2 = np.where(swap, a, b)
                if mom_archive and pool.shape[0] > n:
                    # (A3) OURS -- restore the external archive in the
                    # difference term. [LSMA] Eq. 21 draws BOTH x_b1 and
                    # x_b2 from P, which silently discards the P-union-A
                    # difference that JADE/L-SHADE rely on for diversity
                    # (their Eq. 3 draws x_r2 from P u A). That omission is
                    # the leading candidate explanation for the measured
                    # harm of momentum here (A12 0.455 on the classics;
                    # BR_* numbers withheld), so it is tested, not assumed:
                    # the second point is drawn from P u A and the pair is
                    # then ordered by fitness exactly as before.
                    c2 = rng.integers(0, pool.shape[0], size=n)
                    other = pool[c2]
                    f_other = np.where(c2 < n, fit[np.minimum(c2, n - 1)],
                                       np.inf)
                    take = f_other < fit[b1]
                    diff = np.where(take[:, None], other - X[b1],
                                    X[b1] - other)
                else:
                    diff = X[b1] - X[b2]
                Mom = beta[:, None] * Hm + (1.0 - beta[:, None]) * diff
                V = X + F[:, None] * (X[pb] - X) + F[:, None] * Mom
            else:
                # [L-SHADE] Eq. 3
                V = X + F[:, None] * (X[pb] - X) + F[:, None] * (X[r1] - pool[r2])

            if use_local.any():
                k_loc = int(use_local.sum())
                if local_gen == "ctr":
                    # (A14) OURS -- the controlled test of WHY the second
                    # generator helps. `current-to-rand/1` (Iorio & Li) is
                    # rotation-invariant like the CMA sample but carries NO
                    # learned metric and costs nothing:
                    #     v_i = x_i + K_i (x_r1 - x_i) + F_i (x_r2 - x_r3)
                    # It occupies the same slot, gets the same A7 crossover
                    # treatment and the same allocation rule, so the ONLY
                    # thing that differs is whether a metric is learned.
                    # If MDAC-ctr matches MDAC, the advantage is operator
                    # diversity + rotation invariance, not metric accuracy --
                    # and it is then available at jSO's speed, since the CMA
                    # machinery is 24-60 % of MDAC's runtime (measured).
                    li = np.nonzero(use_local)[0]
                    K = rng.random(k_loc)[:, None]
                    a = rng.integers(0, n, size=k_loc)
                    b = rng.integers(0, pool.shape[0], size=k_loc)
                    c = rng.integers(0, pool.shape[0], size=k_loc)
                    V[li] = (X[li] + K * (X[a] - X[li])
                             + F[li][:, None] * (pool[b] - pool[c]))
                else:
                    # [LSPACMA] CMA-ES offspring, then crossover (Sec. II-D)
                    V[use_local] = cma.sample(k_loc, rng)

            # (A7) the metric generator's own CR, drawn from its own memory
            CR_use = CR.copy()
            loc_idx_mem = None
            if use_local.any() and cma_cr_mode == "adaptive":
                loc_idx_mem = mem_loc.draw_index(int(use_local.sum()), rng)
                CR_use[use_local] = mem_loc.draw_normal("CR", loc_idx_mem, rng)

            V = repair_midpoint(V, X, bounds)
            U = binomial_crossover(V, X, CR_use, rng)

            # (A6) OURS -- rotation-invariance guard on the metric generator.
            #
            # Binomial crossover copies a RANDOM SUBSET OF COORDINATES from
            # the parent. That operation is defined in the coordinate frame,
            # so it is NOT invariant under rotation of the search space: a
            # point drawn from N(m, sigma^2 C), whose whole value is that C
            # encodes the rotated local metric, is destroyed by mixing a
            # random subset of its coordinates back from the parent. The
            # damage is largest exactly where C matters most -- rotated,
            # ill-conditioned landscapes, i.e. BOTH declared target
            # categories (BR unimodal F01-F03 and the rotated BR hybrids).
            #
            # [LSPACMA] applies crossover to its CMA offspring (their
            # Sec. II-D). This flag makes that a tested choice rather than
            # an inherited one: with `cma_cr_full` the individuals assigned
            # to the metric generator bypass crossover (equivalent to CR=1),
            # so their sample reaches the objective in the frame the
            # covariance was learned in, while the DE generator keeps
            # [L-SHADE]'s adapted CR unchanged.
            if (cma_cr_full or cma_cr_mode == "full") and use_local.any():
                U[use_local] = V[use_local]

            fu = ev.evaluate_batch(U)

            # ---- selection: elitist, then [AGDESA] Metropolis if stagnant
            delta = fu - fit
            better = delta < 0.0
            accept = better.copy()
            if metropolis:
                # [AGDESA]'s Fast-SA cooling, with two documented deviations:
                #  (a) scale-invariant -- T carries the current fitness spread,
                #      so acceptance depends on the RELATIVE worsening;
                #  (b) keyed to budget progress t, not to the generation
                #      counter. A counter is not comparable across algorithms
                #      whose per-generation cost differs by 3x, which is the
                #      same reason every schedule in this file is a function
                #      of t = FES/MaxFES.
                T = T0_frac * (1.0 - t) * fitness_scale(fit)
                stagnant = (gen - last_imp) >= n_L
                cand = (~better) & stagnant & np.isfinite(delta)
                if cand.any():
                    prob = np.exp(-np.clip(delta[cand] / max(T, 1e-300),
                                           0.0, 700.0))
                    take = rng.random(int(cand.sum())) < prob
                    sel = np.where(cand)[0][take]
                    accept[sel] = True
                    n_metro_acc += int(take.sum())

            # success statistics BEFORE overwriting, [L-SHADE] Alg. 2 l.16-18
            succ = {}
            dF = np.abs(delta[better])
            if better.any():
                succ["CR"] = CR[better]
                succ["F"] = F[better]
                if momentum:
                    succ["beta"] = beta[better]
                # append the replaced parents, keeping the most recent
                # `cap` of them in insertion order -- identical semantics to
                # `archive.append(...)` followed by `archive[-cap:]`.
                newx = X[better]
                k_new = int(newx.shape[0])
                if k_new:
                    if k_new >= arch_cap:
                        arch[:] = newx[-arch_cap:]
                        arch_n = arch_cap
                    else:
                        room = arch_cap - arch_n
                        if k_new > room:                 # drop the oldest
                            shift = k_new - room
                            arch[:arch_n - shift] = arch[shift:arch_n].copy()
                            arch_n -= shift
                        arch[arch_n:arch_n + k_new] = newx
                        arch_n += k_new
                cap = max(1, int(round(r_arc * n)))
                if arch_n > cap:
                    arch[:cap] = arch[arch_n - cap:arch_n].copy()
                    arch_n = cap

            # [LSPACMA] improvement mass per generator -> allocation,
            # plus (A2) OURS: confidence-widening bounds.
            if fcp and alloc in ("adaptive", "wide"):
                gmask = better & (~use_local)
                lmask = better & use_local
                w_g = float(np.sum(-delta[gmask])) if gmask.any() else 0.0
                w_l = float(np.sum(-delta[lmask])) if lmask.any() else 0.0
                if (w_g + w_l) > 0.0:
                    share = w_g / (w_g + w_l)
                    if alloc == "wide":
                        # (A1) OURS -- evidence-proportional commitment.
                        # [LSPACMA] pins the clamp at [0.2,0.8] so that both
                        # generators always run. That is the right default
                        # when nothing is known, and the wrong one on a
                        # rotated ill-conditioned quadratic (BR unimodal),
                        # where the metric learner should take ~all of the
                        # population. So the clamp WIDENS with evidence:
                        #   lo(c) = max(floor, 0.35 * rho^c)
                        # where c counts consecutive generations in which the
                        # same generator held the larger improvement mass,
                        # and resets to 0 the moment dominance flips. With
                        # rho=0.85 the bound reaches the floor after ~15
                        # consistent generations and snaps back immediately
                        # on any evidence to the contrary, so commitment is
                        # never permanent and never schedule-driven.
                        dom = 1 if share >= 0.5 else -1
                        if dom == dom_sign:
                            dom_run += 1
                        else:
                            dom_sign, dom_run = dom, 1
                        lo_b = max(widen_floor, 0.35 * widen_rho ** dom_run)
                        hi_b = 1.0 - lo_b
                    elif alloc == "dimaware":
                        # (A8) OURS -- dimension-aware floor on the GLOBAL
                        # generator's share. Declared BEFORE the screen that
                        # tests it; see the A8 block above `TARGET_CATEGORIES`.
                        #
                        # Measured weakness it targets: at D=10 the rotated
                        # blocks are our worst cells -- CEC hybrid A12 0.460
                        # (1W/7L) and CEC composition A12 0.411 (1W/10L) --
                        # while D=30 and D=50 are wins. The diagnosis under
                        # test: at D=10 a hybrid partitions ten variables into
                        # 3-6 sub-blocks of 2-4 variables each, and a single
                        # CMA metric over the whole 10-vector cannot represent
                        # that block structure, so population spent on it is
                        # wasted relative to coordinate-wise DE moves.
                        #
                        #     floor(D) = max(fcp_lo, 0.65 * max(0, 30-D) / 20)
                        #
                        # D=10 -> 0.65, D=20 -> 0.325, D>=30 -> fcp_lo.
                        # The rule VANISHES at D>=30 by construction, so it
                        # cannot disturb the dimensions where we already win.
                        lo_b = max(fcp_lo, 0.65 * max(0.0, 30.0 - dim) / 20.0)
                        hi_b = fcp_hi
                    else:
                        lo_b, hi_b = fcp_lo, fcp_hi
                    target = min(hi_b, max(lo_b, share))
                    pi_global = (1.0 - fcp_c) * pi_global + fcp_c * target

            # [LSMA] Eq. 22. The comparison operator is not recoverable from
            # the source PDF's text layer; both readings are implemented and
            # selected by `hm_on_success`, never assumed.
            if momentum:
                step = U - X
                if hm_on_success:
                    Hm = np.where(better[:, None], step, Hm)
                else:
                    Hm = np.where(better[:, None], Hm, step)

            # Full CMA update from ALL local-operator offspring (not only the
            # successful ones): CMA-ES ranks within its own generation, so
            # feeding it a success-filtered subset would corrupt the
            # recombination weights it is built on.
            if use_local.any():
                loc = np.where(use_local)[0]
                n_acc = int(better[loc].sum())
                n_local_acc += n_acc
                # With `local_gen="ctr"` no metric is maintained, so the CMA
                # update is skipped entirely -- that is where the saving is.
                if local_gen != "ctr":
                    cma.tell(U[loc], fu[loc])
                if n_acc:
                    cma_last_imp = gen
                # (A2) OURS -- restart the sub-distribution from the elite
                # pool when it has collapsed or stopped producing. Seeds come
                # from the population/archive elite, i.e. basins the search
                # has already paid to find, so a restart relocates rather
                # than merely re-shrinking. Costs no evaluations.
                if cma_restart and (cma.stalled
                                    or gen - cma_last_imp > cma_stall_gens):
                    elite = np.argsort(fit)[:max(2, n // 4)]
                    seed_pt = X[elite[rng.integers(0, len(elite))]]
                    rad = float(np.sqrt(np.mean(np.sum(
                        (X - X.mean(axis=0)) ** 2, axis=1)))) if n > 1 else 1e-3
                    C0, sig_c = (shrinkage_covariance(
                        np.vstack([X] + ([used] if used is not None else [])),
                        dim) if cma_seed_cov else (None, 0.0))
                    cma.restart(seed_pt,
                                max(rad / max(1.0, math.sqrt(dim)), sig_c, 1e-12),
                                C0)
                    cma_last_imp = gen
                    n_cma_restart += 1

            X = np.where(accept[:, None], U, X)
            fit = np.where(accept, fu, fit)
            last_imp = np.where(better, gen, last_imp)
            mem.update(succ, dF)

            # (A7) update the metric generator's CR memory from ITS OWN
            # successes only. Mixing the two generators' successes would
            # make each learn from moves it never made.
            if loc_idx_mem is not None:
                ok = better[use_local]
                if ok.any():
                    mem_loc.update({"CR": CR_use[use_local][ok]},
                                   np.abs(delta[use_local][ok]))

    except BudgetExhausted:
        pass

    return _finish(name, ev, {"gens": gen, "pi_global": pi_global,
                              "local_acc": n_local_acc,
                              "metro_acc": n_metro_acc,
                              "cma_restarts": n_cma_restart,
                              "cr_loc": float(np.mean(mem_loc.M["CR"]))})


def alg_lshade(ev, dim, bounds, seed, pop_size=None):
    """[L-SHADE] -- canonical. The ablation zero-point: every mechanism we
    add is measured against this."""
    return _shade_core(ev, dim, bounds, seed, "L-SHADE", momentum=False,
                       spa=False, fcp=False, metropolis=False)


def alg_lshade_ma(ev, dim, bounds, seed, pop_size=None):
    """[LSMA] -- L-SHADE plus momentum-to-pbest/1 and adaptive beta."""
    return _shade_core(ev, dim, bounds, seed, "L-SHADE-MA", momentum=True,
                       spa=False, fcp=False, metropolis=False)


def alg_lshade_spacma(ev, dim, bounds, seed, pop_size=None):
    """[LSPACMA] -- semi-parameter adaptation + FCP co-allocation with CMA-ES.

    NOTE ON PARAMETERS: the source states "Pbest, Memory size (H), and
    archive rate were set to (0.11, 1.4, 5) as described in LSHADE". That
    listing is internally inconsistent -- H must be an integer, and neither
    1.4 nor 5 is an L-SHADE value (its tuned settings are p=0.11,
    r_arc=2.6, H=6). We therefore use [L-SHADE]'s published tuned values
    for the shared parameters and say so, rather than silently reordering
    the source's triple. Resolving this needs the authors' released code.
    """
    return _shade_core(ev, dim, bounds, seed, "L-SHADE-SPACMA", momentum=False,
                       spa=True, fcp=True, metropolis=False, alloc="adaptive")


def alg_jso(ev, dim, bounds, seed, pop_size=None):
    r"""jSO  [jSO]  Brest, Maucec & Boskovic, "Single Objective
    Real-Parameter Optimization: Algorithm jSO", IEEE CEC 2017.
    CEC2017 competition top-ranked method.

    TRANSCRIBED FROM THE AUTHORS' OWN C++ SOURCE, not from memory:
    `jSO-SOURCE-RESULTS-29May2017.tar.gz`, shipped inside
    `Codes-of-Top-Methods-and-results.zip` in Suganthan's official CEC2017
    repository. Every constant below carries the source line it came from,
    because an under-powered baseline is the same class of defect as an
    over-tuned proposal -- and this one is the method we must beat.

    main.cc:64-67
        NP_init = round(sqrt(D) * ln(D) * 25)        (not 18*D)
        H  = 5,  arc_rate = 1.0,  p_best_rate = 0.25
    lshade.cc:89,91
        memory_sf init = 0.3,  memory_cr init = 0.8  (not 0.5/0.5)
    lshade.cc:127-129
        the LAST memory slot is fixed: mu_sf = mu_cr = 0.9
    lshade.cc:147-148     CR floors, rising early in the run
        nfes < 0.25*max and CR < 0.7  ->  CR = 0.7
        nfes < 0.50*max and CR < 0.6  ->  CR = 0.6
    lshade.cc:158         F ceiling
        nfes < 0.60*max and F  > 0.7  ->  F  = 0.7
    lshade.cc:165         pbest donor must differ from the target in the
                          first half of the run
    search body (operateCurrentToPBest1BinWithArchive)
        jF = F*0.7 if nfes < 0.2*max
           = F*0.8 if nfes < 0.4*max
           = F*1.2 otherwise
        v_i = x_i + jF*(x_pbest - x_i) + F*(x_r1 - x_{r2 in P u A})
        NOTE the asymmetry: the pbest pull is weighted by jF while the
        difference term keeps the RAW F. That is jSO's defining change
        over L-SHADE and it is easy to get wrong.
    lshade.cc:254-290     memory update = weighted Lehmer mean, then the
                          iL-SHADE average with the previous content:
        M[pos] = (lehmer_new + M_old[pos]) / 2
        CR terminal value is -1 (jSO) rather than L-SHADE's bottom symbol.
    lshade.cc:321-322     p_num = round(NP * 0.25), floored at 2
    """
    rng = np.random.default_rng(seed)
    lo, hi = bounds
    max_fes = ev.max_fes
    H = 5
    n_init = max(8, int(round(math.sqrt(dim) * math.log(dim) * 25)))
    n_min, n = 4, max(8, int(round(math.sqrt(dim) * math.log(dim) * 25)))
    arc_rate, p_best_rate = 1.0, 0.25

    M_sf = np.full(H, 0.3)
    M_cr = np.full(H, 0.8)
    pos = 0
    X = init_pop(n, dim, bounds, rng)
    try:
        fit = ev.evaluate_batch(X)
    except BudgetExhausted:
        return _finish("jSO", ev)
    archive: List[Array] = []

    try:
        while not ev.exhausted():
            t = ev.fes / max_fes
            order = np.argsort(fit)
            p_num = max(2, int(round(n * p_best_rate)))

            r = mem_idx = rng.integers(0, H, size=n)
            mu_cr = M_cr[mem_idx].copy()
            mu_sf = M_sf[mem_idx].copy()
            last = mem_idx == (H - 1)
            mu_sf[last] = 0.9                      # lshade.cc:129
            mu_cr[last] = 0.9

            CR = np.where(mu_cr < 0.0, 0.0,
                          np.clip(rng.normal(mu_cr, 0.1), 0.0, 1.0))
            if t < 0.25:
                CR = np.maximum(CR, 0.7)           # lshade.cc:147
            elif t < 0.50:
                CR = np.maximum(CR, 0.6)           # lshade.cc:148
            F = mu_sf + 0.1 * rng.standard_cauchy(size=n)
            for _ in range(12):
                bad = F <= 0.0
                if not bad.any():
                    break
                F[bad] = mu_sf[bad] + 0.1 * rng.standard_cauchy(size=int(bad.sum()))
            F = np.clip(F, 1e-12, 1.0)
            if t < 0.60:
                F = np.minimum(F, 0.7)             # lshade.cc:158

            jF = F * (0.7 if t < 0.2 else 0.8 if t < 0.4 else 1.2)

            pb = order[rng.integers(0, p_num, size=n)]
            if t < 0.50:                           # lshade.cc:165
                clash = pb == np.arange(n)
                for _ in range(8):
                    if not clash.any():
                        break
                    pb[clash] = order[rng.integers(0, p_num, size=int(clash.sum()))]
                    clash = pb == np.arange(n)

            r1 = rng.integers(0, n, size=n)
            pool = np.vstack([X] + ([np.vstack(archive)] if archive else []))
            r2 = rng.integers(0, pool.shape[0], size=n)

            V = (X + jF[:, None] * (X[pb] - X)
                 + F[:, None] * (X[r1] - pool[r2]))
            V = repair_midpoint(V, X, bounds)
            U = binomial_crossover(V, X, CR, rng)
            fu = ev.evaluate_batch(U)

            better = fu < fit
            if better.any():
                dF = np.abs(fu[better] - fit[better])
                s_cr, s_sf = CR[better], F[better]
                for x_old in X[better]:
                    archive.append(x_old.copy())
                cap = max(1, int(round(arc_rate * n)))
                if len(archive) > cap:
                    sel = rng.choice(len(archive), cap, replace=False)
                    archive = [archive[i] for i in sel]
                old_sf, old_cr = M_sf[pos], M_cr[pos]
                new_sf = lehmer_weighted(s_sf, dF)
                if float(np.sum(s_cr)) == 0.0 or old_cr < 0.0:
                    M_cr[pos] = -1.0               # jSO terminal value
                else:
                    M_cr[pos] = (lehmer_weighted(s_cr, dF) + old_cr) / 2.0
                M_sf[pos] = (new_sf + old_sf) / 2.0   # iL-SHADE averaging
                pos = (pos + 1) % H

            keep_u = fu <= fit
            X = np.where(keep_u[:, None], U, X)
            fit = np.where(keep_u, fu, fit)

            n_t = max(n_min, lpsr(n_init, n_min, ev.fes, max_fes))
            if n_t < n:
                keep = np.argsort(fit)[:n_t]
                X, fit, n = X[keep], fit[keep], n_t
                cap = max(1, int(round(arc_rate * n)))
                if len(archive) > cap:
                    sel = rng.choice(len(archive), cap, replace=False)
                    archive = [archive[i] for i in sel]
            if n < n_min:
                break
    except BudgetExhausted:
        pass
    return _finish("jSO", ev)


def alg_agdesa_ng(ev, dim, bounds, seed, pop_size=None):
    """[AGDESA] without the gradient component ("ng" = no gradient).

    Keeps what is free: DualMutation, the stagnation-gated Metropolis
    acceptance and the cooling schedule. See the header for why the L-BFGS-B
    stage is excluded from the proposed algorithm; `A-GDESA-fd` below is the
    variant that keeps it and pays for it.

    The scale-invariant, budget-keyed temperature of `fitness_scale` is used
    here TOO, identically to MDAC. That is deliberate: with an absolute T the
    acceptance probability is ~1 on CL_RotElliptic (spread ~1e10) and ~0 on
    CL_Rastrigin (spread ~1e0), so an absolute-T competitor would be losing
    to an implementation defect of ours rather than to MDAC. Equipping both
    sides with the same fix is the only comparison that measures anything.
    """
    rng = np.random.default_rng(seed)
    n = pop_size or min(100, 10 * dim)
    X = init_pop(n, dim, bounds, rng)
    pc, F, n_L, T0_frac = 0.9, 0.5, 12, 0.10
    max_fes = ev.max_fes
    try:
        fit = ev.evaluate_batch(X)
        last = np.zeros(n, dtype=np.int64)
        g = 0
        while True:
            g += 1
            T = T0_frac * (1.0 - ev.fes / max_fes) * fitness_scale(fit)
            r = np.array([rng.choice(np.delete(np.arange(n), i), 5,
                                     replace=False) for i in range(n)])
            use2 = rng.random(n) < 0.5          # DualMutation, [AGDESA] Eq. 11
            V = np.where(use2[:, None],
                         X[r[:, 0]] + F * (X[r[:, 1]] - X[r[:, 2]])
                         + F * (X[r[:, 3]] - X[r[:, 4]]),
                         X[r[:, 0]] + F * (X[r[:, 1]] - X[r[:, 2]]))
            V = repair_midpoint(V, X, bounds)
            U = binomial_crossover(V, X, np.full(n, pc), rng)
            fu = ev.evaluate_batch(U)
            delta = fu - fit
            better = delta < 0
            acc = better.copy()
            stag = (g - last) >= n_L
            cand = (~better) & stag & np.isfinite(delta)
            if cand.any():
                p = np.exp(-np.clip(delta[cand] / max(T, 1e-300), 0.0, 700.0))
                sel = np.where(cand)[0][rng.random(int(cand.sum())) < p]
                acc[sel] = True
            X = np.where(acc[:, None], U, X)
            fit = np.where(acc, fu, fit)
            last = np.where(better, g, last)
    except BudgetExhausted:
        pass
    return _finish("A-GDESA-ng", ev)


def alg_agdesa_fd(ev, dim, bounds, seed, pop_size=None):
    """[AGDESA] WITH its gradient stage, by finite differences, every probe
    charged to the shared budget.

    This exists to make a hidden cost visible rather than to win: one
    gradient is dim+1 evaluations and the source runs L-BFGS-B to local
    convergence. Under FES-honest accounting that is exactly the kind of
    bookkeeping overhead this framework was built to expose, so the honest
    thing is to measure it, not to assume it either way.
    """
    rng = np.random.default_rng(seed)
    n = pop_size or min(100, 10 * dim)
    X = init_pop(n, dim, bounds, rng)
    lo, hi = bounds
    pc, F, n_L, T0_frac = 0.9, 0.5, 12, 0.10
    max_fes = ev.max_fes
    p_g = 2.0
    h = 1e-6 * (hi - lo)

    def grad_step(x0, f0, iters=4):
        x, f = x0.copy(), float(f0)
        for _ in range(iters):
            Xp = x[None, :] + h * np.eye(dim)
            vp = ev.evaluate_batch(np.clip(Xp, lo, hi))
            gvec = (vp - f) / h
            gn = np.linalg.norm(gvec)
            if not np.isfinite(gn) or gn < 1e-300:
                break
            step = 0.01 * (hi - lo) / gn
            for _ls in range(4):
                cand = np.clip(x - step * gvec, lo, hi)
                fc = float(ev.evaluate_batch(cand.reshape(1, -1))[0])
                if fc < f:
                    x, f = cand, fc
                    break
                step *= 0.25
            else:
                break
        return x, f

    try:
        fit = ev.evaluate_batch(X)
        last = np.zeros(n, dtype=np.int64)
        cooldown = np.zeros(n)
        g = 0
        while True:
            g += 1
            T = T0_frac * (1.0 - ev.fes / max_fes) * fitness_scale(fit)
            for i in range(n):
                if rng.random() < p_g / n and g >= cooldown[i]:
                    xg, fg = grad_step(X[i], fit[i])
                    cooldown[i] = g + 1.0 / p_g
                    if fg < fit[i]:
                        X[i], fit[i], last[i] = xg, fg, g
            r = np.array([rng.choice(np.delete(np.arange(n), i), 5,
                                     replace=False) for i in range(n)])
            use2 = rng.random(n) < 0.5
            V = np.where(use2[:, None],
                         X[r[:, 0]] + F * (X[r[:, 1]] - X[r[:, 2]])
                         + F * (X[r[:, 3]] - X[r[:, 4]]),
                         X[r[:, 0]] + F * (X[r[:, 1]] - X[r[:, 2]]))
            V = repair_midpoint(V, X, bounds)
            U = binomial_crossover(V, X, np.full(n, pc), rng)
            fu = ev.evaluate_batch(U)
            delta = fu - fit
            better = delta < 0
            acc = better.copy()
            cand = (~better) & ((g - last) >= n_L) & np.isfinite(delta)
            if cand.any():
                p = np.exp(-np.clip(delta[cand] / max(T, 1e-300), 0.0, 700.0))
                acc[np.where(cand)[0][rng.random(int(cand.sum())) < p]] = True
            X = np.where(acc[:, None], U, X)
            fit = np.where(acc, fu, fit)
            last = np.where(better, g, last)
    except BudgetExhausted:
        pass
    return _finish("A-GDESA-fd", ev)


# ===========================================================================
# 6. MDAC -- PROPOSED
# ===========================================================================
MOMENTUM_HM_ON_SUCCESS = True   # decided by `--stage ablation`, not assumed
# (C1) REMOVED 2026-10-09 by its own measurement. It is OFF by default; the
# mechanism stays in the file as a documented negative result, like momentum.
#
# C1 was `pbest_index_diverse`, our diversity-regularized pbest draw. Its
# docstring said it "costs ZERO objective evaluations", which is true and was
# misleading: it costs no FES and it cost most of the wall-clock.
#
#   QUALITY  (mdac_c1, full official suite, D=10/30/50, 25 runs, 6300 tasks)
#       A12(MDAC-noDiv vs MDAC) = 0.495 overall
#       W/T/L = 0/126/0 -- not one significant difference in 126 cells
#       per block: multimodal 0.532 · composition 0.499 · classic-12 0.493
#                  hybrid 0.468 · unimodal 0.500
#
#   COST  (same run, full 10000*D protocol)
#       total CPU   102 035 s -> 17 535 s          5.82x
#       D=50 median    76.69 s -> 8.57 s           8.95x
#       versus jSO   21.0x -> 2.3x  (D=50),  9.5x -> 2.3x  (D=30)
#
# Per generation it computed distances from the elite pool to the WHOLE
# archive -- about 99 x 2340 x 50 = 11.6M operations at D=50 -- which is why
# the overhead grew with dimension instead of staying flat.
#
# The decision rule (remove iff A12 >= 0.48, i.e. no harm) was fixed in
# PLAN_Q1 §5 before the run. Keeping a mechanism with no measured benefit and
# five sixths of the runtime would have been indefensible, and it is the
# direct cause of the equal-wall-clock collapse in RESULTS_FINAL.md §8b.
#
# CONSEQUENCE, stated plainly: C1 was one of three claimed contributions.
# The list is now C2 + A7. `MDAC-noDiv` is from here on identical to `MDAC`.
MDAC_NU = 0.0                   # was 0.6; see above. nu=0 recovers [L-SHADE]

# --------------------------------------------------------------------------
# DECLARED TARGET CATEGORIES -- fixed BEFORE the confirmatory run
# --------------------------------------------------------------------------
# MDAC is a category-targeted algorithm, not a universal one. No Free Lunch
# makes universal superiority unavailable in principle, so the scientifically
# meaningful claim is a specific one, declared in advance and then tested.
#
# Targets:  BR unimodal  (F01-F03)  and  BR hybrid  (F11-F20)   = 13 of 42
#
# Evidence that fixed this choice, from the exploratory BR probe
# (D=30, 10000*D, 3 runs, mean Friedman rank of 4 variants):
#     BR unimodal      MDAC-noMom 2.17  <  L-SHADE 2.78    <- we already win
#     BR hybrid        MDAC-noMom 1.73  <  L-SHADE 2.70    <- we already win
#     BR multimodal    L-SHADE    2.36  <  MDAC-noMom      <- we do not
#     BR composition   L-SHADE    2.30  <  MDAC            <- we do not
#
# Mechanistic reason the targets are these two and not the others: both are
# ROTATED and ill-conditioned, so a learned metric pays; the hybrid block is
# additionally built from unequal sub-blocks, so the right amount of metric
# learning CHANGES during a run -- which is exactly what an allocation rule
# driven by measured improvement can track and a fixed schedule cannot.
#
# The non-target categories are reported in full and are expected to be
# losses. Reporting only the targets would be cherry-picking; declaring them
# in advance and reporting everything is not.
#
# DECLARATION HISTORY -- kept in full, because the first declaration FAILED
# and replacing it silently would be the whole fraud.
#
# H1, declared on the BR_* bridge, tested on the OFFICIAL CEC2017 suite
# (5 algorithms, 42 functions, D = 10/30/50, 25 runs, Holm over 504):
#     targets were  BR unimodal  and  BR hybrid
#     bridge evidence:   a win in BR hybrid (numbers withheld: BR_* is not an official benchmark)
#     official outcome:  unimodal 0/27/0, A12 0.500  -- NOT MET (pure tie;
#                        the rank-1 position came only from CMA-ES's weakness)
#                        hybrid   9/59/22, A12 0.424 -- NOT MET (clear loss
#                        to jSO 0.372 and L-SHADE-SPACMA 0.399)
#     => H1 IS REFUTED. The bridge suite did not merely fail to predict the
#        official suite, it predicted the OPPOSITE SIGN. This is the single
#        clearest justification in this project for refusing to publish any
#        number from a locally generated CEC lookalike.
#
# H2, declared NOW, before the confirmatory run. Two claims:
#     (a) CEC multimodal (F04-F10): MDAC beats each of jSO, L-SHADE-SPACMA
#         and L-SHADE individually.
#     (b) The advantage grows with dimension.
#     Exploratory evidence from the H1 run (which is why H2 needs an
#     INDEPENDENT axis, not a re-reading of the same data):
#         multimodal vs jSO 5/14/2 A12 0.651 · vs SPACMA 5/13/3 A12 0.586
#                     vs L-SHADE 6/14/1 A12 0.667 · pooled 16/41/6 A12 0.635
#         D=10 A12 0.371 -> D=30 0.524 -> D=50 0.559, monotone
#     CONFIRMATORY AXIS: D = 100, which no design decision in this project
#     has ever touched. H2(b) predicts A12 > 0.559 there; H2(a) predicts the
#     multimodal win survives. A result at D=100 is therefore a genuine
#     out-of-sample test and not a post-hoc re-slice.
#     H2 is CONFIRMED only if both hold; partial success is reported as such.
#
# A8, declared 2026-10-08 BEFORE the screen that tests it (rule 4).
#     Mechanism: `alloc="dimaware"` -- a dimension-decaying floor on the
#     global generator's share, floor(D) = max(fcp_lo, 0.65*max(0,30-D)/20),
#     which is 0.65 at D=10 and IDENTICALLY fcp_lo at D>=30.
#     Weakness it targets, measured in `mdac_fixed` (51 runs, 4 top methods):
#         CEC hybrid      D=10  A12 0.460   1W / 7L
#         CEC composition D=10  A12 0.411   1W / 10L
#     while D=30 and D=50 are wins in both blocks.
#     DECISION RULE, fixed now:
#       ACCEPT only if, at D=10, A12(MDAC-dimAware vs MDAC) > 0.5 in BOTH
#       CEC hybrid AND CEC composition, AND >= 0.48 in CEC multimodal
#       (the no-harm condition: the declared target block must not regress).
#       Anything else is a REFUTED hypothesis and is recorded here as such,
#       exactly as H1 was. A partial result is reported as partial, never
#       promoted.
TARGET_CATEGORIES = ("CEC multimodal",)


def alg_mdac(ev, dim, bounds, seed, pop_size=None, *, nu: float = None,
             alloc: str = "adaptive", momentum: bool = False,
             metropolis: bool = True, fcp: bool = True,
             cma_restart: bool = False, cma_seed_cov: bool = False,
             cma_cr_mode: str = "adaptive",
             hm_on_success: bool = None) -> AlgResult:
    """MDAC = Metric-DE with Adaptive Co-allocation.

    CONFIGURATION CHANGE FROM v1, forced by our own measurements
    ------------------------------------------------------------
    v1 shipped with [LSMA]'s momentum ON. Two independent ablations then
    measured it as HARMFUL -- A12 0.455 on the shifted classics (24 cells)
    and on the BR_* bridge suite (numbers withheld: BR_* is not an official benchmark), with `MDAC-noMom` the best of
    ten variants in both. The diagnosed cause: [LSMA] Eq. 21 draws both
    points of its difference vector from P, discarding the P-union-A
    difference that [L-SHADE] Eq. 3 relies on for diversity. Momentum is
    therefore OFF by default and survives only as the flags `momentum` /
    `mom_archive`, which test the repair rather than assume it.

    So the name's "M" is now Metric, not Momentum: what the algorithm
    actually does is decide, continuously and from measured improvement, how
    much of the population should be spent learning a local metric.

    Composition (every row attributed in the file header):
        [L-SHADE]  success-history M_CR/M_F, Lehmer update, LPSR, archive,
                   binomial crossover, midpoint repair, tuned constants
        [LSMA]     momentum-to-pbest/1 with fitness-ordered difference
        [AGDESA]   stagnation-gated Metropolis acceptance, T = T0/(1+g)
        [1+lCMA]   CMA distribution as the local offspring generator
        (C1) OURS  diversity-regularized pbest draw (zero extra FES)
        (C2) OURS  learned global/local allocation, tested against
                   fixed / random / terminal at equal budget

    What is NOT here, deliberately: the unit-normalised direction field of
    our earlier algorithm. Every step above is built from real population
    differences, which contract with the population, so the step-size floor
    that made nine terminal-annealing parameters necessary does not exist
    and those nine parameters are gone.
    """
    return _shade_core(
        ev, dim, bounds, seed, "MDAC",
        momentum=momentum, spa=False, fcp=fcp, metropolis=metropolis,
        diversity_nu=MDAC_NU if nu is None else nu,
        hm_on_success=(MOMENTUM_HM_ON_SUCCESS if hm_on_success is None
                       else hm_on_success),
        alloc=alloc, cma_restart=cma_restart, cma_seed_cov=cma_seed_cov,
        cma_cr_mode=cma_cr_mode)


# ---- ablation handles -----------------------------------------------------
def _mk_mdac(_name: str, **kw):
    """Build an ablation handle.

    The name is a positional parameter and `kw` is COPIED per call. An
    earlier version popped the name out of the closure's own dict, which
    mutated it on the first call and made every later call in the same
    worker process raise KeyError. Those runs were caught by the worker's
    blanket handler, recorded as +inf, and produced a clean-looking but
    entirely fabricated ablation table in which all eight variants tied.
    Hence also the hard failure gate in `analyse`.
    """
    def f(ev, dim, bounds, seed, pop_size=None):
        return _shade_core(ev, dim, bounds, seed, _name, spa=False,
                           **dict(kw))
    return f


# Shipped configuration, each field set by a measurement, not a preference:
#   momentum=False     two BR_* design-stage ablations against it (withheld)
#   mom_archive        the repair hypothesis -- REFUTED, it made things worse
#   alloc="adaptive"   narrow bounds beat the widening ones (BR_*, withheld)
#   cma_restart=False  restarts did not pay on the BR block (withheld)
#   cma_seed_cov=False neutral (BR_*, withheld) -- stays off on parsimony
#   cma_cr_full=True   NEW hypothesis under test (A6), rotation invariance
_BASE = dict(momentum=False, fcp=True, metropolis=True,
             diversity_nu=MDAC_NU, alloc="adaptive",
             cma_restart=False, cma_seed_cov=False, cma_cr_mode="adaptive")


# ===========================================================================
# COMPETITOR: EBOwithCMAR  -- CEC2017 top method
# ===========================================================================
# Transcribed from the authors' own released MATLAB, not from memory, per the
# project rule. Source: `EBOwithCMAR/` inside the same CEC2017 master.zip as
# the benchmark data (Kumar, Misra & Singh; contact address in `LS2.m`).
# File-by-file provenance:
#     Introd_Par.m   CS / Gmax / PopSize=18n / MinPopSize=4 / prob_ls=0.1
#     EBO_BIN.m      the outer loop, PS1/PS2 split, cycle logic, info sharing
#     EBO.m          the DE ("perching and patrolling") phase
#     Scout.m        the CMAR phase
#     init_cma_par.m CMA initialisation constants
#     bestt.m        the pbest-index draw
#     han_boun.m     the two bound-repair variants
#     gnR1R2.m       distinct random index triples
#     updateArchive.m  the L-SHADE archive
#     LS2.m          the SQP local search
#
# TWO QUIRKS OF THE RELEASED SOURCE ARE REPRODUCED DELIBERATELY, NOT FIXED.
# Rule 2 of this project says to transcribe a competitor from its authors'
# released source; "fixing" either one would mean benchmarking against an
# algorithm that was never published and whose CEC2017 numbers nobody can
# check. Both are documented at their site below:
#   (Q1) Scout.m line 55 calls `fliplr` on a COLUMN vector, which is a no-op
#        in MATLAB, so the recombination weights stay in ascending-fitness
#        order -- the best point gets the SMALLEST weight.
#   (Q2) EBO.m increments `hist_pos` between the M_CR write and the M_T /
#        M_freq writes, so T and freq land in the NEXT memory slot.
# The one place we do NOT follow the source is the local search: `LS2.m` uses
# `fmincon`, whose internal finite-difference probes MATLAB does not charge
# against MaxFES. Under this framework every probe is charged (same treatment
# as `A-GDESA-fd`), because an uncharged local search is a budget violation.
def _ebo_bestt_idx(npop: int, dim: int, rng) -> Array:
    """[EBOCMAR] `bestt.m` -- min of D distinct uniform draws, per offspring.

        if 2*D > n:  D = 1,  n = max(round(0.1*n), 2)
        i(j) = min(randperm(n, D))

    Drawing an explicit permutation per offspring is O(npop*n) per generation
    and would dominate the run. The minimum of D distinct draws from
    {0..n-1} has the closed form  P(M >= m) = C(n-m, D)/C(n, D), so it is
    sampled by inverse CDF here: identical distribution, O(n + npop).
    """
    D, n = dim, npop
    if 2 * D > npop:
        D = 1
        n = max(int(round(0.1 * npop)), 2)
    n = min(n, npop)
    D = max(1, min(D, n))
    mm = np.arange(n - D + 1, dtype=np.float64)
    surv = np.concatenate(([1.0], np.cumprod((n - mm - D) / (n - mm))))
    cdf = 1.0 - surv[1:]                       # P(M <= m), last entry 1.0
    return np.searchsorted(cdf, rng.random(npop), side="left").clip(0, n - D)


def _ebo_gn_r1r2(np1: int, np2: int, r0: Array, rng) -> Tuple[Array, Array, Array]:
    """[EBOCMAR] `gnR1R2.m`: r1 != r0; r2 != r1, r0; r3 != r0, r1, r2."""
    n0 = len(r0)
    r1 = rng.integers(0, np1, size=n0)
    for _ in range(1000):
        bad = r1 == r0
        if not bad.any():
            break
        r1[bad] = rng.integers(0, np1, size=int(bad.sum()))
    r2 = rng.integers(0, np2, size=n0)
    for _ in range(1000):
        bad = (r2 == r1) | (r2 == r0)
        if not bad.any():
            break
        r2[bad] = rng.integers(0, np2, size=int(bad.sum()))
    r3 = rng.integers(0, np1, size=n0)
    for _ in range(1000):
        bad = (r3 == r0) | (r3 == r1) | (r3 == r2)
        if not bad.any():
            break
        r3[bad] = rng.integers(0, np1, size=int(bad.sum()))
    return r1, r2, r3


def _ebo_han_boun(x: Array, hi: float, lo: float, ref: Array, mode: int) -> Array:
    """[EBOCMAR] `han_boun.m`. mode 1 = DE midpoint, mode 2 = CMA reflection."""
    x = x.copy()
    if mode == 1:
        p = x < lo
        x[p] = (ref[p] + lo) / 2.0
        p = x > hi
        x[p] = (ref[p] + hi) / 2.0
    else:
        p = x < lo
        x[p] = np.minimum(hi, np.maximum(lo, 2.0 * lo - ref[p]))
        p = x > hi
        # NOTE: the source uses 2*x_L here for BOTH branches (not 2*x_U).
        x[p] = np.maximum(lo, np.minimum(hi, 2.0 * lo - ref[p]))
    return x


def _ebo_lehmer(w: Array, v: Array) -> float:
    d = float(w @ v)
    return float(w @ (v ** 2)) / d if abs(d) > 1e-300 else 0.5


def alg_kmeans_pp(ev: "Evaluator", dim: int, bounds: Bounds, seed: int,
                  pop_size: int = None) -> AlgResult:
    """k-means++ seeding + Lloyd refinement + restarts -- the DOMAIN baseline.

    Omitting this would be the whole dishonesty of an application section: on
    a clustering objective the specialised method is what a reader actually
    compares against, and it is cheap. If it beats every metaheuristic here,
    that is the result and it gets reported.

    It reads the data matrix through `ev.aux`, which the general-purpose
    optimizers never touch. That is deliberate: a domain baseline should be
    given the information it would really have, not handicapped into a straw
    man that is easy to beat.

    BUDGET ACCOUNTING. One Lloyd iteration = one objective call. The
    assignment step and the SSE are the same computation, so each iteration
    reports its centroid set through `ev.evaluate_batch` exactly once and is
    charged exactly once. k-means++ seeding itself touches no candidate
    solution and is charged nothing, which matches how its cost is normally
    reported.
    """
    X = ev.aux.get("X")
    k = ev.aux.get("k")
    if X is None or k is None:
        raise RuntimeError("k-means++ requires ev.aux={'X':..,'k':..}; it is "
                           "a clustering-only baseline")
    rng = np.random.default_rng(seed)
    n, d = X.shape
    try:
        while not ev.exhausted():
            # ---- k-means++ seeding (Arthur & Vassilvitskii) ---------------
            C = np.empty((k, d), dtype=np.float64)
            C[0] = X[rng.integers(0, n)]
            d2 = np.sum((X - C[0]) ** 2, axis=1)
            for j in range(1, k):
                tot = d2.sum()
                if not np.isfinite(tot) or tot <= 0:
                    C[j] = X[rng.integers(0, n)]
                else:
                    C[j] = X[rng.choice(n, p=d2 / tot)]
                d2 = np.minimum(d2, np.sum((X - C[j]) ** 2, axis=1))

            # ---- Lloyd iterations ----------------------------------------
            # `prev` starts as None, NOT as +inf. With prev = inf the
            # tolerance 1e-12*max(1,|prev|) is itself inf, `inf <= inf` is
            # True, and the loop breaks after a single iteration -- every
            # restart would then report the raw k-means++ SEED with no Lloyd
            # refinement at all. That bug made this baseline look ~28 % worse
            # than it is (SSE 62.9 instead of 48.99 on wine k=3) and would
            # have turned the domain comparison into a straw man.
            prev = None
            for _ in range(300):
                if ev.exhausted():
                    break
                sse = float(ev.evaluate_batch(C.ravel().reshape(1, -1))[0])
                dist = np.sum((X[:, None, :] - C[None, :, :]) ** 2, axis=2)
                lab = np.argmin(dist, axis=1)
                for j in range(k):
                    m = lab == j
                    if m.any():
                        C[j] = X[m].mean(axis=0)
                    else:
                        # empty cluster: re-seed it on the worst-served point,
                        # the standard repair
                        C[j] = X[int(np.argmax(dist[np.arange(n), lab]))]
                if prev is not None and \
                        abs(prev - sse) <= 1e-12 * max(1.0, abs(prev)):
                    break
                prev = sse
    except BudgetExhausted:
        pass
    return _finish("k-means++", ev)


def alg_ebo_cmar(ev: "Evaluator", dim: int, bounds: Bounds, seed: int,
                 pop_size: int = None) -> AlgResult:
    """[EBOCMAR] Effective Butterfly Optimizer with CMA-adapted Retreat.

    A CEC2017 top method, added so the comparison is not open to the
    objection that the competition's strongest entries are missing.
    """
    rng = np.random.default_rng(seed)
    lo, hi = bounds
    max_fes = ev.max_fes

    # Introd_Par.m
    cs = {10: 50, 30: 100, 50: 150}.get(dim, 150)
    g_max = {10: 2163, 30: 2745, 50: 3022}.get(dim, 3401)
    ps1 = int(pop_size or 18 * dim)
    ps2 = 4 + int(math.floor(3.0 * math.log(dim)))
    min_pop, prob_ls = 4, 0.1
    init_pop = ps1

    try:
        X = lo + (hi - lo) * rng.random((ps1 + ps2, dim))
        fit = ev.evaluate_batch(X)
        order = np.argsort(fit)
        X, fit = X[order], fit[order]

        ea1, obj1 = X[:ps1].copy(), fit[:ps1].copy()
        ea1_old = ea1[rng.permutation(ps1)].copy()
        ea2, obj2 = X[ps1:].copy(), fit[ps1:].copy()

        # ---- init_cma_par.m ------------------------------------------------
        def init_cma(pop):
            s = {}
            s["xmean"] = pop.mean(axis=0)
            s["insigma"] = 0.3
            s["sigma"] = 0.3
            s["pc"] = np.zeros(dim)
            s["ps"] = np.zeros(dim)
            s["diagD"] = np.ones(dim)
            s["B"] = np.eye(dim)
            s["BD"] = np.eye(dim)
            s["C"] = np.eye(dim)
            s["chiN"] = dim ** 0.5 * (1 - 1 / (4 * dim) + 1 / (21 * dim ** 2))
            mu = int(math.ceil(ps2 / 2))
            s["mu"] = mu
            w = np.log(max(mu, dim / 2) + 0.5) - np.log(np.arange(1, mu + 1))
            s["mueff"] = float(w.sum() ** 2 / (w ** 2).sum())
            me = s["mueff"]
            s["cc"] = (4 + me / dim) / (dim + 4 + 2 * me / dim)
            s["cs"] = (me + 2) / (dim + me + 3)
            s["ccov1"] = 2 / ((dim + 1.3) ** 2 + me)
            s["ccovmu"] = 2 * (me - 2 + 1 / me) / ((dim + 2) ** 2 + me)
            s["damps"] = (0.5 + 0.5 * min(1.0, (0.27 * ps2 / me - 1) ** 2)
                          + 2 * max(0.0, math.sqrt((me - 1) / (dim + 1)) - 1)
                          + s["cs"])
            s["xold"] = s["xmean"].copy()
            return s

        st = init_cma(ea2)

        prob_de = np.full(2, 0.5)
        prob_sc = np.full(2, 0.5)
        arch_rate = 2.6
        arch_np = int(arch_rate * ps1)
        arch_X = np.zeros((0, dim))
        arch_f = np.zeros(0)

        mem = 6
        hist_pos = 0
        m_f = np.full(mem, 0.7)
        m_cr = np.full(mem, 0.5)
        m_t = np.full(mem, 0.1)
        m_freq = np.full(mem, 0.5)

        cy, indx, probs = 0, 0, np.ones(2)
        it = 0

        while not ev.exhausted():
            it += 1
            cy += 1

            # ---- EBO_BIN.m: pick the dominant phase, then share ------------
            if cy == int(math.ceil(cs + 1)):
                qual = np.array([obj1[0], obj2[0]], dtype=np.float64)
                sq = qual.sum()
                nq = 1.0 - (qual / sq if abs(sq) > 1e-300 else np.full(2, 0.5))
                d1 = float(np.mean(np.linalg.norm(ea1[1:] - ea1[0], axis=1))) \
                    if ps1 > 1 else 0.0
                d2 = float(np.mean(np.linalg.norm(ea2[1:] - ea2[0], axis=1))) \
                    if ps2 > 1 else 0.0
                dd = np.array([d1, d2])
                sd = dd.sum()
                nd = dd / sd if sd > 1e-300 else np.full(2, 0.5)
                p = nq + nd
                probs = np.clip(p / p.sum() if p.sum() > 1e-300 else np.full(2, 0.5),
                                0.1, 0.9)
                indx = int(np.argmax(probs))
                if probs[0] == probs[1]:
                    indx = -1              # MATLAB indx=0: no sharing
            elif cy == 2 * int(math.ceil(cs)):
                if indx == 0:              # MATLAB indx==1 -> DE feeds CMA
                    k = min(ps2, ps1)
                    pick = rng.permutation(ps1)[:k]
                    ea2[:k] = ea1[pick]
                    obj2[:k] = obj1[pick]
                    st = init_cma(ea2)
                    st["sigma"] *= (1.0 - ev.fes / max_fes)
                else:
                    if ea2[0].min() > lo and ea2[0].max() < hi:
                        ea1[ps1 - 1] = ea2[0]
                        obj1[ps1 - 1] = obj2[0]
                        o = np.argsort(obj1)
                        obj1, ea1, ea1_old = obj1[o], ea1[o], ea1_old[o]
                cy, probs = 1, np.ones(2)

            # ---- perching and patrolling (EBO.m) ---------------------------
            if not ev.exhausted() and rng.random() < probs[0]:
                upd = int(round(((min_pop - init_pop) / max_fes) * ev.fes + init_pop))
                if ps1 > upd:
                    ps1 = max(min_pop, upd)
                    ea1, ea1_old, obj1 = ea1[:ps1], ea1_old[:ps1], obj1[:ps1]
                    arch_np = int(round(arch_rate * ps1))
                    if arch_X.shape[0] > arch_np:
                        keep = rng.permutation(arch_X.shape[0])[:arch_np]
                        arch_X, arch_f = arch_X[keep], arch_f[keep]

                n = ps1
                ridx = rng.integers(0, mem, size=n)
                mu_sf, mu_cr = m_f[ridx], m_cr[ridx]
                mu_t, mu_fq = m_t[ridx], m_freq[ridx]

                # EBO.m: cr via a sum of two INDEPENDENT arcsine draws
                cr = mu_cr + 0.1 * math.sqrt(math.pi) * (
                    np.arcsin(-rng.random(n)) + np.arcsin(rng.random(n)))
                cr = np.clip(np.where(mu_cr == -1, 0.0, cr), 0.0, 1.0)

                F = mu_sf + 0.1 * np.tan(math.pi * (rng.random(n) - 0.5))
                for _ in range(100):
                    bad = F <= 0
                    if not bad.any():
                        break
                    F[bad] = mu_sf[bad] + 0.1 * np.tan(
                        math.pi * (rng.random(int(bad.sum())) - 0.5))
                F = np.minimum(F, 1.0)

                T = mu_t + 0.05 * math.sqrt(math.pi) * (
                    np.arcsin(-rng.random(n)) + np.arcsin(rng.random(n)))
                T = np.clip(T, 0.0, 0.5)

                freq = mu_fq + 0.1 * np.tan(math.pi * (rng.random(n) - 0.5))
                for _ in range(100):
                    bad = freq <= 0
                    if not bad.any():
                        break
                    freq[bad] = mu_fq[bad] + 0.1 * np.tan(
                        math.pi * (rng.random(int(bad.sum())) - 0.5))
                freq = np.minimum(freq, 1.0)

                # the per-coordinate CR profile, EBO.m lines 33-50
                if dim == 1:
                    CR = cr.reshape(-1, 1)
                else:
                    half = dim // 2
                    base = np.exp(-T[:, None] / dim * np.arange(half)[None, :])
                    if dim % 2 == 0:
                        prof = np.concatenate([base, base[:, ::-1]], axis=1)
                    else:
                        midv = np.exp(-T[:, None] / dim * float(half))
                        prof = np.concatenate([base, midv, base[:, ::-1]], axis=1)
                    prof = cr[:, None] * prof
                    start = rng.integers(0, dim, size=n)
                    cols = (np.arange(dim)[None, :] - start[:, None]) % dim
                    CR = np.take_along_axis(prof, cols, axis=1)

                # the sinusoidal F schedule, first half of the budget only
                if ev.fes <= max_fes / 2:
                    if rng.random() < 0.5:
                        F = np.full(n, 0.5 * (
                            math.tan(2 * math.pi * 0.5 * it + math.pi)
                            * ((g_max - it) / g_max) + 1.0))
                    else:
                        F = 0.5 * (np.tan(2 * math.pi * freq * it)
                                   * (it / g_max) + 1.0)

                pop_all = np.vstack([ea1, arch_X]) if arch_X.size else ea1
                r0 = np.arange(n)
                r1, r2, r3 = _ebo_gn_r1r2(n, pop_all.shape[0], r0, rng)

                bb = rng.random(n)
                op1 = bb <= prob_de[0]
                op2 = (~op1) & (bb <= prob_de[0] + prob_de[1])
                phix = ea1[_ebo_bestt_idx(n, dim, rng)]

                vi = np.zeros((n, dim))
                Fc = F[:, None]
                if op1.any():
                    vi[op1] = ea1[op1] + Fc[op1] * (
                        ea1[r1[op1]] - ea1[op1] + ea1[r3[op1]] - pop_all[r2[op1]])
                if op2.any():
                    vi[op2] = ea1[op2] + Fc[op2] * (
                        phix[op2] - ea1[op2] + ea1[r1[op2]] - ea1[r3[op2]])

                vi = _ebo_han_boun(vi, hi, lo, ea1, 1)
                mask = rng.random((n, dim)) > CR
                mask[np.arange(n), rng.integers(0, dim, size=n)] = False
                ui = np.where(mask, ea1, vi)

                new = ev.evaluate_batch(ui)
                diff = np.abs(obj1 - new)
                imp = new < obj1

                if arch_np > 0 and imp.any():
                    cand = np.vstack([arch_X, ea1[imp]])
                    candf = np.concatenate([arch_f, obj1[imp]])
                    _, uq = np.unique(cand, axis=0, return_index=True)
                    cand, candf = cand[uq], candf[uq]
                    if cand.shape[0] > arch_np:
                        keep = rng.permutation(cand.shape[0])[:arch_np]
                        cand, candf = cand[keep], candf[keep]
                    arch_X, arch_f = cand, candf

                with np.errstate(divide="ignore", invalid="ignore"):
                    d2 = np.maximum(0.0, obj1 - new) / np.abs(obj1)
                d2 = np.where(np.isfinite(d2), d2, 0.0)
                cnt = np.array([
                    max(0.0, float(np.mean(d2[op1]))) if op1.any() else 0.0,
                    max(0.0, float(np.mean(d2[op2]))) if op2.any() else 0.0])
                # EBO.m uses `if count_S~=0`, which in MATLAB is ALL-nonzero
                prob_de = (np.clip(cnt / cnt.sum(), 0.1, 0.9)
                           if np.all(cnt != 0) else np.full(2, 0.5))

                ea1_old[imp] = ea1[imp]
                ea1[imp] = ui[imp]
                obj1[imp] = new[imp]

                if imp.any():
                    w = diff[imp] / diff[imp].sum() if diff[imp].sum() > 0 \
                        else np.full(int(imp.sum()), 1.0 / int(imp.sum()))
                    m_f[hist_pos] = _ebo_lehmer(w, F[imp])
                    gcr = cr[imp]
                    m_cr[hist_pos] = (-1.0 if (gcr.max() == 0 or m_cr[hist_pos] == -1)
                                      else _ebo_lehmer(w, gcr))
                    # (Q2) the source advances hist_pos HERE, so T and freq are
                    # written to the next slot. Reproduced, not corrected.
                    hist_pos = (hist_pos + 1) % mem
                    m_t[hist_pos] = _ebo_lehmer(w, T[imp])
                    gfq = freq[imp]
                    m_freq[hist_pos] = (-1.0 if (gfq.max() == 0
                                                 or m_freq[hist_pos] == -1)
                                        else _ebo_lehmer(w, gfq))

                o = np.argsort(obj1)
                obj1, ea1, ea1_old = obj1[o], ea1[o], ea1_old[o]

            # ---- Scout / CMAR phase (Scout.m) ------------------------------
            if not ev.exhausted() and rng.random() < probs[1]:
                if rng.random() < prob_sc[0]:
                    arz = math.sqrt(math.pi) * (
                        np.arcsin(rng.random((dim, ps2)))
                        + np.arcsin(-rng.random((dim, ps2))))
                else:
                    arz = math.sqrt(math.pi) * np.arcsin(
                        2.0 * rng.random((dim, ps2)) - 1.0)
                arx = st["xmean"][:, None] + st["sigma"] * (st["BD"] @ arz)

                if ev.fes >= 0.5 * max_fes:
                    valid = _ebo_han_boun(arx.T, hi, lo, ea2, 2)
                else:
                    valid = arx.T.copy()

                fv = ev.evaluate_batch(valid)
                o = np.argsort(fv)
                fv, valid, arx, arz = fv[o], valid[o], arx[:, o], arz[:, o]

                mu = st["mu"]
                # (Q1) `fliplr` on a column vector is a no-op in MATLAB, so the
                # weights stay in ASCENDING-fitness order. Reproduced verbatim.
                wq = fv[:mu].astype(np.float64).copy()
                if wq.sum() > 1e25 or not np.isfinite(wq.sum()) or wq.sum() == 0:
                    wq = np.full(mu, 1.0 / mu)
                else:
                    wq = wq / wq.sum()

                st["xold"] = st["xmean"].copy()
                st["xmean"] = arx[:, :mu] @ wq
                if ev.fes >= 0.5 * max_fes:
                    st["xmean"] = _ebo_han_boun(
                        st["xmean"].reshape(1, -1), hi, lo,
                        ea2[0].reshape(1, -1), 2).ravel()
                zmean = arz[:, :mu] @ wq

                cs_, cc_ = st["cs"], st["cc"]
                st["ps"] = ((1 - cs_) * st["ps"]
                            + math.sqrt(cs_ * (2 - cs_) * st["mueff"])
                            * (st["B"] @ zmean))
                hsig = (np.linalg.norm(st["ps"])
                        / math.sqrt(1 - (1 - cs_) ** (2 * it))
                        / st["chiN"]) < (1.4 + 2 / (dim + 1))
                st["pc"] = ((1 - cc_) * st["pc"]
                            + (math.sqrt(cc_ * (2 - cc_) * st["mueff"])
                               / st["sigma"]) * hsig * (st["xmean"] - st["xold"]))

                arpos = (arx[:, :mu] - st["xold"][:, None]) / st["sigma"]
                st["C"] = ((1 - st["ccov1"] - st["ccovmu"]) * st["C"]
                           + st["ccov1"] * np.outer(st["pc"], st["pc"])
                           + st["ccovmu"] * (arpos * wq[None, :]) @ arpos.T)

                st["sigma"] *= _safe_exp(min(
                    1.0, (float(np.linalg.norm(st["ps"])) / st["chiN"] - 1.0)
                    * cs_ / st["damps"]))

                rate = st["ccov1"] + st["ccovmu"]
                if rate > 0 and (it % max(1, int(1.0 / rate / dim / 10)) if
                                 (1.0 / rate / dim / 10) >= 1 else 0) < 1:
                    Cs = np.triu(st["C"]) + np.triu(st["C"], 1).T
                    try:
                        dvals, B = np.linalg.eigh(Cs)
                    except np.linalg.LinAlgError:
                        dvals, B = np.ones(dim), np.eye(dim)
                    if not np.all(np.isfinite(dvals)):
                        dvals, B, Cs = np.ones(dim), np.eye(dim), np.eye(dim)
                    dvals = np.where(dvals < 0, 0.0, dvals)
                    if dvals.min() <= 0:
                        t = dvals.max() / 1e14
                        Cs = Cs + t * np.eye(dim)
                        dvals = dvals + t
                    if dvals.max() > 1e14 * max(dvals.min(), 1e-300):
                        t = dvals.max() / 1e14 - dvals.min()
                        Cs = Cs + t * np.eye(dim)
                        dvals = dvals + t
                    st["C"] = Cs
                    st["B"] = B
                    st["diagD"] = np.sqrt(dvals)
                    st["BD"] = B * st["diagD"][None, :]

                ea2, obj2 = valid, fv

            # ---- LS2.m: SQP local search on the incumbent -------------------
            if ev.fes > 0.75 * max_fes and rng.random() < prob_ls:
                # LS_FE = ceil(20e-3 * Max_FES) in LS2.m
                ls_budget = min(int(math.ceil(0.02 * max_fes)), ev.remaining())
                f0 = ev.best_fitness
                if ls_budget > 0:
                    x0 = np.clip(
                        (ev.best_x if ev.best_x is not None else ea1[0]).copy(),
                        lo, hi)
                    spent = [0]
                    # A flat finite penalty, NOT an exception. Raising a Python
                    # exception from inside SLSQP's Fortran layer kills the
                    # worker process, and a dead worker makes the enclosing
                    # ProcessPoolExecutor wait forever -- which is exactly how
                    # the first version of this port deadlocked a whole sweep.
                    pen = (f0 + abs(f0) * 10.0 + 1e6
                           if math.isfinite(f0) else 1e30)

                    def _obj(z):
                        if spent[0] >= ls_budget or ev.remaining() <= 0:
                            return pen
                        spent[0] += 1
                        v = float(ev.evaluate_batch(
                            np.clip(z, lo, hi).reshape(1, -1))[0])
                        return v if math.isfinite(v) else pen

                    try:
                        minimize(_obj, x0, method="SLSQP",
                                 bounds=[(lo, hi)] * dim,
                                 options={"maxiter": 100, "ftol": 1e-12})
                    except Exception:
                        pass

                if ev.best_fitness < f0:          # LS2 succeeded
                    prob_ls = 0.1
                    ea1[ps1 - 1] = np.clip(ev.best_x, lo, hi)
                    obj1[ps1 - 1] = ev.best_fitness
                    o = np.argsort(obj1)
                    obj1, ea1, ea1_old = obj1[o], ea1[o], ea1_old[o]
                    ea2 = np.repeat(ea1[0].reshape(1, -1), ps2, axis=0)
                    st = init_cma(ea2)
                    st["sigma"] = 1e-05
                    obj2 = np.full(ps2, obj1[0])
                else:
                    prob_ls = 0.01

    except BudgetExhausted:
        pass
    return _finish("EBOwithCMAR", ev)


def _var(name: str, **over):
    kw = dict(_BASE)
    kw.update(over)
    return _mk_mdac(_name=name, **kw)


# Each row removes or replaces exactly ONE mechanism of the shipped
# configuration, so every rank difference is attributable.
MDAC_ABLATIONS: Dict[str, Callable] = {
    "MDAC": alg_mdac,
    # -- the hypothesis currently under test (A7): the three possible rules
    #    for the metric generator's crossover rate.
    "MDAC-crFull":       _var("MDAC-crFull", cma_cr_mode="full"),
    "MDAC-crShared":     _var("MDAC-crShared", cma_cr_mode="shared"),
    # -- our own contributions
    "MDAC-noDiv":        _var("MDAC-noDiv", diversity_nu=0.0),
    "MDAC-allocFixed":   _var("MDAC-allocFixed", alloc="fixed"),
    "MDAC-allocRandom":  _var("MDAC-allocRandom", alloc="random"),
    "MDAC-allocWide":    _var("MDAC-allocWide", alloc="wide"),
    "MDAC-dimAware":     _var("MDAC-dimAware", alloc="dimaware"),   # (A8)
    # (A14) metric accuracy vs operator diversity -- same slot, same A7
    # crossover, same allocation; only the generator differs.
    "MDAC-ctr":          _var("MDAC-ctr", local_gen="ctr"),
    # -- (A3) PARAMETER SENSITIVITY. Four constants that were set once and
    #    never varied; `MDAC_NU` additionally carries a tuning-on-test
    #    disclosure. Each row moves exactly ONE of them, so a rank difference
    #    is attributable. Default values are n_L=12, T0_frac=0.10,
    #    cr_loc_init=0.9, MDAC_NU=0.6 -- the shipped `MDAC` row is the centre
    #    point and is not repeated here.
    "MDAC-nL6":          _var("MDAC-nL6", n_L=6),
    "MDAC-nL24":         _var("MDAC-nL24", n_L=24),
    "MDAC-T0_05":        _var("MDAC-T0_05", T0_frac=0.05),
    "MDAC-T0_20":        _var("MDAC-T0_20", T0_frac=0.20),
    "MDAC-cr07":         _var("MDAC-cr07", cr_loc_init=0.7),
    "MDAC-cr10":         _var("MDAC-cr10", cr_loc_init=1.0),
    "MDAC-nu03":         _var("MDAC-nu03", diversity_nu=0.3),
    "MDAC-nu09":         _var("MDAC-nu09", diversity_nu=0.9),
    "MDAC-allocTerminal": _var("MDAC-allocTerminal", alloc="terminal"),
    "MDAC-cmaRestart":   _var("MDAC-cmaRestart", cma_restart=True),
    "MDAC-seedCov":      _var("MDAC-seedCov", cma_seed_cov=True),
    # -- borrowed mechanisms
    "MDAC-noFCP":        _var("MDAC-noFCP", fcp=False),
    "MDAC-noMetro":      _var("MDAC-noMetro", metropolis=False),
    # -- the momentum repair hypothesis (both readings REFUTED, kept as
    #    documented negative results for the ablation table)
    "MDAC-mom":          _var("MDAC-mom", momentum=True, mom_archive=False),
    "MDAC-momArch":      _var("MDAC-momArch", momentum=True, mom_archive=True),
}

# A SMALL comparison family, for the one question that must reach
# significance. Statistical power, computed rather than hoped for:
#   k competitors x 60 cells = 60k comparisons in the Holm family.
#   Smallest attainable two-sided Wilcoxon p at n runs is 2 / 2^n.
#   Significance requires   2/2^n  <=  0.05 / (60k).
#   k=3, n=9  -> 3.9e-3 vs 2.8e-4   IMPOSSIBLE  (the BR_* design stage)
#   k=3, n=20 -> 1.9e-6 vs 2.8e-4   reachable
#   k=3, n=25 -> 6.0e-8 vs 2.8e-4   comfortable
# This is the same defect we flagged in the earlier study's 5-run control
# and then reproduced twice in our own runs; the fix is fewer variants per
# family and more runs, not a looser correction.
SCREEN_ALGS = ["MDAC", "MDAC-crFull", "MDAC-crShared", "jSO", "L-SHADE"]
# The CEC2017 competition's own top methods, which our declared target
# categories are taken from. Beating L-SHADE is necessary but not
# sufficient: these are what a reviewer will ask about.
# EBOwithCMAR added 2026-10-08: it is a CEC2017 top method and its absence
# was the most visible gap in the comparison. Ported from the authors' own
# MATLAB (see `alg_ebo_cmar`), so the family size for Holm grows to
# 5 competitors x 126 cells = 630; min attainable p at 51 runs is 8.9e-16
# against a threshold of 0.05/630 = 7.9e-5, so power is still comfortable.
TOP_ALGS = ["MDAC", "jSO", "L-SHADE-SPACMA", "L-SHADE", "EBOwithCMAR", "CMA-ES"]

# The definitive comparison set. Adds the two variants that answer the
# open mechanism questions, so the main table and those answers come from
# ONE run rather than from separate screens stitched together:
#   MDAC-ctr    (A14) the metric generator replaced by current-to-rand/1 --
#               rotation-invariant, no learned metric, free. Tests whether
#               the second generator's benefit is METRIC ACCURACY or merely
#               OPERATOR DIVERSITY.
#   MDAC-noFCP  (C2)  the metric generator removed entirely. Tests whether
#               the co-allocation earns its 24-60 % of runtime at all.
# Both occupy the same slot under the same allocation rule and the same A7
# crossover treatment, so the three MDAC rows differ in exactly one thing.
FULL_ALGS = ["MDAC", "MDAC-ctr", "MDAC-noFCP",
             "jSO", "L-SHADE-SPACMA", "L-SHADE", "EBOwithCMAR", "CMA-ES"]

# The official CEC2017 dimensions. D=20 is excluded: only 19 of its 30
# rotation files are installed, and a partial suite is not a suite.
FULL_DIMS = [10, 30, 50, 100]


# ===========================================================================
# 7. ARCHITECTURAL-PARITY CONTROL
# ===========================================================================
BASE_REGISTRY: Dict[str, Callable] = {
    "DE": alg_de,
    "PSO": alg_pso,
    "CMA-ES": alg_cmaes,
    "L-SHADE": alg_lshade,
    "jSO": alg_jso,
    "L-SHADE-MA": alg_lshade_ma,
    "L-SHADE-SPACMA": alg_lshade_spacma,
    "A-GDESA-ng": alg_agdesa_ng,
    "A-GDESA-fd": alg_agdesa_fd,
    "EBOwithCMAR": alg_ebo_cmar,
    "k-means++": alg_kmeans_pp,        # (track B) clustering domain baseline
    "MDAC": alg_mdac,
}

# (Track B) the clustering comparison: the five general-purpose optimizers
# plus the specialised domain baseline. Power, precomputed:
#   m = 6 competitors x (#cells). With 6 cells, m = 36 -> threshold 1.4e-3;
#   smallest attainable two-sided Wilcoxon p at 51 runs is 2/2^51 = 8.9e-16.
#   Reachable with a wide margin.
CLUSTER_ALGS = ["MDAC", "jSO", "L-SHADE-SPACMA", "L-SHADE", "EBOwithCMAR",
                "CMA-ES", "k-means++"]


class SpreadEvaluator(Evaluator):
    """Records the RMS radius of the last multi-row batch, so a baseline's
    local stage can be seeded with the same quantity MDAC seeds its own
    with. Changes no accounting."""
    __slots__ = ("last_spread",)

    def __init__(self, *a, **kw):
        super().__init__(*a, **kw)
        self.last_spread = 0.0

    def evaluate_batch(self, X: Array) -> Array:
        vals = super().evaluate_batch(X)
        X = np.asarray(X)
        if X.ndim == 2 and X.shape[0] >= 2:
            self.last_spread = float(np.sqrt(np.mean(np.sum(
                (X - X.mean(axis=0)) ** 2, axis=1))))
        return vals


def equip_terminal(name: str, handover: float = 0.70) -> Callable:
    """Give ANY algorithm the proposed local stage at the same handover out
    of the same shared budget, then re-rank.

    This is the control that matters. Our earlier study's measured rank
    advantage collapsed under it, which is
    why the claim in this work is about the ALLOCATION RULE and not about
    possessing a local searcher.

    FIX vs the earlier implementation: `pop_size` is passed as None, so
    every algorithm derives its own canonical population size. The previous
    version forced pop_size=30 on L-SHADE, overriding its published
    NP0 = 18*D and handicapping the strongest competitor in the very
    control that was supposed to be fair.
    """
    base = BASE_REGISTRY[name]

    def run(ev: Evaluator, dim, bounds, seed, pop_size=None) -> AlgResult:
        full = ev.max_fes
        ev.max_fes = max(1, int(round(handover * full)))
        try:
            base(ev, dim, bounds, seed, pop_size=None)
        except BudgetExhausted:
            pass
        finally:
            ev.max_fes = full
        if ev.best_x is not None and not ev.exhausted():
            s0 = max(float(getattr(ev, "last_spread", 0.0)), 1e-18)
            cmaes_local(ev, ev.best_x, ev.best_fitness, bounds,
                        sigma0=s0, seed=seed, restarts=True)
        return _finish(f"{name}+cma", ev)

    return run


# ===========================================================================
# 8. STATISTICS
# ===========================================================================
def a12(x: Array, y: Array) -> float:
    """Vargha-Delaney A12 for MINIMISATION: P(x better than y)."""
    x, y = -np.asarray(x, float), -np.asarray(y, float)
    m, n = len(x), len(y)
    r = rankdata(np.concatenate([x, y]), method="average")
    return float((r[:m].sum() / m - (m + 1) / 2.0) / n)


def a12_label(v: float) -> str:
    d = abs(v - 0.5)
    return ("Negligible" if d < 0.06 else "Small" if d < 0.14
            else "Medium" if d < 0.21 else "Large")


def holm(pvals: Dict[str, float], alpha: float = 0.05) -> Dict[str, bool]:
    items = sorted(pvals.items(), key=lambda kv: kv[1])
    m = len(items)
    out, still = {}, True
    for i, (k, p) in enumerate(items, start=1):
        thr = alpha / (m - i + 1)
        rej = bool(still and p <= thr)
        if not rej:
            still = False
        out[k] = rej
    return out


def paired_p(x: Array, y: Array) -> float:
    x, y = np.asarray(x, float), np.asarray(y, float)
    if len(x) < 2 or np.allclose(x, y):
        return 1.0
    try:
        return float(wilcoxon(x, y, zero_method="wilcox",
                              alternative="two-sided").pvalue)
    except ValueError:
        return 1.0


def friedman_ranks(per_alg: Dict[str, Array], algs: List[str]):
    M = np.vstack([per_alg[a] for a in algs]).T
    R = np.vstack([rankdata(row, method="average") for row in M])
    try:
        p = float(friedmanchisquare(*[per_alg[a] for a in algs]).pvalue)
    except ValueError:
        p = 1.0
    return p, dict(zip(algs, R.mean(axis=0)))


def cec_friedman(cells, keys: List[Tuple[str, int]], algs: List[str],
                 ref: str) -> Tuple[List[Dict], Dict]:
    """Friedman test in the form CEC papers use: one rank per (function, dim)
    cell, computed from the MEAN error of the runs (1e-8 convention).

    The per-run ranks in `friedman_ranks` reward an algorithm with good typical
    runs and occasional bad ones; ranking means does not. Both are reported,
    because they disagree on this data.

    Post-hoc, with `ref` as control (Demsar 2006; Derrac et al. 2011):
      * z-test on mean ranks, Holm-corrected over the k-1 comparisons, plus
        the Nemenyi critical difference for all-pairs reading;
      * Wilcoxon signed-rank over the cells on mean errors, Holm-corrected.
    Cells are (function, dim) pairs, so the same function appears once per
    dimension and the cells are not fully independent. Say so when quoting.
    """
    k, n = len(algs), len(keys)
    means = np.array([[float(np.mean(cells[c][a])) for a in algs] for c in keys])
    meds = np.array([[float(np.median(cells[c][a])) for a in algs] for c in keys])
    r_mean = np.vstack([rankdata(row) for row in means]).mean(axis=0)
    r_med = np.vstack([rankdata(row) for row in meds]).mean(axis=0)
    try:
        fr = friedmanchisquare(*means.T)
        chi2, p_fr = float(fr.statistic), float(fr.pvalue)
    except ValueError:
        chi2, p_fr = float("nan"), 1.0
    den = n * (k - 1) - chi2
    f_id = (n - 1) * chi2 / den if den > 0 else float("nan")
    se = math.sqrt(k * (k + 1) / (6.0 * n))
    cd = float(studentized_range.ppf(0.95, k, np.inf)) / math.sqrt(2.0) * se

    i0 = algs.index(ref)
    p_z, p_w, counts = {}, {}, {}
    for j, a in enumerate(algs):
        if a == ref:
            continue
        z = (r_mean[j] - r_mean[i0]) / se
        p_z[a] = float(2.0 * norm.sf(abs(z)))
        x, y = means[:, i0], means[:, j]
        counts[a] = (int(np.sum(x < y)), int(np.sum(x == y)), int(np.sum(x > y)))
        p_w[a] = paired_p(x, y)
    rej_z, rej_w = holm(p_z), holm(p_w)

    order = np.argsort(r_mean)
    rows = []
    for pos, j in enumerate(order, 1):
        a = algs[j]
        rec = {"algorithm": a, "position": pos, "mean_rank": float(r_mean[j]),
               "median_rank": float(r_med[j]),
               "rank_diff_vs_ref": float(r_mean[j] - r_mean[i0])}
        if a != ref:
            b, e, w = counts[a]
            rec.update({"z_p": p_z[a], "z_holm_reject": rej_z[a],
                        "beyond_nemenyi_cd": abs(r_mean[j] - r_mean[i0]) > cd,
                        "ref_better_cells": b, "equal_cells": e,
                        "ref_worse_cells": w, "wilcoxon_p": p_w[a],
                        "wilcoxon_holm_reject": rej_w[a]})
        rows.append(rec)
    summary = {"cells": n, "k": k, "friedman_chi2": chi2, "friedman_p": p_fr,
               "iman_davenport_F": f_id, "nemenyi_cd_0.05": cd,
               "mean_rank": {algs[j]: float(r_mean[j]) for j in range(k)},
               "median_rank": {algs[j]: float(r_med[j]) for j in range(k)}}
    return rows, summary


def write_csv(path: Path, rows: List[Dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    cols, seen = [], set()
    for r in rows:
        for k in r:
            if k not in seen:
                seen.add(k)
                cols.append(k)
    with open(path, "w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=cols)
        w.writeheader()
        for r in rows:
            w.writerow(r)


# ===========================================================================
# 9. WORKER + DRIVER
# ===========================================================================
def resolve(label: str) -> Callable:
    if label.endswith("+cma"):
        return equip_terminal(label[:-4])
    if label in MDAC_ABLATIONS:
        return MDAC_ABLATIONS[label]
    return BASE_REGISTRY[label]


# A1 EQUAL-TIME BUDGET, per (function, dim), in `equal_time_budget.json`.
#
#     Budget[f][D] = median over the SIX algorithms of their median task
#                    seconds on (f, D) in `mdac_fixed`
#
# i.e. the wall-clock the typical competition entry needed for the standard
# 10000*D protocol ON THAT FUNCTION. A flat per-dimension budget was tried
# first and is wrong: on a cheap function every algorithm then runs millions
# of evaluations and the cell saturates into a tie, while an expensive
# function starves. Per-function keeps each cell at the difficulty the
# standard protocol gave it.
#
# Median, not slowest or fastest, so the budget favours no one. Measured
# medians at D=50: MDAC 78.4 s, EBOwithCMAR 10.9, SPACMA 7.8, L-SHADE 5.3,
# CMA-ES 4.8, jSO 3.7 -- so MDAC receives roughly a twelfth of its usual
# evaluations here. That is the honest price of its per-evaluation overhead,
# and whether the advantage survives it is the whole question.
#
# NOT PORTABLE: these seconds were measured on one machine (i5-13400F) at 15
# workers. Re-measure before quoting them anywhere else.
_ETB_CACHE: Dict = {}


def equal_time_budget(function: str, dim: int) -> float:
    if not _ETB_CACHE:
        p = Path(__file__).resolve().parent / "equal_time_budget.json"
        if not p.is_file():
            raise SystemExit(
                "--time_budget needs equal_time_budget.json, which is built "
                "from a completed equal-FES run. Run the standard protocol "
                "first, then regenerate it.")
        _ETB_CACHE.update(json.loads(p.read_text(encoding="utf-8"))["budget"])
    try:
        return float(_ETB_CACHE[function][str(dim)])
    except KeyError:
        raise SystemExit(
            f"no equal-time budget for ({function}, D={dim}); regenerate "
            f"equal_time_budget.json from a run that covers it")


def worker(task: Tuple) -> Dict:
    (label, fid, dim, run_id, fes_mult, base_seed, include_bridge, cec,
     time_budget) = task
    spec = next(s for s in build_registry(dim, base_seed, include_bridge,
                                          cec=cec) if s.fid == fid)
    seed = 42 + run_id * 1000 + fid * 17      # independent of algorithm name
    if time_budget:
        secs = equal_time_budget(spec.name, dim)
        # max_fes is set far beyond anything reachable so that WALL-CLOCK is
        # the only binding constraint.
        ev = SpreadEvaluator(spec.func, dim, max_fes=10 ** 9,
                             f_star=spec.f_star, record_stride=max(1, dim // 2),
                             max_seconds=secs)
    else:
        ev = SpreadEvaluator(spec.func, dim, max_fes=_max_fes(spec, fes_mult, dim),
                             f_star=spec.f_star, record_stride=max(1, dim // 2))
    budget = _max_fes(spec, fes_mult, dim)
    use_thr = spec.category not in RAW_CATEGORIES
    if spec.category == "clustering":
        # domain data for the k-means baseline only; see `alg_kmeans_pp`
        from benchmarks.clustering import load_dataset
        ds = spec.name.split("_")[1]
        ev.aux = {"X": load_dataset(ds), "k": int(spec.name.split("_k")[1])}
    t0 = time.perf_counter()
    try:
        resolve(label)(ev, dim, spec.bounds, seed, pop_size=None)
    except BudgetExhausted:
        pass
    except Exception as exc:                  # never lose a whole sweep
        return {"label": label, "function": spec.name, "category": spec.category,
                "dim": dim, "run": run_id, "error_thr": float("inf"),
                "error_raw": float("inf"), "fes_used": ev.fes,
                "fes_frac": _budget_frac(ev, budget, time_budget),
                "secs": 0.0, "failure": f"{type(exc).__name__}: {exc}"}
    return {"label": label, "function": spec.name, "category": spec.category,
            "dim": dim, "run": run_id,
            "error_thr": ev.error(threshold=use_thr, clamp=use_thr),
            "error_raw": ev.error(threshold=False, clamp=use_thr),
            "fes_used": ev.fes,
            "fes_frac": _budget_frac(ev, budget, time_budget),
            "secs": time.perf_counter() - t0, "failure": ""}


def _budget_frac(ev, max_fes: int, time_budget: bool) -> float:
    """Fraction of the BINDING budget that was consumed.

    In FES mode that is fes/max_fes, which the integrity gate expects to be
    exactly 1.0000 for every algorithm. In equal-time mode the binding budget
    is wall-clock, so the same gate must be applied to the clock instead --
    otherwise an equal-time run would look like a budget-parity violation.
    """
    if time_budget:
        return min(1.0, ev.time_frac())
    return ev.fes / max_fes


MAIN_ALGS = ["MDAC", "jSO", "L-SHADE-SPACMA", "L-SHADE", "L-SHADE-MA",
             "CMA-ES", "A-GDESA-ng", "A-GDESA-fd", "DE", "PSO"]
PARITY_ALGS = ["MDAC", "MDAC-allocTerminal"] + \
              [f"{a}+cma" for a in ["L-SHADE", "L-SHADE-MA",
                                    "L-SHADE-SPACMA", "CMA-ES",
                                    "A-GDESA-ng", "DE", "PSO"]]
ABLATION_ALGS = list(MDAC_ABLATIONS) + ["L-SHADE"]

STAGES = {
    #            algs            dims          runs  fes    bridge
    "smoke":    (["MDAC", "L-SHADE", "CMA-ES"], [5],   2,    300,  False),
    "pilot":    (MAIN_ALGS,      [10],           5,   2000,  False),
    # Design stage: pick the configuration on the TARGET block only, at
    # moderate run count. Run `--only BR_` to restrict to the bridge.
    "ablation": (ABLATION_ALGS,  [10, 30],       9,  10000,  True),
    # High-power screen of ONE hypothesis against the algorithms that matter.
    "screen":   (SCREEN_ALGS,    [10, 30],      20,  10000,  True),
    # Confirmatory run against the CEC2017 competition's top methods.
    "top":      (TOP_ALGS,       [10, 30, 50],  25,  10000,  True),
    # H2 confirmatory axis: D=100, never used for any design decision.
    "confirm":  (TOP_ALGS,       [100],         25,  10000,  True),
    # Confirmatory stage: the declared claim, full suite, high power.
    "main":     (MAIN_ALGS,      [10, 30, 50],  25,  10000,  True),
    "parity":   (PARITY_ALGS,    [10, 30],      15,  10000,  True),
    # ---- the definitive experiment, run once, start to finish -----------
    # Full protocol: official dimensions, the competition's own 51 runs.
    "full":     (FULL_ALGS,  FULL_DIMS,         51,  10000,  True),
    # Every MDAC variant at the SAME power as the main table, so the
    # ablation is not a lower-powered afterthought. D=100 is omitted here
    # only because 24 variants at D=100 would dominate the whole budget.
    "abl51":    (ABLATION_ALGS, [10, 30, 50],   51,  10000,  True),
    # CEC2011 real-world problems (run with --cec cec2011). Each problem has
    # its own fixed D; the budget is the competition's 150 000 evaluations
    # (CEC2011_MAX_FES), so the `fes` column is unused here.
    "rw":       (FULL_ALGS,  CEC2011_DIMS,      51,      0,  False),
    # CEC2020 mechanical engineering problems (run with --cec cec2020mech);
    # official MaxFEs per D, so the `fes` column is unused here.
    "mech":     (FULL_ALGS,  CEC2020_MECH_DIMS, 51,      0,  False),
}


def _benchmark_label(functions) -> str:
    """Describe the suite from the function names that are actually present."""
    fs = set(functions)
    parts = []
    if any(f.startswith("CL_") for f in fs):
        parts.append("CL_* shifted classics")
    if any(f.startswith("CEC2017_") for f in fs):
        parts.append("CEC2017_* official, C-exact (crosscheck_cec2017_c.py "
                     "30/30)")
    if any(f.startswith("BR_") for f in fs):
        parts.append("BR_* bridge (NOT official CEC2017)")
    if any(f.startswith("CL2_") for f in fs):
        parts.append("CL2_* clustering on real UCI data")
    if any(f.startswith("CEC2011_") for f in fs):
        parts.append("CEC2011_* real-world problems, ported from the official "
                     "MATLAB code (crosscheck_cec2011_octave.py)")
    if any(f.startswith("CEC2020_") for f in fs):
        parts.append("CEC2020_* mechanical constrained problems, ported from "
                     "the official MATLAB code (crosscheck_cec2020rw_octave.py)")
    return " + ".join(parts) or "unknown"


def analyse(rows: List[Dict], algs: List[str], ref: str, out: Path,
            tag: str, write_raw: bool = True) -> None:
    out.mkdir(parents=True, exist_ok=True)
    if write_raw:
        write_csv(out / f"raw_{tag}.csv", rows)

    # ---- HARD FAILURE GATE ------------------------------------------------
    # A crashed run is recorded as +inf. Ranking +inf against real values
    # yields a table that LOOKS publishable and is pure fabrication: an
    # earlier bug here crashed 1274 of 1680 runs and produced an ablation
    # in which all eight variants tied at rank 6.5. Any failure now stops
    # the analysis instead of being reported in a line that scrolls past.
    fails = [r for r in rows if r.get("failure")]
    if fails:
        by_label = defaultdict(int)
        by_msg = defaultdict(int)
        for r in fails:
            by_label[r["label"]] += 1
            by_msg[r["failure"]] += 1
        print(f"\n  {'!'*74}")
        print(f"  ABORT: {len(fails)} of {len(rows)} runs failed "
              f"({100*len(fails)/len(rows):.1f}%).")
        print(f"  Failed runs are recorded as +inf and would corrupt every "
              f"rank below.\n  {'!'*74}")
        for lab, c in sorted(by_label.items(), key=lambda kv: -kv[1]):
            print(f"    {lab:24s} {c:5d} failed")
        print("  distinct errors:")
        for msg, c in sorted(by_msg.items(), key=lambda kv: -kv[1])[:8]:
            print(f"    {c:5d} x {msg}")
        print(f"\n  raw rows written to raw_{tag}.csv for diagnosis. "
              f"No statistics produced.")
        if not os.environ.get("MDAC_ALLOW_FAILURES"):
            raise SystemExit(2)
        print("  MDAC_ALLOW_FAILURES set -- continuing on corrupted data.")

    bench = _benchmark_label(r["function"] for r in rows)
    frac = defaultdict(list)
    for r in rows:
        frac[r["label"]].append(r["fes_frac"])

    print(f"\n  budget utilisation (fraction of MaxFES):")
    for a in algs:
        v = frac.get(a, [0.0])
        flag = "" if min(v) > 0.999 else "   <-- under-spends"
        print(f"    {a:22s} min {min(v):.4f}  mean {sum(v)/len(v):.4f}{flag}")

    # ---- functions left out of every statistic (raw rows are kept) --------
    excl = () if os.environ.get("MDAC_KEEP_F2") else EXCLUDED_FROM_STATS
    dropped = sorted({r["function"] for r in rows
                      if r["function"].startswith(excl)})
    if dropped:
        rows = [r for r in rows if not r["function"].startswith(excl)]
        print(f"\n  excluded from statistics (raw rows kept): "
              f"{', '.join(dropped)}")

    # ---- pair runs by RUN INDEX ---------------------------------------------
    # Rows arrive in completion order from the process pool (and from the
    # checkpoint), so a row's list position is not its run. Pairing by
    # position matched run i of one algorithm with run j of another in every
    # cell of the 2026-10-16 analysis (found 2026-10-31). Runs with the same
    # index share a seed, so the index is the pairing the design intends.
    by_run: Dict[Tuple[str, int], Dict[str, Dict[int, Tuple[float, float]]]] = \
        defaultdict(lambda: defaultdict(dict))
    cat_of = {}
    for r in rows:
        key = (r["function"], int(r["dim"]))
        by_run[key][r["label"]][int(r["run"])] = (r["error_thr"],
                                                  r["error_raw"])
        cat_of[r["function"]] = r["category"]
    cells: Dict[Tuple[str, int], Dict[str, List[float]]] = defaultdict(dict)
    cells_raw: Dict[Tuple[str, int], Dict[str, List[float]]] = defaultdict(dict)
    for key, per_alg in by_run.items():
        for a, d in per_alg.items():
            idx = sorted(d)
            cells[key][a] = [d[i][0] for i in idx]
            cells_raw[key][a] = [d[i][1] for i in idx]

    present = [k for k, d in by_run.items()
               if all(len(d.get(a, {})) for a in algs)]
    present.sort()
    unpaired = [k for k in present
                if len({tuple(sorted(by_run[k][a])) for a in algs}) > 1]
    if unpaired:
        print(f"\n  ABORT: {len(unpaired)} cells have different run indices "
              f"across algorithms, so runs cannot be paired. First: "
              f"{unpaired[0]}")
        raise SystemExit(2)

    # ---- Friedman ranks, thresholded and raw ---------------------------
    summary_rows, rank_rows = [], []
    acc_thr = defaultdict(list)
    acc_raw = defaultdict(list)
    acc_cat = defaultdict(lambda: defaultdict(list))
    for key in present:
        fn, dim = key
        per = {a: np.array(cells[key][a]) for a in algs}
        per_raw = {a: np.array(cells_raw[key][a]) for a in algs}
        p_thr, rk_thr = friedman_ranks(per, algs)
        _, rk_raw = friedman_ranks(per_raw, algs)
        for a in algs:
            acc_thr[a].append(rk_thr[a])
            acc_raw[a].append(rk_raw[a])
            acc_cat[cat_of[fn]][a].append(rk_thr[a])
            v = per[a]
            summary_rows.append({
                "function": fn, "dim": dim, "category": cat_of[fn],
                "algorithm": a, "runs": len(v),
                "mean": float(v.mean()), "std": float(v.std()),
                "median": float(np.median(v)), "best": float(v.min()),
                "worst": float(v.max()), "rank_thr": rk_thr[a],
                "rank_raw": rk_raw[a]})
            rank_rows.append({"function": fn, "dim": dim, "algorithm": a,
                              "rank_thr": rk_thr[a], "rank_raw": rk_raw[a],
                              "friedman_p": p_thr})
    write_csv(out / f"summary_{tag}.csv", summary_rows)
    write_csv(out / f"ranks_{tag}.csv", rank_rows)

    if not present:
        print("\n  no complete cells -- nothing to rank")
        return

    print(f"\n  {'='*74}\n  OVERALL FRIEDMAN RANK  ({len(present)} cells, "
          f"ref = {ref})\n  {'='*74}")
    print(f"    {'algorithm':22s} {'rank (1e-8 conv)':>18s} {'rank (raw)':>14s}")
    order = sorted(algs, key=lambda a: np.mean(acc_thr[a]))
    for i, a in enumerate(order, 1):
        mark = "  <<<" if a == ref else ""
        print(f"    {i}. {a:19s} {np.mean(acc_thr[a]):18.3f} "
              f"{np.mean(acc_raw[a]):14.3f}{mark}")

    print(f"\n  {'-'*74}\n  RANK BY CATEGORY (1e-8 convention)"
          f"\n  [T] = declared target category, fixed before this run"
          f"\n  {'-'*74}")
    verdicts = {}
    for cat in sorted(acc_cat):
        vals = {a: float(np.mean(acc_cat[cat][a])) for a in algs}
        o = sorted(vals, key=vals.get)
        pos = o.index(ref) + 1
        tgt = "[T]" if cat in TARGET_CATEGORIES else "   "
        win = "WIN " if pos == 1 else "    "
        verdicts[cat] = (pos, vals[ref], o[0], vals[o[0]],
                         cat in TARGET_CATEGORIES)
        print(f"  {tgt} {win}{cat:18s} {ref} {vals[ref]:5.2f}  "
              f"(pos {pos}/{len(algs)})   best: {o[0]} {vals[o[0]]:5.2f}")

    tg = [c for c in TARGET_CATEGORIES if c in verdicts]
    if tg:
        won = [c for c in tg if verdicts[c][0] == 1]
        print(f"\n  {'='*74}")
        print(f"  TARGET-CATEGORY VERDICT: {len(won)} / {len(tg)} targets won "
              f"outright")
        for c in tg:
            pos, mine, best_a, best_v, _ = verdicts[c]
            status = ("WON" if pos == 1 else
                      f"LOST to {best_a} by {mine - best_v:+.2f} rank")
            print(f"    {c:18s} rank {mine:5.2f}  pos {pos}/{len(algs)}  "
                  f"-> {status}")
        print(f"  {'='*74}")

    # ---- pairwise Wilcoxon + Holm + A12 --------------------------------
    pv, recs = {}, []
    for key in present:
        fn, dim = key
        x = np.array(cells[key][ref])
        for a in algs:
            if a == ref:
                continue
            y = np.array(cells[key][a])
            k = f"{fn}|D{dim}|{a}"
            pv[k] = paired_p(x, y)
            recs.append({"key": k, "function": fn, "dim": dim,
                         "category": cat_of[fn], "competitor": a,
                         "p_value": pv[k], "A12": a12(x, y),
                         "A12_magnitude": a12_label(a12(x, y)),
                         f"{ref}_mean": float(x.mean()),
                         "competitor_mean": float(y.mean())})
    rej = holm(pv)
    for r in recs:
        r["reject_H0"] = rej.get(r["key"], False)
        r["verdict"] = ("T" if not r["reject_H0"]
                        else "W" if r["A12"] > 0.5 else "L")
    write_csv(out / f"pairwise_{tag}.csv", recs)

    print(f"\n  {'-'*74}\n  {ref} vs each competitor -- Holm-corrected W/T/L"
          f" (m={len(pv)})\n  {'-'*74}")
    print(f"    {'competitor':22s} {'W':>4s} {'T':>4s} {'L':>4s} {'meanA12':>9s}")
    for a in algs:
        if a == ref:
            continue
        sub = [r for r in recs if r["competitor"] == a]
        if not sub:
            continue
        c = {v: sum(1 for r in sub if r["verdict"] == v) for v in "WTL"}
        print(f"    {a:22s} {c['W']:4d} {c['T']:4d} {c['L']:4d} "
              f"{np.mean([r['A12'] for r in sub]):9.3f}")

    print(f"\n  {'-'*74}\n  W/T/L BY CATEGORY (all competitors pooled)"
          f"\n  {'-'*74}")
    for cat in sorted({r["category"] for r in recs}):
        sub = [r for r in recs if r["category"] == cat]
        c = {v: sum(1 for r in sub if r["verdict"] == v) for v in "WTL"}
        print(f"    {cat:18s} W/T/L = {c['W']:4d}/{c['T']:4d}/{c['L']:4d}"
              f"   meanA12 = {np.mean([r['A12'] for r in sub]):.3f}")

    # ---- CEC-convention Friedman (rank of MEAN error per cell) ----------
    subsets = [("all", present)]
    cec_only = [k for k in present if k[0].startswith("CEC2017_")]
    if cec_only and len(cec_only) != len(present):
        subsets.append(("CEC2017 only", cec_only))
    cec_rows, cec_meta = [], {}
    for name, keys in subsets:
        fr_rows, fr_sum = cec_friedman(cells, keys, algs, ref)
        cec_meta[name] = fr_sum
        for rec in fr_rows:
            cec_rows.append({"subset": name, **rec})
        print(f"\n  {'-'*74}\n  CEC-CONVENTION FRIEDMAN -- rank of MEAN error per "
              f"cell, subset '{name}' ({fr_sum['cells']} cells)\n  {'-'*74}")
        print(f"    Friedman chi2 = {fr_sum['friedman_chi2']:.2f}  "
              f"p = {fr_sum['friedman_p']:.2e}   Iman-Davenport F = "
              f"{fr_sum['iman_davenport_F']:.2f}   Nemenyi CD(0.05) = "
              f"{fr_sum['nemenyi_cd_0.05']:.3f}")
        print(f"    {'algorithm':22s}{'mean-rk':>8s}{'med-rk':>8s}{'diff':>8s}"
              f"{'z-test p':>10s} {'Holm':>5s}   {'better/eq/worse':>15s}"
              f"{'Wilcoxon p':>11s} {'Holm':>5s}")
        for rec in fr_rows:
            mark = "  <<<" if rec["algorithm"] == ref else ""
            line = (f"    {rec['algorithm']:22s}{rec['mean_rank']:8.3f}"
                    f"{rec['median_rank']:8.3f}{rec['rank_diff_vs_ref']:+8.3f}")
            if rec["algorithm"] != ref:
                bew = (f"{rec['ref_better_cells']}/{rec['equal_cells']}/"
                       f"{rec['ref_worse_cells']}")
                line += (f"{rec['z_p']:10.2e} {str(rec['z_holm_reject'])[0]:>5s}"
                         f"   {bew:>15s}{rec['wilcoxon_p']:11.2e} "
                         f"{str(rec['wilcoxon_holm_reject'])[0]:>5s}")
            print(line + mark)
        print(f"    better/eq/worse counts cells where {ref}'s mean error is "
              f"lower/equal/higher.")
    write_csv(out / f"cec_friedman_{tag}.csv", cec_rows)

    # ---- how much of the record is decided below 1e-8 ------------------
    n_both0 = sum(1 for key in present
                  for a in algs if a != ref
                  if np.median(cells[key][ref]) == 0.0
                  and np.median(cells[key][a]) == 0.0)
    print(f"\n  cells where BOTH medians are 0 under the 1e-8 convention: "
          f"{n_both0} / {len(present)*(len(algs)-1)}")
    print(f"  (these carry no information; ranking them on raw values is the "
          f"artefact\n   the convention exists to remove)")

    with open(out / f"meta_{tag}.json", "w", encoding="utf-8") as fh:
        json.dump({"tag": tag, "reference": ref, "algorithms": algs,
                   "cells": len(present),
                   "overall_rank_thr": {a: float(np.mean(acc_thr[a])) for a in algs},
                   "overall_rank_raw": {a: float(np.mean(acc_raw[a])) for a in algs},
                   "category_rank_thr": {c: {a: float(np.mean(acc_cat[c][a]))
                                             for a in algs} for c in acc_cat},
                   "zero_threshold": ZERO_THRESHOLD,
                   "benchmark": bench,
                   "excluded_from_stats": dropped,
                   "pairing": "runs paired by run index (shared seed)",
                   "cec_friedman": cec_meta},
                  fh, indent=2)
    print(f"\n  artifacts -> {out.resolve()}")


def main() -> None:
    ap = argparse.ArgumentParser(
        description="MDAC: single-file implementation + experiment driver")
    ap.add_argument("--stage", default="smoke", choices=list(STAGES))
    ap.add_argument("--workers", type=int, default=max(1, (os.cpu_count() or 2) - 1))
    ap.add_argument("--runs", type=int, default=None)
    ap.add_argument("--dims", type=int, nargs="+", default=None)
    ap.add_argument("--fes_mult", type=int, default=None)
    ap.add_argument("--ref", default="MDAC")
    ap.add_argument("--out", default="mdac_results")
    ap.add_argument("--limit_functions", type=int, default=None,
                    help="use only the first K functions (debug)")
    ap.add_argument("--only", nargs="+", default=None,
                    help="keep only functions whose name starts with one of "
                         "these prefixes, e.g. 'CL_' or 'CEC2017_F6_ CEC2017_F7_'")
    ap.add_argument("--algs", nargs="+", default=None,
                    help="override the stage's algorithm list")
    ap.add_argument("--only_cats", nargs="+", default=None,
                    help="keep only these categories, e.g. "
                         "--only_cats 'CEC hybrid' classic-12")
    ap.add_argument("--time_budget", action="store_true",
                    help="(A1) equal WALL-CLOCK budget instead of equal FES. "
                         "Every algorithm gets EQUAL_TIME_BUDGET[dim] seconds "
                         "per task and as many evaluations as it can fit. "
                         "Only comparable within one machine and worker count.")
    ap.add_argument("--cec", default="bridge",
                    choices=["bridge", "official", "both", "clustering",
                             "cec2011", "cec2020mech"],
                    help="'official' uses the REAL CEC2017 F1-F30 from "
                         "benchmarks/cec2017_data (verified); 'bridge' uses "
                         "the locally generated BR_* suite")
    args = ap.parse_args()

    algs, dims, runs, fes_mult, bridge = STAGES[args.stage]
    dims = args.dims or dims
    runs = args.runs or runs
    fes_mult = args.fes_mult or fes_mult

    if args.algs:
        algs = args.algs
    specs0 = build_registry(dims[0], BENCH_SEED, bridge, cec=args.cec)
    per_dim_suite = args.cec in ("cec2011", "cec2020mech")
    if args.only and not per_dim_suite:          # those filter per dim below
        specs0 = [s for s in specs0 if s.name.startswith(tuple(args.only))]
        if not specs0:
            raise SystemExit(f"--only {args.only!r} matched no function")
    if args.only_cats:
        want = set(args.only_cats)
        specs0 = [s for s in specs0 if s.category in want]
        if not specs0:
            raise SystemExit(f"--only_cats {args.only_cats} matched nothing")
    fids = [s.fid for s in specs0]
    if args.limit_functions:
        fids = fids[:args.limit_functions]

    tasks = [(a, fid, d, r, fes_mult, BENCH_SEED, bridge, args.cec,
              args.time_budget)
             for a in algs for fid in fids for d in dims for r in range(runs)]
    if per_dim_suite:
        # Every CEC2011 problem has its own fixed D, so the (function, dim)
        # cells are taken per dimension instead of crossing one function list
        # with all dims as above.
        cells, specs0 = [], []
        for d in dims:
            sp = build_registry(d, BENCH_SEED, bridge, cec=args.cec)
            if args.only:
                sp = [s for s in sp if s.name.startswith(tuple(args.only))]
            specs0.extend(sp)
            cells.extend((s.fid, d) for s in sp)
        if not cells:
            raise SystemExit("no CEC2011 problem matches these dims/--only")
        fids = [s.fid for s in specs0]
        tasks = [(a, fid, d, r, fes_mult, BENCH_SEED, bridge, args.cec,
                  args.time_budget)
                 for a in algs for (fid, d) in cells for r in range(runs)]

    print("=" * 78)
    print(f"  MDAC EXPERIMENT -- stage '{args.stage}'")
    print("=" * 78)
    print(f"  algorithms ({len(algs)}): {', '.join(algs)}")
    print(f"  functions  : {len(fids)}   dims: {dims}   runs: {runs}")
    if args.time_budget:
        sv = [equal_time_budget(s.name, d) for s in specs0 if s.fid in set(fids)
              for d in dims]
        print(f"  BUDGET     : EQUAL WALL-CLOCK per (function,dim), "
              f"{min(sv):.2f}-{max(sv):.2f} s/task (FES unbounded)")
    elif args.cec == "cec2011":
        print(f"  MaxFES     : {CEC2011_MAX_FES} per run (CEC2011 rule)")
    elif args.cec == "cec2020mech":
        print("  MaxFES     : official CEC2020 rule (1e5 for D<=10, 2e5 for D<=30)")
    else:
        print(f"  MaxFES     : {fes_mult} * D")
    print(f"  tasks      : {len(tasks)}   workers: {args.workers}")
    bench = "CL_* shifted classics"
    if args.cec == "cec2011":
        bench = "CEC2011_* real-world problems (raw objective, no 1e-8 rule)"
    if args.cec == "cec2020mech":
        bench = ("CEC2020_* mechanical constrained problems (scalar fitness: "
                 "f if feasible, C_k + mean violation otherwise; raw values)")
    if args.cec in ("official", "both"):
        bench += " + CEC2017_* OFFICIAL (verified: f(o_i)=100i for F1-F10)"
    if args.cec == "both" or (args.cec == "bridge" and bridge):
        bench += " + BR_* bridge (NOT official CEC2017)"
    print(f"  benchmark  : {bench}")
    print(f"  categories : " + ", ".join(
        f"{c}({n})" for c, n in sorted(
            __import__("collections").Counter(s.category for s in specs0
                                              if s.fid in set(fids)).items())))
    if per_dim_suite:
        print("  zero conv. : not applied (raw objective)")
    else:
        print(f"  zero conv. : errors < {ZERO_THRESHOLD:g} reported as 0 "
              f"(uniform, both variants written)")
    print("=" * 78, flush=True)

    # ---- CHECKPOINTING ---------------------------------------------------
    # The definitive sweep is tens of hours long and `analyse()` only writes
    # CSVs once every task has finished. On 2026-10-11 the run died at
    # 41 124 of 68 544 tasks after 8.4 h and EVERY completed task was lost,
    # because nothing had been written. Results are now appended to
    # `<out>/checkpoint_<stage>.csv` as they arrive, and a restart with the
    # same `--out` and `--stage` skips whatever is already in that file.
    # Delete the file to force a clean re-run.
    ckpt = Path(args.out) / f"checkpoint_{args.stage}.csv"
    done_rows: List[Dict] = []
    done_keys = set()
    if ckpt.is_file():
        with ckpt.open(newline="", encoding="utf-8-sig") as fh:
            for r in csv.DictReader(fh):
                done_rows.append(r)
                done_keys.add((r["label"], r["function"], int(r["dim"]),
                               int(r["run"])))
    name_of = {s.fid: s.name for s in specs0}
    todo = [t for t in tasks
            if (t[0], name_of[t[1]], t[2], t[3]) not in done_keys]
    if done_keys:
        print(f"  CHECKPOINT : {len(done_keys)} tasks already done, "
              f"{len(todo)} remaining\n", flush=True)

    t0 = time.time()
    rows: List[Dict] = list(done_rows)
    ckpt.parent.mkdir(parents=True, exist_ok=True)
    fresh = not ckpt.is_file()
    fh = ckpt.open("a", newline="", encoding="utf-8-sig")
    wr = None
    n_since = 0

    def _save(rec: Dict) -> None:
        nonlocal wr, n_since
        if wr is None:
            wr = csv.DictWriter(fh, fieldnames=list(rec.keys()))
            if fresh:
                wr.writeheader()
        wr.writerow(rec)
        n_since += 1
        if n_since >= 200:                     # flush to disk, bounded loss
            fh.flush()
            os.fsync(fh.fileno())
            n_since = 0

    step = max(1, len(todo) // 20) if todo else 1
    try:
        if args.workers <= 1:
            for i, t in enumerate(todo, 1):
                rec = worker(t)
                rows.append(rec)
                _save(rec)
                if i % step == 0 or i == len(todo):
                    print(f"  {i}/{len(todo)}  ({time.time()-t0:.0f}s)",
                          flush=True)
        else:
            with ProcessPoolExecutor(max_workers=args.workers) as ex:
                futs = [ex.submit(worker, t) for t in todo]
                for i, f in enumerate(as_completed(futs), 1):
                    rec = f.result()
                    rows.append(rec)
                    _save(rec)
                    if i % step == 0 or i == len(todo):
                        el = time.time() - t0
                        eta = el / i * (len(todo) - i)
                        print(f"  {i}/{len(todo)}  ({el:.0f}s elapsed, "
                              f"~{eta:.0f}s left)", flush=True)
    finally:
        fh.flush()
        try:
            os.fsync(fh.fileno())
        except OSError:
            pass
        fh.close()
    print(f"\n  all tasks done in {time.time()-t0:.1f}s")

    # the checkpoint reader returns strings; the analysis needs numbers
    for r in rows:
        for k in ("dim", "run", "fes_used"):
            r[k] = int(float(r[k]))
        for k in ("error_thr", "error_raw", "fes_frac", "secs"):
            r[k] = float(r[k])

    ref = args.ref if args.ref in algs else algs[0]
    analyse(rows, algs, ref, Path(args.out), args.stage)


if __name__ == "__main__":
    mp.freeze_support()
    main()
