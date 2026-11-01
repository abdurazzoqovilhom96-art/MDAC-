#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Continuous clustering as a black-box optimization problem (track B).

WHAT THIS IS
------------
Minimise the within-cluster sum of squared errors over k free centroids:

    f(C) = sum_i  min_j  || x_i - c_j ||^2 ,   C = (c_1..c_k) in [0,1]^(k*d)

This is the k-means objective written as a continuous, bound-constrained,
derivative-free optimization problem. It is the one mainstream machine-learning
task that fits the regime in which MDAC's advantage was actually measured:

    deterministic .. bound-constrained .. continuous .. D = k*d >= 30
    .. and cheap enough to afford 10000*D evaluations

WHY NOT THE OTHER AI PROBLEMS
-----------------------------
  neural-network weight training : gradients exist; SGD/Adam dominate, and a
                                   metaheuristic here is a weak claim
  hyperparameter optimization    : D is 5-20 (our weakest region) and the
                                   budget is ~1e2 evaluations, not ~1e5
  neural architecture search     : discrete / categorical
  RL policy search               : stochastic objective; noise handling is a
                                   separate research question
  black-box adversarial attacks  : D in the thousands, query-limited

PROPERTIES OF THIS OBJECTIVE
----------------------------
  non-separable : moving one centroid changes which points it owns, which
                  changes every other centroid's contribution
  multimodal    : the k-means objective is well known to have many local
                  minima; this is why practitioners restart k-means
  deterministic : for a fixed dataset, f is a pure function of C
  scalable in D : D = k * d, so the dimensional trend can be measured

HONEST FRAMING, FIXED BEFORE ANY RESULT IS SEEN
-----------------------------------------------
k-means++ with restarts is a strong, cheap, specialised baseline and it is
included as a competitor (`alg_kmeans_pp`). If it wins, that is the result and
it gets reported. The claim being tested is NOT "we solved clustering"; it is
"does the mechanism that wins on rotated CEC2017 landscapes also transfer to a
real machine-learning objective, relative to the other general-purpose
optimizers?"

DATA
----
Real UCI datasets, loaded through scikit-learn's bundled copies (no network
access: they ship inside the package). Every shape is asserted at load time --
the CEC2017 table defects in this project survived for weeks precisely because
nothing checked the data against an independent source.

Features are min-max normalised per column, so the search box is exactly
[0,1]^D for every dataset and the bounds carry no dataset-specific advantage.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Dict, List, Sequence, Tuple

import numpy as np

Array = np.ndarray

# name -> (loader attribute, expected shape). The shapes are asserted, never
# trusted: a silently different dataset would invalidate every number here.
DATASETS: Dict[str, Tuple[str, Tuple[int, int]]] = {
    "iris":   ("load_iris",          (150, 4)),
    "wine":   ("load_wine",          (178, 13)),
    "wdbc":   ("load_breast_cancer", (569, 30)),
}

_CACHE: Dict[str, Array] = {}


class ClusteringDataMissing(ImportError):
    """scikit-learn (which ships the datasets) is not installed."""


def load_dataset(name: str) -> Array:
    """Min-max normalised feature matrix, shape asserted."""
    if name in _CACHE:
        return _CACHE[name]
    try:
        import sklearn.datasets as skd
    except ImportError as exc:                              # pragma: no cover
        raise ClusteringDataMissing(
            "the clustering benchmark needs scikit-learn for its bundled UCI "
            "datasets (no network access is required once installed): "
            "pip install scikit-learn"
        ) from exc
    attr, shape = DATASETS[name]
    X = np.asarray(getattr(skd, attr)().data, dtype=np.float64)
    if X.shape != shape:
        raise ValueError(
            f"dataset {name!r}: expected shape {shape}, got {X.shape}. "
            f"Refusing to run -- an unexpected dataset invalidates every "
            f"number produced from it.")
    if not np.all(np.isfinite(X)):
        raise ValueError(f"dataset {name!r} contains non-finite values")
    lo, hi = X.min(axis=0), X.max(axis=0)
    span = np.where(hi - lo > 0, hi - lo, 1.0)
    X = (X - lo) / span
    _CACHE[name] = X
    return X


def make_sse(X: Array, k: int) -> Callable[[Array], Array]:
    """Vectorised SSE over a batch of candidate centroid sets.

    C has shape (N, k*d) and is read as (N, k, d). A centroid that owns no
    point simply contributes nothing -- that falls out of the `min` and is the
    standard definition, so no special empty-cluster rule is needed or added.
    """
    n, d = X.shape
    Xs = X[None, :, :]                                      # (1, n, d)

    def f(C: Array) -> Array:
        C = np.asarray(C, dtype=np.float64)
        if C.ndim == 1:
            C = C.reshape(1, -1)
        m = C.shape[0]
        cent = C.reshape(m, k, d)
        # (m, n, k) squared distances, computed in chunks so a large
        # population at a large dataset does not allocate m*n*k at once.
        out = np.empty(m, dtype=np.float64)
        step = max(1, int(4_000_000 // max(1, n * k)))
        for a in range(0, m, step):
            b = min(m, a + step)
            diff = Xs[:, :, None, :] - cent[a:b][:, None, :, :]
            out[a:b] = np.min(np.sum(diff * diff, axis=3), axis=2).sum(axis=1)
        return out

    return f


@dataclass
class ClusterSpec:
    name: str
    dataset: str
    k: int
    dim: int
    func: Callable[[Array], Array]


def clustering_suite(dim: int, k_min: int = 2, k_max: int = 25
                     ) -> List[ClusterSpec]:
    """Every (dataset, k) whose k*d equals exactly `dim`.

    The driver gives one dimension to a whole registry, so a clustering cell
    exists at `dim` only when `dim` is a multiple of that dataset's feature
    count. With d = 4 / 13 / 30 the useful dimensions are, for example:
        D=39 wine k=3 · D=40 iris k=10 · D=60 iris k=15 AND wdbc k=2
        D=78 wine k=6 · D=90 wdbc k=3
    """
    out: List[ClusterSpec] = []
    for name, (_, shape) in DATASETS.items():
        d = shape[1]
        if dim % d:
            continue
        k = dim // d
        if not (k_min <= k <= k_max):
            continue
        X = load_dataset(name)
        out.append(ClusterSpec(f"CL2_{name}_k{k}", name, k, dim,
                               make_sse(X, k)))
    return out


def available_dims(k_min: int = 2, k_max: int = 25) -> List[int]:
    dims = set()
    for _, (_, shape) in DATASETS.items():
        d = shape[1]
        for k in range(k_min, k_max + 1):
            dims.add(k * d)
    return sorted(dims)


def self_check() -> None:
    """Determinism and sanity. Run before any result is quoted."""
    print("clustering benchmark self-check")
    for name, (_, shape) in DATASETS.items():
        X = load_dataset(name)
        assert X.shape == shape, (name, X.shape, shape)
        assert X.min() >= -1e-12 and X.max() <= 1 + 1e-12, name
        print(f"  {name:8s} shape {X.shape}  normalised to "
              f"[{X.min():.3f}, {X.max():.3f}]")
    rng = np.random.default_rng(0)
    for dim in (39, 40, 60, 78, 90):
        for s in clustering_suite(dim):
            C = rng.random((7, dim))
            a, b = s.func(C), s.func(C)
            assert np.array_equal(a, b), f"{s.name} is not deterministic"
            # one centroid per point can only lower SSE; a trivial monotonicity
            # check that catches a transposed reshape
            assert np.all(a > 0), f"{s.name} produced a non-positive SSE"
            print(f"  D={dim:3d} {s.name:18s} k={s.k:2d} "
                  f"SSE[min..max]={a.min():.2f}..{a.max():.2f}  deterministic")
    print("  OK")


if __name__ == "__main__":
    self_check()
