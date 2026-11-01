#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
report_final.py -- the publication-facing read-out of a `--stage top` run.

Why this exists as a separate pass
----------------------------------
`analyse()` inside MDAC_EXPERIMENT.py prints its category table under the
heading "W/T/L BY CATEGORY (all competitors pooled)" -- pooled means CMA-ES is
in it. Rule 6 of the project's standing rules says the opposite for any
published category number:

    6. Exclude CMA-ES when pooling category A12 -- it inflates everything.

CMA-ES loses most cells outright (overall rank 4.491 of 5 in the 25-run run),
so every category it is pooled into gains a free win column. The category
claim a reviewer will check is "MDAC vs the CEC2017 top methods", and that
set is {jSO, L-SHADE-SPACMA, L-SHADE}.

The overall Friedman rank is reported over ALL FIVE algorithms, because that
is the number a reviewer computes first and dropping a competitor from it
would be the actual cherry-pick. So: rank over 5, category A12 over 3.

This is a pure post-processing pass over per-cell statistics that the run
already wrote. It re-optimizes nothing and draws no new seeds, so it cannot
disagree with the run it reads.

Usage
-----
    python report_final.py --dir mdac_final
    python report_final.py --dir mdac_official        # reproduces HANDOFF 3
    python report_final.py --dir mdac_final --tag top --check mdac_official
"""
from __future__ import annotations

import argparse
import csv
import json
from collections import defaultdict
from pathlib import Path
from typing import Dict, List

import numpy as np

# Every CSV this project writes is UTF-8 with BOM.
ENC = "utf-8-sig"

# The CEC2017 competition's top methods. CMA-ES is a reference point, not a
# top method, and is excluded from pooled category statistics by rule 6.
# EBOwithCMAR joined on 2026-10-08. It is a CEC2017 top method, so it belongs
# in the pooled category statistics; leaving it out because it happens to beat
# us would be precisely the cherry-pick rule 6 exists to prevent. CMA-ES stays
# out because it is a weak reference point, not a competition entry.
TOP_METHODS = ("jSO", "L-SHADE-SPACMA", "L-SHADE", "EBOwithCMAR")
EXCLUDE_FROM_POOL = ("CMA-ES",)
# Pooled statistics use TOP_METHODS only. Until 2026-10-31 the pool was
# "everything except CMA-ES", which in the 8-algorithm final run let our own
# ablation variants (MDAC-ctr, MDAC-noFCP) in as "rivals" and inflated every
# category A12 (multimodal 0.675 -> 0.704, composition 0.564 -> 0.605).


def pooled(pw: List[Dict[str, str]]) -> List[Dict[str, str]]:
    return [r for r in pw if r["competitor"] in TOP_METHODS]

CATEGORY_ORDER = ("CEC unimodal", "CEC multimodal", "CEC hybrid",
                  "CEC composition", "classic-12")


def read_csv(path: Path) -> List[Dict[str, str]]:
    with path.open(newline="", encoding=ENC) as fh:
        return list(csv.DictReader(fh))


def cat_sort(cats) -> List[str]:
    known = [c for c in CATEGORY_ORDER if c in cats]
    return known + sorted(c for c in cats if c not in CATEGORY_ORDER)


def rule(ch: str = "-", n: int = 78) -> str:
    return ch * n


# ---------------------------------------------------------------------------
# 1. integrity -- never report a number from a run that did not finish clean
# ---------------------------------------------------------------------------
def integrity(raw: List[Dict[str, str]]) -> bool:
    print(rule("="))
    print("  0. RUN INTEGRITY")
    print(rule("="))

    fails = [r for r in raw if r.get("failure")]
    algs = sorted({r["label"] for r in raw})
    print(f"  rows            : {len(raw)}")
    print(f"  algorithms      : {len(algs)}  ({', '.join(algs)})")
    print(f"  functions       : {len({r['function'] for r in raw})}")
    print(f"  dims            : {sorted({int(r['dim']) for r in raw})}")
    print(f"  runs per cell   : {len({int(r['run']) for r in raw})}")
    print(f"  failures        : {len(fails)}")

    ok = not fails
    if fails:
        dist = defaultdict(int)
        for r in fails:
            dist[(r["label"], r["failure"][:60])] += 1
        print("\n  !! FAILURE DISTRIBUTION -- no number below is publishable !!")
        for (a, msg), n in sorted(dist.items(), key=lambda kv: -kv[1]):
            print(f"     {n:6d}  {a:18s} {msg}")

    print("\n  budget utilisation (fraction of MaxFES):")
    for a in algs:
        fr = np.array([float(r["fes_frac"]) for r in raw if r["label"] == a])
        flag = "" if fr.min() >= 0.999 else "   <-- BUDGET PARITY VIOLATION"
        print(f"    {a:20s} min {fr.min():.4f}  mean {fr.mean():.4f}{flag}")
        ok = ok and fr.min() >= 0.999
    return ok


# ---------------------------------------------------------------------------
# 2. overall Friedman rank -- all five algorithms
# ---------------------------------------------------------------------------
def overall_rank(ranks: List[Dict[str, str]], ref: str = "MDAC") -> Dict[str, float]:
    by_alg_thr: Dict[str, List[float]] = defaultdict(list)
    by_alg_raw: Dict[str, List[float]] = defaultdict(list)
    for r in ranks:
        by_alg_thr[r["algorithm"]].append(float(r["rank_thr"]))
        by_alg_raw[r["algorithm"]].append(float(r["rank_raw"]))

    mean_thr = {a: float(np.mean(v)) for a, v in by_alg_thr.items()}
    mean_raw = {a: float(np.mean(v)) for a, v in by_alg_raw.items()}
    n_cells = len(next(iter(by_alg_thr.values())))

    print("\n" + rule("="))
    print(f"  1. OVERALL FRIEDMAN RANK  ({n_cells} cells, all "
          f"{len(mean_thr)} algorithms, 1e-8 convention)")
    print(rule("="))
    print(f"    {'#':<3}{'algorithm':22s}{'rank (1e-8)':>13s}{'rank (raw)':>13s}")
    order = sorted(mean_thr, key=mean_thr.get)
    for i, a in enumerate(order, 1):
        mark = "   <<< proposed" if a == ref else ""
        print(f"    {i:<3}{a:22s}{mean_thr[a]:13.3f}{mean_raw[a]:13.3f}{mark}")

    pos = order.index(ref) + 1
    order_raw = sorted(mean_raw, key=mean_raw.get)
    pos_raw = order_raw.index(ref) + 1
    runner = order[1] if pos == 1 else order[0]
    gap = mean_thr[ref] - mean_thr[runner]

    print(f"\n    {ref} position : {pos} / {len(order)}  "
          f"(1e-8 convention)   |   {pos_raw} / {len(order)}  (raw ranks)")
    print(f"    margin to {'runner-up' if pos == 1 else 'leader':9s}: "
          f"{gap:+.3f} rank units  vs {runner}")
    if abs(gap) < 0.05:
        print("    NOTE: a margin under 0.05 rank units is inside seed-set "
              "noise. Report the\n          position, but do not claim a "
              "separation this data cannot support.")
    return mean_thr


# ---------------------------------------------------------------------------
# 3. per-category mean rank and gap to the category leader
# ---------------------------------------------------------------------------
def category_ranks(ranks: List[Dict[str, str]], cat_of: Dict[str, str],
                   ref: str = "MDAC") -> None:
    acc: Dict[str, Dict[str, List[float]]] = defaultdict(lambda: defaultdict(list))
    for r in ranks:
        acc[cat_of[r["function"]]][r["algorithm"]].append(float(r["rank_thr"]))

    print("\n" + rule("="))
    print("  2. MEAN RANK BY CATEGORY (all 5 algorithms; every category "
          "reported)")
    print(rule("="))
    print(f"    {'category':18s}{ref:>8s}{'leader':>18s}{'leader rank':>13s}"
          f"{'gap':>9s}  verdict")
    for cat in cat_sort(acc):
        vals = {a: float(np.mean(v)) for a, v in acc[cat].items()}
        order = sorted(vals, key=vals.get)
        leader = order[0]
        gap = vals[ref] - vals[leader]
        pos = order.index(ref) + 1
        verdict = (f"LEADER ({pos}/{len(order)})" if pos == 1
                   else f"{pos}/{len(order)}")
        print(f"    {cat:18s}{vals[ref]:8.3f}{leader:>18s}"
              f"{vals[leader]:13.3f}{gap:+9.3f}  {verdict}")


# ---------------------------------------------------------------------------
# 4. pooled category W/T/L and A12 -- CMA-ES EXCLUDED (rule 6)
# ---------------------------------------------------------------------------
def category_pool(pw: List[Dict[str, str]], ref: str = "MDAC") -> None:
    kept = pooled(pw)
    dropped = sorted({r["competitor"] for r in pw} - set(TOP_METHODS))

    print("\n" + rule("="))
    print(f"  3. CATEGORY W/T/L and A12 vs the TOP METHODS ONLY")
    print(rule("="))
    print(f"    pooled competitors : {', '.join(sorted({r['competitor'] for r in kept}))}")
    print(f"    not pooled         : {', '.join(dropped) if dropped else '(none present)'}"
          f"  (CMA-ES by rule 6; own variants are not rivals)")
    print(f"    A12 > 0.5 favours {ref}. W/T/L is Holm-corrected over the "
          f"whole run family.\n")
    print(f"    {'category':18s}{'W':>5s}{'T':>5s}{'L':>5s}{'netW':>7s}"
          f"{'A12':>8s}   verdict")
    for cat in cat_sort({r["category"] for r in kept}):
        sub = [r for r in kept if r["category"] == cat]
        c = {v: sum(1 for r in sub if r["verdict"] == v) for v in "WTL"}
        mean_a = float(np.mean([float(r["A12"]) for r in sub]))
        net = c["W"] - c["L"]
        if net > 0 and mean_a > 0.5:
            verdict = "WIN" if mean_a >= 0.56 else "marginal win"
        elif net < 0 or mean_a < 0.5:
            verdict = "LOSS" if mean_a <= 0.44 else "marginal loss"
        else:
            verdict = "tie"
        print(f"    {cat:18s}{c['W']:5d}{c['T']:5d}{c['L']:5d}{net:+7d}"
              f"{mean_a:8.3f}   {verdict}")


# ---------------------------------------------------------------------------
# 5. category x dimension A12 -- where the advantage lives
# ---------------------------------------------------------------------------
def category_by_dim(pw: List[Dict[str, str]]) -> None:
    kept = pooled(pw)
    dims = sorted({int(r["dim"]) for r in kept})

    print("\n" + rule("="))
    print("  4. CATEGORY x DIMENSION A12   (top methods only; W/L in "
          "parentheses)")
    print(rule("="))
    print(f"    {'category':18s}" + "".join(f"{'D=' + str(d):>18s}" for d in dims))
    for cat in cat_sort({r["category"] for r in kept}):
        line = f"    {cat:18s}"
        for d in dims:
            sub = [r for r in kept
                   if r["category"] == cat and int(r["dim"]) == d]
            if not sub:
                line += f"{'-':>18s}"
                continue
            a = float(np.mean([float(r["A12"]) for r in sub]))
            w = sum(1 for r in sub if r["verdict"] == "W")
            l = sum(1 for r in sub if r["verdict"] == "L")
            line += f"{a:11.3f}({w}/{l}){'':>0s}".rjust(18)
        print(line)


# ---------------------------------------------------------------------------
# 6. MDAC against each top method separately, inside each category
# ---------------------------------------------------------------------------
def head_to_head(pw: List[Dict[str, str]], ref: str = "MDAC") -> None:
    kept = pooled(pw)
    comps = [c for c in TOP_METHODS if c in {r["competitor"] for r in kept}]

    print("\n" + rule("="))
    print(f"  5. {ref} vs EACH TOP METHOD, BY CATEGORY")
    print(rule("="))
    print("    A 'beats each individually' claim needs every row of a "
          "category to favour us:\n    A12 > 0.5 AND more significant wins "
          "than losses (an even W/L record is not a win).\n")
    for cat in cat_sort({r["category"] for r in kept}):
        print(f"    {cat}")
        all_fav = True
        for c in comps:
            sub = [r for r in kept
                   if r["category"] == cat and r["competitor"] == c]
            if not sub:
                continue
            cc = {v: sum(1 for r in sub if r["verdict"] == v) for v in "WTL"}
            a = float(np.mean([float(r["A12"]) for r in sub]))
            all_fav = all_fav and a > 0.5 and cc["W"] > cc["L"]
            print(f"        vs {c:18s} W/T/L {cc['W']:3d}/{cc['T']:3d}/"
                  f"{cc['L']:3d}   A12 {a:.3f}")
        print(f"        -> beats each individually: "
              f"{'YES' if all_fav else 'no'}\n")

    print(f"    OVERALL, vs each top method (all categories):")
    for c in comps:
        sub = [r for r in kept if r["competitor"] == c]
        cc = {v: sum(1 for r in sub if r["verdict"] == v) for v in "WTL"}
        a = float(np.mean([float(r["A12"]) for r in sub]))
        print(f"        vs {c:18s} W/T/L {cc['W']:3d}/{cc['T']:3d}/"
              f"{cc['L']:3d}   A12 {a:.3f}")


# ---------------------------------------------------------------------------
# 7. how much of the record carries no information
# ---------------------------------------------------------------------------
def information_content(raw: List[Dict[str, str]], pw: List[Dict[str, str]],
                        ref: str = "MDAC") -> None:
    med: Dict[tuple, float] = {}
    acc: Dict[tuple, List[float]] = defaultdict(list)
    for r in raw:
        acc[(r["label"], r["function"], int(r["dim"]))].append(
            float(r["error_thr"]))
    for k, v in acc.items():
        med[k] = float(np.median(v))

    kept = pooled(pw)
    both0 = sum(1 for r in kept
                if med.get((ref, r["function"], int(r["dim"])), 1.0) == 0.0
                and med.get((r["competitor"], r["function"], int(r["dim"])),
                            1.0) == 0.0)
    print("\n" + rule("="))
    print("  6. INFORMATION CONTENT")
    print(rule("="))
    print(f"    pairs where BOTH medians are 0 under the 1e-8 convention: "
          f"{both0} / {len(kept)}  ({100.0*both0/max(1,len(kept)):.1f}%)")
    print("    Those pairs are saturated: they can produce a tie and nothing "
          "else, and\n    ranking them on raw values is the artefact the "
          "convention exists to remove.")


# ---------------------------------------------------------------------------
# 8. statistical power -- recomputed from what the run actually contains
# ---------------------------------------------------------------------------
def power(raw: List[Dict[str, str]], pw: List[Dict[str, str]]) -> None:
    n = len({int(r["run"]) for r in raw})
    m = len(pw)
    thr = 0.05 / m
    pmin = 2.0 / (2.0 ** n)
    print("\n" + rule("="))
    print("  7. STATISTICAL POWER (rule 3 -- recomputed, not assumed)")
    print(rule("="))
    print(f"    runs per cell n          : {n}")
    print(f"    Holm family size m       : {m}")
    print(f"    strictest threshold 0.05/m: {thr:.3e}")
    print(f"    min attainable p = 2/2^n : {pmin:.3e}")
    if pmin <= thr:
        print(f"    => REACHABLE. W/T/L from this family is quotable.")
    else:
        print(f"    => NOT REACHABLE. Significance is impossible at this n; "
              f"quote ranks and A12 ONLY.")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dir", required=True)
    ap.add_argument("--tag", default="top")
    ap.add_argument("--ref", default="MDAC")
    args = ap.parse_args()

    d = Path(args.dir)
    raw = read_csv(d / f"raw_{args.tag}.csv")
    ranks = read_csv(d / f"ranks_{args.tag}.csv")
    pw = read_csv(d / f"pairwise_{args.tag}.csv")
    cat_of = {r["function"]: r["category"] for r in raw}

    meta = d / f"meta_{args.tag}.json"
    print(rule("="))
    print(f"  MDAC FINAL REPORT -- {d.resolve()}")
    if meta.exists():
        j = json.loads(meta.read_text(encoding=ENC))
        for k in ("stage", "runs", "dims", "fes_mult", "cec", "seconds"):
            if k in j:
                print(f"  {k:10s}: {j[k]}")
    print(rule("="))

    clean = integrity(raw)
    overall_rank(ranks, args.ref)
    category_ranks(ranks, cat_of, args.ref)
    category_pool(pw, args.ref)
    category_by_dim(pw)
    head_to_head(pw, args.ref)
    information_content(raw, pw, args.ref)
    power(raw, pw)

    print("\n" + rule("="))
    if clean:
        print("  RUN IS CLEAN: zero failures, full budget parity. Numbers "
              "above are reportable.")
    else:
        print("  RUN IS NOT CLEAN. Do not publish any number above until the "
              "cause is fixed.")
    print(rule("="))


if __name__ == "__main__":
    main()
