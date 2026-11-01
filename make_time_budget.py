#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Build `equal_time_budget.json` from a completed equal-FES run.

    Budget[function][dim] = MEDIAN over the algorithms of their median task
                            seconds on that (function, dim)

i.e. the wall-clock the typical entry needed for the standard 10000*D
protocol on that exact function. The median is used rather than the slowest
or the fastest so the budget favours no algorithm.

WHY PER-FUNCTION. A flat per-dimension budget was tried first and is wrong:
on a cheap function every algorithm then runs millions of evaluations and the
cell saturates into an information-free tie, while an expensive function
starves. Per-function keeps each cell at the difficulty the standard protocol
gave it.

NOT PORTABLE. These seconds are measured on one machine at one worker count.
Re-measure before quoting them anywhere else, and never mix a budget built on
one machine with runs from another.

    python make_time_budget.py <fes_run_dir> [--tag top]
"""
from __future__ import annotations

import argparse
import csv
import json
from collections import defaultdict
from pathlib import Path

import numpy as np

ENC = "utf-8-sig"


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("source")
    ap.add_argument("--tag", default="top")
    ap.add_argument("--out", default="equal_time_budget.json")
    a = ap.parse_args()

    rows = list(csv.DictReader(
        open(Path(a.source) / f"raw_{a.tag}.csv", newline="", encoding=ENC)))
    if any(r.get("failure") for r in rows):
        raise SystemExit("source run has failures; refusing to build a budget "
                         "from contaminated timings")

    per = defaultdict(list)
    for r in rows:
        per[(r["function"], int(r["dim"]), r["label"])].append(float(r["secs"]))

    funcs = sorted({r["function"] for r in rows})
    dims = sorted({int(r["dim"]) for r in rows})
    algs = sorted({r["label"] for r in rows})

    budget = {}
    for f in funcs:
        budget[f] = {}
        for d in dims:
            meds = [float(np.median(per[(f, d, al)])) for al in algs
                    if per[(f, d, al)]]
            budget[f][str(d)] = round(float(np.median(meds)), 4)

    Path(a.out).write_text(json.dumps({
        "source": f"{a.source} ({len(rows)} rows, {len(algs)} algorithms, "
                  f"{len({int(r['run']) for r in rows})} runs)",
        "rule": "median over algorithms of their median task seconds on that "
                "(function, dim)",
        "machine": "measured on one machine at one worker count -- NOT "
                   "portable; re-measure elsewhere",
        "budget": budget,
    }, indent=1), encoding="utf-8")

    vals = [budget[f][str(d)] for f in funcs for d in dims]
    tot = sum(vals) * len({int(r["run"]) for r in rows}) * len(algs) / 15.0
    print(f"wrote {a.out}: {len(funcs)} functions x {len(dims)} dims "
          f"({len(vals)} cells)")
    print(f"  range {min(vals):.2f} .. {max(vals):.2f} s per task")
    print(f"  projected equal-time sweep: {tot/3600:.2f} h at 15 workers")


if __name__ == "__main__":
    main()
