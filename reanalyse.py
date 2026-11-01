#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
reanalyse.py -- re-run the statistics of a finished stage from its raw CSV.

Nothing is re-optimised and no seed is drawn: the rows in raw_<tag>.csv are
read back and handed to `MDAC_EXPERIMENT.analyse()`, which rewrites the
summary / ranks / pairwise / cec_friedman / meta files. The raw file itself
is never touched.

Written on 2026-10-31 to apply three analysis fixes to the FINAL_* runs
without repeating 42 h of compute:
  * runs paired by run index instead of by row position,
  * CEC2017 F2 left out of the statistics (withdrawn by the organisers),
  * the CEC-convention Friedman test (rank of mean error) with post-hoc.

Usage
-----
    python reanalyse.py --dir FINAL_fes --tag full
    python reanalyse.py --dir FINAL_fes --tag full --suffix _pub \
        --algs MDAC jSO L-SHADE-SPACMA L-SHADE EBOwithCMAR CMA-ES
    python reanalyse.py --dir FINAL_fes --tag full --out <elsewhere>   # dry run
"""
from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

import MDAC_EXPERIMENT as M  # noqa: E402

ENC = "utf-8-sig"


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dir", required=True, help="stage output directory")
    ap.add_argument("--tag", required=True, help="raw_<tag>.csv to read")
    ap.add_argument("--ref", default="MDAC")
    ap.add_argument("--algs", nargs="+", default=None,
                    help="analyse only these algorithms")
    ap.add_argument("--suffix", default="",
                    help="appended to the tag of the written files")
    ap.add_argument("--out", default=None,
                    help="write results here instead of --dir")
    a = ap.parse_args()

    src = Path(a.dir)
    with (src / f"raw_{a.tag}.csv").open(newline="", encoding=ENC) as fh:
        rows = list(csv.DictReader(fh))

    present = []
    for r in rows:
        if r["label"] not in present:
            present.append(r["label"])
    meta = src / f"meta_{a.tag}.json"
    if a.algs:
        algs = a.algs
    elif meta.is_file():
        algs = json.loads(meta.read_text(encoding=ENC))["algorithms"]
    else:
        algs = present
    missing = [x for x in algs if x not in present]
    if missing:
        raise SystemExit(f"not in raw_{a.tag}.csv: {missing}")
    if a.ref not in algs:
        raise SystemExit(f"--ref {a.ref} is not among the analysed algorithms")

    keep = set(algs)
    rows = [r for r in rows if r["label"] in keep]
    # same conversion as MDAC_EXPERIMENT.main() applies to checkpoint rows
    for r in rows:
        for k in ("dim", "run", "fes_used"):
            r[k] = int(float(r[k]))
        for k in ("error_thr", "error_raw", "fes_frac", "secs"):
            r[k] = float(r[k])

    out = Path(a.out) if a.out else src
    print(f"  reanalyse: {src}/raw_{a.tag}.csv, {len(rows)} rows, "
          f"{len(algs)} algorithms -> {out}/*_{a.tag}{a.suffix}.*")
    M.analyse(rows, algs, a.ref, out, a.tag + a.suffix, write_raw=False)


if __name__ == "__main__":
    main()
