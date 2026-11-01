#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
merge_cfix.py -- replace the rows of the C-exact re-run functions in a FINAL_*
raw file with the rows of the re-run, and keep everything else.

On 2026-10-31 a compiled cross-check showed that F2, F6, F7, F8, F9, F13, F14
and F20 differed from the official C. They were made C-exact and their cells
re-run (run_cfix.ps1). This script swaps exactly those rows: same algorithms,
same dimensions, same run indices. It refuses if anything does not line up.
The original raw file is backed up first.

    python merge_cfix.py --dir FINAL_fes --tag full --fix FINAL_fes_cfix
"""
from __future__ import annotations

import argparse
import csv
import shutil
from collections import Counter
from pathlib import Path

PREFIXES = ("CEC2017_F2_", "CEC2017_F6_", "CEC2017_F7_", "CEC2017_F8_",
            "CEC2017_F9_", "CEC2017_F13_", "CEC2017_F14_", "CEC2017_F20_")
ENC = "utf-8-sig"


def read(p: Path):
    with p.open(newline="", encoding=ENC) as fh:
        r = csv.DictReader(fh)
        return r.fieldnames, list(r)


def fid(name: str) -> str:
    return name.split("_")[1]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dir", required=True)
    ap.add_argument("--tag", required=True)
    ap.add_argument("--fix", required=True)
    ap.add_argument("--fix_tag", default=None)
    ap.add_argument("--backup", default="_backup_2026-10-16")
    a = ap.parse_args()

    src = Path(a.dir) / f"raw_{a.tag}.csv"
    cols, old = read(src)
    _, new = read(Path(a.fix) / f"raw_{a.fix_tag or a.tag}.csv")

    if any(r["failure"] for r in new):
        raise SystemExit("re-run has failed rows; not merging")
    if not all(r["function"].startswith(PREFIXES) for r in new):
        raise SystemExit("re-run contains functions outside the re-run set")
    gone = [r for r in old if r["function"].startswith(PREFIXES)]
    kept = [r for r in old if not r["function"].startswith(PREFIXES)]

    key = lambda r: (r["label"], fid(r["function"]), r["dim"], r["run"])
    if Counter(map(key, gone)) != Counter(map(key, new)):
        raise SystemExit("re-run does not cover exactly the replaced "
                         "(algorithm, function, dim, run) tasks")

    bdir = Path(a.backup) / Path(a.dir).name
    bdir.mkdir(parents=True, exist_ok=True)
    if not (bdir / src.name).exists():
        shutil.copy2(src, bdir / src.name)

    with src.open("w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=cols)
        w.writeheader()
        for r in kept + new:
            w.writerow({c: r.get(c, "") for c in cols})
    print(f"{src}: kept {len(kept)}, replaced {len(gone)} with {len(new)} "
          f"C-exact rows; original backed up to {bdir / src.name}")


if __name__ == "__main__":
    main()
