#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
merge_time_budget.py -- add equal-time budgets for newly built functions to
equal_time_budget.json, keeping every existing entry unchanged.

Used for the C-exact re-run of 2026-10-31: the re-run functions have new names,
and their budgets are measured on the machine and worker count that will run
their equal-time cells. Each cell's budget is shared by all algorithms of that
cell, so per-cell comparisons stay fair; cells measured on different machines
are not comparable in absolute seconds, and that is disclosed.

    python merge_time_budget.py equal_time_budget.json equal_time_budget_cfix.json
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("base")
    ap.add_argument("add")
    ap.add_argument("--machine", default="i7-13700F, 20 workers")
    a = ap.parse_args()

    base = json.loads(Path(a.base).read_text(encoding="utf-8"))
    add = json.loads(Path(a.add).read_text(encoding="utf-8"))
    clash = sorted(set(base["budget"]) & set(add["budget"]))
    if clash:
        raise SystemExit(f"refusing to overwrite existing budgets: {clash}")
    base["budget"].update(add["budget"])
    base.setdefault("amendments", []).append({
        "added": sorted(add["budget"]),
        "source": add.get("source"),
        "rule": add.get("rule"),
        "machine": a.machine,
        "note": "C-exact re-run, 2026-10-31; the original entries were measured "
                "on an i5-13400F at 15 workers",
    })
    Path(a.base).write_text(json.dumps(base, indent=1), encoding="utf-8")
    print(f"added {len(add['budget'])} functions to {a.base}")


if __name__ == "__main__":
    main()
