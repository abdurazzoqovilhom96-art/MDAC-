"""Are our ports of jSO / L-SHADE-SPACMA / EBOwithCMAR as strong as the
originals? Compare our 51 final errors with the 51 the authors submitted to
CEC2017, on the 29 statistics functions (F1, F3-F30). Since the C-exact re-run
of 2026-10-31 our benchmark matches the official C on all 30 functions; F2 is
left out because the organisers withdrew it.

usage: python rivals_vs_official.py <project_dir> <official_results_dir>
"""
import csv, sys
from collections import defaultdict
from pathlib import Path
import numpy as np
from scipy.stats import mannwhitneyu, rankdata

PROJ, OFF = Path(sys.argv[1]), Path(sys.argv[2])
MATCH = [1] + list(range(3, 31))
DIMS = (10, 30, 50, 100)
FILES = {
    "jSO": lambda f, d: OFF / "jso" / "rawResults" / f"jSO_{f}_{d}.txt",
    "L-SHADE-SPACMA": lambda f, d: OFF / "spacma" / "LSHADE_SPACMA_Results" / f"LSHADE_SPACMA_{f}_{d}.txt",
    "EBOwithCMAR": lambda f, d: OFF / "ebo" / "EBOwithCMAR.data" / "data for email" / f"EBOwithCMAR_{f}_{d}.dat",
}


def thr(v):
    v = np.asarray(v, float)
    return np.where(v < 1e-8, 0.0, v)


def a12(x, y):          # P(x better than y), minimisation
    r = rankdata(np.concatenate([-x, -y]))
    return (r[:len(x)].sum() / len(x) - (len(x) + 1) / 2) / len(y)


ours = defaultdict(list)
for r in csv.DictReader(open(PROJ / "FINAL_fes" / "raw_full.csv", encoding="utf-8-sig")):
    if r["function"].startswith("CEC2017_") and r["label"] in FILES:
        fid = int(r["function"].split("_")[1][1:])
        ours[(r["label"], fid, int(r["dim"]))].append(float(r["error_thr"]))

for alg, path in FILES.items():
    rows = []
    for d in DIMS:
        for f in MATCH:
            off = np.loadtxt(path(f, d))[-1]                     # final checkpoint, 51 runs
            o = thr(off)
            m = np.array(ours[(alg, f, d)])
            p = 1.0 if np.allclose(np.sort(m), np.sort(o)) else mannwhitneyu(m, o).pvalue
            rows.append((f, d, a12(m, o), p, m.mean(), o.mean(), np.median(m), np.median(o)))
    n = len(rows)
    a = np.array([r[2] for r in rows])
    p = np.array([r[3] for r in rows])
    worse = sum(1 for r in rows if r[3] < 0.05 / n and r[2] < 0.5)
    better = sum(1 for r in rows if r[3] < 0.05 / n and r[2] > 0.5)
    sat = sum(1 for r in rows if r[6] == 0 and r[7] == 0)
    print(f"\n=== {alg}: ours vs the authors' submitted CEC2017 results ({n} cells)")
    print(f"  mean A12 (ours better than original) = {a.mean():.3f}   "
          f"Bonferroni-significant: ours WORSE in {worse}, ours BETTER in {better}, "
          f"both medians 0 in {sat}")
    for d in DIMS:
        ad = [r[2] for r in rows if r[1] == d]
        wd = sum(1 for r in rows if r[1] == d and r[3] < 0.05 / n and r[2] < 0.5)
        bd = sum(1 for r in rows if r[1] == d and r[3] < 0.05 / n and r[2] > 0.5)
        print(f"    D={d:<3d} A12 {np.mean(ad):.3f}   worse {wd:2d}  better {bd:2d}")
    bad = sorted([r for r in rows if r[3] < 0.05 / n and r[2] < 0.5], key=lambda r: r[2])[:8]
    if bad:
        print("  largest significant deficits of our port:")
        for f, d, aa, pp, mm, om, md, od in bad:
            print(f"    F{f:<2} D{d:<3} A12 {aa:.3f}  mean ours {mm:.3g} vs orig {om:.3g}"
                  f"   median ours {md:.3g} vs orig {od:.3g}")
    good = sorted([r for r in rows if r[3] < 0.05 / n and r[2] > 0.5], key=lambda r: -r[2])[:5]
    if good:
        print("  largest significant advantages of our port:")
        for f, d, aa, pp, mm, om, md, od in good:
            print(f"    F{f:<2} D{d:<3} A12 {aa:.3f}  mean ours {mm:.3g} vs orig {om:.3g}"
                  f"   median ours {md:.3g} vs orig {od:.3g}")
