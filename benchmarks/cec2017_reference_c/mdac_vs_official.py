"""MDAC (our runs) against the results the rivals' AUTHORS submitted to CEC2017,
on the 29 statistics functions (F1, F3-F30). Since the C-exact re-run of
2026-10-31 our benchmark equals the official C on all 30 functions; F2 is left
out because the organisers withdrew it. No port of ours is involved on the
rival side, so a weak port cannot help MDAC here.

usage: python mdac_vs_official.py <project_dir> <official_results_dir>
"""
import csv, sys
from collections import defaultdict
from pathlib import Path
import numpy as np
from scipy.stats import mannwhitneyu, rankdata, wilcoxon

PROJ, OFF = Path(sys.argv[1]), Path(sys.argv[2])
MATCH = [1] + list(range(3, 31))
CAT = {**{f: "unimodal" for f in (1, 3)}, **{f: "multimodal" for f in range(4, 11)},
       **{f: "hybrid" for f in range(11, 21)},
       **{f: "composition" for f in range(21, 31)}}
NC = len(MATCH) * 4
DIMS = (10, 30, 50, 100)
FILES = {
    "jSO": lambda f, d: OFF / "jso" / "rawResults" / f"jSO_{f}_{d}.txt",
    "L-SHADE-SPACMA": lambda f, d: OFF / "spacma" / "LSHADE_SPACMA_Results" / f"LSHADE_SPACMA_{f}_{d}.txt",
    "EBOwithCMAR": lambda f, d: OFF / "ebo" / "EBOwithCMAR.data" / "data for email" / f"EBOwithCMAR_{f}_{d}.dat",
}
thr = lambda v: np.where(np.asarray(v, float) < 1e-8, 0.0, np.asarray(v, float))


def a12(x, y):
    r = rankdata(np.concatenate([-x, -y]))
    return (r[:len(x)].sum() / len(x) - (len(x) + 1) / 2) / len(y)


mdac, ours_rival = defaultdict(list), defaultdict(list)
for r in csv.DictReader(open(PROJ / "FINAL_fes" / "raw_full.csv", encoding="utf-8-sig")):
    if r["function"].startswith("CEC2017_"):
        fid = int(r["function"].split("_")[1][1:])
        if r["label"] == "MDAC":
            mdac[(fid, int(r["dim"]))].append(float(r["error_thr"]))
        elif r["label"] in FILES:
            ours_rival[(r["label"], fid, int(r["dim"]))].append(float(r["error_thr"]))

off = {(a, f, d): thr(np.loadtxt(p(f, d))[-1]) for a, p in FILES.items() for f in MATCH for d in DIMS}

# per-cell Mann-Whitney, Holm over the whole family
recs = []
for a in FILES:
    for f in MATCH:
        for d in DIMS:
            x, y = np.array(mdac[(f, d)]), off[(a, f, d)]
            p = 1.0 if np.allclose(np.sort(x), np.sort(y)) else mannwhitneyu(x, y).pvalue
            recs.append({"a": a, "f": f, "d": d, "A12": a12(x, y), "p": p})
order = sorted(range(len(recs)), key=lambda i: recs[i]["p"])
m, still = len(recs), True
for k, i in enumerate(order):
    rej = still and recs[i]["p"] <= 0.05 / (m - k)
    still = still and rej
    recs[i]["v"] = ("W" if recs[i]["A12"] > .5 else "L") if rej else "T"


def pool(rs):
    c = {v: sum(r["v"] == v for r in rs) for v in "WTL"}
    return f"{np.mean([r['A12'] for r in rs]):.3f} ({c['W']}/{c['T']}/{c['L']})"


print(f"MDAC (ours) vs the authors' submitted results -- {len(MATCH)} C-matching functions x 4 D, "
      f"Holm m={m}\n")
print(f"{'':16s}{'overall':>18s}" + "".join(f"{c:>18s}" for c in ("multimodal", "composition", "hybrid", "unimodal")))
for a in list(FILES) + ["ALL 3"]:
    rs = recs if a == "ALL 3" else [r for r in recs if r["a"] == a]
    print(f"{a:16s}{pool(rs):>18s}" + "".join(
        f"{pool([r for r in rs if CAT[r['f']] == c]):>18s}" for c in ("multimodal", "composition", "hybrid", "unimodal")))
print("\nby dimension (all 3 rivals pooled):")
for d in DIMS:
    rs = [r for r in recs if r["d"] == d]
    mm = [r for r in rs if CAT[r["f"]] == "multimodal"]
    print(f"  D={d:<3d} overall {pool(rs)}   multimodal {pool(mm)}")

# same comparison, but against OUR ports: how much does the port choice move it?
print(f"\nfor reference -- MDAC vs OUR ports on the same {NC} cells (unpaired A12):")
for a in FILES:
    for c in ("overall", "multimodal"):
        keys = [(f, d) for f in MATCH for d in DIMS if c == "overall" or CAT[f] == c]
        v_off = np.mean([a12(np.array(mdac[k]), off[(a,) + k]) for k in keys])
        v_our = np.mean([a12(np.array(mdac[k]), np.array(ours_rival[(a,) + k])) for k in keys])
        print(f"  {a:16s} {c:10s}  vs authors' results {v_off:.3f}   vs our port {v_our:.3f}")

# mean-error Friedman ranking: MDAC + the three authors' results, all cells
algs = ["MDAC"] + list(FILES)
R = []
for f in MATCH:
    for d in DIMS:
        R.append(rankdata([np.mean(mdac[(f, d)])] + [off[(a, f, d)].mean() for a in FILES]))
R = np.array(R).mean(axis=0)
Rm = []
for f in MATCH:
    for d in DIMS:
        Rm.append(rankdata([np.median(mdac[(f, d)])] + [np.median(off[(a, f, d)]) for a in FILES]))
Rm = np.array(Rm).mean(axis=0)
print(f"\nFriedman over {NC} cells, MDAC + authors' results:")
print("  mean-error ranks  : " + "  ".join(f"{a} {v:.3f}" for a, v in sorted(zip(algs, R), key=lambda t: t[1])))
print("  median-error ranks: " + "  ".join(f"{a} {v:.3f}" for a, v in sorted(zip(algs, Rm), key=lambda t: t[1])))
for a in FILES:
    x = np.array([np.mean(mdac[(f, d)]) for f in MATCH for d in DIMS])
    y = np.array([off[(a, f, d)].mean() for f in MATCH for d in DIMS])
    print(f"  cell-level Wilcoxon on means, MDAC vs {a:15s}: better {int((x<y).sum())} / "
          f"equal {int((x==y).sum())} / worse {int((x>y).sum())}   p = {wilcoxon(x, y).pvalue:.2e}")
