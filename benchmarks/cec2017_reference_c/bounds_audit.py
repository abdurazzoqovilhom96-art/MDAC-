"""Does any algorithm evaluate points outside the [-100,100]^D box?
Short runs (2000*D FES), D=10 and 30, four CEC functions, 2 seeds."""
import sys
from pathlib import Path
import numpy as np

PROJ = Path(sys.argv[1])
sys.path.insert(0, str(PROJ))
import MDAC_EXPERIMENT as M  # noqa: E402

ALGS = ["MDAC", "jSO", "L-SHADE-SPACMA", "L-SHADE", "EBOwithCMAR", "CMA-ES"]
FUNCS = ("CEC2017_F5_", "CEC2017_F10_", "CEC2017_F12_", "CEC2017_F22_")
stats = {a: [0, 0, 0.0] for a in ALGS}           # points, outside, max |x|
for dim in (10, 30):
    specs = [s for s in M.build_registry(dim, M.BENCH_SEED, True, cec="official")
             if s.name.startswith(FUNCS)]
    for s in specs:
        lo, hi = s.bounds
        for a in ALGS:
            for run in range(2):
                st = stats[a]

                def wrapped(X, f=s.func, st=st):
                    X = np.atleast_2d(X)
                    st[0] += X.shape[0]
                    st[1] += int(np.sum(np.any((X < lo) | (X > hi), axis=1)))
                    st[2] = max(st[2], float(np.max(np.abs(X))))
                    return f(X)

                ev = M.SpreadEvaluator(wrapped, dim, max_fes=2000 * dim, f_star=s.f_star)
                try:
                    M.resolve(a)(ev, dim, s.bounds, 42 + run * 1000, pop_size=None)
                except M.BudgetExhausted:
                    pass
print(f"{'algorithm':16s}{'points':>10s}{'outside box':>13s}{'max |x|':>10s}")
for a, (n, out, mx) in stats.items():
    print(f"{a:16s}{n:10d}{out:13d}{mx:10.3f}")
