# Protocol — CEC2020 real-world constrained, mechanical engineering problems

**Written 2026-10-31, before any optimisation run on these problems.**
Nothing below may be changed after results are seen without saying so in the
report, with the reason.

## Why this stage exists

AJSE readers expect classic engineering design problems (pressure vessel,
welded beam, spring, speed reducer, ...). The CEC2011 stage (FINAL_REPORT §5a)
did not include constrained problems.

## Problems — no hand selection

**All 19 problems of the "mechanical engineering" category of the CEC2020
real-world constrained suite, RC15–RC33** (Kumar et al., Swarm Evol. Comput.
56 (2020) 100693; official code `github.com/P-N-Suganthan/2020-RW-Constrained-Optimisation`,
`SOFTWARE.zip`, SHA-256 `AD5F679D…B825`). This includes all the classic design
problems (speed reducer RC15, spring RC17/RC30, pressure vessel RC18, welded
beam RC19, three-bar truss RC20, clutch brake RC21, gear train RC31, ...).

A problem may be dropped **only** if its official MATLAB code cannot be
reproduced (cross-check against the official code in Octave fails and the
cause cannot be fixed). Any drop is reported with its reason. Performance is
never a reason.

## Code fidelity

Each problem is ported from `cec20_func.m` and must match the official MATLAB
code run in GNU Octave 11.3.0 (same procedure as CEC2011: uniform points plus
points near good solutions; objective and every constraint compared). Quirks of
the official code are kept. Problems are evaluated one point at a time
(ps = 1), so a result never depends on which other points share a batch
(RC22's `if beta == real(beta)` is population-wide in MATLAB).

## Constraint handling — identical for every algorithm

The algorithms are unconstrained optimisers, so constraints enter through one
fixed scalar fitness, applied identically to all eight algorithms:

    v(x) = ( sum_i max(0, g_i(x)) + sum_j H_j(x) ) / m        (official violation)
    H_j(x) = |h_j(x)| if |h_j(x)| > 1e-4, else 0             (official tolerance)
    F(x)  = f(x)          if v(x) = 0
    F(x)  = C_k + v(x)    otherwise

This reproduces the official solution ordering exactly (feasible before
infeasible; feasible by f; infeasible by mean violation) as long as C_k
exceeds every feasible f of problem k. C_k is fixed per problem **before the
runs**: a power of ten at least 1000 times the largest feasible |f| found in
200 000 uniform samples plus points near good solutions. m counts the
constraint columns the official function returns (inequality) plus its
declared equality constraints. Non-finite values get F = 1e20.

## Protocol

- Algorithms: the 8 of stage 1 (MDAC, MDAC-ctr, MDAC-noFCP, jSO,
  L-SHADE-SPACMA, L-SHADE, EBOwithCMAR, CMA-ES), unchanged.
- Budget: the official MaxFEs = 1e5 (D <= 10), 2e5 (10 < D <= 30).
- Runs: 51 per problem (the official 25 is a minimum; 51 matches every other
  stage).
- Search space: the official bounds (`Cal_par.m`), mapped from [0,1]^D as in
  CEC2011. Uniform initialisation, as the guidelines require.
- Seeds: as every other stage (independent of the algorithm).

## Analysis plan (fixed now)

Primary: per-run Friedman rank of F over the 19 problems, MDAC's position
among the 6 published algorithms and among all 8; Holm-corrected per-problem
Mann-Whitney W/T/L and A12 of MDAC vs each algorithm.
Also reported, per problem and algorithm: feasibility rate (FR, official
definition), best/median/mean/worst/std of f over feasible runs, and the gap
to the official best-known value f*.

## Amendment 1 (2026-10-31, while porting, before any run)

**RC24 (robot gripper) is dropped.** Its objective calls MATLAB's `fminbnd`
inside the function (`OBJ11`). MATLAB's `fminbnd` is closed source, and the
only runnable reference here, Octave's `fminbnd`, differs from it: its step
tolerance uses eps instead of sqrt(eps), and for complex intermediate values
(acos of arguments beyond ±1, common for random gripper geometries) Octave
orders complex numbers by modulus then argument while MATLAB compares real
parts. The official RC24 values can therefore neither be reproduced exactly
nor verified without MATLAB. 18 problems remain.

## Amendment 2 (2026-10-31, after the code cross-check, before any run)

Cross-check: the 18 ported problems match the official MATLAB code in Octave
on the objective and every constraint, and on feasibility, at 400 points each
(`crosscheck_cec2020rw.log`; max relative difference 2.4e-9, RC33).

C_k. The sampling rule alone is not safe where sampling finds no feasible
point (RC23, RC26: 0 of 200 000) or few. C_k is therefore the power of ten at
least 1000 × max(B_meas, B_anal), where B_meas is the largest feasible |f|
measured (`scan_cec2020rw.log`) and B_anal a crude upper bound of |f| over the
whole box, derived by hand where needed: RC17 88, RC22 3 845 (i2 with
N6 − N4 = 1), RC23 26.2, RC26 3.3e3.

| RC | 15 | 16 | 17 | 18 | 19 | 20 | 21 | 22 | 23 | 25 | 26 | 27 | 28 | 29 | 30 | 31 | 32 | 33 |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| C_k | 1e7 | 1e10 | 1e5 | 1e9 | 1e5 | 1e6 | 1e4 | 1e7 | 1e5 | 1e8 | 1e7 | 1e7 | 1e8 | 1e11 | 1e5 | 1e6 | 1e8 | 1e12 |

RC31 and RC33 have no binding constraints (every point is feasible), so their
C_k is never used.

## Amendment 3 (2026-10-31, after a one-run integration test, before the runs)

The integration test (8 algorithms × 18 problems × 1 run) crashed 3 of 144
runs, all EBOwithCMAR, with index errors: our EBOwithCMAR port evaluates
points outside the search box (already documented, FINAL_REPORT §0.5), and
problems with lookup tables (RC22, RC26, RC30) cannot be evaluated there — the
official MATLAB code would fail too. The search range is a hard rule of the
competition, so **points outside the official box are invalid and get the
same 1e20 penalty as non-finite values**, for every algorithm. Only an
algorithm that leaves the box is affected. This decision rests on the crash,
not on any performance number (none was looked at).

## Decision on the paper (2026-11-01, after the results)

The commitment below asked for this decision before the results; it was taken
after them. It is recorded here as such. **Decision: the stage goes into the
paper, in full, together with the CEC2011 stage**, although MDAC is mid-table
on it (FINAL_REPORT §5b). Leaving it out after seeing it would have been
selective reporting.

## Reporting commitment

The results go into FINAL_REPORT.md in full **whatever they are**. The
decision whether this stage appears in the paper should be taken before the
results are seen; if it is left out of the paper, the paper must still say
that it was run and where the results are.
