# Official CEC2017 C reference — compiled cross-check

`cec17_test_func.cpp` is the organisers' C implementation, **unmodified**. It
comes from `CEC17_fast_pow-C++.zip` inside
`github.com/P-N-Suganthan/CEC2017-BoundContrained` `master.zip`:

- master.zip: 35.51 MB, SHA-256 `827E805E221869B8B89C51960B38E75A56729FECCF38BE4042AB250BF6EC5BE3`
  (the same archive the `cec2017_data/` files come from; all 328 data files
  are byte-identical)
- cec17_test_func.cpp: SHA-256 `8FAC3480680FE39009E810E5A05A8CFF16CBC5AA16F100028C989B6B29814382`

`cec17_eval.cpp` is our harness. It reads points from a binary file, calls
`cec17_test_func`, and writes the values back.

## Build and run (Windows, WinLibs GCC from winget)

```bash
g++ -O2 -std=gnu++17 -static -w -I benchmarks/cec2017_reference_c benchmarks/cec2017_reference_c/cec17_eval.cpp -o cec17_eval.exe
python crosscheck_cec2017_c.py --exe cec17_eval.exe
```

The script compares `benchmarks/cec2017_official.py` with the compiled C for
all 30 functions at D = 10/30/50/100. Per cell it uses 301–400 points:
uniform, at the optimum, near the optimum at five scales, and at every
composition component's optimum.

## Comparison with the authors' own submitted results

`master.zip` → `Codes-of-Top-Methods-and-results.zip` → `Results for all
papers.zip` holds the files the jSO, LSHADE-SPACMA and EBOwithCMAR authors
submitted to CEC2017: 14 checkpoints × 51 runs per function and dimension.
Extract `jSO-rawResults.tar.gz`, `LSHADE_SPACMA_Results_2.zip` and
`EBOwithCMAR.data.zip` into `<dir>/jso`, `<dir>/spacma` and `<dir>/ebo`, then:

```bash
python benchmarks/cec2017_reference_c/rivals_vs_official.py . <dir>   # our ports vs the originals
python benchmarks/cec2017_reference_c/mdac_vs_official.py   . <dir>   # MDAC vs the originals
python benchmarks/cec2017_reference_c/bounds_audit.py       .         # out-of-box evaluations
```

All 29 analysed functions (F1, F3–F30; F2 is withdrawn) are used. Results:
`FINAL_REPORT.md` §0.5.

## Result on 2026-10-31

First check: 22/30 functions matched to ≤ 4e-11 relative (most ≤ 1e-13).
**8 differed**: F2, F6, F7, F8, F9, F13, F14, F20. Causes are in
`FINAL_REPORT.md` §0.4.

The same day all eight were made C-exact and their cells re-run
(`run_cfix.ps1`, `merge_cfix.py`). **Now 30/30 match** at D = 10/30/50/100
(worst relative difference 3.6e-11, F27 D=30).
