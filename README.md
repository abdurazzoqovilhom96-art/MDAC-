# MDAC — learned global–local co-allocation in a DE–CMA-ES hybrid

Code, verified benchmark implementations and the raw results of every run behind the paper

> I. Abdurazzokov, Q. Ibragimov, H. Norqobilov, R. Mamatmurotov, F. Nuraliyeva.
> *Learned Global–Local Co-allocation in a Differential Evolution–CMA-ES Hybrid:
> Gains on Multimodal Benchmarks and Limits on Real-World Engineering Problems.*
> Submitted to the Arabian Journal for Science and Engineering, 2026.

MDAC is an L-SHADE-based algorithm in which a differential-evolution generator and a
(μ/μ_w, λ)-CMA-ES generator share one population, and the share of offspring given to
each is learned from the improvement it produces (the FCP co-allocation rule of
LSHADE-SPACMA), with a separate crossover-rate memory for the CMA-ES trials. The paper
reports where this helps (CEC2017 multimodal functions at equal evaluations) and,
with the same weight, where it does not (equal wall-clock time, CEC2011 real-world
problems, CEC2020 mechanical design problems, clustering).

## Contents

| Path | What it is |
|---|---|
| `MDAC_EXPERIMENT.py` | MDAC, all competitors (jSO, LSHADE-SPACMA, L-SHADE, EBOwithCMAR, CMA-ES, ablation variants), the evaluator that charges every objective call, the experiment driver and the statistics |
| `benchmarks/cec2017_official.py` | CEC2017, C-exact (30/30 functions match the organisers' C code) |
| `benchmarks/cec2011_official.py` | CEC2011 P1, P2, P5, P6, P7, P10, P12, P13, ported from the official MATLAB code (8/8 match) |
| `benchmarks/cec2020rw_official.py` | CEC2020 real-world constrained suite, mechanical category RC15–RC33 except RC24 (18/18 match) |
| `benchmarks/clustering.py` | SSE clustering on Iris, Wine, WDBC (loaded from scikit-learn) |
| `benchmarks/cec2017_data/` | official CEC2017 shift/rotation/shuffle files (third-party, see below) |
| `benchmarks/*_reference*/` | how each suite was verified against the official code, with sources and SHA-256 |
| `crosscheck_*.py`, `verify_cec2017_data.py` | the verification scripts |
| `FINAL_*/` | raw results and analysis of every stage of the paper (see below) |
| `FINAL_*_cfix/` | the re-run of the eight CEC2017 functions corrected to match the C code, already merged into `FINAL_*` |
| `logs/` | verification logs, analysis logs and the official-format CEC2020 tables |
| `PROTOCOL_CEC2020_MECH.md` | the protocol of the CEC2020 stage, written before the runs, with its three amendments |
| `run_*.ps1` | the PowerShell scripts that produced the results (Windows) |

## Installation

Python 3.13 or newer.

```bash
pip install -r requirements.txt
python verify_cec2017_data.py          # checks the CEC2017 data files, D = 10/30/50/100
python MDAC_EXPERIMENT.py --stage smoke --cec official --dims 10 --runs 1   # a few minutes
```

## Results and how to reproduce them

Every stage uses 51 runs. Each `FINAL_<stage>/raw_<tag>.csv` holds one row per run:
algorithm, function, category, dimension, run, final error with and without the
1e-8 rule (`error_thr`, `error_raw`), evaluations used, seconds and a failure flag.
`summary_*` has per-cell statistics, `pairwise_*` the Wilcoxon p-values, A12 and Holm
verdicts, `ranks_*` and `cec_friedman_*` the rankings. Files ending in `_pub`
restrict the analysis to the six published algorithms.

| Stage | Folder | Paper | Command |
|---|---|---|---|
| CEC2017 + classics, equal evaluations | `FINAL_fes` | Sect. 5.1–5.4 | `python MDAC_EXPERIMENT.py --stage full --cec official --out FINAL_fes` |
| Equal wall-clock time | `FINAL_time` | Sect. 5.7 | `python make_time_budget.py FINAL_fes --tag full`, then `python MDAC_EXPERIMENT.py --stage full --cec official --time_budget --out FINAL_time` |
| Ablation | `FINAL_abl` | Sect. 5.5 | `python MDAC_EXPERIMENT.py --stage abl51 --cec official --out FINAL_abl` |
| Architectural parity | `FINAL_parity` | Sect. 5.6 | `python MDAC_EXPERIMENT.py --stage parity --cec official --runs 51 --out FINAL_parity` |
| CEC2011 | `FINAL_rw` | Sect. 6.1 | `python MDAC_EXPERIMENT.py --stage rw --cec cec2011 --out FINAL_rw` |
| CEC2020 mechanical | `FINAL_mech` | Sect. 6.2 | `python MDAC_EXPERIMENT.py --stage mech --cec cec2020mech --out FINAL_mech` |
| Clustering | `FINAL_cluster` | Sect. 6.3 | `python MDAC_EXPERIMENT.py --stage top --cec clustering --dims 40 44 60 65 78 90 --runs 51 --algs MDAC MDAC-ctr jSO L-SHADE-SPACMA L-SHADE EBOwithCMAR CMA-ES k-means++ --out FINAL_cluster` |

Add `--workers N` to set the number of processes. The analysis is redone from the raw
files with `python reanalyse.py --dir FINAL_fes --tag full` (and the corresponding
folder and tag for the other stages; add `--suffix _pub --algs MDAC jSO L-SHADE-SPACMA
L-SHADE EBOwithCMAR CMA-ES` for the six published algorithms) and
`python report_final.py --dir FINAL_fes --tag full`. A full reproduction took about
70 hours on a 16–24-thread desktop. Wall-clock results depend on the machine.

**Hardware and software.** Most stages ran on an Intel i5-13400F with 15 worker
processes (Python 3.13, NumPy 2.5.3). The re-run of the eight corrected CEC2017
functions and the CEC2011 and CEC2020 stages ran on an Intel i7-13700F with 20 workers
(Python 3.14, NumPy 2.4.4). Within a cell all algorithms ran on the same machine.

## Verification of the benchmark code

- CEC2017: `crosscheck_cec2017_c.py` compares the Python suite with the organisers'
  compiled C code at 301–400 points per function and dimension
  (`logs/crosscheck_cfix.log`). Build instructions:
  `benchmarks/cec2017_reference_c/README.md`.
- CEC2011 and CEC2020: `crosscheck_cec2011_octave.py` and
  `crosscheck_cec2020rw_octave.py` run the official MATLAB code unmodified in GNU
  Octave and compare every objective and constraint value
  (`logs/crosscheck_cec2011.log`, `logs/crosscheck_cec2020rw.log`).

The official C and MATLAB sources are not redistributed here. Each
`benchmarks/*_reference*/README.md` gives the download location and the SHA-256 of the
archives that were used.

## Notes

- `MDAC_EXPERIMENT.py` keeps its development history in its comments and docstrings,
  including mechanisms that were tested and switched off. The configuration evaluated
  in the paper is the `MDAC` entry: `fcp=True`, `metropolis=True`,
  `alloc="adaptive"`, `cma_cr_mode="adaptive"`, `momentum=False`, `diversity_nu=0`.
- During development a locally generated CEC2017 look-alike (`BR_*`, "bridge") was
  used. It is **not** an official benchmark. No result from it appears in the paper
  or in this repository, and numbers from it have been removed from the code comments.
  The executable code is identical to the version that produced the results.
- In the parity control and in the "terminal allocation" ablation variant, the local
  stage is the restarting (1+λ)-CMA-ES (`cmaes_local`), started at 70 % of the budget.
- The success-history memories M_F and M_CR are updated from all successful trials,
  including those of the CMA-ES generator.

## Third-party material

- `benchmarks/cec2017_data/`: the official CEC2017 data files by the competition
  organisers (Awad, Ali, Liang, Qu, Suganthan), unmodified, from
  `github.com/P-N-Suganthan/CEC2017-BoundContrained`. They are not covered by this
  repository's license.
- jSO and EBOwithCMAR were transcribed from their authors' released code; L-SHADE and
  LSHADE-SPACMA follow their papers. The original publications are cited in the paper
  and in the header of `MDAC_EXPERIMENT.py`.

## License

The code is released under the MIT License (`LICENSE`). The result files (`FINAL_*/`,
`logs/`) are released under CC BY 4.0
(<https://creativecommons.org/licenses/by/4.0/>).

## Citation

See `CITATION.cff`. Until the paper is published, please cite this repository.
