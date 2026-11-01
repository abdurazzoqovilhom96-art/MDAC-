# Official CEC2011 real-world problems — reference code and cross-check

## Source

`github.com/P-N-Suganthan/CEC-2011--Real_World_Problems`, branch `master`,
downloaded 2026-10-31 into `download/`:

| file | bytes | SHA-256 |
|---|---|---|
| Tech-Rep.pdf | 295 690 | `B0B1242B7FB50CC9CED56EAE328FEBE1BBCB4E82009D41409454CF65E160EB7F` |
| Linux-C.rar | 193 992 | `8D3DB8E4E6A6F7BB4D10F7D218B6015AF9D53294E4C4C0AD731B98BE5AEE11BA` |
| Matlab.rar | 1 723 423 | `AC7B743F45895E0DBA6B7B03F41461DB809532D8D312167C5092AEDE6E611095` |

Extracted, unmodified, into `src_Linux-C/` and `src_Matlab/`. The "C" package
is only a wrapper (`mCEC_Function.cpp`) around the MATLAB code compiled with
`mcc`; it needs the MATLAB runtime. **The MATLAB files are therefore the
reference implementation.** Every file that the two packages share
(`bench_func.m`, `antennafunccircular.m`, `array_factorcir.m`, `mga_dsm.m`,
`messengerfull.mat`, `cassini2.mat`, ...) is byte-identical between them.

## What is used

`benchmarks/cec2011_official.py` ports, line by line, every problem that
Table 1 of the technical report lists as bound-constrained with D >= 6:
P1, P2, P5, P6, P7, P10, P12, P13. P3/P4 (1-D, MATLAB ode45 at RelTol 0.1)
and the constrained P8, P9, P11.x are not used. The module docstring lists the
quirks of the official code that are reproduced on purpose.

Bounds follow the technical report. For Messenger (P12) the report's Table 2
and `getlimit_messenger.m` give t0 in [1900, 2300] and Vinf in [2.5, 4.05];
the `bounds` field inside `messengerfull.mat` (ESA's original box) says
[1900, 2200] and [2.5, 5.0]. The report defines the competition, so its box
is used. Cassini 2 (P13): Table 3, `getlimit_cassini2.m` and the .mat agree.

The P10 arguments are those of the official wrapper: null = [50, 120],
phi_desired = 180, distance = 0.5.

## Cross-check against the official code

The MATLAB files were run unmodified in GNU Octave 11.3.0 (portable build
`octave-11.3.0-w64.7z` from ftpmirror.gnu.org, 481 127 074 bytes, SHA-256
`FD3CF0E885467A15211B8CEDED42800432A5462481324D49A896EF257E05D1A0`):

```bash
python crosscheck_cec2011_octave.py --octave <path>/octave-cli.exe
```

Per problem: 300 uniform points, 100 points near a good solution, plus P1's
official target point. Result on 2026-10-31 (`crosscheck_cec2011.log`):

| problem | points | both finite | max relative difference | non-finite sets agree |
|---|---|---|---|---|
| P1 FM sound waves | 401 | 401 | 7.7e-16 | 401/401 |
| P2 Lennard-Jones | 400 | 400 | 0 | 400/400 |
| P5 Tersoff Si(B) | 400 | 144 | 0 | 400/400 |
| P6 Tersoff Si(C) | 400 | 84 | 0 | 400/400 |
| P7 radar polyphase | 400 | 400 | 0 | 400/400 |
| P10 circular antenna | 400 | 400 | 8.4e-15 | 400/400 |
| P12 Messenger | 400 | 400 | 0 | 400/400 |
| P13 Cassini 2 | 400 | 400 | 0 | 400/400 |

**8/8 problems match.** Octave produced no complex values and no errors on
these points.

The Tersoff NaNs are a property of the official code, not of the port: where
fcr(i,k) = 0 and the exp() term overflows, 0*Inf = NaN, and Octave returns
NaN on exactly the same points. In the experiments every non-finite value is
replaced by a 1e20 death penalty (`unit_box`), identically for all
algorithms.

Octave and MATLAB use different maths libraries, so MATLAB itself may differ
from these values in the last bits; a porting error would show as an O(1)
difference.
