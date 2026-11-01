# Official CEC2020 real-world constrained suite — reference code and cross-check

## Source

`github.com/P-N-Suganthan/2020-RW-Constrained-Optimisation`, branch `master`,
downloaded 2026-10-31 into `download/`:

| file | bytes | SHA-256 |
|---|---|---|
| Problem-Definitions.pdf | 265 813 | `30649BC4E4E4ACA5CC8047EBE0A4E5DEC82657E6F336FA433963EC934987F19C` |
| Guidelines_Real_World_Constrained.pdf | 86 477 | `3C76FBD382714F0697BA1723D3C427B5EAD3F770C1C3B0F28A9D525E01253E0B` |
| SOFTWARE.zip | 42 941 | `AD5F679D6CF46E690FCE27F0E5CD79FDCEB3AA4323948ADB1B65E4863ED9B825` |
| Pyhton-C++.docx | 12 111 | `EA9F4DC001EC51161828109C53E345F83AD39A54942BDD1BA7ADA58B3A265A33` |

`SOFTWARE.zip` was extracted, unmodified, into `src/`. The reference
implementation is `src/Software/CEC2020/cec20_func.m` with `Cal_par.m`.

## What is used

`benchmarks/cec2020rw_official.py` ports the mechanical-engineering category,
RC15–RC33, except RC24 (its objective runs MATLAB's closed-source `fminbnd`;
see `PROTOCOL_CEC2020_MECH.md`, amendment 1). 18 problems.

## Cross-check against the official code

`cec20_func.m` run unmodified in GNU Octave 11.3.0, one point at a time,
objective and every constraint compared, 400 points per problem (300 uniform,
100 near a good solution):

```bash
python crosscheck_cec2020rw_octave.py --octave <path>/octave-cli.exe
```

Result on 2026-10-31 (`crosscheck_cec2020rw.log`): **18/18 problems match**;
feasibility agrees at all 400 points of every problem; largest relative
difference 2.4e-9 (RC33, dense vs sparse linear solve), 1.5e-13 (RC27,
generalised eigenproblem), at most 2.3e-16 elsewhere. No point produced a
complex value or an error in Octave.

Note: `cec20_func.m` returns the constraints transposed (`g = g'; h = h';`),
one column per point.
