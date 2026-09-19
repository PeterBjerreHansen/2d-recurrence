# Deep-supervision training ablation report

## Question

Does training the recurrent model with normalized auxiliary predictions from
nonfinal passes improve the final-only evaluation surface?

The auxiliary condition used the first planned objective:

```text
L = (L_final + 0.25 * mean(L_intermediate)) / 1.25
```

One-pass trajectories contribute `L_final` only. The control is the existing
architecture-A final-only run at
`experiments/ablations/architecture_sites/results/separated`.

## Matched protocol

- architecture A: 1 prelude, 1 buffer, 4 core, 1 source, 1 coda;
- full `chess_143K_v1` dataset;
- seed 1337, recurrence schedule seed 1729;
- same optimizer, learning-rate schedule, recurrence probabilities, batch,
  frozen selection panel, and 10,000-update budget;
- checkpoints at 0, 1,000, 2,000, 5,000, 8,000, and 10,000;
- final-only evaluation on the nine recurrence cells;
- selection panel: 128 fixed validation rows, three mask seeds.

The implementation and regression tests are in `models/recurrent_2d.py`,
`train.py`, and `tests/test_recurrence.py`. The full test suite passed with
78 tests.

## Selection results

NLL is lower-is-better. The `best` column is the best of the nine recurrence
cells, not a separately selected model.

| Updates | Condition | (0,0) NLL | (1,1) NLL | (3,3) NLL | Best NLL |
| ---: | --- | ---: | ---: | ---: | ---: |
| 1,000 | final-only | 0.90250 | 0.82352 | 0.81861 | 0.81778 |
| 1,000 | auxiliary | 0.83658 | 0.79655 | 0.79589 | 0.79426 |
| 2,000 | final-only | 0.67857 | 0.62324 | 0.62438 | 0.62256 |
| 2,000 | auxiliary | 0.62720 | 0.60334 | 0.60671 | 0.60321 |
| 5,000 | final-only | 0.50457 | 0.47587 | 0.47568 | 0.47497 |
| 5,000 | auxiliary | 0.48860 | 0.47075 | 0.47176 | 0.47059 |
| 8,000 | final-only | 0.44461 | 0.42035 | 0.41992 | 0.41972 |
| 8,000 | auxiliary | 0.43373 | 0.41839 | 0.41810 | 0.41797 |
| 10,000 | final-only | 0.43084 | 0.40725 | 0.40661 | 0.40661 |
| 10,000 | auxiliary | 0.42181 | 0.40701 | 0.40624 | 0.40624 |

At 10,000 updates, averaging all nine cells gives the following descriptive
cross-backend values:

| Condition | Mean NLL | Mean accuracy |
| --- | ---: | ---: |
| Final-only | 0.41287 | 0.84740 |
| Auxiliary λ=0.25 | 0.41079 | 0.84798 |

These values cannot establish that the auxiliary run is better because the
conditions were evaluated on different backends. The apparent differences are
concentrated in the no-update and shallow-update cells: the `(0,0)` values
differ by 0.00903, while `(3,3)` differs by only 0.00036. Treat these as
observations requiring a matched rerun, not evidence of an optimization,
generalization, or recurrent-computation benefit.

## Cost and diagnostics

The original timing comparison is not a valid estimate of auxiliary-supervision
overhead. The auxiliary run used MPS, while its reused final-only control used
CUDA on an A6000. It also compares single-update timings. The reported 0.178 s
and 1.120 s values must therefore be retained as raw observations only, not
converted into a ``roughly 6x slower`` claim. The same hardware/backend issue
also means that the small NLL differences in the selection table are
cross-backend observations and should not be treated as a clean objective
effect.

The auxiliary run's own final training diagnostics were:

- final-pass NLL: 0.37801;
- intermediate-pass NLL: 0.38563;
- normalized objective: 0.37952;
- fixed four-batch validation NLL: 0.42430.

No nonfinite values or training failures occurred. The existing forward
matrix-multiply estimator remains appropriate for final-only evaluation of
either checkpoint. It does not describe deep-supervision training cost, which
adds intermediate source/coda executions, readout and normalization work, and
backward work.

## Scope and next step

The large confirmation split was not run: it contains 1,317 rows, compared
with 128 selection rows, and would multiply the already expensive nine-cell
evaluation. The selection result is sufficient for triage, but not for a
final claim.

Recommendation: retain final-only as the practical default pending a matched
same-backend cost comparison. If the objective effect is important, run a
separately budgeted benchmark with identical device, dtype, checkpoint,
batching, schedules, and measured-update protocol for final-only and
normalized λ=0.25 supervision. Until then, treat λ=0.25 as an optional
optimization variant rather than a new baseline.
