# Live temporal feedback

The 20B temporal model failed in live execution because training never showed it the settled memory that live execution uses. Two fixes work, both in the [20B report](../../long_runs/20B_recurrence/REPORT.md#post-hoc-live-alignment):

- **After training:** [`align.py`](align.py) retrains only the temporal mixer on settled memory. It takes 200 updates.
- **During training:** [`aligned_decay.py`](aligned_decay.py) redoes the decay phase with 25% warm-start batches (the trainer's `warm_start_fraction`). This fixes live execution at almost no training-graph cost.

## Open questions

1. **Does a gap-free, broad update support prevent the failure without a fix?**
   - **Run:** use the [1B update-schedule ablation](../1B_update_schedule/README.md) to compare:
     - support {0, 1, 3} with the narrowing curriculum;
     - {0, 1, 2, 3, 4} with at least 5–10% on every count;
     - the same with a fixed distribution.
2. **Do damped memory writes, `m ← ½(m + F(m))`, bring few-pass training states close to live ones?** The fixed point is unchanged, so live execution is unaffected. It costs nothing; test it at 1B alongside question 1.

## Metrics for every run

Log these at each evaluation, not only at the end:

- **Live NLL:** 32 rows is enough to catch the failure.
- **Long chains:** training-graph NLL at 3 and at 63 updates.
- **Parity gap:** the difference between neighbouring even and odd update counts.
- **Move-number NLL:** where this failure showed most clearly.
