# Live temporal feedback: follow-ups

Status: the post-hoc repair is done; the follow-ups below are open. The [20B report](../../long_runs/20B_recurrence/REPORT.md) has the findings and numbers.

## What is settled

- **The failure.** As trained, the 20B temporal arm fails in live execution: 0.48 NLL, against 0.21 in the training graph. About 95% of the penalty is on move-number characters.
- **The cause is the reader, not the memory.**
  - The live memory is well defined: it is the fixed point that enough training-graph passes reach.
  - The path there oscillates. Successive passes alternate between two states.
  - Training used only odd update counts ({1, 3} after step 39,101), so the mixer learned to read only one of the two states and misreads the fixed point.
  - The decay phase made the reader more selective.
- **The repair.** [`align.py`](align.py) retrains only the temporal mixer on settled memory: 200 updates, 3.3M characters. On confirmation rows, live NLL falls to 0.2176, and the training graph gets worse by 0.0038.
- **The hybrid.** It does not need the repair at four core iterations.

The original plan's 5,000-update fine-tune arms (T-warm, T-plain, hybrid control) are superseded by this repair.

## Open questions

1. **Can alignment be built into training instead of repaired afterwards?**
   - **Run:** resume temporal from the retained pre-decay checkpoint (step 175,954, optimiser state included) and redo the 19,550-update decay with about 25% warm-start batches. Each warm-start batch settles the memory without gradients, stopping at a tolerance with a cap of about 64 passes, then trains 1–3 passes from it.
   - **Controls:** the original final checkpoint and the post-hoc aligned one.
   - **Data:** matched; no extra characters.
   - **Cost:** about $4–5 per arm on a Secure RTX 4090. Repeat for the hybrid if the answer is useful.
2. **Does a gap-free, broad update support prevent the failure?**
   - **Run:** use the [1B update-schedule ablation](../1B_update_schedule/README.md) to compare:
     - support {0, 1, 3} with the current narrowing curriculum;
     - {0, 1, 2, 3, 4} with a floor of 5–10% on every count;
     - the same with a fixed distribution.
   - **Metrics:** the metrics listed below.
3. **Do damped memory writes make few-pass training states match live ones?**
   - **Change:** write `m ← ½(m + F(m))` in the training graph. The fixed point is unchanged, so live execution is unaffected, but the oscillation should die out within a few passes.
   - **Cost:** none. Test at 1B alongside question 2.
4. **Optional: does deep supervision help once live NLL is the metric?** The earlier rejection was early in training and measured only the training graph.

## Metrics for every run

Log these at each evaluation, not only at the end:

- **Live NLL:** 32 rows is enough to catch the failure.
- **Long chains:** training-graph NLL at 3 and at 63 updates.
- **Parity gap:** the difference between neighbouring even and odd update counts.
- **Memory usefulness:** NLL with memory off, minus NLL at 3 updates.
- **By character class:** in particular move numbers, where this failure showed most clearly.
