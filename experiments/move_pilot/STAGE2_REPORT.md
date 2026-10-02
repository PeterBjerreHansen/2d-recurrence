# Stage 2: Leela's search distribution, full runs

**Question.** At the deep end of the target range (Leela's search distribution, which rewards calculation), which recurrence axis pays off? Stage 1, the shallow end, ended in a tie ([report](STAGE1_REPORT.md)).

**Answer at this scale.**
- **Temporal recurrence pays off, by a small margin.** On Leela's distribution, temporal and hybrid end ahead of the transformer by 0.014–0.016 KL. The margin is the same after three times as much training.
- **On a tactics panel, temporal and hybrid follow a tactic through better** at all 7 scored checkpoints: they put more probability on the deep move and lose less win probability. They are no better at finding the first move of a tactic.
- **Depth recurrence doesn't pay off.** Depth-only ends level with the transformer. Within-position passes help up to J = 2–4 but never overtake the transformer.

One seed per arm. Runs `full_engine_{arm}` (one pass) and `full_engine_long_{arm}` (three passes) in [`pilot.py`](pilot.py); analyses in [`stratify_stage2.py`](stratify_stage2.py), [`hybrid_grid.py`](hybrid_grid.py) and [`tactics.py`](tactics.py).

## Setup

- **Data:** `leela_full_v1`, 400M positions from 56 Leela test80 archives (2024-04-01 to 04-03), 2,011,683 rows. Dev and test sets are whole held-out games.
- **Target:** Leela's search probabilities over the legal moves; softmax over legal moves, cross-entropy against those probabilities, final pass only. KL from Leela is the loss minus the target's entropy.
- **Start:** each arm's final stage-1 model (`full_legal_{arm}`), with a fresh output layer and optimizer.
- **Training:** 400 rows per update, AdamW at 1e-3, 3% warmup, flat, the last 10% decaying linearly to 1e-4. Stage 1's final update mixture throughout (U = 1–4: 0.15, 0.20, 0.30, 0.35; hybrid 80% on the diagonal).
  - **Short:** 5,030 updates, one pass.
  - **Long:** resumes the short run at its decay start (update 4,527, still at the peak learning rate) and runs to 15,090 updates, three passes, decaying from 13,581. Checkpoints at 6,000, 8,000, 10,000, 12,000, 13,581 and the end.
- **Evaluation** every 500 updates: training graph on Leela dev games; live decoding on a subset, with the training graph scored on the same subset.

## Results on Leela's distribution

**At the end** (Leela dev set, training graph):

| Arm | KL, short | KL, long | Top-move agreement, short | Top-move agreement, long |
| --- | --- | --- | --- | --- |
| Temporal | **0.672** | **0.609** | **0.465** | **0.491** |
| Hybrid | 0.678 | 0.611 | 0.460 | 0.488 |
| Transformer | 0.686 | 0.625 | 0.460 | 0.483 |
| Depth | 0.689 | 0.627 | 0.459 | 0.484 |

- **Three times as much training lowers every arm by about 0.06 KL, and the gaps barely move.** Temporal leads the transformer by 0.014 (short) and 0.016 (long). The ordering is stable from update 1,000.
- **Most of the long run's gain comes in the decay:** the transformer goes from 0.679 (update 12,000) to 0.625.
- **The longer runs don't reach clearly different plateaus.**

**Passes,** live decoding on the dev subset, long run, end (KL):

| Arm | J = 1 | J = 2 | J = 4 | J = 5 | Training graph, same subset |
| --- | --- | --- | --- | --- | --- |
| Temporal | 0.583 | — | — | — | 0.583 |
| Depth | 0.712 | 0.608 | 0.599 | 0.599 | 0.599 |
| Hybrid | 0.627 | 0.586 | **0.580** | 0.581 | 0.581 |

- Live execution matches the training graph for all three recurrent arms.
- Gains stop after J = 4, and most of them come by J = 2. The short runs show the same pattern.
- J = 1 is out of distribution for depth and hybrid (no U = 0 in stage 2).

**By position type** (short runs, `stratify_stage2.py`, 2.15M dev positions):
- Temporal leads in every stratum.
- The gain from more passes (J = 2 → 4) is largest where Leela's top move is a check (0.020–0.024 KL) and smallest in check or where Leela is unsure (0.006).

**The hybrid's (U_T, U_D) grid** (short runs, `hybrid_grid.py`): KL is flat from (2, 2) at 0.666; (4, 4) gives 0.664, as does temporal alone. Cells far from the diagonal were barely trained, so the grid's edges don't measure either axis alone.

## Tactics panel

**Building it** (`tactics.py`):
- Every position of 5,000 human dev games is searched by Stockfish 17.1 (official AVX-512 build, SHA-256 recorded) at depth 1 and at 200k nodes.
- A **tactic** is a position where the depth-1 move loses at least 0.10 Q (win probability) under the deeper search. It is followed for up to three steps along the deep line: plies +0, +2 and +4, same side to move, with the line so far in the context.
- **Controls** are positions where both searches pick the same move.
- At every step, the teacher labels every legal move at 200k nodes.
- The panel: 8,000 tactics (8,000 / 6,382 / 4,022 steps at plies +0 / +2 / +4) and 2,000 controls.

**Metrics.** The model picks its top legal move. Three metrics:
- *agreement*: the pick is the deep move;
- *regret*: Q lost by the pick against the best move;
- *p_best*: the probability given to the deep move. It is smoother than agreement, which flips when two moves are nearly tied.

**Checkpoints.** Seven per arm:
- the five constant-learning-rate checkpoints of the long run (6,000 to 13,581);
- two trained to the end of their decay, the final checkpoints of the long and short runs.

Below, each arm minus the transformer, averaged over each group; 95% intervals by a bootstrap over tactics; ✱ marks an interval that excludes zero. Depth and hybrid are shown at J = 4.

**p_best** (transformer: 0.23–0.25 at step 0, 0.25–0.27 at steps 1–2, 0.33–0.36 on controls):

| Group | Checkpoints | Temporal | Depth J4 | Hybrid J4 |
| --- | --- | --- | --- | --- |
| Finding the tactic (step 0) | constant | +0.006 ✱ | −0.002 ✱ | −0.001 |
| | decayed | +0.010 ✱ | +0.001 | +0.002 |
| Following through (steps 1–2) | constant | **+0.012 ✱** | +0.001 | **+0.012 ✱** |
| | decayed | **+0.009 ✱** | +0.002 ✱ | **+0.011 ✱** |
| Controls | constant | +0.001 | −0.007 ✱ | +0.003 |
| | decayed | +0.001 | −0.006 ✱ | +0.000 |

**Regret** (transformer: 0.087–0.099 at step 0, 0.047–0.055 at steps 1–2):

| Group | Checkpoints | Temporal | Depth J4 | Hybrid J4 |
| --- | --- | --- | --- | --- |
| Finding the tactic | constant | −0.0025 ✱ | +0.0003 | −0.0010 |
| | decayed | −0.0018 ✱ | −0.0003 | +0.0013 |
| Following through | constant | **−0.0027 ✱** | −0.0007 | **−0.0019 ✱** |
| | decayed | **−0.0037 ✱** | −0.0008 | **−0.0032 ✱** |

**Following through:**
- Temporal and hybrid J4 are ahead of the transformer on p_best and on regret at **all seven checkpoints**:
  - temporal p_best +0.006 to +0.020;
  - hybrid J4 p_best +0.0005 to +0.019.
- The effect is small: about 5% relative on p_best and 6% on regret.
- It is absent on controls. So it is specific to tactics, not a general lead.

**Finding the tactic:**
- Temporal is slightly ahead on average, but changes sign across checkpoints on p_best.
- The hybrid is level with the transformer, and below it at the middle checkpoints.
- No arm finds tactics consistently better than the transformer.

**Depth-only** has no advantage on tactics and is slightly worse on controls. Within-position passes help depth and hybrid up to J = 2. Beyond that:
- depth J4 gains +0.005 agreement over J2 at step 0 (final checkpoint), but loses 0.022 at steps 1–2;
- the hybrid is unchanged from J2 to J4.

**Agreement points the same way but is unreliable from one checkpoint to the next.**
- Between neighbouring checkpoints, arm differences move by 0.02–0.03, against bootstrap intervals of about ±0.007.
- At the long run's end, temporal leads by +0.034 on follow-through. At 12,000 the lead is +0.006, and at step 0 it is +0.034 at 12,000 but −0.007 at the end.
- So a single checkpoint's agreement can't rank the arms; averaged over checkpoints:
  - follow-through: temporal +0.012 (constant) and +0.021 (decayed), both ✱;
  - finding: not significant once decayed.

The full tables: `tactics.py compare` over `results/tactics_large/panel_scores_*.npz`, saved in `results/tactics_large/compare.txt`.

## Cost

Four runs shared one A100 80 GB spot instance (Verda, $0.90/h).
- **Median seconds per update, long run:** transformer 1.09, depth 1.89, hybrid 2.10, temporal 2.17. That is 1.7–2× the transformer, matched data, not compute.
- **Wall time:** the transformer's long run took 3.2 hours; the others 6.5–7 hours of training, plus the time lost to the reclaim.
- **The spot instance was reclaimed** at update 13,000 (temporal and hybrid) and 14,500 (depth). The disk was kept; the runs resumed from their recovery checkpoints on a new instance booted from it. The resumed runs read rows in a different order than an unbroken run would. Lost updates were rerun; the logs were trimmed to the checkpoint.
- **Scoring one checkpoint on the panel** takes 21 minutes on the laptop (MPS).

## Limits

- **One seed per arm.** The seven checkpoints come from one run per arm, so they don't measure seed noise. In stage 1's pilot, two hybrid seeds ended 0.036 apart on human games. A second seed per arm, at least for temporal and the transformer, is the next check.
- **Models trained on Leela games are scored on human games,** labelled by Stockfish, not Leela.
- **Follow-through steps follow the engine's best line,** not what the human played. The context therefore holds a strong line, which is the situation temporal recurrence should exploit.
- **Effect sizes are small:** 0.016 KL, and about 5% relative on the tactics panel.
- **Matched data, not compute** (see Cost). Temporal-only has no configuration with the hybrid's block count at J = 4.

## Next

- A second seed per arm, at least temporal and transformer, long schedule.
- The deferred controls in the [plan](../../docs/engine_policy_plan.md): the same rows with the shallow target, and the played-move (imitation) target.
