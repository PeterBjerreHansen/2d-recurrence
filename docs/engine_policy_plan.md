# Engine-policy objective: plan

> **Status: proposal.** Nothing here has been run. The MVP is a fine-tune of the existing 20B checkpoints; the later sections describe the full version and what it would show.

## Why

The 20B study left one question open: **does chess need the depth loop on top of temporal feedback?**

- In the training graph, the hybrid ties temporal-only (0.2136 each), and depth-only trails by 0.0024.
- At the transformer's block count, aligned temporal is the best deployed model. The hybrid wins only when it runs four core iterations per token, and then by 0.0012 over depth-only.
- The gains are in **move choice**, and they grow over the game ([20B report](../experiments/long_runs/20B_recurrence/REPORT.md#where-the-gains-come-from)).

A plausible reason the depth axis adds so little is the objective, not the architecture:

- **In chess, extra computation per move goes into calculating lines,** that is, into choosing good moves.
- **Next-token prediction on Lichess games rewards imitating the player,** and the training data has no rating filter. The targets contain only as much calculation as the players did, so lookahead has little to gain.
- **Temporal feedback has a clear job under either objective:** the model reads PGN from move 1 and must track the board.

An objective derived from engine evaluations keeps the input and the board-tracking problem unchanged, but makes **move quality** the target. That gives the task we want: state tracking (temporal) plus calculation at the moment of the move (depth), inside chess itself.

It also needs no RL loop. The target below is the closed-form solution of one step of KL-regularised policy improvement, computed offline on the same games the models were pretrained on.

## The objective

### Move-level target

For a position $s$, Stockfish gives each legal move $a$ a win probability $Q(s,a)$ for the side to move:

$$
Q = \tfrac12 + \tfrac12\left(\frac{2}{1 + e^{-0.00368208\,\mathrm{cp}}} - 1\right).
$$

This is the Lichess conversion from centipawns. Forced mates map to 1 or 0. The target distribution over moves is

$$
\pi^*(a \mid s) \propto \exp\!\big(A(s,a)/\tau\big), \qquad A(s,a) = Q(s,a) - \max_{a'} Q(s,a').
$$

The temperature $\tau$ sets how far mass spreads over near-best moves. With $\tau = 0.02$, a move that loses 5% win probability gets about 8% of the best move's weight.

### Character-level decomposition

The model predicts characters, not moves. The legal moves' SAN strings, each followed by the character that ends the move, form a prefix tree (trie). A node $u$ carries the mass of every move below it, $m(u) = \sum_{a \,:\, u \preceq \mathrm{SAN}(a)} \pi^*(a)$. At each character of the **played** move, the target for the next character $c$ is the exact conditional

$$
P^*(c \mid u) = \frac{m(uc)}{m(u)}.
$$

For example:
- two good knight moves put their combined mass on a first character of `N`;
- if the player (a weak one, say) actually played a bishop move, the target for the second character is spread over the bishop moves only, re-weighted by their values;
- the same continues through the whole move, including the end character that separates `O-O` from `O-O-O`.

The loss is cross-entropy against these soft targets on move characters. Move numbers and delimiters are deterministic and keep ordinary next-token targets.

**Properties:**
- **Every conditional target is exact.** The played move decides only which trie nodes get trained, not what their targets are, so the best the model can do is match $\pi^*$ at every node it visits.
- **Illegal continuations get zero target mass,** so the objective also trains legality.
- **The input is unchanged:** the game continues with the moves actually played, so temporal state and live execution see the same distribution as in pretraining.

## MVP

**Question:** after the same engine-policy fine-tune, does the hybrid choose better moves than temporal, and does the gain concentrate where calculation matters?

### Data

- **Positions:** rows from `chess_8M_v1` train, the games the 20B arms were pretrained on. The target is about 2M annotated positions, roughly 12k rows, with every move in a chosen row annotated.
- **Held-out evaluation set:** about 20k positions from the confirmation rows.
- **Annotation:** a pinned Stockfish version, MultiPV over all legal moves, and a fixed node budget so labels are deterministic. The budget is set by a local benchmark.
- **Storage:** per position, the legal moves and their $Q$ values. $\pi^*$ is computed at training time, so $\tau$ can change without re-annotating.

### Training

- **Arms:** the 20B checkpoints.
  - **Primary:** aligned temporal and the hybrid.
  - **Secondary:** depth-only and the transformer.
- **Identical fine-tune for every arm:** same rows, order, update count and learning-rate schedule. Each recurrent arm uses its own 20B `(U_T, U_D)` sampler.
- **Target:** the plain temperature form of $\pi^*$, the same for every arm. One or two values of $\tau$.
- **Live alignment:** training-graph fine-tuning may undo temporal's live alignment. Live NLL is checked after the fine-tune, and the settled-memory alignment is repeated if needed.

### Evaluation

Live execution. The hybrid is evaluated at J = 1, 2, 4 and 8.

- **Expected regret:** $\sum_a p_\theta(a \mid s)\,\big(\max_{a'} Q(s,a') - Q(s,a)\big)$, the win probability the model's own move distribution gives up. $p_\theta(a \mid s)$ comes from the trie branching in `evaluation/live_legality.py`, extended from legal mass to per-move probabilities.
- **Best-move agreement** and **KL to $\pi^*$.**
- **Strata:**
  - how forcing the position is, measured by the gap between the best and second-best move ("only-move" positions need calculation);
  - ply;
  - whether the engine's best move is a capture, a check, or quiet.
- **Axes:** every metric is plotted against positions seen, and FLOPs per token are reported alongside. Hybrid J=1 costs the same as temporal, so it is the matched-compute comparison.
- **Zero-training baseline:** the same metrics on the unmodified 20B checkpoints. This already shows whether extra loops help on forcing positions before any fine-tuning.

### Decision rule

Fixed before the fine-tune:

- **Depth is needed** if hybrid J=4 has lower expected regret than aligned temporal on forcing positions, with a paired 95% interval excluding zero, **and** the gain from J=1 to J=4 is larger on forcing positions than on quiet ones.
- **Depth is not needed in chess** if the hybrid is within noise of temporal on forcing positions at every J.
- **The main open question is still the depth loop** if depth-only beats temporal but the hybrid does not beat temporal. The hybrid would then be failing to combine the two axes.

### Steps

1. **Benchmark:** Stockfish MultiPV throughput on a few hundred positions. Choose the node budget and estimate annotation cost.
2. **Annotation pipeline:** SAN trie targets, with tests that every node's targets sum to one, match the move-level $\pi^*$ when chained, and handle castling, promotions, disambiguation and check suffixes.
3. **Evaluation:** annotate the held-out set, extend the live evaluation to per-move probabilities, and run the zero-training baseline.
4. **Fine-tune:** annotate the training positions (on a rented CPU machine if the benchmark says so), then fine-tune the four arms on an A100 or 4090.
5. **Report:** evaluate against the decision rule and write the report next to the scripts.

**Rough cost.**
- **Annotation:** on the order of 100 core-hours per 1M positions, pending the benchmark. For the MVP that's about a day on a rented 64-core machine, tens of dollars.
- **Fine-tuning:** a few GPU-hours per arm.

## The best version

If the MVP shows a depth signal, the full experiment would change the following.

- **Keep the moves players consider.** Tilt a reference policy instead of replacing it:

  $$
  \pi^*(a \mid s) \propto p_{\text{ref}}(a \mid s)\,\exp\!\big(A(s,a)/\tau\big).
  $$

  - This keeps the moves humans consider and shifts mass towards good ones: the move a strong player would make. That fits the view that depth computation goes into probable moves.
  - $p_{\text{ref}}$ is one fixed model, the same for every arm, so the targets stay identical.
- **Cover the engine's line too.** Add a second teacher-forced path along the engine's best move, as a side branch sharing the prefix. Subtrees that players rarely enter then get trained where $\pi^*$ puts its mass.
- **A value head from the same annotations:** binned win probability at every move boundary. The extra supervision is free and helps separate "can evaluate" from "can choose".
- **From scratch, not fine-tuned.** Pretrain the four arms with next-token prediction mixed with the engine-policy objective, under the 20B protocol with a gap-free update support. Three seeds for the transformer, temporal and hybrid.
- **Harder evaluations:**
  - Lichess puzzles with their game histories, split by solution length and rating;
  - playing strength against fixed Stockfish levels;
  - hybrid J beyond the trained maximum.
- **A complementary control:** the same objective with FEN input. With the board given, temporal feedback has nothing to track, so its benefit should vanish while depth's remains.

### The optimistic case

**What the paper could say.** Training both recurrence axes costs almost nothing extra (0.2% FLOPs to add depth to a temporal model). Each axis does a different job in the same model:
- **temporal feedback** carries the board state across the game;
- **the depth loop** turns extra inference compute into better moves exactly where calculation is required.

**The figure.** The hybrid's expected regret falls as J grows, with a steeper fall on forcing positions and puzzles with longer solutions. Temporal-only has no such control.

**Further gains:**
- **Data efficiency:** the hybrid might reach temporal's regret with fewer annotated positions, measured per position, which is where the real cost lies (engine calls).
- **A general recipe:** the character-level trie decomposition trains any character- or token-level language model from action values, without an RL loop and without changing the input format.
