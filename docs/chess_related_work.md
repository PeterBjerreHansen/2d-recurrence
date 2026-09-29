# Chess language models and chess transformers: related work

Surveyed 2026-09-29 for the [move-target plan](engine_policy_plan.md). A targeted survey, not an exhaustive one. Summaries come from each paper's abstract or main text; nothing was reproduced.

## Where our work sits

| | Input | Output | Architectures |
| --- | --- | --- | --- |
| Chess language models (Toshniwal, Karvonen, Transcendence) | move text | next token | transformer |
| State-tracking benchmarks (Chess-World-Model) | move tokens | board | transformer vs linear RNNs |
| Searchless chess (Ruoss, Chessformer, Maia) | board | move / values | transformer or CNN |
| Human-move models from history (Allie) | move tokens | move, time, value | transformer |
| **This plan** | **move text** | **move distribution: legal, then engine-weighted** | **transformer vs temporal, depth and joint recurrence** |

We found no work that compares recurrence **across tokens** with recurrence **within a token** on chess, or that trains a model to read game text and output a policy distilled from an engine.

## 1. Language models on chess text: state tracking and world models

| Paper | What it shows | Relevance |
| --- | --- | --- |
| [Toshniwal et al., 2022](https://arxiv.org/abs/2102.13249) (AAAI), *Chess as a Testbed for Language Model State Tracking* | Transformers trained on UCI move sequences learn to track pieces and predict legal moves. With little data, giving board information during training helps significantly. Approximating full attention hurts. | Earliest chess state-tracking testbed. Stronger supervision helping small-data training is the precedent for our "faster board learning" hypothesis. |
| [Li et al., 2023](https://arxiv.org/abs/2210.13382) (ICLR), Othello-GPT | Trained on synthetic games of uniformly random legal moves, the model learns a board representation that nonlinear probes recover. | Next-token prediction on uniformly random legal games has the uniform legal distribution as its expected target. Our stage 1 is the exact, noise-free version of that target, on human games. |
| [Nanda et al., 2023](https://arxiv.org/abs/2309.00941) | The Othello board is linearly decodable in mine/theirs coordinates. | The origin of the relative board encoding our probes use. |
| [Karvonen, 2024](https://arxiv.org/abs/2403.15498) | Character-level GPTs on Lichess PGN learn a linearly decodable board and player skill; interventions on both change play. | Our backbone, data and probe baseline. |
| [Karvonen et al., 2024](https://arxiv.org/abs/2408.00113) | Uses chess and Othello board features as ground truth for evaluating sparse autoencoders. | Tools if the mechanism work resumes. |
| [HaileyStorm, 2024](https://huggingface.co/HaileyStorm/chess-mamba-vs-xformer/blob/main/Report/REPORT.md) (informal report) | On Karvonen's setup, Mamba beats a transformer on win rate and illegal moves. | Early, uncontrolled evidence that recurrence across tokens helps on chess text. |
| [Walker & Lyons, 2026](https://arxiv.org/abs/2605.30100), *Chess-World-Model* | Predict the exact board after each move, from single-token UCI moves: 10M Lichess games, game-disjoint splits. Linear RNNs beat the transformer at 3–8M parameters. In-distribution accuracy saturates (≈99.9%) but uniformly random legal games stay discriminative (transformer 73%, Mamba-3 81%). | **The closest comparison.** Recurrent vs transformer on chess state tracking, but with a board target and linear RNNs, with no loops within a token and no decision target. Its random-legal-game test is adopted in our plan. |
| [Harang et al., 2025](https://arxiv.org/abs/2508.19851) (ICML workshop) | Evaluates state tracking by comparing legal-move distributions, the positions' "affordances". | Similar in spirit to our legal-move target as a measure of state tracking. |
| [Zhang et al., 2024](https://arxiv.org/abs/2406.11741) (NeurIPS), *Transcendence* | A 50M transformer trained on PGN from players up to 1,000 or 1,300 Elo plays at about 1,500 with low-temperature sampling, by averaging out individual players' errors. | A caution for the next-character reference: its move quality depends on the sampling temperature. |

## 2. Theory: state tracking and recurrence

| Paper | What it shows | Relevance |
| --- | --- | --- |
| [Merrill et al., 2024](https://arxiv.org/abs/2404.08819) (ICML), *The Illusion of State* | Transformers and linear state-space models are both limited to TC⁰. Chess in source–target notation is among the state-tracking problems they cannot solve exactly at fixed depth. | Temporal feedback in our models passes a nonlinear state from each character to the next, so it is not bound by this limit in principle. Our input is SAN, but our move head outputs source–target moves. |
| [Grazzi et al., 2025](https://arxiv.org/abs/2411.12537) (ICLR) | Linear RNNs need negative eigenvalues in their state transitions to track state; with them they can recognise any regular language. | Explains Chess-World-Model's ablations. |

The recurrence-architecture precedents (Huginn, Ouro, T²MLR, Latent Recurrent Transformer, the full-bandwidth transformer) are reviewed separately in the paper-positioning notes.

## 3. Board-input transformers, searchless play and engine distillation

| Paper | What it shows | Relevance |
| --- | --- | --- |
| [Ruoss et al., 2024](https://arxiv.org/abs/2402.04494) (NeurIPS), *Amortized Planning with Large-Scale Transformers* (ChessBench) | Decoder transformers (9M–270M) read a fixed-length FEN and are trained on Stockfish 16 labels for 10M Lichess games (February 2023): 15.3B action values. Predicting the value of every move works best (2,895 Lichess blitz Elo against humans); state values are weaker, and imitating the best move is weakest. The missing history means the models can't see threefold repetition and wander between winning plans. | **The main precedent for stage 2.** We use the move history instead of the board; that is exactly the information they lack. Their moves come from a list of 1,968 from/to moves, which we adopt so their labels map directly. Our soft engine target lies between their best-move imitation and full action values. |
| [ChessBench data](https://github.com/google-deepmind/searchless_chess) | The action-value training set is 1.1 TB; the best-move and state-value sets are 34–36 GB. Records are keyed by position; game IDs and move histories are not kept. | To use its labels we would need the February 2023 Lichess games: replay them and look up each position. That is feasible but heavy; see the plan. |
| [Ye et al., 2025](https://arxiv.org/abs/2502.19805) (ICLR), DiffuSearch | A discrete diffusion model that imagines future positions beats both a one-step policy and a policy with explicit tree search (MCTS) on action accuracy. | Evidence that computing future states inside the model helps chess decisions: a rival mechanism for "extra computation" to depth recurrence. |
| [Monroe et al., 2026](https://arxiv.org/abs/2605.19091), *Chessformer* | An encoder over square tokens with a geometric attention bias and a from/to policy head. It sets the state of the art in human-move prediction (Maia3) and strengthens Leela by over 100 Elo. | The current architecture of Leela and Maia; its policy head is a from/to design like ours. |
| [Tang et al., 2024](https://arxiv.org/abs/2409.20553) (NeurIPS), Maia-2 | Human-move prediction across skill levels with skill-aware attention. | Background on human-move targets. |
| [Zhang et al., 2025](https://arxiv.org/abs/2410.03893) (ICLR), Allie | A 355M decoder transformer on 91M Lichess blitz games as move tokens, with a policy head, a thinking-time head and a value head. Adding a little search scaled to predicted thinking time plays close to human strength across 1,000–2,600 Elo. | The closest design to ours on the input side: game history in, policy head out. But its targets are human moves, its input is move tokens rather than text, and it is a plain transformer. |
| [Miłosz et al., 2026](https://arxiv.org/abs/2608.27757) | Self-play reinforcement learning fine-tune of Leela's search-free network. Puzzle accuracy and playing strength move apart; training on puzzles alone costs about 260 Elo. | A caution for evaluation: tactical accuracy is not playing strength. |

## 4. Mechanisms of look-ahead

All four study Leela, a transformer over board squares.

| Paper | What it shows | Relevance |
| --- | --- | --- |
| [Jenner et al., 2024](https://arxiv.org/abs/2406.00877) (NeurIPS) | Leela represents future optimal moves internally. A probe predicts the best move two turns ahead with 92% accuracy, and attention heads move information between future-move squares. | The reference for the forcing-line and imagined-board questions. |
| [Cruz, 2025](https://arxiv.org/abs/2505.21552) | Extends the analysis to look-ahead of up to seven moves, over several lines at once. | Suggests looking for more than one line. |
| [Sandmann et al., 2025](https://arxiv.org/abs/2508.21380) | Correct solutions appear in intermediate layers but are overridden by a learned preference for safe play in late layers. Steering recovers 62% of these. | Readable internal computation does not guarantee behaviour; supports evaluating output quality, not just probes. |
| [Lin et al., 2026](https://arxiv.org/abs/2604.10158) | Sparse replacement layers for both the MLPs and attention in Leela reveal tactical features and parallel reasoning. | Method reference. |

## 5. General-purpose language models playing chess

| Paper | What it shows | Relevance |
| --- | --- | --- |
| [Dionisopoulos et al., 2026](https://arxiv.org/abs/2604.05134) (ICML) | A 7B language model fine-tuned then trained with reinforcement learning on chess. Training directly on moves gives stronger play but unfaithful reasoning. | Different scale and question; cite for context only. |
| [Fauber, 2024](https://arxiv.org/abs/2410.02426) | Small instruction-tuned language models learn chess rules and legal moves from examples. | Context only. |

## What this changes in the plan

- **Adopt ChessBench's 1,968-move vocabulary** (queen promotions explicit), so its labels map directly.
- **Add uniformly random legal games as an out-of-distribution test.** Following Chess-World-Model, this separates arms after in-distribution metrics saturate.
- **Position stage 2 against Ruoss et al.** directly. History input fixes their repetition blindness, and it makes the model reconstruct the board itself.
- **Use Allie as the reference design** for a history-in, policy-out model.
- **Use Chess-World-Model and Merrill et al.** for the claim that recurrence across tokens should help chess state tracking. Both concern recurrence across tokens; neither covers loops within a token or a decision target.
