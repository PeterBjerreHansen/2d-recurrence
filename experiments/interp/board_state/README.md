# Board-state probing

Where, and how early, do the four 20B arms represent the board and the side to move? This experiment replicates the basic probes of Karvonen (2024, *Emergent World Models and Latent Variable Estimation in Chess-Playing Language Models*) on every residual-stream site of each arm, and checks them causally by editing the residual stream along probe directions.

## Arms and execution

Every arm runs at the **training-graph fixed point**, not live token by token. The prelude runs once. Temporal memory is settled with gradient-free passes until no position's memory changes by more than 0.1% (relative L2). One final pass then reads the settled memory. Settled memory equals live memory (see `experiments/ablations/live_warm_start/align.py` and its tests), so this is the deployed computation.

| Arm | Checkpoint | Core iterations | Extra sites |
| --- | --- | --- | --- |
| `transformer` | `transformer_20B` step 195,504 | 1 | none: `emb`, `L1`–`L8` |
| `temporal` | `live_warm_start/results/temporal_aligned.pt` | 1 | `Tmix` after `L1` |
| `depth` | `depth_20B` step 195,504 | 4 | `L3@i`–`L6@i` per iteration, `Dmix@i` before iterations 2–4 |
| `hybrid` | `hybrid_20B` step 195,504 | 4 | `Tmix` and the looped core |
| `karvonen` | `adamkarvonen/chess_llms` `lichess_8layers_ckpt_no_optimizer.pt` | 1 | none; same architecture and data, 61.4B characters |
| `random_init` | `transformer_20B` step 0 | 1 | none; control for what a probe reads from untrained features |

The temporal arm uses the live-aligned export, because the as-trained reader misreads settled memory (live NLL 0.48, see the [20B report](../../long_runs/20B_recurrence/REPORT.md)). Every site records the number of transformer blocks applied so far (`applications`), so looped and single-pass arms can be compared by compute as well as by layer.

The site executor is `interp/sites.py`. Tests in `tests/test_interp.py` check that it reproduces the model's own forward pass for the transformer and depth arms, and the alignment code's settled fixed point for the temporal and hybrid arms. Fixed-point NLL on the probe rows matches the 20B report to within about 0.001 per arm; `capture.py` records it in each arm's `run.json`.

## Probe points and labels

`interp/boards.py` replays each row with python-chess and labels three kinds of input character:

- `dot`: the `.` of a white move number. White is to move. This is Karvonen's probe position.
- `space_black`: the space after white's move. Black is to move, and the next character starts black's move.
- `space_white`: the space after black's move. White is to move, and the next characters are a move number. A fixed third of these is kept.

The two space kinds are the same input token, so side to move can't be read from the current character.

Rows are 1,000 validation rows of `chess_8M_v1`, chosen with seed 0. They come from the upstream 1% split, which Karvonen's model also did not train on. The first 800 rows train the probes and the last 200 test them.

## Probes

`probe.py` fits linear probes (`interp/probes.py`) at every site:

- `board`: 64 squares × 13 classes, side to move first ("mine"/"theirs"), on both pre-move kinds across whole rows. One probe has to read both colours' boards at any point in a game.
- `board_karvonen`: Karvonen's setting. It uses `dot` points of each row's first game, within its first 365 characters, because his probe games are truncated to 365 characters. With the same pipeline, his model scores 0.980 at L6; his paper reports 0.991 with about ten times more probe games. The random-init model scores 0.754 (paper: 0.750).
- `turn`: side to move on the two space kinds, class-balanced.

Board accuracy is reported over all squares, which is Karvonen's metric, and over *changed* squares, whose class differs from the starting position. The per-square majority class is the trivial baseline.

## Causal checks

`steer.py` edits the residual stream at the probed character and every character generated after it. Recurrent arms apply the edit on every settling pass. The edit adds `k · c · d`:

- `d` is the probe direction from the source class to the target class.
- `c` sets the probe's target-minus-source logit margin to 5 at that site. This is Karvonen's dynamic-scale rule.
- `k` is a multiplier.

Every condition is compared with a random direction of the same norm.

- `turn`: at held-out spaces, push the side-to-move probe to the other player. The model should switch between predicting a move number (digit) and a move (letter).
- `board`: Karvonen's intervention. Decode the model's greedy move, delete the moved piece (its square goes from "mine" piece to empty in the `board` probe), then decode again. Success means the new move is legal on the modified board, although the PGN text is unchanged. Examples come from the first game of held-out rows at plies 10–60. Only positions where deleting the piece makes the original move illegal are used, so the unedited success rate is 0.

## Commands

From the repository root:

```bash
uv run python -m experiments.interp.board_state.capture
uv run python -m experiments.interp.board_state.probe
uv run python -m experiments.interp.board_state.steer turn --arm transformer
uv run python -m experiments.interp.board_state.steer board --arm transformer --groups L6 L4+L5+L6+L7 --multipliers 1 2 4
uv run --with matplotlib python -m experiments.interp.board_state.figures
```

Outputs go to the ignored `results/` directory. Activations take about 17 GB across the six arms.
