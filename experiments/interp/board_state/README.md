# Board-state probing

Where, and how early, do the four 20B arms represent the board and the side to move? How does the temporal memory carry the board, and do the models use it? This experiment replicates the linear probes of Karvonen (2024, *Emergent World Models and Latent Variable Estimation in Chess-Playing Language Models*) at every residual-stream site, and tests the probe directions causally. Results are in the [report](REPORT.md).

## Arms and execution

Every arm runs at the **training-graph fixed point**, not live token by token:

1. The prelude runs once.
2. Temporal memory is settled with gradient-free passes until no position's memory changes by more than 0.1% (relative L2).
3. One final pass reads the settled memory.

Settled memory equals live memory (see `experiments/ablations/live_warm_start/align.py` and its tests), so this is the deployed computation.

| Arm | Checkpoint | Core iterations | Sites |
| --- | --- | --- | --- |
| `transformer` | `transformer_20B` step 195,504 | 1 | `emb`, `L1`–`L8` |
| `temporal` | `live_warm_start/results/temporal_aligned.pt` | 1 | adds `Tmix`, the temporal mixer output, after `L1` |
| `depth` | `depth_20B` step 195,504 | 4 | `L3@i`–`L6@i` per iteration; `Dmix@i` before iterations 2–4 |
| `hybrid` | `hybrid_20B` step 195,504 | 4 | `Tmix` and the looped core |
| `karvonen` | `adamkarvonen/chess_llms`, `lichess_8layers_ckpt_no_optimizer.pt` | 1 | same architecture and data, 61.4B training characters |
| `random_init` | `transformer_20B` step 0 | 1 | control for what a probe reads from untrained features |
| `temporal_decay` | `live_warm_start/results/aligned_decay_temporal/ckpt-step195504.pt` | 1 | temporal model whose decay phase trained the whole network with 25% of batches on settled memory |

The temporal arm is the live-aligned export, because the as-trained reader misreads settled memory (see the [20B report](../../long_runs/20B_recurrence/REPORT.md)). Each site records the transformer blocks applied so far, so looped and single-pass arms can be compared by compute as well as by layer.

The site executor is `interp/sites.py`. `tests/test_interp.py` checks that it reproduces each model's own forward pass, and the settled fixed point of the alignment code. Fixed-point NLL on the probe rows matches the 20B report to within about 0.001 per arm. `capture.py` records it in each arm's `run.json`.

## Data and labels

- **Rows:** 1,000 validation rows of `chess_8M_v1`, chosen with seed 0, from the upstream 1% split that Karvonen's model did not train on either. The first 800 train probes and the last 200 test them.
- **Decision points** (`interp/boards.py`), where a move is about to be written:
  - `dot`: the `.` of a white move number, where White decides. This is Karvonen's probe position.
  - `space_black`: the space after White's move, where Black decides.
  - `space_white`: the space after Black's move. A third of these is kept, for the side-to-move probe.
- **The board is represented relative to the side to move** (Karvonen's mine/theirs). Every probe either uses the relative encoding, or fits one probe per side to move.

## Analyses

| Script | Question | Output under `results/` |
| --- | --- | --- |
| `capture.py` | Residual stream at every site, at every decision point, for every arm | `labels.npz`, `activations/<arm>/` |
| `probe.py` | Board (`board`, whole rows; `board_karvonen`, first 365 characters of each first game) and side-to-move probes at every site | `probes/<arm>/` |
| `figures.py` | Layer curves of the board probes | `report_figures/probe_curves.png` (tracked) |
| `cycle.py` | The board at every character of the move cycle: in the memory, after the mixer, and at L2, L5 and L7; mixer gates per character role | `cycle/<arm>.json` |
| `mixer.py` | How much of the board survives the temporal mixer; the size of each mixer input's term; the depth mixer's previous-iteration versus fresh-start terms | `mixer/<arm>.json` |
| `mixer_terms.py` | Is the board lost in the mixer, or only hard to read? Probes on the memory term and current-character term separately, linear and MLP | `mixer_terms/<arm>.json` |
| `memory_scale.py` | What if the mixer passed more memory? NLL and board readouts with the memory term scaled | `memory_scale/<arm>.json` |
| `gates.py` | What the mixer's gating selects: gates fixed at their means; board and current character in the memory term; gates for memories of 1–3 updates versus settled | `gates/<test>-<arm>.json` |
| `attention.py` | Does a character get the board by attending to the previous one? Attention weights, and the board with that attention blocked | `attention/<arm>.json` |
| `update.py` | Where the latest move is applied: probes for the board before and after it | `update/<arm>.json` |
| `decision.py` | Where the move choice becomes readable: from- and to-square probes, and the logit lens | `decision/<arm>.json` |
| `swap.py` | Does the model use the board in its memory? Swap in another game's memory over the last *k* characters | `swap/<arm>.json` |
| `steer.py` | Are the probe directions causal? Flip side to move; delete a piece (Karvonen's intervention) | `steer/<test>-<arm>.json` |

**Probe training.** Every board probe gets at least 4,000 optimizer steps: about 4,900 for `board`, 6,300 for `board_karvonen`, and at least 4,000 per role in `cycle.py`. At about 1,650 steps, probes were still about 0.02 short of convergence.

**Steering edits.** Each edit adds `k · c · d` at the probed character and at every character generated after it, on every settling pass:

- `d` is the probe direction from the source class to the target class.
- `c` sets the probe's logit margin to 5 at that site (Karvonen's dynamic scale).
- `k` is a multiplier.

Every condition is paired with a random direction of the same norm.

## Commands

From the repository root, in order:

```bash
uv run python -m experiments.interp.board_state.capture
uv run python -m experiments.interp.board_state.probe
uv run --with matplotlib python -m experiments.interp.board_state.figures
uv run python -m experiments.interp.board_state.cycle
uv run python -m experiments.interp.board_state.mixer
uv run python -m experiments.interp.board_state.mixer_terms
uv run python -m experiments.interp.board_state.memory_scale
uv run python -m experiments.interp.board_state.attention
uv run python -m experiments.interp.board_state.gates constant --arms temporal temporal_decay
uv run python -m experiments.interp.board_state.update
uv run python -m experiments.interp.board_state.decision
uv run python -m experiments.interp.board_state.swap --arm temporal --pairs 200
uv run python -m experiments.interp.board_state.steer board --arm temporal --groups L6 L4+L5+L6+L7 --multipliers 2 4
```

The report's steering conditions are listed in its causal-checks section. Outputs go to the ignored `results/` directory. Activations take about 17 GB across the six arms. The recurrent arms settle for 11–41 passes per forward, so the hybrid is the slowest arm throughout.
