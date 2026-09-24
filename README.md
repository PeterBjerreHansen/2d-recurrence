# 2D recurrence: training looped and feedback transformers at once

**Looped transformers and temporally recurrent (feedback) transformers are both trained the same way: run the model several times over the whole sequence, and let each pass read state left behind by the previous one. If the training loop is the same, you can train both mechanisms in one model, in one trajectory, for little extra cost.**

This repository tests that idea on a small character-level chess language model: the [ChessGPT](https://github.com/adamkarvonen/train_ChessGPT) backbone, with 8 layers and width 512, trained on PGN text.

![Training-time information flow in four models: vanilla, looped, temporally recurrent, and hybrid](docs/figures/training_time_layer_wise.png)

## The idea in four pictures

Read the figure as a 2×2 grid. The columns switch **depth recurrence** off and on. The rows switch **temporal recurrence** off and on. Each recurrent panel shows two consecutive training passes over the same three positions.

- **Top left, vanilla transformer.** One pass, bottom to top. Position *t* sees earlier positions only through attention.
- **Top right, looped (depth-recurrent) model.** A shared *recurrent core* runs several times. On pass *i*, the core output from pass *i−1* at the **same position** is mixed in before the core (orange **D**). This is the [Huginn](https://arxiv.org/abs/2502.05171)-style looped transformer: more computation per token, no new core parameters.
- **Bottom left, temporally recurrent model.** A late-layer state from pass *i−1* at the **previous position** *t−1* is mixed in early on pass *i* (purple **T**). A high-level state flows forward in time, as in feedback transformers.
- **Bottom right, hybrid.** Both reads at once. Depth state comes from the top of the recurrent core, and temporal state from a separate *T-source* layer above it. Temporal state is injected below a *T-buffer* layer, depth state above it.

The two recurrent models differ in only two ways: **where** the state is read, and whether it is **shifted by one position**. Both are trained by unrolling passes over the whole sequence in parallel, so one training loop serves both.

In this repository, the temporal-only and depth-only models are restrictions of the hybrid. They keep the same eight-layer layout, including the T-source layer. The bottom-left panel shows the generic feedback idea.

## One sampler trains both

For every microbatch, training draws a pair of update counts `(U_T, U_D)`, for example `(3, 1)`. It then runs `max(U_T, U_D) + 1` passes and writes each state after a random subset of the non-final passes:

```text
pass:              1   2   3   4
temporal write:    ✓   ✓   ✓   –
depth write:       –   ✓   –   –
```

Every pass reads whatever state exists; a state that isn't rewritten is simply held. Only the final pass is supervised, and gradients flow through the whole trajectory. The model is never told how many passes it will get.

So `(0,0)` is an ordinary transformer, `(U,0)` trains the temporal axis, `(0,U)` trains the depth axis, and mixed pairs train the two together. A single checkpoint can be evaluated anywhere on the `(U_T, U_D)` surface.

**How cheap is it?** In the 5B-character study, with the same distribution of pass counts, a hybrid update took about as long as a temporal-only update (1.00 s vs. 1.03 s) and about 11% longer than a depth-only update (0.90 s). Each arm ran on its own RTX 3090 pod. Most of the cost is the extra passes themselves, which both single-axis models already pay.

## At inference, the two axes separate again

![Inference-time information flow in the same four models](docs/figures/inference_time_layer_wise.png)

Parallel multi-pass training is only a way to *train* recurrence. When generating, the model runs token by token:

- **Depth** loops the recurrent core `d` times *within* the current token. The depth state is discarded at the next token.
- **Temporal** state is written once per token by the T-source and read by the *next* token, across the whole sequence.

During training, `U` passes chain the temporal state only `U` positions back. At inference the chain runs through every earlier token. The repository implements both executions and reports them separately. It does not assume they agree.

## What has been found so far

| Scale | Comparison | Result |
| --- | --- | --- |
| 1B characters | Hybrid checkpoint vs. transformer | The `(3,3)` hybrid path beat the transformer by about 0.003–0.004 NLL. Temporal-only and depth-only execution of the same checkpoint each helped. |
| 5B characters | Separately trained temporal, depth and hybrid models | Depth trailed by about 0.002 NLL. Temporal led the hybrid early on; the gap shrank to about 0.0006 by the end. |
| 20B characters | Four arms, curriculum toward four passes | Prepared, not yet launched. See the [20B study](experiments/long_runs/20B_recurrence/README.md). |

These are single-seed results. Protocols and caveats are in the [experiment index](experiments/README.md).

## Quick start

```sh
uv sync --frozen --python 3.11
uv run pytest -q
uv run python data/chess_v1/prepare.py --file lichess_100mb_blocks.zip --out-dir data/chess_143K_v1
uv run python train.py configs/local/recurrent_mps.py
```

This runs a small local check on Apple MPS. The [usage guide](docs/usage.md) covers the full-corpus setup, resume rules, evaluation and generation.

## Where to go next

| If you want to… | Read |
| --- | --- |
| Understand the model step by step | [Concepts](docs/concepts.md) |
| See the exact training equations, initialization and masking | [Recurrence contract](docs/RECURRENCE_CONTRACT.md) |
| Understand token-by-token generation and KV caches | [Inference contract](docs/INFERENCE_CONTRACT.md) |
| Train, resume, evaluate or generate | [Usage guide](docs/usage.md) |
| Find experiments, protocols and results | [Experiment index](experiments/README.md) |
| See what is done and what comes next | [Implementation plan and status](docs/implementation_plan.md) |
| Read the original research proposal | [Proposal](docs/proposal.md) (historical) |

## Background

- Looped / recurrent-depth transformers: [Huginn](https://arxiv.org/abs/2502.05171) ([code](https://github.com/seal-rg/recurrent-pretraining)), [Ouro](https://arxiv.org/abs/2510.25741)
- Temporal feedback: [Full-Bandwidth Transformer](https://arxiv.org/abs/2608.08888), [multipass-transformer-training](https://github.com/PeterBjerreHansen/multipass-transformer-training), [multipass-transformer-memory](https://github.com/PeterBjerreHansen/multipass-transformer-memory)
- Chess language modeling: [train_ChessGPT](https://github.com/adamkarvonen/train_ChessGPT), [chess world models](https://adamkarvonen.github.io/machine_learning/2024/01/03/chess-world-models.html)
