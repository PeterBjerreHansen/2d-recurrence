# Live inference contract

Stage 14 adds a sequential, fixed-depth inference path beside the existing
parallel training graph. `Recurrent2DGPT.forward()` remains the training
interface and always receives an explicit `RecurrenceSchedule`. Live
inference is exposed by `inference.live` and is intentionally mode-aware.

## Execution modes

| Checkpoint mode | Temporal feedback | Core depth per token | Required live policy |
| --- | --- | ---: | --- |
| baseline | no | 1 | `ordinary` |
| `temporal` | yes | exactly 1 | `final_depth` (or `ordinary`) |
| `depth` | no | fixed `J >= 1` | `final_depth` or `depth_specialized` |
| `hybrid` | yes | fixed `J >= 1` | `final_depth` or `depth_specialized` |

The depth budget is fixed for the whole session. There is no adaptive exit,
iteration-index input, state-age input, or convergence loss. Depth state is
absent at the start of every physical token; it is not carried between
tokens. A temporal state, when enabled, is carried from the preceding token.

## One-token execution

For token `x_t`, the live path performs the following operations:

1. Embed `x_t` at its absolute physical position `t`.
2. Run the prelude blocks once. If temporal memory `m_{t-1}` exists, mix it
   with the current prelude representation. If it does not exist, bypass the
   temporal mixer and use the prelude representation directly.
3. Run the buffer once and hold the resulting anchor fixed.
4. Run the shared core `J` times. The first pass uses the anchor directly;
   later passes mix the preceding depth output with the same anchor.
5. Run the temporal-source segment once. Its raw output is the outgoing
   temporal memory `m_t` when temporal feedback is enabled.
6. Feed that same source output through the coda, final normalization, and
   language-model head to produce logits for `x_{t+1}`.

The source and coda are outside the depth loop. A zero-valued temporal memory
is still a valid memory; only an absent memory bypasses the mixer. A layout
with no source blocks uses the final core output as both source and temporal
candidate.

## Prompt prefill and positions

Prompt tokens are prefixed through the same one-token path, in order, starting
at position zero. The final prompt token produces the first continuation
logits. The sampled continuation token is then consumed through the live path
before the next sample is drawn. This means the outgoing temporal memory and
all committed KV histories always correspond to the most recently consumed
token.

Every transformer block uses the absolute physical position supplied by
`GPT.embed_step`. Attention history is causal and consists of the committed
physical-token entries for that block plus the current token. Live sessions
stop at the model context limit and cannot be resumed with a different model
layout, recurrence mode, depth budget, or cache policy.

## KV cache policies

The cache has independent physical streams for the baseline blocks, prelude,
buffer, source, and coda. Core streams follow one of two policies:

* `final_depth`: one physical core history is shared by the token stream. The
  current token's intermediate depth K/V entries are candidates for attention
  but are ephemeral. Only the final core depth commits K/V for the next token.
* `depth_specialized`: every recurrent depth has its own physical core
  history. Each depth commits its current token K/V to its own stream.

`ordinary` is the baseline policy and is also accepted for the one-step
temporal-only reduction. It is not accepted for depth-recurrent or hybrid
sessions. The cache records mode, strategy, depth budget, and a model-layout
signature; mismatched state reuse raises an error.

## Correctness references

`inference.reference` is a slow live oracle. It keeps historical block inputs
and reruns each block on the full historical prefix plus the current input,
without using incremental attention K/V. Cached live logits must match this
oracle up to backend floating-point tolerances for baseline, temporal, depth,
and hybrid modes.

The slow live oracle is distinct from the parallel training graph. The latter
uses a supplied write schedule over full sequences; differences between the
two executions are expected and are useful evaluation results, not cache
correctness failures.

## CLI and reports

Use `sample.py --execution live` for checkpoint generation. Recurrent live
checkpoints require `--depth-steps` for depth or hybrid modes; optionally pass
`--kv-strategy final_depth` or `--kv-strategy depth_specialized`. Reports label
the execution mode, recurrence mode, prompt prefill, depth budget, KV policy,
temporal-feedback setting, final cache length, and cache bytes.

`evaluation/live_inference.py` provides a small deterministic comparison of
cached live decoding against the slow oracle for supplied token sequences.
