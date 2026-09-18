# Recurrent training contract

Variation A is the default: one prelude block, one block between temporal and depth injection, four recurrent-core blocks, one temporal-source block, and one coda block. Width is 512, with eight attention heads, final LayerNorm, and tied embedding/unembedding weights. This is a practical choice after a near-tied A/B comparison, not a finding that separation is intrinsically safer or more accurate. Historical experiments retain their original configurations.

```text
embeddings -> L1 -> fixed p
                       |
shifted temporal ----> T -> L2 -> anchor q
memory from L7                       |
held depth from L6 ----------------> D -> L3-L6 -> h
                                                   | depth write, if scheduled
                                                   v
                                                  L7 -> temporal write, if scheduled
                                                   |
                                          final pass only: L8 -> norm -> head
```

## Configurable sites

The implementation uses one ordered `ModuleList` and five block counts. It does not create separate model classes for each layout. Counts select the source and destination boundaries without changing the mixers or write-mask semantics:

| Field | Role | Default A | B | Original experiments |
| --- | --- | ---: | ---: | ---: |
| `n_prelude` | Blocks before temporal injection | 1 | 1 | 2 |
| `n_buffer` | Blocks between temporal and depth injection | 1 | 0 | 0 |
| `n_core` | Blocks after depth injection through the depth source | 4 | 6 | 4 |
| `n_source` | Blocks between the depth and temporal sources | 1 | 0 | 1 |
| `n_coda` | Blocks after the temporal source | 1 | 1 | 1 |

Counts must be nonnegative integers, sum to `n_layer`, and leave a nonempty core. A zero buffer makes the destinations adjacent; a zero source makes both state candidates the same core output. Buffer and source segments may contain more than one block. Temporal injection always precedes depth injection, and the temporal source is at or after the depth source. Arbitrary crossed connections are outside this interface.

## Recurrence modes

`recurrence_mode` selects which axes are available without introducing separate
model classes:

| Mode | Available axis | Structural modules |
| --- | --- | --- |
| `hybrid` | temporal and depth | `TemporalMixer` and `DepthMixer` |
| `temporal` | temporal only | `TemporalMixer`; no `DepthMixer` |
| `depth` | depth only | `DepthMixer`; no `TemporalMixer` |

`hybrid` is the compatibility default. Historical recurrent checkpoints that
do not contain `recurrence_mode` load as hybrid. Specialized models are exact
limiting cases of hybrid for compatible schedules: temporal models accept only
`(U_T, 0)`, and depth models accept only `(0, U_D)`. Inactive axes are rejected
by both training configuration validation and the model forward boundary.

The `(0, 0)` schedule remains the ordinary physical backbone path for all
modes. Source and coda blocks remain part of the prediction path even when
temporal recurrence is disabled. Matched component-ablation schedules use the
same active-axis count `K` for both axes and therefore update the active state
after every nonfinal pass; this matches pass-count distributions but is not a
FLOP-matched comparison. Inactive mixers are absent from specialized
state-dicts, parameter counts, and optimizer groups.

The implementation exposes half-open segment boundaries (`core_start:core_stop`,
`source_start:source_stop`, and `coda_start:n_layer`). The separately named
`temporal_source_output_index` identifies the block whose output is stored as
temporal memory; when `n_source=0`, it is the final core block. The older
`source_index` property remains a compatibility alias. Diagnostic hooks should
use `temporal_source_output_index` so the temporal-memory source is not confused
with the coda.

In one-based block numbering, temporal injection is after `n_prelude`; depth injection is after `n_prelude + n_buffer`; the depth source is after those blocks plus `n_core`; the temporal source is after those blocks plus `n_source`. Thus A injects temporal state after L1 and depth state after L2, and reads sources after L6 and L7. B injects both after L1 and takes both candidates after L7. At zero updates, every layout executes the same ordinary backbone once in physical block order.

## Schedule and reads

For nonnegative update counts $U_T,U_D$, execute $B=\max(U_T,U_D)+1$ core passes. Each mask contains $B-1$ Boolean entries, with exactly its specified count of true entries, sampled uniformly without replacement. There is no final-pass write slot. Counts and pass length are derived from the masks.

Compute $p=P(x)$ once and start both states absent on every forward call. Each pass reads every available state. Masks control only writes; held tensors retain their values and gradient connections. No iteration index, update count, mask embedding, or state age is supplied to the model. There is no implicit carry of the latest core output around the masks.

## Temporal mixing and the buffer

If temporal state exists, set $r_b=\operatorname{ShiftRight}(m_T)$ and $a_b=T(p,r_b)$; otherwise use $a_b=p$. Shift the stored memory once for each read without modifying it. At position zero, bypass temporal mixing and return raw $p$ because no predecessor exists. A zero-valued memory elsewhere remains a valid input. Preserve causal context across internal game markers within a row.

Use $(\alpha,\beta)=\sigma(G([N_r(r);N_p(p)]))$ and $T(p,r)=\alpha\odot W_mN_r(r)+\beta\odot W_pN_p(p)$. The controller reads normalized sources before value projection. The feature-wise coefficients need not sum to one. The gate is one dense $2D\rightarrow2D$ map with zero initial weights and biases giving $\alpha=0.1$, $\beta=0.9$. Both bias-free value projections start as identity matrices.

Then compute the depth anchor $q_b=Q(a_b)$ through the buffer segment. For A, $Q$ is L2; with `n_buffer=0`, $Q$ is the identity. The buffer runs on every training pass, including when temporal memory is absent or held. Position-zero bypass applies only to the temporal mixer, not to the buffer.

## Depth mixing and writes

If depth state exists, compute $z_b=W_hN_h(h_D)+W_aN_a(q_b)$; otherwise use $z_b=q_b$. Execute $h_b=R(z_b)$ using the shared core. Depth projections start at $0.5I$ each and remain unconstrained learned matrices. Normalization is baseline LayerNorm with epsilon $10^{-5}$, learned scale, and optional backbone-controlled bias.

After a nonfinal pass, store $h_D\leftarrow h_b$ when its write mask is true. Store $m_T\leftarrow S(h_b)$ when its write mask is true, where $S$ is the source segment or identity when empty. A source segment is executed on a nonfinal pass only if its output is consumed by a temporal write. Both writes happen after all reads. With a shared source, different write masks can still give the stored states different ages.

After the final pass, compute logits through $C(S(h_B))$, final normalization, and the head. Backpropagate final-output cross entropy through the complete trajectory and all held-state reads. There are no detached states or intermediate losses. The model performs $L_P+B(L_Q+L_R)+(U_T+1)L_S+L_C$ transformer-block applications. A uses $3+5B+U_T$; B uses $2+6B$.

## Execution scope

This implements the parallel Jacobi-style training graph. Fixed-depth
live-feedback generation is a separate execution path documented in
[`INFERENCE_CONTRACT.md`](INFERENCE_CONTRACT.md). For A at token time, mix
incoming temporal memory into the prelude and run the buffer once; hold that
resulting depth anchor fixed while iterating only the core. Then run the source
and coda once. With $J$ core calls, A uses $4+4J$ block applications; B uses
$2+6J$. Shared-source B emits the final core state directly. These block
counts do not establish measured decoding latency.

## Reproducibility

The sampler checkpoints its RNG, support, probability matrix, draw count, and histograms. Under DDP, rank zero samples each microbatch schedule and broadcasts it. Accumulated microbatches can have different schedules. Evaluation uses independent fixed schedules and does not advance training RNG. Eager execution handles schedule-dependent unused parameters.

New checkpoints store every block count explicitly. Load old checkpoints through `RecurrentGPTConfig.from_checkpoint`: omitted buffer/source counts mean zero/one, preserving the original semantics rather than adopting today's defaults. Resume checks normalized model configurations, all training settings, dataset identity, panel content hash, and world size. Moving a panel file without changing its bytes is allowed. CPU exact resume is tested; GPU kernels may introduce numerical nondeterminism. Raw historical artifacts retain their original paths and hashes; the repository relocation map is in `experiments/relocations.json`.
