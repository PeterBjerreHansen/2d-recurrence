# Recurrent training contract

This is the implementation reference for stages 2–8. The model is a causal character-level language model. Its stored block order remains identical to the baseline: two prelude blocks, four shared core blocks, one temporal-source block, and one coda block, followed by final normalization and the tied LM head. Small tests may change the block counts; the source remains one block. Prelude and coda may be empty, but the core must contain at least one block.

## Schedule and reads

For nonnegative integer update counts $U_T,U_D$, execute $B=\max(U_T,U_D)+1$ core passes. Each write mask contains $B-1$ Boolean entries, with exactly $U_T$ or $U_D$ true entries sampled uniformly without replacement. There is no final-pass write slot. Counts and pass length are derived from the masks rather than separately mutable schedule fields.

Compute the prelude $p=P(x)$ once. Both states start absent for every forward call. Every pass reads every available state. Masks control only writes after core computation; a held tensor retains its value and gradient connection. States do not persist between training microbatches. The learned model receives no iteration index, update count, mask embedding, or state-age input.

## Temporal mixer

If temporal memory exists, set $r_b=\operatorname{ShiftRight}(m_T)$ and $a_b=T(p,r_b)$; otherwise use $a_b=p$. Shift the stored memory by exactly one position for each read, without modifying it. At position zero, return raw $p$ because no predecessor exists. Other zero-valued memories remain valid inputs. Preserve the inherited causal context across game markers inside a data row.

Use $(\alpha,\beta)=\sigma(G([N_r(r);N_p(p)]))$ and $T(p,r)=\alpha\odot W_mN_r(r)+\beta\odot W_pN_p(p)$. The gate reads normalized sources before value projection. Coefficients are feature-wise and unconstrained in their sum. The MVP uses a single dense $2D\rightarrow2D$ gate followed by sigmoid; an extra hidden layer is not required by the contract. Its weights start at zero and its biases give $\alpha=0.1$, $\beta=0.9$. Both bias-free value projections start as identity matrices.

## Depth mixer and state writes

If depth state exists, compute $z_b=W_hN_h(h_D)+W_aN_a(a_b)$; otherwise use $z_b=a_b$. Execute $h_b=R(z_b)$ using the same core blocks on every pass. The two bias-free depth projections start at $0.5I$ each. They remain unconstrained learned matrices, not a convex gate. Normalization uses the baseline LayerNorm implementation (epsilon $10^{-5}$, learned scale, optional bias controlled by the backbone configuration). These initializations are experimental defaults, not claims of optimality or convergence.

After a nonfinal pass, assign $h_D\leftarrow h_b$ when the depth mask is true. Assign $m_T\leftarrow S(h_b)$ when the temporal mask is true. The source $S$ is a normal transformer block with parameters distinct from the core and coda, reused across all temporal writes. It stores its raw output with no additional writer. If a write mask is false, leave that state untouched, including leaving it absent before its first write. There is no additional carry of the last core output that bypasses the masks.

After the final pass, compute logits through $C(S(h_B))$, final normalization, and the LM head. This final source computation is part of prediction, not a counted state write. A trajectory therefore executes the source $U_T+1$ times and each coda block once. Backpropagate final-output cross entropy through the complete trajectory and all held-state reads; do not detach states or add intermediate losses.

## Exact reductions and execution scope

$(0,0)$ is the ordinary backbone executed once in block order. $(U_T>0,0)$ never creates depth state. $(0,U_D>0)$ never creates temporal memory. A hybrid creates both, including when one state is refreshed once and read repeatedly alongside the other evolving state.

This is exact execution of the parallel Jacobi-style training graph. It is not live-feedback generation. In that later inference mode, the temporal source and coda run once per token after all depth iterations. Do not treat a training-graph pass count as a token-time inference setting or silently use the ordinary sampler on recurrent checkpoints.

## Reproducibility and distributed training

The sampler owns a dedicated RNG and checkpoints its state, support, probability matrix, draw count, and pair/pass histograms. Under DDP, rank zero samples one schedule per microbatch and broadcasts it to all ranks. Only rank zero advances the sampler; its saved state is authoritative on resume. Microbatches in the same accumulated optimizer update can have different schedules.

Some schedules bypass mixer parameters. DDP must therefore handle unused parameters on each iteration; a static graph cannot be assumed. The MVP uses eager execution rather than compiling many schedule-dependent graphs. Evaluation uses its own fixed schedule RNG and explicit update pair, without advancing the training sampler. Checkpoint resume preserves the training schedule sequence as well as model, optimizer, and random-generator state.
