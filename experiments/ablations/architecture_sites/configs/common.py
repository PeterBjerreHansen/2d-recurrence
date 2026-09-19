"""Shared CUDA protocol. Architecture A/B are unrelated to the earlier LR A/B."""
COMMON = dict(
    architecture='recurrent', dataset='chess_143K_v1',
    eval_panel_path='experiments/ablations/architecture_sites/panel.json',
    n_layer=8, n_head=8, n_embd=512, n_prelude=1, n_coda=1,
    block_size=1023, bias=False, dropout=0.0,
    batch_size=2, gradient_accumulation_steps=4,
    recurrence_support=[0, 1, 3],
    recurrence_probabilities=[[.10, .12, .04], [.12, .26, .08], [.04, .08, .16]],
    seed=1337, recurrence_seed=1729, eval_u_t=3, eval_u_d=3,
    learning_rate=3e-4, min_lr=3e-5, weight_decay=.1, beta1=.9, beta2=.95,
    grad_clip=1.0, decay_lr=True, warmup_iters=100, lr_decay_iters=10000,
    max_iters=10000, eval_interval=100, eval_iters=4, log_interval=1,
    keep_checkpoints=True, checkpoint_steps=[0, 100, 1000, 2000, 5000, 8000, 10000],
    compile=False, device='cuda', dtype='float32', num_threads=4,
)
