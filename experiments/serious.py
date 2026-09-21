"""Shared, explicit protocol for the next CUDA experiments. Run from repo root."""
from copy import deepcopy
import json
import math
from pathlib import Path

from train import DEFAULTS

ROOT = Path('experiments/ablations/deep_supervision')
REVISION = '1a932e1abca935aae585f417ede39ecde4f2a620'
TOKENS_PER_UPDATE = 100 * 1023
LONG_RUN_ROOTS = {
    10**9: 'experiments/long_runs/1B_baseline',
    64 * 10**9: 'experiments/long_runs/64B_core',
}
COMMON = dict(
    dataset='chess_8M_v1', block_size=1023, n_layer=8, n_head=8, n_embd=512,
    bias=False, dropout=0.0, n_prelude=1, n_buffer=1, n_core=4, n_source=1, n_coda=1,
    batch_size=5, gradient_accumulation_steps=20,
    learning_rate=3e-4, min_lr=3e-5, weight_decay=.1, beta1=.9, beta2=.95,
    grad_clip=1.0, decay_lr=True, seed=1337, recurrence_seed=1729,
    update_support=[0, 1, 3],
    update_probabilities=[[.10, .12, .04], [.12, .26, .08], [.04, .08, .16]],
    eval_u_t=3, eval_u_d=3, deep_supervision=False, deep_supervision_lambda=.25,
    device='cuda', dtype='bfloat16', compile=False, num_threads=4,
    eval_interval=500, eval_iters=16, log_interval=10,
    eval_panel_path=str(ROOT / 'results/panel.json'),
    keep_checkpoints=True,
)


def base(architecture='recurrent'):
    config = deepcopy({**DEFAULTS, **COMMON, 'architecture': architecture})
    if architecture == 'baseline':
        config.update(update_support=[], update_probabilities=[], eval_u_t=0, eval_u_d=0)
    return config


def ablation(variant):
    if variant not in ('final', 'deep', 'deep_more'):
        raise ValueError(variant)
    config = base()
    config.update(out_dir=str(ROOT / 'results' / variant),
                  deep_supervision=variant != 'final', max_iters=1_000_000,
                  lr_decay_iters=10000, warmup_iters=200, checkpoint_steps=[0])
    if variant == 'deep_more':
        # Keep the conditional distribution of (U_T,U_D) at each max-update count.
        config['update_probabilities'] = [[.10, .072, .06], [.072, .156, .12], [.06, .12, .24]]
    return config


def supervision_choice():
    decision = json.loads((ROOT / 'results/decision.json').read_text())
    if decision.get('mode') not in ('final', 'deep') or not decision.get('reason', '').strip():
        raise ValueError('Record a reviewed supervision choice before launching long runs')
    return decision['mode'] == 'deep'


def long_run(model, characters):
    if model not in ('transformer', 'recurrent_a') or characters not in LONG_RUN_ROOTS:
        raise ValueError('Supported series: transformer/recurrent_a at 1B/64B')
    selected = supervision_choice()  # Both members of the queued pair require the decision.
    steps = math.ceil(characters / TOKENS_PER_UPDATE)
    label = f'{model}_{characters // 10**9}B'
    config = base('baseline' if model == 'transformer' else 'recurrent')
    config.update(out_dir=f'{LONG_RUN_ROOTS[characters]}/{label}/results', max_iters=steps,
                  lr_decay_iters=steps, warmup_iters=min(2000, round(.02 * steps)),
                  deep_supervision=selected if model == 'recurrent_a' else False,
                  checkpoint_steps=sorted({0, steps, *(round(steps * f) for f in (.1, .25, .5, .75))}))
    return config
