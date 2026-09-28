"""Expected forward matmul FLOPs per sequence for each 20B arm, relative to the transformer.

Averages `evaluation.recurrence_grid.compute_estimate` over each arm's update
distribution, every write placement, and the curriculum phases weighted by
their length. Backward work scales with forward work, so the ratios also hold
for training compute.

    python -m experiments.long_runs.20B_recurrence.training_flops
"""
import itertools
from importlib import import_module
from types import SimpleNamespace

from evaluation.recurrence_grid import compute_estimate
from recurrence.schedule import RecurrenceSchedule

study = import_module('experiments.long_runs.20B_recurrence.study')
L, d, V = 1023, 512, 32
cfg = SimpleNamespace(n_embd=d, n_prelude=1, n_buffer=1, n_core=4, n_source=1, n_coda=1, vocab_size=V)
support = study.UPDATE_SUPPORT


def cell_flops(u_t, u_d):
    """Mean over every write placement, as sample_schedule draws them uniformly."""
    slots = max(u_t, u_d)
    if slots == 0:
        return compute_estimate(cfg, RecurrenceSchedule((), ()), L)['estimated_forward_matmul_flops_per_sequence']
    values = []
    for t in itertools.combinations(range(slots), u_t):
        for dd in itertools.combinations(range(slots), u_d):
            schedule = RecurrenceSchedule(tuple(i in t for i in range(slots)), tuple(i in dd for i in range(slots)))
            values.append(compute_estimate(cfg, schedule, L)['estimated_forward_matmul_flops_per_sequence'])
    return sum(values) / len(values)


transformer = 8 * (24 * L * d * d + 4 * L * L * d) + 2 * L * d * V
starts = list(study.CURRICULUM_STARTS) + [study.UPDATES]
results = {}
for arm in ('temporal', 'depth', 'hybrid'):
    phases = study.update_probability_schedule(arm)['phases']
    per_phase = [sum(m[i][j] * cell_flops(support[i], support[j])
                     for i in range(len(support)) for j in range(len(support)) if m[i][j])
                 for m in (phase['update_probabilities'] for phase in phases)]
    weights = [starts[k + 1] - starts[k] for k in range(len(phases))]
    results[arm] = (sum(w * f for w, f in zip(weights, per_phase)) / sum(weights), per_phase[-1])
    print(f'{arm:9} whole curriculum {results[arm][0] / transformer:.3f}x  late phase {results[arm][1] / transformer:.3f}x')
for a, b in (('hybrid', 'temporal'), ('hybrid', 'depth'), ('temporal', 'depth')):
    print(f'{a} vs {b}: {results[a][0] / results[b][0] - 1:+.1%} whole, {results[a][1] / results[b][1] - 1:+.1%} late')
