"""Shared configuration helper for the temporal gate initialization ablation."""

from importlib import import_module


GATE_INITIALIZATIONS = {
    'conservative': .10,
    'higher_memory': .25,
}


def gate_config(name):
    if name not in GATE_INITIALIZATIONS:
        raise ValueError(name)
    study = import_module('experiments.ablations.temporal_gate_init.study')
    return study.run_config(name)
