from importlib import import_module


study = import_module('experiments.ablations.temporal_gate_init.study')


def test_temporal_gate_initialization_study_is_two_arm_and_matched():
    assert study.UPDATES == 2444
    assert study.ACTUAL_CHARACTERS == 250_021_200
    assert study.WARMUP_UPDATES == 49
    assert study.CHECKPOINT_STEPS[-1] == study.UPDATES
    conservative = study.run_config('conservative')
    higher_memory = study.run_config('higher_memory')
    assert conservative['temporal_memory_gate_init'] == .10
    assert higher_memory['temporal_memory_gate_init'] == .25
    for key in conservative:
        if key not in {'out_dir', 'temporal_memory_gate_init'}:
            assert conservative[key] == higher_memory[key], key
