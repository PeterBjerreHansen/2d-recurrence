from importlib import import_module


pass_schedule_config = import_module(
    'experiments.ablations.1B_pass_schedule.configs.common').pass_schedule_config

globals().update(pass_schedule_config('temporal_hard_2to4'))
