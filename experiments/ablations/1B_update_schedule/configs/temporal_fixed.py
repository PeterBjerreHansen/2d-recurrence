from importlib import import_module


update_schedule_config = import_module(
    'experiments.ablations.1B_update_schedule.configs.common').update_schedule_config

globals().update(update_schedule_config('temporal_fixed'))
