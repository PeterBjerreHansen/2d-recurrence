from importlib import import_module


run_config = import_module('experiments.long_runs.5B_axis.study').run_config


globals().update(run_config('temporal'))
