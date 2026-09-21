from importlib import import_module


axis_config = import_module('experiments.long_runs.5B_axis.configs.common').axis_config


globals().update(axis_config('temporal'))
