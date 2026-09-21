"""Shared recurrence-axis schedules and config entry-point helper for 5B_axis."""


UPDATE_SUPPORT = [0, 1, 3]

UPDATE_PROBABILITIES = {
    'temporal': [
        [.10, .00, .00],
        [.50, .00, .00],
        [.40, .00, .00],
    ],
    'depth': [
        [.10, .50, .40],
        [.00, .00, .00],
        [.00, .00, .00],
    ],
    'hybrid_matched': [
        [.10, .05, .01],
        [.05, .40, .03],
        [.01, .03, .32],
    ],
}

EVALUATION_COUNTS = {
    'temporal': (3, 0),
    'depth': (0, 3),
    'hybrid_matched': (3, 3),
}


def axis_config(name):
    """Return the frozen 5B config for one of the axis entry points."""
    if name not in UPDATE_PROBABILITIES:
        raise ValueError(name)
    # Import lazily so study.py can import the schedule constants without a
    # module cycle.  The wrappers in this directory are the canonical config
    # entry points and must resolve to the same frozen 5B configs as run.py.
    from ..study import run_config

    return run_config('hybrid' if name == 'hybrid_matched' else name)
