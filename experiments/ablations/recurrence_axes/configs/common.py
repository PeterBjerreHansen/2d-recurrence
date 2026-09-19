from experiments.serious import base


SUPPORT = [0, 1, 3]

PROBABILITIES = {
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
    if name not in PROBABILITIES:
        raise ValueError(name)
    mode = 'hybrid' if name == 'hybrid_matched' else name
    u_t, u_d = EVALUATION_COUNTS[name]
    config = base('recurrent')
    config.update(
        out_dir=f'experiments/ablations/recurrence_axes/results/{name}',
        recurrence_mode=mode,
        recurrence_support=SUPPORT,
        recurrence_probabilities=PROBABILITIES[name],
        eval_u_t=u_t,
        eval_u_d=u_d,
    )
    return config
