import importlib
import json


def test_frozen_config_serialization_round_trips_nested_tuples():
    run = importlib.import_module('experiments.ablations.1B_update_schedule.run')
    config = {
        'update_probabilities': ((0.0, 0.85), (0.15, 0.0)),
        'update_probability_schedule': {
            'phases': [{'update_probabilities': ((1.0, 0.0), (0.0, 0.0))}],
        },
    }

    recorded = json.loads(json.dumps(run._json_ready(config)))

    assert recorded == {
        'update_probabilities': [[0.0, 0.85], [0.15, 0.0]],
        'update_probability_schedule': {
            'phases': [{'update_probabilities': [[1.0, 0.0], [0.0, 0.0]]}],
        },
    }
