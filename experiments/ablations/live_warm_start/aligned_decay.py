"""Redo the 20B temporal arm's decay phase with warm-start batches.

Resumes the frozen temporal run from its retained pre-decay checkpoint (step
175,954, with optimizer, data and sampler state) and trains the same 19,550
decay updates. The only change is that a quarter of each update's
microbatches (5 of 20) start from settled temporal memory, the memory live
execution uses, instead of from none. The data, the sampled schedules and
the learning rate are those of the original decay. See PLAN.md, question 1.

    python -m experiments.ablations.live_warm_start.aligned_decay
"""

import json
from importlib import import_module
from pathlib import Path
import shutil

from data_loader import file_hash
from train import train

study = import_module('experiments.long_runs.20B_recurrence.study')

SOURCE = Path('experiments/long_runs/20B_recurrence/temporal_20B/results/ckpt-step175954.pt')
OUTPUT = Path('experiments/ablations/live_warm_start/results/aligned_decay_temporal')
WARM_START = dict(warm_start_fraction=0.25, warm_start_max_passes=64, warm_start_tolerance=0.01)


def config():
    settings = study.run_config('temporal')
    settings.update(WARM_START, out_dir=str(OUTPUT), init_from='resume',
                    checkpoint_steps=[180_000, 185_000, 190_000, study.UPDATES])
    return settings


def main():
    OUTPUT.mkdir(parents=True, exist_ok=True)
    plan = OUTPUT / 'plan.json'
    if not (OUTPUT / 'ckpt.pt').exists():
        if plan.exists():
            raise SystemExit(f'{plan} exists but ckpt.pt is missing; refusing to restart from the source')
        shutil.copyfile(SOURCE, OUTPUT / 'ckpt.pt')
        plan.write_text(json.dumps(dict(source=str(SOURCE), source_sha256=file_hash(SOURCE),
                                        warm_start=WARM_START, updates=study.UPDATES,
                                        decay_start=study.WSD_DECAY_START), indent=2) + '\n')
    print(train(config()))


if __name__ == '__main__':
    main()
