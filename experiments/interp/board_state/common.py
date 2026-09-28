"""Shared arms, rows and paths for the board-state probing experiment."""

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch

from data_loader import ChessData
from evaluation.live_inference import load_checkpoint_model
from interp.sites import SiteRunner
from model import GPT, GPTConfig

HERE = Path(__file__).parent
RESULTS = HERE / 'results'
RUNS = Path('experiments/long_runs/20B_recurrence')


@dataclass(frozen=True)
class Arm:
    checkpoint: Path
    depth_steps: int


# Karvonen's released 8-layer Lichess model: same architecture, data file and
# vocabulary, trained for 600,000 updates (61.4B characters).
KARVONEN = ('adamkarvonen/chess_llms', 'lichess_8layers_ckpt_no_optimizer.pt')


# Final 20B checkpoints read at the training-graph fixed point. The temporal arm
# is the live-aligned export: the as-trained reader misreads settled memory
# (live NLL 0.48, see the 20B report). ``random_init`` is the transformer's
# step-0 checkpoint, the control for what a probe can read from untrained features.
ARMS = {
    'transformer': Arm(RUNS / 'transformer_20B/results/ckpt-step195504.pt', 1),
    'temporal': Arm(Path('experiments/ablations/live_warm_start/results/temporal_aligned.pt'), 1),
    'depth': Arm(RUNS / 'depth_20B/results/ckpt-step195504.pt', 4),
    'hybrid': Arm(RUNS / 'hybrid_20B/results/ckpt-step195504.pt', 4),
    'random_init': Arm(RUNS / 'transformer_20B/results/ckpt-step000000.pt', 1),
    'karvonen': Arm(Path('/'.join(KARVONEN)), 1),
}

# Validation rows: the first TRAIN_ROWS of the selection train the probes, the rest test them.
ROW_SEED = 0
ROWS = 1000
TRAIN_ROWS = 800
CONTEXT = 1023
# Board probes read the two pre-move kinds; ``space_white`` points only serve the
# side-to-move probe, so a fixed third of them is kept to bound storage.
SPACE_WHITE_KEEP = 3


def device():
    return 'mps' if torch.backends.mps.is_available() else 'cuda' if torch.cuda.is_available() else 'cpu'


def load_arm(name, device_name, **runner_options):
    """A site runner for one arm and its checkpoint (``None`` for Karvonen's model)."""
    arm = ARMS[name]
    if name == 'karvonen':
        from huggingface_hub import hf_hub_download
        checkpoint = torch.load(hf_hub_download(*KARVONEN), map_location='cpu', weights_only=True)
        model = GPT(GPTConfig(**checkpoint['model_args']))
        model.load_state_dict({key.removeprefix('_orig_mod.'): value for key, value in checkpoint['model'].items()})
        return SiteRunner(model.to(device_name).eval(), 1, **runner_options), None
    model, checkpoint = load_checkpoint_model(arm.checkpoint, device_name)
    return SiteRunner(model, arm.depth_steps, **runner_options), checkpoint


def load_rows(checkpoint=None):
    """The selected validation rows as uint8 [ROWS, 1024] and the vocabulary."""
    data = ChessData(Path('data/chess_8M_v1'), CONTEXT, verify_hashes=False)
    if checkpoint is not None and data.manifest_hash != checkpoint['manifest_hash']:
        raise SystemExit('Dataset manifest differs from the checkpoint')
    val = data.rows['val']
    selection = np.random.default_rng(ROW_SEED).choice(len(val), ROWS, replace=False)
    return np.array(val[selection]), selection, data.meta


def decode(row, meta):
    return ''.join(meta['itos'][int(c)] for c in row)
