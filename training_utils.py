"""Run provenance and complete checkpoint state for the training loop."""
import importlib.metadata
import json
import os
from pathlib import Path
import random
import subprocess

import numpy as np
import torch


def provenance():
    def git(*args):
        return subprocess.check_output(['git', *args], text=True, stderr=subprocess.DEVNULL).strip()
    try:
        commit, branch = git('rev-parse', 'HEAD'), git('branch', '--show-current')
        dirty = bool(git('status', '--porcelain'))
    except (FileNotFoundError, subprocess.CalledProcessError):
        # Frozen transfer bundles carry their source receipt separately from Git.
        commit, branch, dirty = None, None, None
    return dict(commit=commit, branch=branch, dirty=dirty, cuda=torch.version.cuda,
                packages={name: importlib.metadata.version(name)
                          for name in ['torch', 'numpy', 'chess', 'datasets', 'huggingface-hub']})


def capture_rng(generator, device):
    state = dict(python=random.getstate(), numpy=np.random.get_state(),
                 torch=torch.get_rng_state(), batches=generator.get_state())
    if str(device).startswith('cuda'):
        state['cuda'] = torch.cuda.get_rng_state(device)
    if str(device).startswith('mps'):
        state['mps'] = torch.mps.get_rng_state()
    return state


def restore_rng(state, generator, device):
    random.setstate(state['python'])
    np.random.set_state(state['numpy'])
    torch.set_rng_state(state['torch'])
    generator.set_state(state['batches'])
    if 'cuda' in state:
        torch.cuda.set_rng_state(state['cuda'], device)
    if 'mps' in state:
        torch.mps.set_rng_state(state['mps'])


def atomic_save(checkpoint, path):
    path = Path(path)
    temporary = path.with_suffix('.tmp')
    torch.save(checkpoint, temporary)
    os.replace(temporary, path)


def append_json(path, record):
    with open(path, 'a') as stream:
        stream.write(json.dumps(record, allow_nan=False) + '\n')
