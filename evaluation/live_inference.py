"""Compare cached live decoding with the slow live reference implementation.

Example:
    python -m evaluation.live_inference --checkpoint checkpoint.pt \
        --prompt ';1.e4' --depth-steps 3 --kv-strategy final_depth
"""

import argparse
import json
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

from data_loader import ChessData
from evaluation.panels import load_panel
from inference.live import (create_live_state, decode_live_step,
                            validate_live_inference_spec)
from inference.reference import create_reference_state, decode_reference_step
from model import GPT, GPTConfig
from models.recurrent_2d import Recurrent2DGPT, RecurrentGPTConfig


def load_checkpoint_model(checkpoint_path, device='cpu'):
    """Load a trusted checkpoint using the same model selection as sampling."""
    checkpoint = torch.load(checkpoint_path, map_location='cpu', weights_only=False)
    recurrent = checkpoint['config'].get('architecture', 'baseline') == 'recurrent'
    model = (Recurrent2DGPT(RecurrentGPTConfig.from_checkpoint(checkpoint['model_args']))
             if recurrent else GPT(GPTConfig(**checkpoint['model_args'])))
    model.load_state_dict(checkpoint['model'])
    return model.to(device).eval(), checkpoint


def _prompt_tokens(prompt, meta, device):
    try:
        values = [meta['stoi'][char] for char in prompt]
    except KeyError as error:
        raise ValueError(f'Prompt character outside vocabulary: {error.args[0]!r}') from error
    if not values:
        raise ValueError('Prompts must contain at least one token')
    return torch.tensor(values, dtype=torch.long, device=device)


@torch.no_grad()
def evaluate_sequences(model, sequences, *, depth_steps=None, kv_strategy=None):
    """Return cached-versus-reference live errors for one-token sequences."""
    spec = validate_live_inference_spec(model.config, depth_steps, kv_strategy)
    samples = []
    for sequence_index, sequence in enumerate(sequences):
        if sequence.ndim != 1 or sequence.numel() == 0:
            raise ValueError('Each sequence must be a nonempty one-dimensional tensor')
        if sequence.numel() > model.config.block_size:
            raise ValueError('A sequence exceeds the model context limit')
        cached = create_live_state(model, spec.depth_steps, spec.kv_strategy)
        reference = create_reference_state(model, spec.depth_steps, spec.kv_strategy)
        maximum = 0.0
        total = 0.0
        count = 0
        for token in sequence:
            cached_logits = decode_live_step(model, token[None], cached, spec)
            reference_logits = decode_reference_step(model, token[None], reference, spec)
            difference = (cached_logits - reference_logits).abs()
            maximum = max(maximum, difference.max().item())
            total += difference.sum().item()
            count += difference.numel()
        samples.append(dict(
            sequence_index=sequence_index,
            token_count=sequence.numel(),
            max_abs_logit_error=maximum,
            mean_abs_logit_error=total / count,
            cache_length=cached.cache.cache_length,
            cache_bytes=cached.cache.bytes,
        ))
    if not samples:
        raise ValueError('At least one sequence is required')
    return dict(
        execution='live',
        recurrence_mode=spec.recurrence_mode,
        prefill='sequential_live',
        depth_steps=spec.depth_steps,
        kv_strategy=spec.kv_strategy,
        reference='slow_live_full_prefix_block_recomputation',
        sequence_count=len(samples),
        max_abs_logit_error=max(item['max_abs_logit_error'] for item in samples),
        mean_abs_logit_error=sum(item['mean_abs_logit_error'] for item in samples) / len(samples),
        samples=samples,
    )


@torch.no_grad()
def evaluate_teacher_forced(model, rows, *, depth_steps=None, kv_strategy=None):
    """Evaluate sequential next-character prediction on independent rows.

    Each row is a ``(inputs, targets)`` pair. State is reset before every row,
    matching the stored-row boundary used by training.
    """
    spec = validate_live_inference_spec(model.config, depth_steps, kv_strategy)
    device = next(model.parameters()).device
    block_count = (model.config.n_layer if spec.recurrence_mode == 'baseline' else
                   model.config.n_prelude + model.config.n_buffer +
                   spec.depth_steps * model.config.n_core + model.config.n_source +
                   model.config.n_coda)
    total_loss = 0.0
    total_correct = 0
    target_count = 0
    row_reports = []
    for row_index, (inputs, targets) in enumerate(rows):
        if inputs.ndim != 1 or targets.ndim != 1 or inputs.shape != targets.shape:
            raise ValueError('Each row must contain matching one-dimensional input and target tensors')
        if inputs.numel() == 0 or inputs.numel() > model.config.block_size:
            raise ValueError('Each row must fit in the model context and contain one token')
        state = create_live_state(model, spec.depth_steps, spec.kv_strategy)
        row_loss = 0.0
        row_correct = 0
        inputs, targets = inputs.to(device), targets.to(device)
        for token, target in zip(inputs, targets):
            logits = decode_live_step(model, token[None], state, spec)
            row_loss += F.cross_entropy(logits, target[None], reduction='sum').item()
            row_correct += (logits.argmax(-1) == target).sum().item()
        row_count = targets.numel()
        total_loss += row_loss
        total_correct += row_correct
        target_count += row_count
        row_reports.append(dict(row_index=row_index, target_count=row_count,
                               nll=row_loss / row_count, accuracy=row_correct / row_count,
                               cache_length=state.cache.cache_length,
                               cache_bytes=state.cache.bytes))
    if not row_reports:
        raise ValueError('At least one row is required')
    return dict(
        execution='live',
        recurrence_mode=spec.recurrence_mode,
        prefill='sequential_live',
        depth_steps=spec.depth_steps,
        kv_strategy=spec.kv_strategy,
        cache_strategy=spec.kv_strategy,
        temporal_feedback_enabled=spec.recurrence_mode in ('temporal', 'hybrid'),
        row_count=len(row_reports),
        target_count=target_count,
        nll=total_loss / target_count,
        accuracy=total_correct / target_count,
        cache_bytes=max(row['cache_bytes'] for row in row_reports),
        core_block_applications=target_count * (
            spec.depth_steps * model.config.n_core if spec.recurrence_mode != 'baseline'
            else model.config.n_layer),
        physical_transformer_block_applications=target_count * block_count,
        rows=row_reports,
    )


def evaluate_checkpoint(checkpoint_path, prompts, *, depth_steps=None, kv_strategy=None,
                        device='cpu'):
    """Load a checkpoint and compare its cached and reference prompt traces."""
    model, checkpoint = load_checkpoint_model(checkpoint_path, device)
    sequences = [_prompt_tokens(prompt, checkpoint['meta'], device) for prompt in prompts]
    report = evaluate_sequences(model, sequences, depth_steps=depth_steps, kv_strategy=kv_strategy)
    report.update(checkpoint=str(Path(checkpoint_path).resolve()),
                  checkpoint_step=checkpoint['iter_num'], model_args=checkpoint['model_args'],
                  manifest_hash=checkpoint['manifest_hash'], prompts=list(prompts))
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--checkpoint', required=True)
    parser.add_argument('--device', default='cpu')
    parser.add_argument('--prompt', default=';1.')
    parser.add_argument('--prompts', help='JSON list of prompt strings')
    parser.add_argument('--data-dir', help='Prepared dataset directory for teacher-forced validation')
    parser.add_argument('--context-length', type=int)
    parser.add_argument('--panel-file', help='Frozen validation panel JSON')
    parser.add_argument('--panel-split', choices=['selection', 'confirmation'], default='selection')
    parser.add_argument('--depth-steps', type=int)
    parser.add_argument('--kv-strategy', choices=['final_depth', 'depth_specialized', 'ordinary'])
    parser.add_argument('--output')
    args = parser.parse_args()
    if args.data_dir:
        if args.prompts:
            parser.error('--prompts cannot be combined with --data-dir')
        model, checkpoint = load_checkpoint_model(args.checkpoint, args.device)
        context_length = model.config.block_size if args.context_length is None else args.context_length
        data = ChessData(args.data_dir, context_length)
        if args.panel_file:
            panel = load_panel(args.panel_file, data, args.panel_split)
            row_indices = panel['row_indices']
        else:
            panel = None
            row_indices = range(len(data.rows['val']))
        block = np.array(data.rows['val'][list(row_indices), :context_length + 1], dtype=np.int64)
        rows = [(torch.from_numpy(values[:-1].copy()), torch.from_numpy(values[1:].copy()))
                for values in block]
        report = evaluate_teacher_forced(model, rows, depth_steps=args.depth_steps,
                                         kv_strategy=args.kv_strategy)
        report.update(checkpoint=str(Path(args.checkpoint).resolve()),
                      checkpoint_step=checkpoint['iter_num'], model_args=checkpoint['model_args'],
                      manifest_hash=checkpoint['manifest_hash'], data_dir=str(Path(args.data_dir).resolve()),
                      data_manifest_hash=data.manifest_hash, context_length=context_length,
                      panel=panel)
    else:
        if args.panel_file or args.context_length is not None:
            parser.error('--context-length and --panel-file require --data-dir')
        prompts = json.loads(Path(args.prompts).read_text()) if args.prompts else [args.prompt]
        if not prompts or not all(isinstance(prompt, str) for prompt in prompts):
            parser.error('Prompts must be a nonempty list of strings')
        report = evaluate_checkpoint(args.checkpoint, prompts, depth_steps=args.depth_steps,
                                     kv_strategy=args.kv_strategy, device=args.device)
    output = Path(args.output) if args.output else None
    if output is not None:
        if output.exists():
            parser.error('Output exists; choose a new report path')
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(json.dumps(report, indent=2) + '\n')
    print(json.dumps(report, indent=2))


if __name__ == '__main__':
    main()
