"""Generate chess characters from a local checkpoint and stop on the first invalid move."""
import argparse
from collections import Counter
import json
from pathlib import Path
import random

import torch

from evaluation.chess import MoveValidator
from inference.live import (LiveInferenceSpec, consume_prompt_live, create_live_state,
                            decode_live_step, validate_live_inference_spec)
from model import GPT, GPTConfig
from models.recurrent_2d import (Recurrent2DGPT, RecurrentGPTConfig,
                                 validate_recurrence_counts, validate_recurrence_mode)
from recurrence.schedule import sample_schedule


def effective_training_graph_counts(mode, u_t=None, u_d=None):
    validate_recurrence_mode(mode)
    defaults = {'hybrid': (3, 3), 'temporal': (3, 0), 'depth': (0, 3)}[mode]
    u_t = defaults[0] if u_t is None else u_t
    u_d = defaults[1] if u_d is None else u_d
    validate_recurrence_counts(mode, u_t, u_d)
    return u_t, u_d


def effective_generation_counts(mode, u_t=None, u_d=None):
    """Backward-compatible name for training-graph schedule defaults."""
    return effective_training_graph_counts(mode, u_t, u_d)


@torch.no_grad()
def generate(model, meta, prompt=';1.', *, seed=1337, max_new_tokens=512, temperature=1.0,
             top_k=0, schedule=None, live_spec: LiveInferenceSpec | None = None):
    if temperature < 0 or top_k < 0 or max_new_tokens < 0:
        raise ValueError('temperature, top_k and max_new_tokens must be nonnegative')
    if schedule is not None and live_spec is not None:
        raise ValueError('Generation cannot combine training-graph and live execution')
    validator = MoveValidator()
    validator.read_prompt(prompt)
    model.eval()
    device = next(model.parameters()).device
    try:
        tokens = [meta['stoi'][char] for char in prompt]
    except KeyError as error:
        raise ValueError(f'Prompt character outside vocabulary: {error.args[0]!r}') from error
    if len(tokens) > model.config.block_size:
        raise ValueError('Prompt exceeds model context')
    # Sample on CPU with a dedicated generator: reproducible within a device/version,
    # without advancing the training process RNG or masking illegal moves.
    generator = torch.Generator().manual_seed(seed)
    reason = 'generation_limit'
    live_state = None
    logits = None
    if live_spec is not None:
        live_state = create_live_state(model, live_spec.depth_steps, live_spec.kv_strategy)
        logits = consume_prompt_live(model, torch.tensor(tokens, dtype=torch.long, device=device),
                                     live_state, live_spec)
    for _ in range(max_new_tokens):
        if len(tokens) >= model.config.block_size:
            reason = 'context_limit'
            break
        if live_spec is None:
            x = torch.tensor([tokens], dtype=torch.long, device=device)
            logits, _ = model(x, **(dict(schedule=schedule) if schedule is not None else {}))
            logits = logits[0, -1]
        logits = logits.float().cpu()
        if temperature == 0:
            token = int(logits.argmax())
        else:
            logits = logits / temperature
            if top_k:
                threshold = torch.topk(logits, min(top_k, len(logits))).values[-1]
                logits[logits < threshold] = -float('inf')
            token = int(torch.multinomial(logits.softmax(-1), 1, generator=generator))
        tokens.append(token)
        validator.feed(meta['itos'][token])
        if validator.reason:
            break
        if live_spec is not None:
            logits = decode_live_step(model, torch.tensor([token], dtype=torch.long, device=device),
                                      live_state, live_spec)
    report = validator.finish(reason)
    execution = 'live' if live_spec is not None else 'training_graph' if schedule is not None else 'baseline'
    mode = live_spec.recurrence_mode if live_spec is not None else getattr(model.config, 'recurrence_mode', 'baseline')
    report.update(prompt=prompt, seed=seed, generated_characters=len(tokens) - len(prompt),
                  temperature=temperature, top_k=top_k, max_new_tokens=max_new_tokens,
                  execution=execution,
                  recurrence_mode=mode,
                  prefill='sequential_live' if live_spec is not None else 'full_prefix_recomputation_each_character',
                  depth_budget=(dict(kind='training_graph_core_passes', value=schedule.rounds)
                                if schedule else
                                dict(kind='fixed_core_steps_per_token', value=live_spec.depth_steps)
                                if live_spec is not None else None),
                  u_t=schedule.u_t if schedule else None, u_d=schedule.u_d if schedule else None,
                  temporal_write_mask=schedule.temporal_write_mask if schedule else None,
                  depth_write_mask=schedule.depth_write_mask if schedule else None,
                  depth_steps=live_spec.depth_steps if live_spec is not None else None,
                  kv_strategy=live_spec.kv_strategy if live_spec is not None else None,
                  temporal_feedback_enabled=(live_spec.recurrence_mode in ('temporal', 'hybrid')
                                             if live_spec is not None else None),
                  cache_length=live_state.cache.cache_length if live_state is not None else None,
                  cache_bytes=live_state.cache.bytes if live_state is not None else None)
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--checkpoint', required=True, help='Trusted local training checkpoint')
    parser.add_argument('--device', default='cpu')
    parser.add_argument('--execution', choices=['baseline', 'training_graph', 'live'])
    parser.add_argument('--u-t', type=int)
    parser.add_argument('--u-d', type=int)
    parser.add_argument('--mask-seed', type=int)
    parser.add_argument('--depth-steps', type=int)
    parser.add_argument('--kv-strategy', choices=['final_depth', 'depth_specialized', 'ordinary'])
    parser.add_argument('--num-threads', type=int, default=4)
    parser.add_argument('--prompt', default=';1.')
    parser.add_argument('--prompts', help='JSON list of game-prefix strings')
    parser.add_argument('--num-samples', type=int, default=10)
    parser.add_argument('--seed', type=int, default=1337)
    parser.add_argument('--max-new-tokens', type=int, default=512)
    parser.add_argument('--temperature', type=float, default=1.0)
    parser.add_argument('--top-k', type=int, default=0)
    parser.add_argument('--output', help='Defaults to generation-<checkpoint stem>.json beside the checkpoint')
    args = parser.parse_args()
    if args.num_samples < 1:
        parser.error('--num-samples must be positive')
    if args.output is None:
        checkpoint_path = Path(args.checkpoint)
        args.output = str(checkpoint_path.with_name(f'generation-{checkpoint_path.stem}.json'))
    if Path(args.output).exists():
        parser.error('Output exists; choose a new report path')
    torch.set_num_threads(args.num_threads)
    checkpoint = torch.load(args.checkpoint, map_location='cpu', weights_only=False)
    recurrent = checkpoint['config'].get('architecture', 'baseline') == 'recurrent'
    model = (Recurrent2DGPT(RecurrentGPTConfig.from_checkpoint(checkpoint['model_args'])) if recurrent
             else GPT(GPTConfig(**checkpoint['model_args']))).to(args.device)
    model.load_state_dict(checkpoint['model'])
    if args.execution is None:
        if recurrent:
            parser.error('Recurrent checkpoints require --execution=training_graph or --execution=live')
        args.execution = 'baseline'
    schedule = None
    live_spec = None
    if args.execution == 'training_graph':
        if not recurrent:
            parser.error('Training-graph recurrence requires a recurrent checkpoint')
        if args.depth_steps is not None or args.kv_strategy is not None:
            parser.error('--depth-steps and --kv-strategy are valid only with --execution=live')
        try:
            args.u_t, args.u_d = effective_training_graph_counts(
                checkpoint['model_args'].get('recurrence_mode', 'hybrid'), args.u_t, args.u_d)
        except ValueError as error:
            parser.error(str(error))
        args.mask_seed = 11 if args.mask_seed is None else args.mask_seed
        schedule = sample_schedule(args.u_t, args.u_d, random.Random(args.mask_seed))
    elif args.execution == 'live':
        if args.u_t is not None or args.u_d is not None or args.mask_seed is not None:
            parser.error('--u-t, --u-d, and --mask-seed are invalid with --execution=live')
        try:
            live_spec = validate_live_inference_spec(model.config, args.depth_steps, args.kv_strategy)
        except ValueError as error:
            parser.error(str(error))
    elif args.execution == 'baseline':
        if recurrent:
            parser.error('Recurrent checkpoints require --execution=training_graph or --execution=live')
        if args.u_t is not None or args.u_d is not None or args.mask_seed is not None:
            parser.error('Recurrence schedule options require --execution=training_graph')
        if args.depth_steps is not None or args.kv_strategy is not None:
            parser.error('Live cache options require --execution=live')
    prompts = json.loads(Path(args.prompts).read_text()) if args.prompts else [args.prompt]
    if not prompts:
        parser.error('Prompt list is empty')
    results = [generate(model, checkpoint['meta'], prompts[i % len(prompts)], seed=args.seed + i,
                        max_new_tokens=args.max_new_tokens, temperature=args.temperature, top_k=args.top_k,
                        schedule=schedule, live_spec=live_spec)
               for i in range(args.num_samples)]
    attempts = sum(r['attempted_moves'] for r in results)
    legal = sum(r['legal_moves'] for r in results)
    summary = dict(samples=len(results), completed_move_legality=legal / attempts if attempts else None,
                   mean_legal_continuation=legal / len(results),
                   pgn_parse_success_rate=sum(r['pgn_parse_success'] for r in results) / len(results),
                   valid_termination_rate=sum(r['valid_termination'] for r in results) / len(results),
                   termination_reasons=dict(Counter(r['termination_reason'] for r in results)))
    out = Path(args.output)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(dict(checkpoint=str(Path(args.checkpoint).resolve()),
                                  checkpoint_step=checkpoint['iter_num'], model_args=checkpoint['model_args'],
                                  manifest_hash=checkpoint['manifest_hash'], settings=vars(args),
                                  summary=summary, samples=results), indent=2) + '\n')
    print(json.dumps(summary, indent=2))


if __name__ == '__main__':
    main()
