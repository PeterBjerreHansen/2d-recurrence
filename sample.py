"""Generate chess characters from a local checkpoint and stop on the first invalid move."""
import argparse
from collections import Counter
import json
from pathlib import Path

import torch

from evaluation.chess import MoveValidator
from model import GPT, GPTConfig


@torch.no_grad()
def generate(model, meta, prompt=';1.', *, seed=1337, max_new_tokens=512, temperature=1.0, top_k=0):
    if temperature < 0 or top_k < 0 or max_new_tokens < 0:
        raise ValueError('temperature, top_k and max_new_tokens must be nonnegative')
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
    for _ in range(max_new_tokens):
        if len(tokens) >= model.config.block_size:
            reason = 'context_limit'
            break
        x = torch.tensor([tokens], dtype=torch.long, device=device)
        logits, _ = model(x)
        logits = logits[0, -1].float().cpu()
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
    report = validator.finish(reason)
    report.update(prompt=prompt, seed=seed, generated_characters=len(tokens) - len(prompt),
                  temperature=temperature, top_k=top_k, max_new_tokens=max_new_tokens)
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--checkpoint', required=True, help='Trusted local training checkpoint')
    parser.add_argument('--device', default='cpu')
    parser.add_argument('--prompt', default=';1.')
    parser.add_argument('--prompts', help='JSON list of game-prefix strings')
    parser.add_argument('--num-samples', type=int, default=10)
    parser.add_argument('--seed', type=int, default=1337)
    parser.add_argument('--max-new-tokens', type=int, default=512)
    parser.add_argument('--temperature', type=float, default=1.0)
    parser.add_argument('--top-k', type=int, default=0)
    parser.add_argument('--output', default='out-evaluation/generation.json')
    args = parser.parse_args()
    if args.num_samples < 1:
        parser.error('--num-samples must be positive')
    checkpoint = torch.load(args.checkpoint, map_location='cpu', weights_only=False)
    model = GPT(GPTConfig(**checkpoint['model_args'])).to(args.device)
    model.load_state_dict(checkpoint['model'])
    prompts = json.loads(Path(args.prompts).read_text()) if args.prompts else [args.prompt]
    if not prompts:
        parser.error('Prompt list is empty')
    results = [generate(model, checkpoint['meta'], prompts[i % len(prompts)], seed=args.seed + i,
                        max_new_tokens=args.max_new_tokens, temperature=args.temperature, top_k=args.top_k)
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
