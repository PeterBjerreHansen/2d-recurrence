"""Where is computation happening? Content-free measures by character role.

For every character role of the move cycle (``cycle.ROLES``), over held-out rows:

``update``   size of each block's (and mixer's) change to the residual stream,
             ``|out - in| / |in|``, where ``in`` is the site before it in execution order
``pass``     looped arms: change of the core output between passes,
             ``|L6@k - L6@(k-1)| / |L6@k|``
``exit``     looped arms: finish the network (L7, L8, readout) from the core output
             after pass ``k``, and compare with the final prediction: KL(final || exit)
             and top-1 agreement. Exiting after the last pass reproduces the final
             prediction, which is checked.
``nll``      next-character NLL per role, to show which characters are trivial to predict

These say where the network changes its state and whether later passes change
the prediction; they say nothing about what is computed.

Writes ``results/latents/<arm>.json``.

Example:
    python -m experiments.interp.board_state.latents --arms random_init transformer temporal depth hybrid
"""

import argparse
import json

import numpy as np
import torch
import torch.nn.functional as F

from .common import CONTEXT, RESULTS, TRAIN_ROWS, decode, device, load_arm, load_rows
from .cycle import ROLES, label_row


def block_inputs(runner):
    """(site, input site, kind) for every block and mixer site, in execution order."""
    names = runner.site_names
    return [(site.name, names[i - 1], site.kind) for i, site in enumerate(runner.sites) if i > 0]


@torch.no_grad()
def early_exit_logits(model, core_output):
    """Finish the network from a core output: T-source, coda, final norm and head."""
    return model.lm_head(model.transformer.ln_f(model.coda(model.temporal_source(core_output))))


@torch.no_grad()
def analyse(name, rows, meta, device_name, batch_rows):
    runner, _ = load_arm(name, device_name)
    model = runner.model
    pairs = block_inputs(runner)
    looped = runner.depth_steps > 1
    n_roles = len(ROLES)
    update_sum = {site: np.zeros(n_roles) for site, _, _ in pairs}
    pass_sum = {k: np.zeros(n_roles) for k in range(2, runner.depth_steps + 1)}
    exit_kl = {k: np.zeros(n_roles) for k in range(1, runner.depth_steps + 1)}
    exit_top1 = {k: np.zeros(n_roles) for k in range(1, runner.depth_steps + 1)}
    nll_sum, count = np.zeros(n_roles), np.zeros(n_roles)
    last_exit_error = 0.0
    for start in range(0, len(rows), batch_rows):
        block = rows[start:start + batch_rows]
        x = torch.from_numpy(block[:, :CONTEXT].astype(np.int64)).to(device_name)
        y = torch.from_numpy(block[:, 1:CONTEXT + 1].astype(np.int64)).to(device_name)
        b, t = x.shape
        index = (torch.arange(b, device=device_name).repeat_interleave(t), torch.arange(t, device=device_name).repeat(b))
        # float32 captures, so that exiting after the last pass reproduces the final prediction exactly.
        run = runner.run(x, capture=runner.site_names, index=index, capture_dtype=torch.float32)
        states = {site: v.to(device_name).float().view(b, t, -1) for site, v in run.captures.items()}
        labels = [label_row(decode(row, meta), CONTEXT) for row in block]
        roles = np.stack([label[0] for label in labels])
        valid = np.stack([label[3] for label in labels])
        one_hot = torch.zeros(b, t, n_roles, device=device_name)
        role_t = torch.from_numpy(np.where(valid, roles, 0).astype(np.int64)).to(device_name)
        one_hot.scatter_(2, role_t[..., None], torch.from_numpy(valid).to(device_name)[..., None].float())

        def per_role(values):
            return (values[..., None] * one_hot).sum((0, 1)).cpu().numpy()

        count += one_hot.sum((0, 1)).cpu().numpy()
        final_log = F.log_softmax(run.logits.float(), -1)
        nll_sum += per_role(-final_log.gather(-1, y[..., None])[..., 0])
        for site, source, _ in pairs:
            delta = (states[site] - states[source]).norm(dim=-1) / states[source].norm(dim=-1).clamp_min(1e-6)
            update_sum[site] += per_role(delta)
        if looped:
            for k in range(1, runner.depth_steps + 1):
                core = states[f'L6@{k}']
                if k > 1:
                    previous = states[f'L6@{k - 1}']
                    pass_sum[k] += per_role((core - previous).norm(dim=-1) / core.norm(dim=-1).clamp_min(1e-6))
                exit_log = F.log_softmax(early_exit_logits(model, core).float(), -1)
                kl = (final_log.exp() * (final_log - exit_log)).sum(-1)
                exit_kl[k] += per_role(kl)
                exit_top1[k] += per_role((exit_log.argmax(-1) == final_log.argmax(-1)).float())
                if k == runner.depth_steps:
                    last_exit_error = max(last_exit_error, float((exit_log - final_log).abs().max()))
        print(f'{name}: rows {start + b}/{len(rows)}', flush=True)
    per = lambda sums: {ROLES[i]: float(sums[i] / count[i]) for i in range(n_roles) if count[i]}
    result = dict(arm=name, depth_steps=runner.depth_steps,
                  characters={ROLES[i]: int(count[i]) for i in range(n_roles)},
                  nll=per(nll_sum), update={site: per(v) for site, v in update_sum.items()},
                  kinds={site: kind for site, _, kind in pairs})
    if looped:
        result['pass_change'] = {str(k): per(v) for k, v in pass_sum.items()}
        result['exit_kl'] = {str(k): per(v) for k, v in exit_kl.items()}
        result['exit_top1'] = {str(k): per(v) for k, v in exit_top1.items()}
        result['last_exit_max_logprob_error'] = last_exit_error
        if last_exit_error > 1e-3:
            raise AssertionError(f'Exiting after the last pass differs from the final prediction ({last_exit_error})')
    out = RESULTS / 'latents'
    out.mkdir(parents=True, exist_ok=True)
    (out / f'{name}.json').write_text(json.dumps(result, indent=2) + '\n')
    print(f'{name}: NLL by role ' + ', '.join(f'{r} {v:.3f}' for r, v in result['nll'].items()), flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--arms', nargs='+', default=['random_init', 'transformer', 'temporal', 'depth', 'hybrid'])
    parser.add_argument('--rows', type=int, default=40)
    parser.add_argument('--batch-rows', type=int, default=4)
    parser.add_argument('--device', default=device())
    args = parser.parse_args()
    rows, _, meta = load_rows()
    rows = rows[TRAIN_ROWS:TRAIN_ROWS + args.rows]
    for name in args.arms:
        analyse(name, rows, meta, args.device, args.batch_rows)


if __name__ == '__main__':
    main()
