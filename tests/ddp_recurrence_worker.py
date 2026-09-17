"""Two-rank gradient oracle, run by test_distributed.py."""
import copy
import os
import random

import torch
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel

from models.recurrent_2d import Recurrent2DGPT, RecurrentGPTConfig
from recurrence.schedule import sample_schedule


def main():
    torch.set_num_threads(1)
    dist.init_process_group('gloo')
    rank = int(os.environ['RANK'])
    torch.manual_seed(13)
    raw = Recurrent2DGPT(RecurrentGPTConfig(n_layer=4, n_prelude=1, n_core=1,
                                           n_coda=1, n_head=2, n_embd=16, block_size=8))
    reference = copy.deepcopy(raw)
    model = DistributedDataParallel(raw, find_unused_parameters=True)
    generator = torch.Generator().manual_seed(19)
    batches = torch.randint(32, (2, 2, 1, 8), generator=generator)
    # Switching between used and unused branches across an accumulated step is the hard case.
    for pairs in [((3, 1), (0, 0)), ((0, 0), (1, 3)), ((3, 0), (0, 3)), ((0, 3), (3, 0))]:
        model.zero_grad(set_to_none=True)
        reference.zero_grad(set_to_none=True)
        for micro, pair in enumerate(pairs):
            selected = [sample_schedule(*pair, random.Random(micro)) if rank == 0 else None]
            dist.broadcast_object_list(selected, src=0)
            model.require_backward_grad_sync = micro == 1
            x = batches[rank, micro]
            _, loss = model(x, x, schedule=selected[0])
            (loss / 2).backward()
            for other_rank in range(2):
                all_x = batches[other_rank, micro]
                _, expected_loss = reference(all_x, all_x, schedule=selected[0])
                (expected_loss / 4).backward()
        for name, parameter in raw.named_parameters():
            expected = dict(reference.named_parameters())[name].grad
            if expected is None:
                assert parameter.grad is None, name
            else:
                torch.testing.assert_close(parameter.grad, expected, rtol=2e-5, atol=2e-7, msg=name)
    dist.destroy_process_group()


if __name__ == '__main__':
    main()
