import pytest
import torch

from inference.branch import branch_from_live, branch_step, select_branches
from inference.live import create_live_state, decode_live_step
from model import GPT, GPTConfig
from models.recurrent_2d import Recurrent2DGPT, RecurrentGPTConfig


def _model(mode):
    torch.manual_seed(211)
    if mode == 'baseline':
        return GPT(GPTConfig(n_layer=3, n_head=2, n_embd=8, block_size=16)).eval()
    return Recurrent2DGPT(RecurrentGPTConfig(
        n_layer=5, n_prelude=1, n_buffer=1, n_core=1, n_source=1, n_coda=1,
        n_head=2, n_embd=8, block_size=16, recurrence_mode=mode)).eval()


def _live_logits(model, tokens, depth_steps, strategy):
    state = create_live_state(model, depth_steps, strategy)
    return [decode_live_step(model, token[None], state) for token in tokens]


SETTINGS = [('baseline', None, None), ('temporal', 1, 'final_depth'),
            ('depth', 3, 'final_depth'), ('depth', 3, 'depth_specialized'),
            ('hybrid', 1, 'final_depth'), ('hybrid', 3, 'depth_specialized')]


@pytest.mark.parametrize('mode, depth_steps, strategy', SETTINGS)
def test_branches_match_independent_live_decoding(mode, depth_steps, strategy):
    model = _model(mode)
    prefix = torch.randint(32, (6,))
    # A two-level trie: the first two continuations share their first token.
    continuations = torch.tensor([[4, 9, 2], [4, 7, 7], [11, 3, 5]])
    root = create_live_state(model, depth_steps, strategy)
    for token in prefix:
        decode_live_step(model, token[None], root)
    root_length = root.cache.cache_length

    level_one = branch_from_live(root, 2)
    first = branch_step(model, torch.tensor([4, 11]), level_one)
    level_two = select_branches(level_one, [0, 0, 1])
    second = branch_step(model, continuations[:, 1], level_two)
    third = branch_step(model, continuations[:, 2], level_two)

    assert root.cache.cache_length == root_length, 'branching must not modify the shared prefix'
    for index, continuation in enumerate(continuations):
        expected = _live_logits(model, torch.cat((prefix, continuation)), depth_steps, strategy)
        parent = 0 if index < 2 else 1
        torch.testing.assert_close(first[parent], expected[len(prefix)][0], rtol=1e-5, atol=1e-5)
        torch.testing.assert_close(second[index], expected[len(prefix) + 1][0], rtol=1e-5, atol=1e-5)
        torch.testing.assert_close(third[index], expected[len(prefix) + 2][0], rtol=1e-5, atol=1e-5)


def test_branch_context_limit_is_enforced():
    model = _model('temporal')
    root = create_live_state(model, 1, 'final_depth')
    for token in torch.randint(32, (16,)):
        decode_live_step(model, token[None], root)
    with pytest.raises(ValueError, match='context'):
        branch_step(model, torch.tensor([1]), branch_from_live(root, 1))
