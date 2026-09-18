import torch

from inference.cache import LayerKVCache
from model import GPT, GPTConfig


def test_embed_step_matches_absolute_full_sequence_position():
    torch.manual_seed(71)
    model = GPT(GPTConfig(n_layer=1, n_head=2, n_embd=8, block_size=12)).eval()
    tokens = torch.randint(32, (2, 6))
    full = model.embed(tokens)
    for position in range(tokens.shape[1]):
        stepped = model.embed_step(tokens[:, position], position)
        torch.testing.assert_close(stepped, full[:, position:position + 1], rtol=0, atol=0)


def test_block_forward_step_matches_full_causal_outputs():
    torch.manual_seed(73)
    model = GPT(GPTConfig(n_layer=1, n_head=2, n_embd=8, block_size=12)).eval()
    tokens = torch.randint(32, (2, 6))
    full_inputs = model.embed(tokens)
    full_outputs = model.transformer.h[0](full_inputs)
    cache = LayerKVCache()
    stepped_outputs = []
    for position in range(tokens.shape[1]):
        stepped_outputs.append(model.transformer.h[0].forward_step(
            full_inputs[:, position:position + 1], cache, commit=True))
    stepped = torch.cat(stepped_outputs, dim=1)
    torch.testing.assert_close(stepped, full_outputs, rtol=1e-5, atol=1e-6)
    assert cache.length == tokens.shape[1]


def test_commit_false_keeps_history_but_attends_to_current_token():
    torch.manual_seed(79)
    model = GPT(GPTConfig(n_layer=1, n_head=2, n_embd=8, block_size=12)).eval()
    inputs = model.embed(torch.randint(32, (1, 3)))
    block = model.transformer.h[0]
    cache = LayerKVCache()
    block.forward_step(inputs[:, :1], cache, commit=True)
    before = cache.length
    held_output = block.forward_step(inputs[:, 1:2], cache, commit=False)
    assert cache.length == before
    committed_cache = LayerKVCache()
    block.forward_step(inputs[:, :1], committed_cache, commit=True)
    committed_output = block.forward_step(inputs[:, 1:2], committed_cache, commit=True)
    torch.testing.assert_close(held_output, committed_output, rtol=0, atol=0)
    assert committed_cache.length == before + 1
