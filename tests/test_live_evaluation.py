import torch

from evaluation.live_inference import evaluate_sequences, evaluate_teacher_forced
from inference.live import LiveInferenceSpec
from model import GPT, GPTConfig
from models.recurrent_2d import Recurrent2DGPT, RecurrentGPTConfig
from sample import generate


def recurrent_model(mode):
    return Recurrent2DGPT(RecurrentGPTConfig(
        n_layer=5, n_prelude=1, n_buffer=1, n_core=1, n_source=1, n_coda=1,
        n_head=2, n_embd=8, block_size=12, recurrence_mode=mode)).eval()


def test_live_evaluation_reports_reference_error_and_cache_metadata():
    torch.manual_seed(101)
    report = evaluate_sequences(recurrent_model('hybrid'), [torch.tensor([1, 2, 3, 4])],
                                depth_steps=2, kv_strategy='depth_specialized')
    assert report['execution'] == 'live'
    assert report['reference'] == 'slow_live_full_prefix_block_recomputation'
    assert report['depth_steps'] == 2
    assert report['kv_strategy'] == 'depth_specialized'
    assert report['max_abs_logit_error'] < 1e-4
    assert report['samples'][0]['cache_length'] == 4
    assert report['samples'][0]['cache_bytes'] > 0


def test_baseline_live_evaluation_uses_ordinary_policy():
    torch.manual_seed(103)
    report = evaluate_sequences(GPT(GPTConfig(
        n_layer=2, n_head=2, n_embd=8, block_size=12)).eval(),
        [torch.tensor([1, 2, 3])])
    assert report['recurrence_mode'] == 'baseline'
    assert report['kv_strategy'] == 'ordinary'
    assert report['max_abs_logit_error'] < 1e-5


def test_live_generation_report_labels_execution_and_depth_budget():
    torch.manual_seed(107)
    model = recurrent_model('hybrid')
    characters = ';1.e23456789abcdefgh '
    meta = {'stoi': {char: index for index, char in enumerate(characters)},
            'itos': {index: char for index, char in enumerate(characters)}}
    report = generate(model, meta, max_new_tokens=1, temperature=0,
                      live_spec=LiveInferenceSpec('hybrid', 2, 'final_depth'))
    assert report['execution'] == 'live'
    assert report['prefill'] == 'sequential_live'
    assert report['depth_budget'] == dict(kind='fixed_core_steps_per_token', value=2)
    assert report['kv_strategy'] == 'final_depth'
    assert report['cache_length'] == len(report['prompt']) + report['generated_characters']


def test_live_teacher_forced_evaluation_resets_each_row_and_reports_metrics():
    torch.manual_seed(109)
    model = recurrent_model('depth')
    rows = [(torch.tensor([1, 2, 3]), torch.tensor([2, 3, 4])),
            (torch.tensor([5, 6]), torch.tensor([6, 7]))]
    report = evaluate_teacher_forced(model, rows, depth_steps=2,
                                    kv_strategy='depth_specialized')
    assert report['row_count'] == 2
    assert report['target_count'] == 5
    assert report['cache_strategy'] == 'depth_specialized'
    assert report['temporal_feedback_enabled'] is False
    assert report['core_block_applications'] == 10
    assert report['physical_transformer_block_applications'] == 30
    assert report['cache_bytes'] > 0
    assert report['nll'] > 0
    assert 0 <= report['accuracy'] <= 1
