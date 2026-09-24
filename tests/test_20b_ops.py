from datetime import datetime, timedelta, timezone
from importlib import import_module
import json

import pytest


cli = import_module('experiments.long_runs.20B_recurrence.ops.cli')
policy = import_module('experiments.long_runs.20B_recurrence.ops.policy')
pods = import_module('experiments.long_runs.20B_recurrence.ops.pods')
study = import_module('experiments.long_runs.20B_recurrence.study')
runner = import_module('experiments.long_runs.20B_recurrence.run')


def _config(**overrides):
    return dict(cli.DEFAULT_CONFIG, **overrides)


def _arm(phase='running', **overrides):
    arm = cli.new_state()['arms']['hybrid']
    arm.update(phase=phase, pod_id='pod', pod_host='host', **overrides)
    return arm


def _integrity(ok=True, exists=True, valid=True, step=4_000, protocol='frozen', problems=()):
    return dict(ok=ok, protocol_sha256=protocol, problems=list(problems),
                checkpoint=dict(exists=exists, valid=valid, step=step, sha256='x'))


def _observation(**overrides):
    base = dict(pod='RUNNING', gpu_ok=True, job_alive=True, done=False, last_step=1_000,
                disk_used_fraction=0.5, log_tail='', frozen_protocol_sha256='frozen',
                integrity=_integrity())
    return dict(base, **overrides)


def _decide(arm, observation=None, config=None, blocker=None):
    return policy.decide(arm, observation, config or _config(), blocker=blocker)[0]


# Policy ----------------------------------------------------------------------

def test_pending_arm_acquires_unless_blocked():
    assert _decide(_arm('pending')) == 'acquire'
    assert _decide(_arm('pending'), blocker='all Pod slots are in use') == 'wait'


@pytest.mark.parametrize('observation, action', [
    (None, 'alert'),
    (dict(pod='missing'), 'alert'),
    (dict(pod='EXITED'), 'alert'),
    (dict(pod='RUNNING_WITHOUT_GPU'), 'alert'),
    (_observation(gpu_ok=False), 'alert'),
    (_observation(integrity=_integrity(protocol='drifted')), 'alert'),
    (_observation(integrity=_integrity(ok=False, problems=['ckpt.pt is invalid'])), 'alert'),
    (_observation(disk_used_fraction=0.95), 'alert'),
    (_observation(done=True), 'collect'),
    (_observation(last_step=2_000), 'healthy'),
])
def test_running_arm_actions(observation, action):
    assert _decide(_arm(last_step=1_000), observation) == action


def test_stall_alerts_only_after_repeated_ticks_and_not_during_evaluation():
    assert _decide(_arm(last_step=1_000), _observation()) == 'stalled'
    assert _decide(_arm(last_step=1_000, stall_ticks=1), _observation()) == 'alert'
    finished = _observation(last_step=study.UPDATES)
    assert _decide(_arm(last_step=study.UPDATES, stall_ticks=5), finished) == 'healthy'


def test_unreachable_pod_alerts_only_after_repeated_ticks():
    observation = dict(pod='RUNNING', error='ssh timeout', frozen_protocol_sha256='frozen')
    assert _decide(_arm(), observation) == 'unreachable'
    assert _decide(_arm(unreachable_ticks=1), observation) == 'alert'


def test_restart_resumes_only_from_a_valid_checkpoint():
    enabled = _config(resume_enabled=True)
    stopped = _observation(job_alive=False)
    assert _decide(_arm(), stopped, enabled) == 'restart'
    assert _decide(_arm(), stopped) == 'alert'  # resume gate not passed
    assert _decide(_arm(restarts=3), stopped, enabled) == 'alert'
    # Never start fresh: a missing or invalid checkpoint always alerts.
    missing = _observation(job_alive=False, integrity=_integrity(exists=False, valid=False, step=None))
    assert _decide(_arm(), missing, enabled) == 'alert'


def test_release_needs_explicit_permission_and_alerts_are_sticky():
    assert _decide(_arm('collected')) == 'ready_to_release'
    assert _decide(_arm('collected'), config=_config(allow_release=True)) == 'release'
    assert _decide(_arm('alert', alert='x'), _observation(done=True)) == 'none'


# Spend -----------------------------------------------------------------------

def test_ledger_spend_includes_disk_and_open_pods():
    start = datetime(2026, 9, 25, tzinfo=timezone.utc)
    ledger = [
        dict(pod_id='a', cost_per_hr=0.34, created_utc=start.isoformat(),
             ended_utc=(start + timedelta(hours=10)).isoformat()),
        dict(pod_id='b', cost_per_hr=None, created_utc=start.isoformat()),
    ]
    config = _config(storage_usd_per_gb_month=0.2)
    disk_rate = 60 * 0.2 / policy.HOURS_PER_MONTH
    expected = 10 * (0.34 + disk_rate) + 20 * (0.34 + disk_rate)
    assert policy.ledger_spend(ledger, config, start + timedelta(hours=20)) == pytest.approx(expected)
    assert [pod['pod_id'] for pod in policy.open_pods(ledger)] == ['b']


def test_pods_stop_before_the_cap_is_reached():
    config = _config(spend_cap_usd=80, stop_margin_hours=1.5)
    assert not policy.must_stop_all(77.0, 1.36, config)   # 77 + 2.04 < 80
    assert policy.must_stop_all(78.0, 1.36, config)       # 78 + 2.04 >= 80


def test_acquire_blockers_cover_slots_cap_and_balance():
    config = _config(spend_cap_usd=80, max_pods=4, min_balance_hours=24)
    assert policy.acquire_blocker(0, 0, 100, 4, config) == 'all Pod slots are in use'
    assert 'spend cap' in policy.acquire_blocker(30, 1.0, 100, 3, config)
    assert 'balance' in policy.acquire_blocker(0, 1.0, 5.10, 0, config)
    assert policy.acquire_blocker(0, 0.34, 100, 1, config) is None


def test_pending_arms_wait_until_balance_covers_the_projected_campaign():
    config = _config()
    four_arm_hours = [config['projected_hours_per_arm']] * len(study.ARM_ORDER)
    projected = policy.projected_campaign_cost(four_arm_hours, config)
    assert projected == pytest.approx(75.28, abs=0.02)
    assert 'campaign budget' in policy.acquire_blocker(
        0, 0, 44.26, 0, config, remaining_arm_hours=four_arm_hours)
    assert policy.acquire_blocker(
        0, 0, 80, 0, config, remaining_arm_hours=four_arm_hours) is None


def test_remaining_campaign_hours_reduce_with_pod_elapsed_time():
    state = cli.new_state()
    started = datetime(2026, 9, 25, tzinfo=timezone.utc)
    state['arms']['hybrid'].update(phase='running', pod_id='pod-1')
    state['pods'].append(dict(pod_id='pod-1', created_utc=(started - timedelta(hours=20)).isoformat()))
    remaining = cli.remaining_arm_hours(state, _config(projected_hours_per_arm=48), started)
    assert sorted(remaining) == [28, 48, 48, 48]


# Configuration ---------------------------------------------------------------

def _valid_config_files(tmp_path, **overrides):
    key = tmp_path / 'key'
    key.write_text('key')
    bundle = tmp_path / 'bundle.tar.gz'
    bundle.write_bytes(b'bundle')
    (tmp_path / 'bundle.tar.gz.sha256').write_text(f'{cli.file_sha256(bundle)}  bundle.tar.gz\n')
    path = tmp_path / 'config.json'
    path.write_text(json.dumps(dict(dict(ssh_key=str(key), bundle=str(bundle)), **overrides)))
    return path, bundle


def test_config_is_validated_before_any_pod_is_rented(tmp_path):
    path, bundle = _valid_config_files(tmp_path)
    config = cli.load_config(path)
    assert config['bundle_sha256'] == cli.file_sha256(bundle)
    state = cli.new_state()
    cli.verify_bundle(config, state)
    assert state['bundle_verified']

    path, _ = _valid_config_files(tmp_path, arm_priority=['hybrid', 'hybrid', 'depth', 'temporal'])
    with pytest.raises(ValueError, match='each arm exactly once'):
        cli.load_config(path)
    path, _ = _valid_config_files(tmp_path, bundle=str(tmp_path / 'missing.tar.gz'))
    with pytest.raises(FileNotFoundError, match='Bundle'):
        cli.load_config(path)


def test_changed_bundle_fails_verification(tmp_path):
    path, bundle = _valid_config_files(tmp_path)
    config = cli.load_config(path)
    bundle.write_bytes(b'tampered')
    with pytest.raises(ValueError, match='sidecar'):
        cli.verify_bundle(config, cli.new_state())


# Remote plumbing -------------------------------------------------------------

def test_remote_output_is_read_between_markers_not_from_terminal_echo():
    echoed = 'root@pod:/# echo "__OPS_""BEGIN__"\r\n'
    output = f'motd\r\n{echoed}{pods.BEGIN}\r\n{{"cuinit": 0}}\r\n{pods.END}\r\nexit\r\n'
    assert json.loads(pods.extract_marked(output)) == {'cuinit': 0}
    with pytest.raises(RuntimeError):
        pods.extract_marked('connection closed')


def test_croc_code_parsing():
    assert pods.parse_croc_code("sending 'x' (1 kB)\ncode is: data-65c21d0ee399df46-6\n") == 'data-65c21d0ee399df46-6'
    assert pods.parse_croc_code('hashing...') is None


def test_first_start_is_fresh_and_restarts_resume():
    fresh, resumed = pods.job_script('depth', resume=False), pods.job_script('depth', resume=True)
    assert 'run train depth\n' in fresh
    assert 'run train depth --resume\n' in resumed
    for script in (fresh, resumed):
        assert 'run evaluate depth --step $step' in script
        assert ' '.join(str(step) for step in study.EVALUATION_CHECKPOINT_STEPS) in script
        assert script.rstrip().endswith('depth.done')


def test_ops_code_is_not_part_of_the_frozen_source():
    paths = set(runner.source_snapshot()['files'])
    assert not any(path.startswith('experiments/long_runs/20B_recurrence/ops/') for path in paths)
    assert 'experiments/long_runs/20B_recurrence/run.py' in paths


def test_pod_is_usable_only_with_working_gpu_and_new_enough_driver():
    assert pods.driver_supports(dict(cuinit=0, driver_cuda=13000), '13.0')
    assert not pods.driver_supports(dict(cuinit=0, driver_cuda=12080), '13.0')  # seen 2026-09-24
    assert not pods.driver_supports(dict(cuinit=999, driver_cuda=13000), '13.0')


def test_download_probe_requires_a_complete_range_and_minimum_speed():
    good = dict(status=206, transferred_bytes=50_000_000, mb_per_second=25.0)
    slow = dict(status=206, transferred_bytes=50_000_000, mb_per_second=1.0)
    truncated = dict(status=206, transferred_bytes=12_000_000, mb_per_second=25.0)
    ignored_range = dict(status=200, transferred_bytes=50_000_000, mb_per_second=25.0)
    assert pods.download_meets_minimum(good, 50_000_000, 20.0)
    assert not pods.download_meets_minimum(slow, 50_000_000, 20.0)
    assert not pods.download_meets_minimum(truncated, 50_000_000, 20.0)
    assert not pods.download_meets_minimum(ignored_range, 50_000_000, 20.0)


def test_bandwidth_probe_tracks_the_locked_python311_linux_torch_wheel():
    assert pods.locked_torch_wheel_url().endswith(
        'cp311-cp311-manylinux_2_28_x86_64.whl')


def test_candidate_health_rejects_a_slow_locked_torch_download(monkeypatch):
    candidate = pods.Pod('pod', 'host', '~/.runpod/ssh/key')
    monkeypatch.setattr(pods, 'locked_torch_wheel_url', lambda: 'https://example.invalid/torch.whl')
    monkeypatch.setattr(pods.Pod, 'gpu_check', lambda self: dict(cuinit=0, driver_cuda=13000))
    monkeypatch.setattr(pods.Pod, 'download_probe', lambda self, url, size:
                        dict(status=206, transferred_bytes=size, mb_per_second=1.0))
    usable, reason = candidate.check_usable('13.0', minimum_download_mb_per_second=20.0,
                                            download_probe_bytes=50_000_000)
    assert not usable
    assert '1.0 MB/s' in reason
