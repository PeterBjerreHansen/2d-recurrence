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


NOW = datetime(2026, 9, 25, 12, tzinfo=timezone.utc)


def _ago(minutes):
    return (NOW - timedelta(minutes=minutes)).isoformat()


def test_stall_and_startup_alerts_use_minutes_not_tick_counts():
    config = _config(stall_minutes=90, startup_minutes=60)
    decide = lambda arm, observation: policy.decide(arm, observation, config, at=NOW)[0]
    assert decide(_arm(last_step=1_000, progress_utc=_ago(20)), _observation()) == 'stalled'
    assert decide(_arm(last_step=1_000, progress_utc=_ago(95)), _observation()) == 'alert'
    finished = _observation(last_step=study.UPDATES)
    assert decide(_arm(last_step=study.UPDATES, progress_utc=_ago(600)), finished) == 'healthy'
    starting = _observation(last_step=None)
    assert decide(_arm(job_started_utc=_ago(10)), starting) == 'healthy'
    assert decide(_arm(job_started_utc=_ago(65)), starting) == 'alert'


def test_unreachable_pod_alerts_after_a_time_threshold():
    config = _config(unreachable_minutes=60)
    observation = dict(pod='RUNNING', error='ssh timeout', frozen_protocol_sha256='frozen')
    decide = lambda arm: policy.decide(arm, observation, config, at=NOW)[0]
    assert decide(_arm()) == 'unreachable'
    assert decide(_arm(unreachable_since_utc=_ago(30))) == 'unreachable'
    assert decide(_arm(unreachable_since_utc=_ago(61))) == 'alert'


def test_cadence_is_fast_until_every_arm_has_left_pending():
    config = _config(fast_tick_minutes=10, slow_tick_minutes=60)
    arms = cli.new_state()['arms']
    assert policy.next_tick_minutes(arms, config) == 10
    for name in ('hybrid', 'temporal', 'depth'):
        arms[name]['phase'] = 'running'
    assert policy.next_tick_minutes(arms, config) == 10
    arms['transformer']['phase'] = 'alert'
    assert policy.next_tick_minutes(arms, config) == 60


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
    assert policy.acquire_blocker('depth', 0, 0, 100, 4, config) == 'all Pod slots are in use'
    assert 'spend cap' in policy.acquire_blocker('depth', 30, 1.0, 100, 3, config, committed_cost=40)
    assert 'balance' in policy.acquire_blocker('depth', 0, 1.0, 5.10, 0, config)
    assert policy.acquire_blocker('depth', 0, 0.34, 100, 1, config, committed_cost=17) is None


def test_arm_cloud_types_price_each_arm_at_its_own_rate():
    config = _config(arm_cloud_types={'hybrid': 'SECURE'})
    assert policy.arm_cloud('hybrid', config) == 'SECURE'
    assert policy.arm_cloud('depth', config) == 'COMMUNITY'
    disk = policy.disk_rate(config)
    assert policy.arm_cost('hybrid', 48, config) == pytest.approx(48 * (0.74 + disk))
    assert policy.arm_cost('transformer', policy.arm_hours('transformer', config), config) == pytest.approx(24 * (0.34 + disk))
    # Hybrid on Secure fits an $83 balance on its own; not-yet-started arms do not block it.
    assert policy.acquire_blocker('hybrid', 0, 0, 83.39, 0, config) is None
    # With hybrid, temporal and depth committed, the transformer waits for a top-up.
    committed = policy.arm_cost('hybrid', 48, config) + 2 * policy.arm_cost('depth', 48, config)
    assert 'top up' in policy.acquire_blocker('transformer', 0, 0, 83.39, 3, config, committed_cost=committed)


def test_committed_cost_counts_only_running_arms_minus_elapsed_time():
    state = cli.new_state()
    started = datetime(2026, 9, 25, tzinfo=timezone.utc)
    state['arms']['hybrid'].update(phase='running', pod_id='pod-1')
    state['pods'].append(dict(pod_id='pod-1', created_utc=(started - timedelta(hours=20)).isoformat()))
    config = _config(arm_cloud_types={'hybrid': 'SECURE'})
    assert cli.committed_cost(state, config, started) == pytest.approx(policy.arm_cost('hybrid', 28, config))


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
    monkeypatch.setattr(pods.Pod, 'run', lambda self, script, timeout=900: '450.00, 450.00')
    monkeypatch.setattr(pods.Pod, 'download_probe', lambda self, url, size:
                        dict(status=206, transferred_bytes=size, mb_per_second=1.0))
    usable, reason = candidate.check_usable('13.0', minimum_download_mb_per_second=20.0,
                                            download_probe_bytes=50_000_000)
    assert not usable
    assert '1.0 MB/s' in reason


def test_early_tick_is_not_due_and_touches_nothing(tmp_path, monkeypatch):
    state = cli.new_state()
    state['next_due_utc'] = (datetime.now(timezone.utc) + timedelta(minutes=50)).isoformat()
    monkeypatch.setattr(cli, 'PAUSED', tmp_path / 'PAUSED')
    monkeypatch.setattr(cli, 'load_config', lambda: _config())
    monkeypatch.setattr(cli, 'load_state', lambda: state)
    monkeypatch.setattr(cli, 'verify_bundle', lambda config, state: None)
    monkeypatch.setattr(cli.pods, 'account', lambda: pytest.fail('an early tick must not act'))
    report = cli.tick()
    assert report['not_due'] and report['next_tick_minutes'] == 10
    # Two minutes of slack absorb scheduler jitter.
    state['next_due_utc'] = (datetime.now(timezone.utc) + timedelta(minutes=1)).isoformat()
    monkeypatch.setattr(cli.pods, 'account', lambda: (_ for _ in ()).throw(RuntimeError('reached account')))
    with pytest.raises(RuntimeError, match='reached account'):
        cli.tick()


def test_faulted_machines_are_remembered_for_a_while():
    assert policy.machine_id('30pka6640iqv7f-64411a5a') == '64411a5a'
    faulted = {'64411a5a': _ago(60)}
    assert policy.recently_faulted(faulted, '64411a5a', NOW, hours=12)
    assert not policy.recently_faulted(faulted, '6441176e', NOW, hours=12)
    assert not policy.recently_faulted({'64411a5a': _ago(13 * 60)}, '64411a5a', NOW, hours=12)


def test_a_failed_acquisition_stops_further_attempts_this_tick(monkeypatch):
    created = []
    monkeypatch.setattr(cli, 'save_state', lambda state: None)
    monkeypatch.setattr(cli, 'log_event', lambda **event: None)
    monkeypatch.setattr(cli.pods, 'create_pod', lambda config, name, cloud: created.append(name) or f'pod{len(created)}')
    monkeypatch.setattr(cli.pods, 'wait_for_host', lambda pod_id: dict(costPerHr=0.34, machine=dict(podHostId=f'{pod_id}-64411a5a')))
    monkeypatch.setattr(cli.pods, 'delete_pod', lambda pod_id: (True, 'deleted'))
    state = cli.new_state()
    state['faulted_machines'] = {'64411a5a': datetime.now(timezone.utc).isoformat()}
    context = {}
    assert cli.acquire('hybrid', state, _config(), context)[0] is None
    assert cli.acquire('temporal', state, _config(), context)[0] is None
    assert created == ['20b-hybrid']  # the second arm did not rent a Pod
    assert state['pods'][0]['ended_utc']


def test_bundle_test_exclusions_name_real_tests():
    from pathlib import Path
    for option in pods.BUNDLE_TEST_EXCLUSIONS:
        path = option.split('=', 1)[1].split('::')[0]
        assert Path(path).is_file(), option
        if '::' in option:
            assert f"def {option.split('::')[1]}(" in Path(path).read_text(), option


def test_power_capped_gpus_are_rejected_and_alert_mid_run():
    assert pods.parse_power('150.00, 450.00\n') == (150.0, 450.0)
    assert not pods.power_supports(150.0, 450.0, 0.9)   # seen 2026-09-25: ~3x slower updates
    assert not pods.power_supports(193.0, 450.0, 0.9)
    assert pods.power_supports(450.0, 450.0, 0.9)
    capped = _observation(power_limit_w=193.0, power_default_w=450.0)
    assert _decide(_arm(last_step=1_000), capped) == 'alert'
    assert _decide(_arm(last_step=1_000), _observation(power_limit_w=450.0, power_default_w=450.0,
                                                        last_step=2_000)) == 'healthy'


def test_candidate_health_rejects_a_power_capped_gpu(monkeypatch):
    candidate = pods.Pod('pod', 'host', '~/.runpod/ssh/key')
    monkeypatch.setattr(pods.Pod, 'gpu_check', lambda self: dict(cuinit=0, driver_cuda=13000))
    monkeypatch.setattr(pods.Pod, 'run', lambda self, script, timeout=900: '150.00, 450.00')
    monkeypatch.setattr(pods.Pod, 'download_probe', lambda self, url, size: pytest.fail('capped GPU must be rejected first'))
    usable, reason = candidate.check_usable('13.0')
    assert not usable and '150 W of 450 W' in reason
