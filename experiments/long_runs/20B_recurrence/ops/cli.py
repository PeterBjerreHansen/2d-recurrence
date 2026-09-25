"""Schedule and monitor the 20B arms on Runpod Community GPUs.

Run ``tick`` about once an hour. Each tick observes every arm, chooses one
action per arm with ``policy.decide`` and carries it out: acquire a healthy
Pod, bootstrap and start an arm, resume a stopped job from its checkpoint,
collect and verify a finished arm, or raise an alert for a human. Alerts are
the only output an operator needs to read. See ``ops/README.md``.

Local state lives in the ignored ``results/ops/`` directory: ``config.json``
(copy ``ops/config.example.json``), ``state.json``, ``events.jsonl``,
``tick.lock``, ``PAUSED`` and ``incoming/``.
"""
import argparse
from contextlib import contextmanager
from datetime import datetime, timedelta, timezone
import fcntl
import hashlib
from importlib import import_module
import json
from pathlib import Path

study = import_module('experiments.long_runs.20B_recurrence.study')
policy = import_module('experiments.long_runs.20B_recurrence.ops.policy')
pods = import_module('experiments.long_runs.20B_recurrence.ops.pods')

OPS_ROOT = Path(study.RESULTS_ROOT) / 'ops'
CONFIG = OPS_ROOT / 'config.json'
STATE = OPS_ROOT / 'state.json'
EVENTS = OPS_ROOT / 'events.jsonl'
LOCK = OPS_ROOT / 'tick.lock'
PAUSED = OPS_ROOT / 'PAUSED'
INCOMING = OPS_ROOT / 'incoming'
FROZEN_PROTOCOL = Path(study.RESULTS_ROOT) / 'protocol.json'
DEFAULT_CONFIG = dict(
    spend_cap_usd=80.0,
    max_pods=4,
    arm_priority=['hybrid', 'temporal', 'depth', 'transformer'],
    bundle=None,
    gpu_id='NVIDIA GeForce RTX 4090',
    cloud_type='COMMUNITY',
    image='runpod/pytorch:1.0.2-cu1281-torch280-ubuntu2404',
    min_cuda_version='13.0',  # the locked torch build is cu130
    minimum_download_mb_per_second=20.0,
    download_probe_bytes=50_000_000,
    volume_gb=30,
    container_disk_gb=30,
    projected_hours_per_arm=48,
    projected_budget_margin_fraction=0.10,
    planning_rate_usd_per_hr=0.34,
    arm_cloud_types={},  # e.g. {'hybrid': 'SECURE'}; others use cloud_type
    cloud_rates_usd_per_hr={'COMMUNITY': 0.34, 'SECURE': 0.74},
    arm_projected_hours={'transformer': 24},
    storage_usd_per_gb_month=0.20,
    stop_margin_hours=1.5,
    min_balance_hours=24,
    acquire_attempts_per_tick=2,
    resume_enabled=False,
    max_restarts_per_arm=3,
    stall_minutes=90,
    startup_minutes=60,
    unreachable_minutes=60,
    fast_tick_minutes=10,
    faulted_machine_hours=12,
    min_power_fraction=0.9,
    # CPU models with desktop/workstation single-thread speed. The recurrent arms
    # are CPU-dispatch bound: on EPYC 7532 hosts they ran ~1.6x slower.
    fast_cpu_patterns=[r'Ryzen', r'Threadripper', r'Core\(TM\) i[79]-1[2-4]', r'Core\(TM\) Ultra',
                       r'Xeon\(R\) w[579]-', r'EPYC 9\d{3}', r'EPYC 4\d{3}'],
    fast_host_survey_minutes=60,  # 0 disables the hourly fast-host sample
    macos_notifications=True,  # post alerts and news to Notification Center
    notify_repeat_hours=6,  # re-notify a persisting alert at most this often
    daily_summary_hour=9,  # local hour of the once-a-day status notification
    slow_tick_minutes=60,
    disk_alert_fraction=0.9,
    allow_release=False,
    ssh_key='~/.runpod/ssh/runpodctl-ssh-key',
)


def now():
    return datetime.now(timezone.utc)


def file_sha256(path):
    digest = hashlib.sha256()
    with open(path, 'rb') as stream:
        for block in iter(lambda: stream.read(1 << 24), b''):
            digest.update(block)
    return digest.hexdigest()


# --------------------------------------------------------------------------
# Configuration and state

def load_config(path=CONFIG):
    """Load and validate everything that can be checked before renting a GPU."""
    if not path.is_file():
        raise FileNotFoundError(f'{path} is missing; copy ops/config.example.json there and edit it')
    config = dict(DEFAULT_CONFIG, **json.loads(path.read_text()))
    if sorted(config['arm_priority']) != sorted(study.ARM_ORDER):
        raise ValueError(f'arm_priority must list each arm exactly once: {study.ARM_ORDER}')
    if config['minimum_download_mb_per_second'] <= 0 or config['download_probe_bytes'] <= 0:
        raise ValueError('download bandwidth threshold and probe size must be positive')
    if not Path(config['ssh_key']).expanduser().is_file():
        raise FileNotFoundError(f"SSH key not found: {config['ssh_key']}")
    if not config['bundle']:
        raise ValueError('config.bundle must point to the frozen transfer bundle')
    bundle = Path(config['bundle']).expanduser()
    sidecar = Path(str(bundle) + '.sha256')
    if not bundle.is_file() or not sidecar.is_file():
        raise FileNotFoundError(f'Bundle or its .sha256 sidecar is missing: {bundle}')
    config['bundle_sha256'] = sidecar.read_text().split()[0]
    return config


def verify_bundle(config, state):
    """Hash the bundle against its sidecar; cached by size and mtime."""
    bundle = Path(config['bundle']).expanduser()
    stat = bundle.stat()
    key = f'{bundle}:{stat.st_size}:{stat.st_mtime_ns}'
    if state.get('bundle_verified') != key:
        if file_sha256(bundle) != config['bundle_sha256']:
            raise ValueError(f'{bundle} does not match its .sha256 sidecar')
        state['bundle_verified'] = key


def new_state():
    return dict(arms={name: dict(phase='pending', pod_id=None, pod_host=None, restarts=0,
                                 last_step=None, progress_utc=None, job_started_utc=None,
                                 unreachable_since_utc=None, alert=None)
                      for name in study.ARM_ORDER},
                pods=[], campaign_start_utc=None, bundle_verified=None)


def load_state(path=STATE):
    return json.loads(path.read_text()) if path.is_file() else new_state()


def save_state(state, path=STATE):
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix('.tmp')
    temporary.write_text(json.dumps(state, indent=2) + '\n')
    temporary.replace(path)


def log_event(**event):
    EVENTS.parent.mkdir(parents=True, exist_ok=True)
    with EVENTS.open('a') as stream:
        stream.write(json.dumps(dict(utc=now().isoformat(), **event)) + '\n')


@contextmanager
def tick_lock():
    LOCK.parent.mkdir(parents=True, exist_ok=True)
    with LOCK.open('w') as handle:
        try:
            fcntl.flock(handle, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            raise RuntimeError('Another tick or command holds the lock; do nothing this hour') from None
        yield


def arm_pod(arm, config):
    return pods.Pod(arm['pod_id'], arm['pod_host'], config['ssh_key'])


def end_pod(state, pod_id):
    for record in state['pods']:
        if record['pod_id'] == pod_id and not record.get('ended_utc'):
            record['ended_utc'] = now().isoformat()


def committed_cost(state, config, at):
    """Projected remaining GPU and disk cost of the arms that are running."""
    by_pod = {record['pod_id']: record for record in state['pods']}
    total = 0.0
    for name, arm in state['arms'].items():
        if arm['phase'] == 'running':
            record = by_pod.get(arm.get('pod_id'))
            elapsed = policy.pod_hours(record, at) if record else 0.0
            total += policy.arm_cost(name, max(0.0, policy.arm_hours(name, config) - elapsed), config)
    return total


# --------------------------------------------------------------------------
# Spend

def spend_now(state, config):
    """Total spend: the larger of the local ledger and Runpod's billing."""
    ledger = policy.ledger_spend(state['pods'], config, now())
    billed = pods.billed_since(state['campaign_start_utc']) if state['campaign_start_utc'] else 0.0
    return max(ledger, billed), dict(ledger=round(ledger, 2), billed=round(billed, 2))


def stop_everything(state, alerts):
    failures = []
    for record in policy.open_pods(state['pods']):
        stopped, detail = pods.stop_pod(record['pod_id'])
        log_event(event='cap_stop', pod=record['pod_id'], stopped=stopped, detail=detail)
        if not stopped:
            failures.append(f"{record['pod_id']} ({detail})")
    if failures:
        alerts.append('Spend cap: FAILED to stop ' + ', '.join(failures) + ' - stop them by hand now')
    else:
        alerts.append('Spend cap: every Pod was stopped (volumes kept)')


# --------------------------------------------------------------------------
# Actions

def _discard(name, state, pod_id, host, event, reason, cpu=None):
    deleted, detail = pods.delete_pod(pod_id)
    if deleted:
        end_pod(state, pod_id)
    save_state(state)
    log_event(arm=name, event=event, pod=pod_id, host=host, cpu=cpu, deleted=deleted, health=reason, detail=detail)
    if not deleted:
        raise RuntimeError(f'Faulted Pod {pod_id} could not be deleted: {detail}')


def acquire(name, state, config, context):
    """Create a Pod and keep it only if its GPU, driver and network are usable.

    Faulted machines are remembered for ``faulted_machine_hours``; a Pod that
    lands on one is deleted without further checks. After any failure the
    rest of this tick stops acquiring, because Community stock is usually the
    same few machines.
    """
    cloud = policy.arm_cloud(name, config)
    stopped = context.setdefault('stop_acquiring', {})  # per cloud type
    if stopped.get(cloud):
        return None, stopped[cloud]
    faulted = state.setdefault('faulted_machines', {})
    for attempt in range(config['acquire_attempts_per_tick']):
        pod_id = pods.create_pod(config, f'20b-{name}', cloud)
        if pod_id is None:
            log_event(arm=name, event='no_capacity', cloud=cloud, attempt=attempt)
            stopped[cloud] = f'no {cloud.title()} 4090 with a CUDA 13 driver available'
            return None, stopped[cloud]
        record = dict(pod_id=pod_id, arm=name, cloud=cloud, cost_per_hr=None, created_utc=now().isoformat())
        state['pods'].append(record)
        save_state(state)
        info = pods.wait_for_host(pod_id)
        record['cost_per_hr'] = (info or {}).get('costPerHr')
        host = info['machine']['podHostId'] if info else None
        machine = policy.machine_id(host)
        cpu = pods.cpu_name(info)
        if machine:
            state.setdefault('machine_cpus', {})[machine] = cpu
        if machine and policy.recently_faulted(faulted, machine, now(), config['faulted_machine_hours']):
            _discard(name, state, pod_id, host, 'known_faulted_pod', f'machine {machine} failed recently', cpu)
            stopped[cloud] = f'only known-faulted {cloud.title()} machines offered (e.g. {machine})'
            return None, stopped[cloud]
        candidate = pods.Pod(pod_id, host, config['ssh_key']) if host else None
        usable, health_reason = (candidate.check_usable(
            config['min_cuda_version'],
            minimum_download_mb_per_second=config['minimum_download_mb_per_second'],
            download_probe_bytes=config['download_probe_bytes'],
            min_power_fraction=config['min_power_fraction']) if candidate else (False, 'Pod host is unavailable'))
        if host and usable:
            log_event(arm=name, event='pod_acquired', pod=pod_id, host=host, cpu=cpu,
                      fast_cpu=policy.cpu_is_fast(cpu, config))
            return (pod_id, host, cpu), 'acquired'
        if machine:
            faulted[machine] = now().isoformat()
        _discard(name, state, pod_id, host, 'faulted_pod', health_reason, cpu)
        stopped[cloud] = f'offered {cloud.title()} machine {machine} failed its health check ({health_reason})'
        return None, stopped[cloud]
    return None, 'no healthy Pod this tick'


def survey_fast_host(state, config, alerts):
    """Rent one 4090 per cloud, read its CPU, check it if fast, and delete it.

    Runs at most every ``fast_host_survey_minutes`` while a running arm is on
    a slow CPU. Each sample lives about a minute. A healthy fast host raises a
    FAST HOST alert so a human can migrate a slow arm (pause, copy results,
    ``train <arm> --resume`` on the new Pod).
    """
    state['last_survey_utc'] = now().isoformat()
    faulted = state.setdefault('faulted_machines', {})
    for cloud in ('COMMUNITY', 'SECURE'):
        pod_id = pods.create_pod(config, 'fast-host-survey', cloud)
        if pod_id is None:
            log_event(event='survey_no_capacity', cloud=cloud)
            continue
        record = dict(pod_id=pod_id, arm='survey', cloud=cloud, cost_per_hr=None, created_utc=now().isoformat())
        state['pods'].append(record)
        save_state(state)
        usable, health, cpu, host = False, 'no host', None, None
        try:
            info = pods.wait_for_host(pod_id, timeout=300)
            record['cost_per_hr'] = (info or {}).get('costPerHr')
            host = info['machine']['podHostId'] if info else None
            machine, cpu = policy.machine_id(host), pods.cpu_name(info)
            if machine:
                state.setdefault('machine_cpus', {})[machine] = cpu
            fast = policy.cpu_is_fast(cpu, config)
            if fast and not policy.recently_faulted(faulted, machine, now(), config['faulted_machine_hours']):
                usable, health = pods.Pod(pod_id, host, config['ssh_key']).check_usable(
                    config['min_cuda_version'],
                    minimum_download_mb_per_second=config['minimum_download_mb_per_second'],
                    download_probe_bytes=config['download_probe_bytes'],
                    min_power_fraction=config['min_power_fraction'])
            else:
                health = 'slow CPU' if not fast else f'machine {machine} failed recently'
        finally:
            deleted, detail = pods.delete_pod(pod_id)
            if deleted:
                end_pod(state, pod_id)
            save_state(state)
            log_event(event='fast_host_survey', cloud=cloud, host=host, cpu=cpu, usable=usable,
                      health=health, deleted=deleted)
            if not deleted:
                alerts.append(f'Survey Pod {pod_id} could not be deleted ({detail}); delete it by hand')
        if usable:
            alerts.append(f"FAST HOST: healthy {cloud.title()} RTX 4090 with {cpu} "
                          f"(machine {policy.machine_id(host)}, ${record['cost_per_hr']}/h) is available now; "
                          'a slow-CPU arm could migrate to it')
            return


def launch(name, arm, state, config, context):
    acquired, reason = acquire(name, state, config, context)
    if acquired is None:
        return f'not started: {reason}'
    arm['pod_id'], arm['pod_host'], arm['cpu'] = acquired
    arm['phase'] = 'running'
    save_state(state)
    pod = arm_pod(arm, config)
    try:
        pods.bootstrap(pod, name, config['bundle'], config['bundle_sha256'], OPS_ROOT / f'send-{name}.log')
        pods.start_job(pod, name, resume=False)
    except Exception:
        # Keep the Pod for inspection, but stop it so it does not bill GPU time.
        stopped, detail = pods.stop_pod(arm['pod_id'])
        log_event(arm=name, event='bootstrap_failed_pod_stopped', stopped=stopped, detail=detail)
        raise
    arm.update(job_started_utc=now().isoformat(), progress_utc=now().isoformat())
    log_event(arm=name, event='started', pod=arm['pod_id'])
    return 'started'


def collect(name, arm, config):
    run = import_module('experiments.long_runs.20B_recurrence.run')
    return pods.collect(arm_pod(arm, config), name, INCOMING,
                        lambda results: run.arm_integrity(name, results, require_complete=True))


def check_bundle(config):
    """Run the Pod bootstrap's test command inside an extracted copy of the bundle.

    Catches bundle-only failures on this Mac before any GPU is rented. The
    training data file is skipped to save disk; no test needs it.
    """
    import subprocess
    import tempfile
    with tempfile.TemporaryDirectory(prefix='20b-bundle-check-') as directory:
        subprocess.run(['tar', '-xzf', str(Path(config['bundle']).expanduser()), '-C', directory,
                        '--exclude=data/chess_8M_v1/train.bin'], check=True)
        environment = {key: value for key, value in __import__('os').environ.items() if key != 'VIRTUAL_ENV'}
        subprocess.run(['uv', 'sync', '--frozen', '--python', '3.11', '-q'], cwd=directory, check=True, env=environment)
        completed = subprocess.run(pods.BUNDLE_TEST_COMMAND.split(), cwd=directory, env=environment,
                                   capture_output=True, text=True)
        return completed.returncode, completed.stdout[-2000:]


def notify(title, message, sound=None):
    """Post a macOS notification; never let a notification failure break a tick."""
    import subprocess
    quote = lambda text: json.dumps(' '.join(str(text).split()))  # AppleScript-compatible string
    script = f'display notification {quote(message[:240])} with title {quote(title)}'
    if sound:
        script += f' sound name {quote(sound)}'
    try:
        subprocess.run(['osascript', '-e', script], capture_output=True, timeout=20)
    except Exception as error:
        log_event(event='notify_failed', reason=str(error)[:200])


def send_notifications(report, state, config):
    """Notify new alerts, arm news, and a daily summary."""
    if not config['macos_notifications']:
        return
    fresh, state['notified'] = policy.fresh_alerts(
        report['alerts'], state.get('notified', {}), now(), config['notify_repeat_hours'])
    for alert in fresh:
        urgent = 'FAST HOST' in alert or 'failed' in alert or 'FAILED' in alert
        notify('20B campaign ALERT', policy.alert_key(alert), 'Glass' if urgent else 'Ping')
    for line in report.get('news', []):
        notify('20B campaign', line, 'Hero')
    local_now = datetime.now()
    if policy.daily_summary_due(state.get('last_daily_summary_date'), local_now, config['daily_summary_hour']):
        state['last_daily_summary_date'] = local_now.date().isoformat()
        arms = '; '.join(f"{name} {arm['phase']}" + (f" @{arm['step']}" if arm.get('step') else '')
                         for name, arm in report['arms'].items())
        notify('20B campaign daily status',
               f"{arms}. Spend ${report['spend_usd']} of ${report['spend_cap_usd']}, balance ${report['balance_usd']:.2f}")


def alert_text(name, reason, observation):
    tail = (observation or {}).get('log_tail', '').strip().splitlines()[-5:]
    return f'{name}: {reason}' + (('\n    log: ' + '\n    log: '.join(tail)) if tail else '')


# --------------------------------------------------------------------------
# Tick

def tick(dry_run=False, force=False):
    if PAUSED.exists():
        return dict(utc=now().isoformat(), paused=True, arms={}, alerts=[])
    config = load_config()
    state = load_state()
    due = state.get('next_due_utc')
    # A tick called early (for example, a 10-minute schedule during hourly
    # monitoring) does nothing; two minutes of slack absorb scheduler jitter.
    if due and not (dry_run or force) and now() < datetime.fromisoformat(due) - timedelta(minutes=2):
        return dict(utc=now().isoformat(), not_due=True, next_due_utc=due,
                    next_tick_minutes=policy.next_tick_minutes(state['arms'], config), arms={}, alerts=[])
    alerts = []
    news = []  # arms started, resumed or collected this tick
    if not dry_run:
        verify_bundle(config, state)
        state['campaign_start_utc'] = state['campaign_start_utc'] or now().isoformat()
    account = pods.account()
    spend, spend_detail = spend_now(state, config)
    frozen_protocol = file_sha256(FROZEN_PROTOCOL)
    if account['balance'] is not None and account['balance'] < account['burn_per_hr'] * config['min_balance_hours']:
        alerts.append(f"Runpod balance ${account['balance']:.2f} covers less than "
                      f"{config['min_balance_hours']} h at ${account['burn_per_hr']:.2f}/h; "
                      'top up - Runpod stops Pods at zero balance')
    stopping = policy.must_stop_all(spend, account['burn_per_hr'], config)
    if stopping and not dry_run:
        stop_everything(state, alerts)
    summary = {}
    context = {}  # per-tick scratch, e.g. stop acquiring after a faulted Pod
    for name in config['arm_priority']:
        arm = state['arms'][name]
        observation = None
        if arm['phase'] == 'running' and arm['pod_id']:
            info = pods.pod_info(arm['pod_id'])
            if not arm.get('cpu'):
                arm['cpu'] = pods.cpu_name(info)
            observation = dict(pod=pods.pod_status(info), frozen_protocol_sha256=frozen_protocol)
            if observation['pod'] == 'RUNNING':
                try:
                    observation.update(pods.observe(arm_pod(arm, config), name))
                except Exception as error:
                    observation['error'] = str(error)[:300]
        blocker = ('spend cap reached; Pods stopped' if stopping else
                   policy.acquire_blocker(name, spend, account['burn_per_hr'], account['balance'],
                                          len(policy.open_pods(state['pods'])), config,
                                          committed_cost=committed_cost(state, config, now())))
        action, reason = policy.decide(arm, observation, config, blocker=blocker, at=now())
        summary[name] = dict(phase=arm['phase'], action=action, reason=reason,
                             step=(observation or {}).get('last_step'), cpu=arm.get('cpu'),
                             fast_cpu=policy.cpu_is_fast(arm.get('cpu'), config) if arm.get('cpu') else None)
        if action == 'none' and arm['phase'] == 'alert':
            alerts.append(f'{name}: {reason}')
        if dry_run or action in ('none', 'wait'):
            continue
        try:
            if action == 'acquire':
                summary[name]['reason'] = launch(name, arm, state, config, context)
                if summary[name]['reason'] == 'started':
                    news.append(f"{name} started on {policy.arm_cloud(name, config).title()} ({arm.get('cpu')})")
            elif action == 'healthy':
                if observation['last_step'] != arm.get('last_step'):
                    arm.update(last_step=observation['last_step'], progress_utc=now().isoformat())
                arm['unreachable_since_utc'] = None
            elif action == 'stalled':
                arm['unreachable_since_utc'] = None
            elif action == 'unreachable':
                arm['unreachable_since_utc'] = arm.get('unreachable_since_utc') or now().isoformat()
            elif action == 'restart':
                arm['restarts'] += 1
                pods.start_job(arm_pod(arm, config), name, resume=True)
                arm.update(job_started_utc=now().isoformat(), progress_utc=now().isoformat())
                log_event(arm=name, event='resumed', restarts=arm['restarts'], reason=reason)
                news.append(f'{name} resumed: {reason}')
            elif action == 'collect':
                destination = collect(name, arm, config)
                arm['phase'] = 'collected'
                log_event(arm=name, event='collected', destination=str(destination))
                news.append(f'{name} finished and was collected')
            elif action == 'release':
                deleted, detail = pods.delete_pod(arm['pod_id'])
                if not deleted:
                    raise RuntimeError(f'delete failed: {detail}')
                end_pod(state, arm['pod_id'])
                arm['phase'] = 'released'
                log_event(arm=name, event='released', pod=arm['pod_id'])
            elif action == 'ready_to_release':
                alerts.append(f"{name}: {reason} (Pod {arm['pod_id']})")
            elif action == 'alert':
                arm['phase'], arm['alert'] = 'alert', reason
                alerts.append(alert_text(name, reason, observation))
                log_event(arm=name, event='alert', reason=reason)
        except Exception as error:
            arm['phase'], arm['alert'] = 'alert', f'{action} failed: {str(error)[:300]}'
            alerts.append(alert_text(name, arm['alert'], observation))
            log_event(arm=name, event='alert', reason=arm['alert'])
        save_state(state)
    if not (dry_run or stopping) and policy.survey_due(state, config, now()):
        try:
            survey_fast_host(state, config, alerts)
        except Exception as error:
            alerts.append(f'fast-host survey failed: {str(error)[:200]}')
            log_event(event='fast_host_survey_failed', reason=str(error)[:300])
    cadence = policy.next_tick_minutes(state['arms'], config)
    if not dry_run:
        state['next_due_utc'] = (now() + timedelta(minutes=cadence)).isoformat()
        save_state(state)
    spend, spend_detail = spend_now(state, config)
    report = dict(utc=now().isoformat(), dry_run=dry_run, next_tick_minutes=cadence,
                  pending_arms=[name for name, arm in state['arms'].items() if arm['phase'] == 'pending'],
                  spend_usd=round(spend, 2), spend_detail=spend_detail,
                  spend_cap_usd=config['spend_cap_usd'], balance_usd=account['balance'],
                  burn_usd_per_hr=account['burn_per_hr'], arms=summary, alerts=alerts, news=news)
    if not dry_run:
        send_notifications(report, state, config)
        save_state(state)
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = parser.add_subparsers(dest='command', required=True)
    tick_parser = sub.add_parser('tick', help='Observe every arm and act once')
    tick_parser.add_argument('--dry-run', action='store_true', help='Observe and decide, but act on nothing')
    tick_parser.add_argument('--force', action='store_true', help='Run a full tick even if the next one is not due yet')
    sub.add_parser('status', help='Print the local state')
    sub.add_parser('check-bundle', help="Run the Pod bootstrap's tests on an extracted bundle, locally")
    sub.add_parser('notify-test', help='Post a test macOS notification')
    sub.add_parser('pause', help='Make ticks do nothing (use before any manual Pod work)')
    sub.add_parser('unpause', help='Let ticks act again')
    clear = sub.add_parser('clear-alert', help='Return an arm from alert after a human fix')
    clear.add_argument('arm', choices=study.ARM_ORDER)
    clear.add_argument('--phase', choices=('pending', 'running', 'collected'), required=True)
    args = parser.parse_args()
    if args.command == 'tick':
        with tick_lock():
            report = tick(args.dry_run, args.force)
        print(json.dumps(report, indent=2))
        if report.get('paused'):
            print('PAUSED: no action taken')
        if report.get('not_due'):
            print(f"NOT DUE: next full tick at {report['next_due_utc']}")
        print(f"NEXT_TICK_MINUTES {report.get('next_tick_minutes')}")
        for alert in report['alerts']:
            print(f'ALERT {alert}')
    elif args.command == 'notify-test':
        notify('20B campaign', 'Test notification: alerts from the campaign tick will look like this.', 'Glass')
    elif args.command == 'check-bundle':
        code, tail = check_bundle(load_config())
        print(tail)
        print('BUNDLE TESTS PASSED' if code == 0 else f'BUNDLE TESTS FAILED (exit {code})')
        raise SystemExit(code)
    elif args.command == 'status':
        print(json.dumps(load_state(), indent=2))
    elif args.command in ('pause', 'unpause'):
        with tick_lock():
            if args.command == 'pause':
                PAUSED.parent.mkdir(parents=True, exist_ok=True)
                PAUSED.write_text(now().isoformat() + '\n')
            else:
                PAUSED.unlink(missing_ok=True)
            log_event(event=args.command)
    else:
        with tick_lock():
            state = load_state()
            state['arms'][args.arm].update(phase=args.phase, alert=None, unreachable_since_utc=None,
                                           progress_utc=now().isoformat())
            state['next_due_utc'] = None
            save_state(state)
            log_event(arm=args.arm, event='alert_cleared', phase=args.phase)


if __name__ == '__main__':
    main()
