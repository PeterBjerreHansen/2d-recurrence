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
from datetime import datetime, timezone
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
    storage_usd_per_gb_month=0.20,
    stop_margin_hours=1.5,
    min_balance_hours=24,
    acquire_attempts_per_tick=2,
    resume_enabled=False,
    max_restarts_per_arm=3,
    stall_ticks=2,
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
                                 last_step=None, stall_ticks=0, unreachable_ticks=0, alert=None)
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


def remaining_arm_hours(state, config, at):
    """Conservative remaining runtime estimate for pending and running arms."""
    by_pod = {record['pod_id']: record for record in state['pods']}
    hours = []
    for arm in state['arms'].values():
        if arm['phase'] == 'pending':
            hours.append(config['projected_hours_per_arm'])
        elif arm['phase'] == 'running':
            record = by_pod.get(arm.get('pod_id'))
            elapsed = policy.pod_hours(record, at) if record else 0.0
            hours.append(max(0.0, config['projected_hours_per_arm'] - elapsed))
    return hours


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

def acquire(name, state, config):
    """Create Pods until one initializes CUDA; delete faulted ones at once."""
    for attempt in range(config['acquire_attempts_per_tick']):
        pod_id = pods.create_pod(config, f'20b-{name}')
        if pod_id is None:
            log_event(arm=name, event='no_capacity', attempt=attempt)
            return None
        record = dict(pod_id=pod_id, arm=name, cost_per_hr=None, created_utc=now().isoformat())
        state['pods'].append(record)
        save_state(state)
        info = pods.wait_for_host(pod_id)
        record['cost_per_hr'] = (info or {}).get('costPerHr')
        host = info['machine']['podHostId'] if info else None
        candidate = pods.Pod(pod_id, host, config['ssh_key']) if host else None
        usable, health_reason = (candidate.check_usable(
            config['min_cuda_version'],
            minimum_download_mb_per_second=config['minimum_download_mb_per_second'],
            download_probe_bytes=config['download_probe_bytes']) if candidate else (False, 'Pod host is unavailable'))
        if host and usable:
            log_event(arm=name, event='pod_acquired', pod=pod_id, host=host)
            return pod_id, host
        deleted, detail = pods.delete_pod(pod_id)
        if deleted:
            end_pod(state, pod_id)
        save_state(state)
        log_event(arm=name, event='faulted_pod', pod=pod_id, host=host, deleted=deleted,
                  health=health_reason, detail=detail)
        if not deleted:
            raise RuntimeError(f'Faulted Pod {pod_id} could not be deleted: {detail}')
    return None


def launch(name, arm, state, config):
    acquired = acquire(name, state, config)
    if acquired is None:
        return 'no healthy Pod available this hour'
    arm['pod_id'], arm['pod_host'] = acquired
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
    log_event(arm=name, event='started', pod=arm['pod_id'])
    return 'started'


def collect(name, arm, config):
    run = import_module('experiments.long_runs.20B_recurrence.run')
    return pods.collect(arm_pod(arm, config), name, INCOMING,
                        lambda results: run.arm_integrity(name, results, require_complete=True))


def alert_text(name, reason, observation):
    tail = (observation or {}).get('log_tail', '').strip().splitlines()[-5:]
    return f'{name}: {reason}' + (('\n    log: ' + '\n    log: '.join(tail)) if tail else '')


# --------------------------------------------------------------------------
# Tick

def tick(dry_run=False):
    if PAUSED.exists():
        return dict(utc=now().isoformat(), paused=True, arms={}, alerts=[])
    config = load_config()
    state = load_state()
    alerts = []
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
    for name in config['arm_priority']:
        arm = state['arms'][name]
        observation = None
        if arm['phase'] == 'running' and arm['pod_id']:
            observation = dict(pod=pods.pod_status(pods.pod_info(arm['pod_id'])),
                               frozen_protocol_sha256=frozen_protocol)
            if observation['pod'] == 'RUNNING':
                try:
                    observation.update(pods.observe(arm_pod(arm, config), name))
                except Exception as error:
                    observation['error'] = str(error)[:300]
        blocker = ('spend cap reached; Pods stopped' if stopping else
                   policy.acquire_blocker(spend, account['burn_per_hr'], account['balance'],
                                          len(policy.open_pods(state['pods'])), config,
                                          remaining_arm_hours=remaining_arm_hours(state, config, now())))
        action, reason = policy.decide(arm, observation, config, blocker=blocker)
        summary[name] = dict(phase=arm['phase'], action=action, reason=reason,
                             step=(observation or {}).get('last_step'))
        if action == 'none' and arm['phase'] == 'alert':
            alerts.append(f'{name}: {reason}')
        if dry_run or action in ('none', 'wait'):
            continue
        try:
            if action == 'acquire':
                summary[name]['reason'] = launch(name, arm, state, config)
            elif action == 'healthy':
                arm.update(last_step=observation['last_step'], stall_ticks=0, unreachable_ticks=0)
            elif action == 'stalled':
                arm['stall_ticks'] = arm.get('stall_ticks', 0) + 1
            elif action == 'unreachable':
                arm['unreachable_ticks'] = arm.get('unreachable_ticks', 0) + 1
            elif action == 'restart':
                arm['restarts'] += 1
                pods.start_job(arm_pod(arm, config), name, resume=True)
                log_event(arm=name, event='resumed', restarts=arm['restarts'], reason=reason)
            elif action == 'collect':
                destination = collect(name, arm, config)
                arm['phase'] = 'collected'
                log_event(arm=name, event='collected', destination=str(destination))
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
    if not dry_run:
        save_state(state)
    spend, spend_detail = spend_now(state, config)
    return dict(utc=now().isoformat(), dry_run=dry_run, spend_usd=round(spend, 2), spend_detail=spend_detail,
                spend_cap_usd=config['spend_cap_usd'], balance_usd=account['balance'],
                burn_usd_per_hr=account['burn_per_hr'], arms=summary, alerts=alerts)


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = parser.add_subparsers(dest='command', required=True)
    tick_parser = sub.add_parser('tick', help='Observe every arm and act once')
    tick_parser.add_argument('--dry-run', action='store_true', help='Observe and decide, but act on nothing')
    sub.add_parser('status', help='Print the local state')
    sub.add_parser('pause', help='Make ticks do nothing (use before any manual Pod work)')
    sub.add_parser('unpause', help='Let ticks act again')
    clear = sub.add_parser('clear-alert', help='Return an arm from alert after a human fix')
    clear.add_argument('arm', choices=study.ARM_ORDER)
    clear.add_argument('--phase', choices=('pending', 'running', 'collected'), required=True)
    args = parser.parse_args()
    if args.command == 'tick':
        with tick_lock():
            report = tick(args.dry_run)
        print(json.dumps(report, indent=2))
        if report.get('paused'):
            print('PAUSED: no action taken')
        for alert in report['alerts']:
            print(f'ALERT {alert}')
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
            state['arms'][args.arm].update(phase=args.phase, alert=None, stall_ticks=0, unreachable_ticks=0)
            save_state(state)
            log_event(arm=args.arm, event='alert_cleared', phase=args.phase)


if __name__ == '__main__':
    main()
