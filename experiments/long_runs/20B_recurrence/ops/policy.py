"""Pure scheduling policy and spend accounting for the 20B arms (no I/O)."""
from datetime import datetime
from importlib import import_module

study = import_module('experiments.long_runs.20B_recurrence.study')

HOURS_PER_MONTH = 730


def pod_hours(pod, at):
    start = datetime.fromisoformat(pod['created_utc'])
    end = datetime.fromisoformat(pod['ended_utc']) if pod.get('ended_utc') else at
    return max(0.0, (end - start).total_seconds()) / 3600


def ledger_spend(pods, config, at):
    """GPU plus disk spend estimated from the local Pod ledger."""
    return sum(pod_hours(pod, at) * ((pod.get('cost_per_hr') or config['planning_rate_usd_per_hr']) + disk_rate(config))
               for pod in pods)


def open_pods(pods):
    return [pod for pod in pods if not pod.get('ended_utc')]


def must_stop_all(spend, burn_per_hr, config):
    """Stop before the cap: leave room for another monitoring interval at the current burn."""
    return spend + burn_per_hr * config['stop_margin_hours'] >= config['spend_cap_usd']


def disk_rate(config):
    disk_gb = config['volume_gb'] + config['container_disk_gb']
    return disk_gb * config['storage_usd_per_gb_month'] / HOURS_PER_MONTH


def arm_cloud(name, config):
    """Cloud type for one arm; ``arm_cloud_types`` overrides the default."""
    return config['arm_cloud_types'].get(name, config['cloud_type'])


def arm_rate(name, config):
    return config['cloud_rates_usd_per_hr'][arm_cloud(name, config)]


def arm_hours(name, config):
    return config['arm_projected_hours'].get(name, config['projected_hours_per_arm'])


def arm_cost(name, hours, config):
    """GPU plus disk cost of ``hours`` of this arm at its cloud's rate."""
    return hours * (arm_rate(name, config) + disk_rate(config))


def acquire_blocker(name, spend, burn_per_hr, balance, pods_in_use, config, *, committed_cost=0.0):
    """Reason not to start this arm now, or None.

    ``committed_cost`` is the projected remaining cost of arms already running.
    The spend cap must fit spend so far plus committed and this arm's cost; the
    balance must cover the same with margin (Runpod stops Pods at zero balance).
    Arms that are not yet started do not block this one; they wait their turn.
    """
    if pods_in_use >= config['max_pods']:
        return 'all Pod slots are in use'
    candidate = arm_cost(name, arm_hours(name, config), config)
    projected = committed_cost + candidate
    if spend + projected > config['spend_cap_usd']:
        return (f'starting this arm could exceed the spend cap (${spend:.2f} spent, '
                f'${projected:.2f} projected for running arms plus this one)')
    reserve = (burn_per_hr + arm_rate(name, config)) * config['min_balance_hours']
    needed = max(reserve, projected * (1 + config['projected_budget_margin_fraction']))
    if balance is not None and balance < needed:
        return (f'account balance ${balance:.2f} is below ${needed:.2f} needed for running arms '
                f'plus this one (including storage and margin); top up to start it')
    return None


def minutes_since(timestamp, at):
    return (at - datetime.fromisoformat(timestamp)).total_seconds() / 60 if timestamp else 0.0


def machine_id(pod_host):
    """Runpod's podHostId is '<pod id>-<machine id>'; the machine outlives Pods."""
    return pod_host.rsplit('-', 1)[-1] if pod_host else None


def recently_faulted(faulted_machines, machine, at, hours):
    """True if this machine failed a health check within the last ``hours``."""
    seen = faulted_machines.get(machine)
    return bool(seen) and minutes_since(seen, at) < hours * 60


def alert_key(alert):
    """The first line identifies an alert; attached log tails vary between ticks."""
    return alert.splitlines()[0]


def fresh_alerts(alerts, notified, at, repeat_hours):
    """Alerts not notified within ``repeat_hours``, plus the updated notified map."""
    fresh, updated = [], dict(notified)
    for alert in alerts:
        key = alert_key(alert)
        if minutes_since(notified.get(key), at) >= repeat_hours * 60 or key not in notified:
            fresh.append(alert)
            updated[key] = at.isoformat()
    return fresh, updated


def daily_summary_due(last_date, local_now, hour):
    """True once per local day, at the first tick at or after ``hour``."""
    return local_now.hour >= hour and last_date != local_now.date().isoformat()


def next_tick_minutes(arms, config):
    """Poll often while an arm still needs a Pod, then settle to hourly monitoring."""
    pending = any(arm['phase'] == 'pending' for arm in arms.values())
    return config['fast_tick_minutes'] if pending else config['slow_tick_minutes']


def decide(arm, observation, config, *, blocker=None, at=None):
    """Choose one action for one arm.

    ``observation`` is None when the arm has no Pod. ``blocker`` is the
    account-level reason, if any, that a pending arm may not acquire a Pod.
    Stall, startup and unreachable thresholds are in minutes (from ``at``),
    so they do not depend on how often ticks run.
    """
    phase = arm['phase']
    if phase in ('alert', 'released'):
        return 'none', arm.get('alert') or phase
    if phase == 'pending':
        return ('wait', blocker) if blocker else ('acquire', 'no Pod yet')
    if phase == 'collected':
        if config['allow_release']:
            return 'release', 'results collected and verified against the frozen protocol'
        return 'ready_to_release', 'results collected and verified; release the Pod manually'
    # phase == 'running'
    if observation is None or observation['pod'] == 'missing':
        return 'alert', 'Pod no longer exists; check for a surviving volume before doing anything'
    if observation['pod'] != 'RUNNING':
        return 'alert', (f"Pod is {observation['pod']}; keep it and use zero-GPU recovery "
                         'to retrieve the latest checkpoint')
    if observation.get('error'):
        if minutes_since(arm.get('unreachable_since_utc'), at) >= config['unreachable_minutes']:
            return 'alert', f"Pod unreachable for {config['unreachable_minutes']}+ min: {observation['error']}"
        return 'unreachable', f"Pod unreachable: {observation['error']}"
    if not observation['gpu_ok']:
        return 'alert', 'GPU no longer initializes (cuInit failed); keep the Pod and recover the checkpoint'
    integrity = observation['integrity']
    if integrity['protocol_sha256'] != observation['frozen_protocol_sha256']:
        return 'alert', 'the Pod\'s protocol.json differs from the frozen protocol'
    if not integrity['ok']:
        return 'alert', 'integrity check failed: ' + '; '.join(integrity['problems'])
    if ('power_limit_w' in observation and
            observation['power_limit_w'] < config['min_power_fraction'] * observation['power_default_w']):
        return 'alert', (f"GPU power capped at {observation['power_limit_w']:.0f} W of "
                         f"{observation['power_default_w']:.0f} W; training is running several times slower")
    if observation['disk_used_fraction'] >= config['disk_alert_fraction']:
        return 'alert', f"volume is {observation['disk_used_fraction']:.0%} full"
    if observation['done']:
        return 'collect', 'training and evaluation finished'
    if observation['job_alive']:
        step = observation['last_step']
        if step == study.UPDATES:
            return 'healthy', 'training complete; evaluating retained checkpoints'
        if step is None:
            if minutes_since(arm.get('job_started_utc'), at) >= config['startup_minutes']:
                return 'alert', f"no training step logged {config['startup_minutes']}+ min after the job started"
            return 'healthy', 'job starting'
        if step == arm.get('last_step'):
            if minutes_since(arm.get('progress_utc'), at) >= config['stall_minutes']:
                return 'alert', f"no training progress since step {step} for {config['stall_minutes']}+ min"
            return 'stalled', f'no progress since step {step}'
        return 'healthy', f'step {step}'
    checkpoint = integrity['checkpoint']
    if not (checkpoint['exists'] and checkpoint['valid']):
        return 'alert', 'job stopped and there is no valid checkpoint to resume; never start fresh automatically'
    if not config['resume_enabled']:
        return 'alert', 'job stopped before finishing; automatic resume is disabled until the resume gate passes'
    if arm['restarts'] >= config['max_restarts_per_arm']:
        return 'alert', f"job stopped again after {arm['restarts']} restarts"
    return 'restart', f"job stopped; resuming from step {checkpoint['step']}"
