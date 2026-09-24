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
    disk_gb = config['volume_gb'] + config['container_disk_gb']
    disk_rate = disk_gb * config['storage_usd_per_gb_month'] / HOURS_PER_MONTH
    return sum(pod_hours(pod, at) * ((pod.get('cost_per_hr') or config['planning_rate_usd_per_hr']) + disk_rate)
               for pod in pods)


def open_pods(pods):
    return [pod for pod in pods if not pod.get('ended_utc')]


def must_stop_all(spend, burn_per_hr, config):
    """Stop before the cap: leave room for another monitoring interval at the current burn."""
    return spend + burn_per_hr * config['stop_margin_hours'] >= config['spend_cap_usd']


def projected_campaign_cost(remaining_arm_hours, config):
    """Estimate the remaining GPU and disk cost, plus a configured margin."""
    disk_gb = config['volume_gb'] + config['container_disk_gb']
    disk_rate = disk_gb * config['storage_usd_per_gb_month'] / HOURS_PER_MONTH
    base = sum(remaining_arm_hours) * (config['planning_rate_usd_per_hr'] + disk_rate)
    return base * (1 + config['projected_budget_margin_fraction'])


def acquire_blocker(spend, burn_per_hr, balance, pods_in_use, config, *, remaining_arm_hours=None):
    """Reason not to start another arm, or None."""
    rate = config['planning_rate_usd_per_hr']
    if pods_in_use >= config['max_pods']:
        return 'all Pod slots are in use'
    projected = (pods_in_use + 1) * config['projected_hours_per_arm'] * rate
    if spend + projected > config['spend_cap_usd']:
        return f'starting this arm could exceed the spend cap (${spend:.2f} spent, ${projected:.2f} projected)'
    reserve = (burn_per_hr + rate) * config['min_balance_hours']
    remaining_arm_hours = (remaining_arm_hours if remaining_arm_hours is not None
                           else [config['projected_hours_per_arm']])
    campaign = projected_campaign_cost(remaining_arm_hours, config)
    needed = max(reserve, campaign)
    if balance is not None and balance < needed:
        return (f'account balance ${balance:.2f} is below the projected remaining campaign budget '
                f'${needed:.2f} (including storage and margin)')
    return None


def decide(arm, observation, config, *, blocker=None):
    """Choose one action for one arm.

    ``observation`` is None when the arm has no Pod. ``blocker`` is the
    account-level reason, if any, that a pending arm may not acquire a Pod.
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
        if arm.get('unreachable_ticks', 0) + 1 >= config['stall_ticks']:
            return 'alert', f"Pod unreachable: {observation['error']}"
        return 'unreachable', f"Pod unreachable: {observation['error']}"
    if not observation['gpu_ok']:
        return 'alert', 'GPU no longer initializes (cuInit failed); keep the Pod and recover the checkpoint'
    integrity = observation['integrity']
    if integrity['protocol_sha256'] != observation['frozen_protocol_sha256']:
        return 'alert', 'the Pod\'s protocol.json differs from the frozen protocol'
    if not integrity['ok']:
        return 'alert', 'integrity check failed: ' + '; '.join(integrity['problems'])
    if observation['disk_used_fraction'] >= config['disk_alert_fraction']:
        return 'alert', f"volume is {observation['disk_used_fraction']:.0%} full"
    if observation['done']:
        return 'collect', 'training and evaluation finished'
    if observation['job_alive']:
        step = observation['last_step']
        if step == study.UPDATES:
            return 'healthy', 'training complete; evaluating retained checkpoints'
        if step is not None and step == arm.get('last_step'):
            if arm.get('stall_ticks', 0) + 1 >= config['stall_ticks']:
                return 'alert', f'no training progress since step {step}'
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
