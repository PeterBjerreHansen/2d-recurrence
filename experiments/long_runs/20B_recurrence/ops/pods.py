"""Runpod API, SSH-proxy shell, file transfers and arm lifecycle on a Pod."""
import base64
from dataclasses import dataclass
from datetime import datetime
import hashlib
from importlib import import_module
import json
import os
from pathlib import Path
import re
import shutil
import subprocess
import time
import tomllib

study = import_module('experiments.long_runs.20B_recurrence.study')

REMOTE_REPO = '/workspace/repo'
REMOTE_OPS = '/workspace/ops'
REMOTE_STAGE = '/root/stage'
STUDY_DIR = 'experiments/long_runs/20B_recurrence'
RUN_MODULE = 'experiments.long_runs.20B_recurrence.run'
BEGIN, END = '__OPS_BEGIN__', '__OPS_END__'
# The bundle has no Git metadata and no ops/ directory (it is excluded from the
# frozen source). These tests cannot pass there; they cover local tooling and
# freeze-time provenance, not training.
BUNDLE_TEST_EXCLUSIONS = (
    '--ignore=tests/test_20b_ops.py',
    '--deselect=tests/test_20b_recurrence_protocol.py::test_freeze_records_fixed_gate_and_is_repeatable',
    '--deselect=tests/test_update_schedule_study.py::test_source_snapshot_status_is_scoped_to_study_runtime',
)
BUNDLE_TEST_COMMAND = 'uv run pytest -q ' + ' '.join(BUNDLE_TEST_EXCLUSIONS)
GRAPHQL = 'https://api.runpod.io/graphql'


# --------------------------------------------------------------------------
# Runpod API and CLI

def _api_key():
    text = (Path.home() / '.runpod' / 'config.toml').read_text()
    match = re.search(r'apikey\s*=\s*["\']?([^"\'\n]+)', text)
    if not match:
        raise RuntimeError('No Runpod API key in ~/.runpod/config.toml')
    return match.group(1).strip()


def graphql(query):
    # The key travels in a header, never in the URL.
    completed = subprocess.run(
        ['curl', '-sS', '--max-time', '60', GRAPHQL, '-H', 'Content-Type: application/json',
         '-H', f'Authorization: Bearer {_api_key()}', '-d', json.dumps(dict(query=query))],
        capture_output=True, text=True, check=True)
    payload = json.loads(completed.stdout)
    if payload.get('errors') or 'data' not in payload:
        raise RuntimeError(f'Runpod API error: {completed.stdout[:300]}')
    return payload['data']


def runpodctl(*args, timeout=300):
    completed = subprocess.run(['runpodctl', *args], capture_output=True, text=True, timeout=timeout)
    return completed.returncode, completed.stdout + completed.stderr


def runpodctl_json(*args):
    # Deprecation notices go to stderr, so only stdout is parsed.
    completed = subprocess.run(['runpodctl', *args], capture_output=True, text=True, timeout=300)
    if completed.returncode != 0:
        raise RuntimeError(f"runpodctl {' '.join(args)} failed: {completed.stderr[-300:]}")
    return json.loads(completed.stdout)


def pod_info(pod_id):
    return graphql('query { pod(input:{podId:"%s"}) { id desiredStatus costPerHr '
                   'machine { podHostId } runtime { uptimeInSeconds gpus { id } } } }' % pod_id)['pod']


def pod_status(info):
    if info is None:
        return 'missing'
    if info['desiredStatus'] == 'RUNNING' and info['runtime']:
        return 'RUNNING' if info['runtime'].get('gpus') else 'RUNNING_WITHOUT_GPU'
    return info['desiredStatus']


def account():
    """Balance and current account-wide burn rate (GPU and storage)."""
    user = runpodctl_json('user')
    return dict(balance=user.get('clientBalance'), burn_per_hr=user.get('currentSpendPerHr') or 0.0)


def billed_since(start_utc):
    """Pod spend Runpod has billed since ``start_utc`` (may lag the ledger)."""
    start = datetime.fromisoformat(start_utc).strftime('%Y-%m-%dT%H:%M:%SZ')
    rows = runpodctl_json('billing', 'pods', '--start-time', start, '--bucket-size', 'hour')
    return sum(row.get('amount', 0.0) for row in rows)


def create_pod(config, name):
    code, output = runpodctl(
        'pod', 'create', '--name', name, '--cloud-type', config['cloud_type'],
        '--gpu-id', config['gpu_id'], '--image', config['image'],
        '--container-disk-in-gb', str(config['container_disk_gb']),
        '--volume-in-gb', str(config['volume_gb']), '--ports', '22/tcp',
        '--min-cuda-version', config['min_cuda_version'])
    match = re.search(r'"id"\s*:\s*"([a-z0-9]+)"', output)
    return match.group(1) if code == 0 and match else None


def stop_pod(pod_id):
    """Stop a Pod and confirm it; the volume is kept."""
    code, output = runpodctl('pod', 'stop', pod_id)
    if code != 0:
        return False, output[-200:]
    for _ in range(12):
        status = pod_status(pod_info(pod_id))
        if status in ('EXITED', 'missing'):
            return True, status
        time.sleep(5)
    return False, f'still {status} after stop'


def delete_pod(pod_id):
    code, output = runpodctl('pod', 'delete', pod_id)
    return code == 0, output[-200:]


def wait_for_host(pod_id, timeout=600):
    deadline = time.time() + timeout
    while time.time() < deadline:
        info = pod_info(pod_id)
        if info and info['runtime'] and (info.get('machine') or {}).get('podHostId'):
            return info
        time.sleep(10)
    return None


def driver_supports(check, min_cuda_version):
    major, minor = (int(part) for part in min_cuda_version.split('.'))
    return check['cuinit'] == 0 and check['driver_cuda'] >= major * 1000 + minor * 10


def locked_torch_wheel_url():
    """Return the exact Python 3.11 Linux wheel URL used for the bandwidth probe."""
    lock_path = Path(__file__).resolve().parents[4] / 'uv.lock'
    with lock_path.open('rb') as stream:
        packages = tomllib.load(stream)['package']
    torch = next((package for package in packages if package['name'] == 'torch'), None)
    suffix = 'cp311-cp311-manylinux_2_28_x86_64.whl'
    urls = [wheel['url'] for wheel in (torch or {}).get('wheels', [])
            if wheel['url'].endswith(suffix)]
    if len(urls) != 1:
        raise RuntimeError(f'uv.lock must contain one Linux Python 3.11 Torch wheel; found {len(urls)}')
    return urls[0]


def download_meets_minimum(result, expected_bytes, minimum_mb_per_second):
    """Require a complete byte-range response at the configured decimal MB/s rate."""
    return (result.get('status') == 206 and
            result.get('transferred_bytes') == expected_bytes and
            result.get('mb_per_second', 0) >= minimum_mb_per_second)


POWER_QUERY = ('nvidia-smi --query-gpu=power.limit,power.default_limit '
               '--format=csv,noheader,nounits')


def parse_power(text):
    """Parse the POWER_QUERY output into watts: (enforced limit, default limit)."""
    limit, default = (float(part) for part in text.strip().splitlines()[-1].split(','))
    return limit, default


def power_supports(limit, default, min_fraction):
    """Community hosts may cap GPU power (150-193 W of 450 W seen), making updates ~3x slower."""
    return limit >= min_fraction * default


def parse_croc_code(text):
    match = re.search(r'code is: (\S+)', text)
    return match.group(1) if match else None


# --------------------------------------------------------------------------
# Remote shell through the Runpod SSH proxy. The proxy is interactive-only,
# so the script is piped in base64 and its output is framed by markers.

def extract_marked(output):
    """Return text between the last BEGIN/END markers printed by a remote script."""
    text = output.replace('\r', '')
    start, end = text.rfind(BEGIN), text.rfind(END)
    if start < 0 or end < start:
        raise RuntimeError(f'Remote output was not framed: {text[-300:]}')
    return text[start + len(BEGIN):end].strip()


@dataclass(frozen=True)
class Pod:
    pod_id: str
    host: str
    ssh_key: str

    def run(self, script, timeout=900):
        # Splitting the markers keeps the terminal's echo from matching them.
        body = f'echo "{BEGIN[:6]}""{BEGIN[6:]}"\n{script}\necho "{END[:6]}""{END[6:]}"\n'
        encoded = base64.b64encode(body.encode()).decode()
        command = f'echo {encoded} | base64 -d > /tmp/ops_cmd.sh && bash /tmp/ops_cmd.sh; exit\n'
        completed = subprocess.run(
            ['ssh', '-tt', '-o', 'StrictHostKeyChecking=accept-new', '-o', 'ConnectTimeout=30',
             '-i', os.path.expanduser(self.ssh_key), f'{self.host}@ssh.runpod.io'],
            input=command, capture_output=True, text=True, timeout=timeout)
        return extract_marked(completed.stdout)

    def run_json(self, script, timeout=900):
        return json.loads(self.run(script, timeout).splitlines()[-1])

    def gpu_check(self):
        """``cuInit`` result and the driver's supported CUDA version (e.g. 13000)."""
        return self.run_json('''python3 - <<'PY'
import ctypes, json
cuda = ctypes.CDLL("libcuda.so.1")
version = ctypes.c_int(0)
cuda.cuDriverGetVersion(ctypes.byref(version))
print(json.dumps(dict(cuinit=cuda.cuInit(0), driver_cuda=version.value)))
PY''', timeout=120)

    def download_probe(self, url, probe_bytes):
        """Read a byte range from the locked Torch wheel and report its rate."""
        script = f'''python3 - <<'PY'
import json, time, urllib.request
request = urllib.request.Request({json.dumps(url)}, headers={{
    "Range": "bytes=0-{probe_bytes - 1}", "Accept-Encoding": "identity",
}})
started = time.perf_counter()
with urllib.request.urlopen(request, timeout=120) as response:
    status = response.status
    content = response.read({probe_bytes})
elapsed = max(time.perf_counter() - started, 1e-9)
print(json.dumps(dict(status=status, transferred_bytes=len(content), seconds=elapsed,
                      mb_per_second=len(content) / elapsed / 1_000_000)))
PY'''
        return self.run_json(script, timeout=180)

    def check_usable(self, min_cuda_version, *, minimum_download_mb_per_second=20.0,
                     download_probe_bytes=50_000_000, min_power_fraction=0.9):
        """Require CUDA support, an uncapped GPU and adequate access to the locked Torch CDN."""
        try:
            check = self.gpu_check()
            if not driver_supports(check, min_cuda_version):
                return False, f'cuInit={check["cuinit"]}, driver CUDA={check["driver_cuda"]}'
            limit, default = parse_power(self.run(POWER_QUERY, timeout=120))
            if not power_supports(limit, default, min_power_fraction):
                return False, f'GPU power capped at {limit:.0f} W of {default:.0f} W'
            result = self.download_probe(locked_torch_wheel_url(), download_probe_bytes)
            if not download_meets_minimum(result, download_probe_bytes,
                                          minimum_download_mb_per_second):
                speed = result.get('mb_per_second', 0)
                return False, (f'Torch wheel download was {speed:.1f} MB/s; '
                               f'requires {minimum_download_mb_per_second:.1f} MB/s')
            return True, f'Torch wheel download {result["mb_per_second"]:.1f} MB/s'
        except Exception as error:
            return False, f'health check failed: {str(error)[:200]}'

    def usable(self, min_cuda_version):
        """Compatibility wrapper for the default CUDA and bandwidth thresholds."""
        return self.check_usable(min_cuda_version)[0]

    def start_background(self, name, script):
        """Run ``script`` detached; its exit code lands in ``<name>.exit``."""
        encoded = base64.b64encode(script.encode()).decode()
        self.run(f'''mkdir -p {REMOTE_OPS}
echo {encoded} | base64 -d > {REMOTE_OPS}/{name}.sh
rm -f {REMOTE_OPS}/{name}.exit
setsid nohup bash -c 'bash {REMOTE_OPS}/{name}.sh > {REMOTE_OPS}/{name}.log 2>&1; echo $? > {REMOTE_OPS}/{name}.exit' > /dev/null 2>&1 &
echo started''')

    def wait_background(self, name, timeout, poll=30):
        deadline = time.time() + timeout
        while time.time() < deadline:
            result = self.run(f'cat {REMOTE_OPS}/{name}.exit 2>/dev/null || echo running')
            if result != 'running':
                return int(result), self.run(f'tail -n 30 {REMOTE_OPS}/{name}.log')
            time.sleep(poll)
        raise TimeoutError(f'{name} did not finish within {timeout} s')


# --------------------------------------------------------------------------
# Arm lifecycle

def arm_dir(name):
    return study.RUNS[name]['directory']


def job_script(name, resume):
    steps = ' '.join(str(step) for step in study.EVALUATION_CHECKPOINT_STEPS)
    return f'''set -euo pipefail
cd {REMOTE_REPO}
uv run python -m {RUN_MODULE} train {name}{' --resume' if resume else ''}
for step in {steps}; do
  uv run python -m {RUN_MODULE} evaluate {name} --step $step
done
touch {REMOTE_OPS}/{name}.done
'''


def observe(pod, name):
    """Process, progress, disk, log tail, and the frozen-protocol integrity check."""
    results = f'{REMOTE_REPO}/{STUDY_DIR}/{arm_dir(name)}/results'
    status = pod.run_json(f'''python3 - <<'PY'
import ctypes, json, os, shutil, subprocess
step = None
try:
    with open("{results}/metrics.jsonl") as stream:
        for line in stream:
            if '"event": "train"' in line:
                step = json.loads(line)["step"]
except FileNotFoundError:
    pass
alive = subprocess.run(["pgrep", "-f", "{REMOTE_OPS}/job-{name}.sh"], capture_output=True).returncode == 0
usage = shutil.disk_usage("/workspace")
try:
    with open("{REMOTE_OPS}/job-{name}.log", errors="replace") as stream:
        tail = stream.read()[-1500:]
except FileNotFoundError:
    tail = ""
print(json.dumps(dict(gpu_ok=ctypes.CDLL("libcuda.so.1").cuInit(0) == 0, job_alive=alive,
                      done=os.path.exists("{REMOTE_OPS}/{name}.done"), last_step=step,
                      disk_used_fraction=round(usage.used / usage.total, 3), log_tail=tail)))
PY''', timeout=180)
    status['power_limit_w'], status['power_default_w'] = parse_power(pod.run(POWER_QUERY, timeout=120))
    status['integrity'] = pod.run_json(f'cd {REMOTE_REPO} && uv run python -m {RUN_MODULE} integrity {name} 2>/dev/null',
                                       timeout=600)
    return status


def send_bundle(pod, bundle, digest, log_path):
    bundle = Path(bundle).expanduser()
    # Output goes to a file: an undrained pipe would stall the transfer.
    with log_path.open('w') as stream:
        sender = subprocess.Popen(['runpodctl', 'send', str(bundle)], stdout=stream,
                                  stderr=subprocess.STDOUT)
    deadline = time.time() + 1800  # hashing a large bundle takes a while
    while (code := parse_croc_code(log_path.read_text(errors='replace'))) is None:
        if sender.poll() is not None or time.time() > deadline:
            sender.kill()
            raise RuntimeError(f'runpodctl send gave no code: {log_path.read_text(errors="replace")[-300:]}')
        time.sleep(2)
    pod.start_background('receive', f'mkdir -p {REMOTE_STAGE} && cd {REMOTE_STAGE} && runpodctl receive {code}')
    status, tail = pod.wait_background('receive', timeout=3 * 3600)
    sender.wait(timeout=600)
    received = pod.run(f'sha256sum {REMOTE_STAGE}/{bundle.name} | cut -d" " -f1').splitlines()[-1]
    if status != 0 or received != digest:
        raise RuntimeError(f'Bundle transfer failed or its hash differs: {tail}')
    return bundle.name


def bootstrap(pod, name, bundle, digest, log_path):
    bundle_name = send_bundle(pod, bundle, digest, log_path)
    pod.start_background('bootstrap', f'''set -euo pipefail
mkdir -p {REMOTE_REPO}
tar -xzf {REMOTE_STAGE}/{bundle_name} -C {REMOTE_REPO}
rm -f {REMOTE_STAGE}/{bundle_name}
pip install -q uv
cd {REMOTE_REPO}
uv sync --frozen --python 3.11
{BUNDLE_TEST_COMMAND}
uv run python -m {RUN_MODULE} preflight {name}
''')
    status, tail = pod.wait_background('bootstrap', timeout=3600)
    if status != 0:
        raise RuntimeError(f'Bootstrap failed: {tail}')


def start_job(pod, name, resume):
    alive = pod.run(f'pgrep -f {REMOTE_OPS}/job-{name}.sh > /dev/null && echo yes || echo no')
    if alive.splitlines()[-1] == 'yes':
        raise RuntimeError(f'A worker for {name} is already running on this Pod')
    pod.start_background(f'job-{name}', job_script(name, resume))


def collect(pod, name, incoming, verify):
    """Copy the arm's results home, check transfer hashes, then ``verify`` them.

    ``verify(results_dir)`` must return an integrity report against the frozen
    protocol (``run.arm_integrity`` with ``require_complete``). Results are
    moved into place only if both checks pass.
    """
    directory = arm_dir(name)
    destination = Path(STUDY_DIR) / directory / 'results'
    if destination.exists():
        raise RuntimeError(f'{destination} already exists locally; not overwriting')
    pod.start_background('send-results', f'''set -euo pipefail
cd {REMOTE_REPO}/{STUDY_DIR}
cp {REMOTE_OPS}/job-{name}.log {directory}/results/ops_job.log
find {directory} -type f -exec sha256sum {{}} + > {REMOTE_OPS}/{name}.sha256
runpodctl send {directory}
''')
    code = None
    for _ in range(90):
        code = parse_croc_code(pod.run(f'cat {REMOTE_OPS}/send-results.log 2>/dev/null'))
        if code:
            break
        time.sleep(20)
    if not code:
        raise RuntimeError('The Pod did not produce a transfer code')
    manifest = pod.run(f'cat {REMOTE_OPS}/{name}.sha256')
    staging = Path(incoming) / name
    if staging.exists():
        shutil.rmtree(staging)
    staging.mkdir(parents=True)
    subprocess.run(['runpodctl', 'receive', code], cwd=staging, check=True, timeout=6 * 3600)
    for line in manifest.splitlines():
        digest, relative = line.split(maxsplit=1)
        local = staging / relative
        if not local.is_file() or hashlib.sha256(local.read_bytes()).hexdigest() != digest:
            raise RuntimeError(f'Transfer hash mismatch: {relative}')
    report = verify(staging / directory / 'results')
    if not report['ok']:
        raise RuntimeError('Collected results fail the frozen-protocol check: ' + '; '.join(report['problems']))
    destination.parent.mkdir(parents=True, exist_ok=True)
    (staging / directory / 'results').rename(destination)
    return destination
