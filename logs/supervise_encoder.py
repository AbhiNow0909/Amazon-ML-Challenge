"""Observe one existing PID; never launch training or inference jobs."""
import json
import os
from pathlib import Path
import subprocess
import sys
import time

import psutil

ROOT = Path(__file__).resolve().parents[1]
pid = int((ROOT / 'logs/encoder_pipeline.pid').read_text())
process = psutil.Process(pid)
birth = process.create_time()
print(json.dumps({'supervising_pid': pid, 'process_created': birth}), flush=True)
while psutil.pid_exists(pid):
    try:
        if process.create_time() != birth or process.status() == psutil.STATUS_ZOMBIE:
            break
        rss = process.memory_info().rss
        for child in process.children(recursive=True):
            try:
                rss += child.memory_info().rss
            except psutil.NoSuchProcess:
                pass
        sample = {'time': time.strftime('%Y-%m-%d %H:%M:%S'), 'pid': pid, 'rss_gib': rss / 2**30}
        gpu = subprocess.run(['nvidia-smi', '--query-gpu=utilization.gpu,memory.used',
                              '--format=csv,noheader,nounits'], capture_output=True, text=True)
        if gpu.returncode == 0:
            utilization, memory = gpu.stdout.strip().split(',')
            sample.update(gpu_utilization=int(utilization), gpu_memory_mib=int(memory))
        with (ROOT / 'logs/encoder_resources.jsonl').open('a') as stream:
            stream.write(json.dumps(sample) + '\n')
        print(json.dumps(sample), flush=True)
    except psutil.NoSuchProcess:
        break
    time.sleep(30)

state = json.loads((ROOT / 'cache/encoder_native_v1/state.json').read_text())
if state.get('stage') != 'COMPLETE':
    print(json.dumps({'stopped_before_completion': True, 'state': state}), flush=True)
    sys.exit(1)
if not state.get('delivery_complete'):
    env = dict(os.environ, PYTHONPATH=str(ROOT / 'code/business_entity_resolution/src'))
    subprocess.run([sys.executable, '-u', '-m', 'business_entity_resolution.encoder_delivery'],
                   cwd=ROOT, env=env, check=True)
print('Supervision and audited delivery complete.', flush=True)
