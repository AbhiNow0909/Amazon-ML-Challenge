"""Report real background compute activity via the current server's Hub API.

This does not change idle-culler policy, server configuration, or permissions.
It stops when the monitored pipeline exits. Credentials stay in the environment.
"""
import datetime
import json
import os
from pathlib import Path
import time
import urllib.error
import urllib.request

import psutil

root = Path(__file__).resolve().parents[1]
pid = int((root / 'logs/encoder_pipeline.pid').read_text())
process = psutil.Process(pid)
birth = process.create_time()
url = os.environ['JUPYTERHUB_ACTIVITY_URL']
token = os.environ['JUPYTERHUB_API_TOKEN']
server = os.environ.get('JUPYTERHUB_SERVER_NAME', '')
while psutil.pid_exists(pid):
    try:
        if process.create_time() != birth or process.status() == psutil.STATUS_ZOMBIE:
            break
        stamp = datetime.datetime.now(datetime.timezone.utc).isoformat().replace('+00:00', 'Z')
        body = {'last_activity': stamp, 'servers': {server: {'last_activity': stamp}}}
        request = urllib.request.Request(url, data=json.dumps(body).encode(), method='POST',
                   headers={'Authorization': 'token ' + token, 'Content-Type': 'application/json'})
        with urllib.request.urlopen(request, timeout=15) as response:
            print(json.dumps({'pid': pid, 'active_computation_reported': stamp,
                              'http_status': response.status}), flush=True)
    except psutil.NoSuchProcess:
        break
    except urllib.error.HTTPError as error:
        print(json.dumps({'http_status': error.code}), flush=True)
        if error.code in (401, 403):
            raise SystemExit('Activity API denied access; no policy changes attempted.')
    except urllib.error.URLError:
        print('Activity endpoint temporarily unavailable.', flush=True)
    time.sleep(60)
print('Pipeline exited; stopping activity reports.', flush=True)
