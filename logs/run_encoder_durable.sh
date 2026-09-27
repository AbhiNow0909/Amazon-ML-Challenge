#!/bin/sh
cd /home/jupyter-1nt23cb058/Amazon-ML-Challenge || exit 1
export PATH=/opt/tljh/user/bin:/usr/local/bin:/usr/bin:/bin
nohup setsid sh logs/run_encoder.sh >> logs/encoder_pipeline.log 2>&1 < /dev/null &
encoder_pid=$!
echo "$encoder_pid" > logs/encoder_pipeline.pid
sleep 5
if ! kill -0 "$encoder_pid" 2>/dev/null; then exit 1; fi
nohup python -u logs/supervise_encoder.py >> logs/encoder_supervisor.log 2>&1 < /dev/null &
observer_pid=$!
echo "$observer_pid" > logs/encoder_supervisor.pid
if [ -n "$JUPYTERHUB_ACTIVITY_URL" ]; then
    nohup python -u logs/report_encoder_activity.py >> logs/encoder_activity.log 2>&1 < /dev/null &
    echo $! > logs/encoder_activity.pid
fi
wait "$encoder_pid"
encoder_exit=$?
wait "$observer_pid"
exit "$encoder_exit"
