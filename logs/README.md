# Encoder operations

Read [the Claude handoff](../CLAUDE_HANDOFF.md) first. The live pipeline updates
`ASTRA_STATUS.md`, `encoder_pipeline.log`, and `encoder_experiments.jsonl` here.
These mutable files and PIDs are deliberately untracked. A versioned snapshot is
in `handoff/encoder/logs/`, with its capture time and inventory in
`handoff/encoder/checkpoint_inventory.json`.

- `run_encoder.sh`: thread settings and encoder entry point.
- `run_encoder_durable.sh`: nohup/setsid launch, PID capture, resource supervisor,
  and genuine background activity reporting for the current JupyterHub server.
- `supervise_encoder.py`: observe one existing PID, log resources, and finish
  audited delivery after successful production/validation. Never starts inference.
- `report_encoder_activity.py`: use the current server's environment-provided
  activity endpoint and token; credentials are never written to files or logs.
  It reports only while the pipeline PID remains alive and stops on access denial.
- `RECOVERY_AUDIT.md`: prior termination/recovery details. Read the more recent
  handoff and live status for current progress and PIDs.

Always inspect the host processes before launching. Do not start another job if
the existing encoder is active. `nohup` survives a terminal disconnect but cannot
survive termination of its enclosing notebook service. This host's notebook
service uses `KillMode=control-group`; the activity reporter lets JupyterHub know
that real computation is still active, without changing server policy. A forced
restart still requires checkpoint resumption.
