# Encoder recovery audit — 2026-09-27

Host inspection confirmed old PID 1308325 was absent; no encoder/production job or GPU compute was running. Last durable checkpoint: India feed 3,194,880/4,717,565 rows. The log stopped without traceback; termination cause is unconfirmed.

Frozen selection is unchanged: calibration 0.967616211546784; single holdout 0.9665124399102869 (1,980 Source1 entities). Baseline exceeded, 0.98 not reached. No further holdout-based tuning or retraining was performed.

All 52 France shards (259,452 Source1) were preserved and verified by Parquet row counts. India Source1 embeddings and the partial feed are reused. Existing git modifications, raw datasets, all models and candidate caches were preserved.

Recovery additions: advisory production lock; skip fully completed countries; four feature workers (previous exact-equality benchmark); expanded reports/docs/package audit. All 61 repository tests passed. A first nohup launch exited before logging; a second launch with setsid and a brief verification delay survived.

Active pipeline PID: 1316607; logs/encoder_pipeline.pid and logs/encoder_pipeline.log.
Detached resource/delivery supervisor PID: see logs/encoder_supervisor.pid; logs/encoder_supervisor.log. The supervisor observes this one process only and never launches training/inference. It records process-tree RAM and GPU utilization every 30 seconds, then finishes audited packaging only after both official validator modes PASS.

Status: logs/ASTRA_STATUS.md (pipeline-owned, updated each embedding checkpoint/shard).
Final audit: logs/ENCODER_FINAL_REPORT.json and .md, after completion.
Model license/parameter evidence: logs/encoder_model_audit.json.
Package requirements: student_resource/README.md and Documentation_template.md inspected directly. Team identity requested; if unavailable it remains explicitly marked as not provided.

Resume: verify host processes and GPU first. If stopped, run the existing logs/run_encoder.sh with nohup + setsid. Do not launch if the PID is live. The pipeline lock also rejects concurrent copies. Do not delete partial or completed checkpoints.

## Second resumption

{
  "time": "2026-09-27 08:29:42 IST",
  "old_pid": 1316607,
  "new_pid": 1328767,
  "saved_india_source1": 440000,
  "saved_france_source1": 259452,
  "failure": "Pipeline and supervisor absent after notebook service restart; completed and partial shard files preserved.",
  "diagnosis": "Notebook service has KillMode=control-group and no memory limit; nohup cannot survive service termination. A transient independent system service requires unavailable administrator authentication.",
  "mitigation": "Resume under nohup/setsid; report genuine running-job activity through existing JUPYTERHUB_ACTIVITY_URL. Initial activity POST returned HTTP 200; reporter stops when pipeline exits."
}
