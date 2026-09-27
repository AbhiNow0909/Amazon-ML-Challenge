# Claude handoff: encoder-first entity resolution

## User's latest instruction

Stop the interactive supervision, save the work and context, and commit/push it to
the new branch `gpt-approach`. The detached encoder job was deliberately left
running; the user did not request killing it. Continue from disk, not this static
snapshot. Do not duplicate a live job, delete valid checkpoints, retrain an already
frozen model, or claim the submission is finished before validation and packaging.

Repository: `/home/jupyter-1nt23cb058/Amazon-ML-Challenge`.
Remote: `git@github.com:AbhiNow0909/Amazon-ML-Challenge.git`.
Work in this repository; standard model/package caches are the only exceptions.

## First actions: reconstruct live state

```sh
cd /home/jupyter-1nt23cb058/Amazon-ML-Challenge
cat logs/ASTRA_STATUS.md
cat logs/encoder_pipeline.pid logs/encoder_supervisor.pid logs/encoder_activity.pid
tail -30 logs/encoder_pipeline.log
tail -3 logs/encoder_resources.jsonl
tail -3 logs/encoder_activity.log
ps -eo pid,ppid,sid,stat,etime,pcpu,rss,args | rg 'encoder_pipeline|supervise_encoder|report_encoder_activity'
nvidia-smi
git status --short
df -h .
free -h
```

**Host access matters:** the restricted execution sandbox cannot see the real GPU
or host processes. An empty sandbox `ps` or failed sandbox `nvidia-smi` does not
prove a job is dead. Use host/escalated execution for those checks and GPU work.

At handoff the active PIDs were pipeline **1328767**, resource/delivery supervisor
**1328808**, and activity reporter **1328895**. The launcher has its own PID file.
Always re-read and verify these; they can become stale.

The exact captured progress, file sizes, complete markers and model hashes are in
[`handoff/encoder/checkpoint_inventory.json`](handoff/encoder/checkpoint_inventory.json).
At preparation time France was complete and India had passed **580,000 / 809,986**
Source1 rows; US had not started. Both structured and address-only India embeddings
were complete. Final TSVs, official validation, and the archive were still pending.

Some older fields in `ASTRA_STATUS.md` (especially `checkpoints_completed` and
`current_embedding`) are inherited descriptions; the newest shard markers,
`completed_source1_in_country`, and log events determine actual progress.

## Scores and validation integrity

| Measure | Frozen result |
|---|---:|
| Historical baseline | 0.965614 |
| Real target | 0.980000 |
| Best calibration macro F0.5 | 0.967616211546784 |
| Clean holdout macro F0.5 | **0.9665124399102869** |
| India holdout macro F0.5 | 0.9533537586405949 |
| US holdout macro F0.5 | 0.9748151992616739 |
| Holdout candidate pair recall | 0.9855178466939731 |
| Holdout oracle macro F0.5 | 0.9948744995749075 |
| Holdout pair precision / recall | 0.985111281657713 / 0.9388531304856641 |
| Holdout TP / FP / FN | 6418 / 97 / 418 |

The baseline was exceeded numerically; **0.98 was not reached**. Retrieval recall
and oracle score do not count as achieving the target. France has no validation
labels and no claimed score.

The development universe has 10,000 Source1 and 1,034,502 feed records, including
the selected entities' true links and distractors. This bounded feed is not a
hidden-test leaderboard estimate. Stable hashing of whole Source1 IDs with seed
260926 assigned 6,033 training, 1,987 calibration, and 1,980 holdout entities.
Every pair for one Source1 stays in the same split. Models, epochs, blend and
thresholds were selected on calibration; the holdout was evaluated **once** after
freezing. Do not reopen this holdout for iterative selection or call subsequent
tuning against its known results an untouched evaluation.

The frozen configuration is already in production. The current continuation
prioritizes finishing that submission, preserving honest validation and artifacts.

## Encoder mechanism

Entry point: `code/business_entity_resolution/src/business_entity_resolution/encoder_pipeline.py`.

1. **Normalize** official test records and partition by open-set country. The
   encoder imports `_partition_normalized_test` from `production_submission.py`
   solely to reuse normalization/country caches. It is not running the old
   LightGBM/XGBoost ensemble production model.
2. **Structured E5:** `intfloat/multilingual-e5-base`, pinned revision
   `d128750597153bb5987e10b1c3493a34e5a4502a`; MIT; 768 dimensions; max length 128;
   L2-normalized output; float16 disk storage; mixed-precision GPU inference.
   Source1 text is `query: business name: {name_full} | address: {address_full} | country: {country_norm}`;
   feed uses `passage:`. The independent address view uses the same format with
   an empty business name.
3. **Retrieve:** same-country exact blocked GPU cosine, structured top40 plus
   address-only top20. Query blocks are 256 and feed matmul blocks 100,000;
   running top-k avoids a full dense universe matrix. Bounded exact normalized
   full/core names and full addresses supplement retrieval; posting lists are
   capped at 100. Sparse rare-address-token recovery adds up to 20. Deduplicate
   `(qi, fi)` while preserving channel provenance. Do not arbitrarily cut the
   semantic top-k: it was selected before freezing.
4. **Features:** 53 float32 features listed exactly in `selection.json`. These
   include E5 rank/cosine/gaps, candidate count and competition, address-view rank,
   name/address exactness, token Jaccard/overlap, hashed character cosine, lengths,
   number/postcode conflicts, source flags and retrieval provenance. Sparse
   vectorization and factorized unique strings avoid Python loops over pairs.
   CPU features use four spawned workers in 50,000-pair chunks. The prior worker
   benchmark established exact equality with serial features.
5. **GPU XGBoost:** `encoder_xgb_address.json`, histogram/CUDA, depth 7, eta .04,
   subsample .85, colsample .9, min child weight 8, lambda 2, seed 260926.
   Best iteration 849 (prediction includes trees 0 through 849).
6. **Cross-encoder:** an XLM-R sequence classifier initialized from multilingual
   E5 base, **278,044,417 parameters**. Native paired inputs are
   `name | address | country` for each record. Max length 128, FP16 autocast,
   batch 128 at inference. First eight transformer layers are frozen; final four
   and classifier trained for two epochs. Hard negatives came from the highest
   XGBoost probabilities per training Source1 (up to five negatives per positive,
   with a minimum allowance for singleton entities). No in-sample CE stacker.
7. **Gate/blend:** cross-encode only pairs whose XGBoost probability is at least
   **0.02**. CE contribution is zero for gated-out pairs. Final score is
   **0.5 × XGBoost + 0.5 × cross-encoder**.
8. **Decode:** India threshold **0.5555462184873949**; US and pooled/global
   threshold **0.6977310924369747**. France uses the pooled/global threshold.
   Allow zero or multiple matches; no top-k cap on predictions. Every prediction
   is selected from that entity's saved candidate list.
9. **Checkpoint/serialize:** each 5,000-Source1 shard saves candidates, features,
   probabilities, output rows, and an atomic complete JSON marker. Country order
   is France, India, US. Final assembly restores original Source1 row order and
   writes chunked TSVs.

## Files and checkpoints — preserve them

Large artifacts are deliberately **local, Git-ignored, and not deleted**. The
encoder work directory was approximately 29 GiB at handoff; do not confuse a Git
checkout with a complete copy of model/data artifacts.

| Path | Contents |
|---|---|
| `cache/encoder_native_v1/selection.json` | Frozen model, feature order, decoder and gate |
| `cache/encoder_native_v1/production_configuration.json` | Frozen configuration fingerprint |
| `cache/encoder_native_v1/entity_split.parquet` | Immutable Source1 split assignments |
| `cache/encoder_native_v1/clean_holdout_metrics.json` | Single frozen holdout evaluation |
| `cache/encoder_native_v1/encoder_xgb_address.json` | Selected XGBoost weights |
| `cache/encoder_native_v1/cross_encoder/best/` | Selected CE weights and tokenizer |
| `cache/encoder_native_v1/development_*` | Development candidates/features/probabilities |
| `cache/encoder_native_v1/address_view/` | Development address-view embeddings |
| `cache/encoder_native_v1/full/{country}/` | Test embeddings and inference shards |
| `cache/production/e23_ensemble_production_v1/normalized/` | Reusable normalized country records and manifest |
| `cache/embeddings/e26_multilingual_e5_source1_f16.npy` | Existing 10,000 × 768 development embeddings |
| `cache/embeddings/e26_multilingual_e5_feed_f16.npy` | Existing 1,034,502 × 768 development embeddings |
| `cache/candidates/e26_*.parquet` | Historical E5/union candidates |

Each embedding has a JSON identity/row marker. A `.partial.npy` is preallocated at
the full shape: **only rows below the marker's saved offset are valid**. The
encoder flushes data before advancing the atomic marker every 8,192 rows. Do not
rename a partial file to complete based on its size. Completed country/shard files
are reused automatically. A production-config mismatch intentionally aborts.

Small frozen metadata and the entity split are copied into
`handoff/encoder/validation/` for review from Git. Model weight SHA256 values and
all local artifact paths/sizes are in the inventory; CE weights and embeddings
are not uploaded to GitHub. A new machine would need those local checkpoints
transferred separately or regenerated under the documented clean procedure.

## Logs, recovery and surviving disconnection

Live logs are under `logs/`; the versioned capture is under
[`handoff/encoder/logs/`](handoff/encoder/logs/). Consult the live files on this
machine. The committed files are a static handoff snapshot and will not advance.

- `encoder_pipeline.log`: complete stdout/stderr and stage timings.
- `encoder_experiments.jsonl`: training, calibration, frozen holdout and production events.
- `ASTRA_STATUS.md` / `cache/encoder_native_v1/state.json`: most recent stage/PID.
- `encoder_resources.jsonl`: sampled process-tree RSS and GPU usage.
- `encoder_supervisor.log`: detached monitor; no inference/training duplication.
- `encoder_activity.log`: Hub activity reports (HTTP 200 at handoff).
- `encoder_recovery_events.jsonl` and `RECOVERY_AUDIT.md`: termination/recovery history.
- `france_checkpoint_audit.json`: all 259,452 France rows passed ID membership,
  duplicate, source-row alignment and prediction-subset checks. This is **not**
  the official full-test validator. France has 19,964,316 candidates,
  1,399,877 predicted links and 5,316 empty predictions.
- `encoder_model_audit.json`: cached upstream MIT license evidence and parameter count.

Old PID 1308325 had stopped during India feed encoding. First recovery PID
1316607 completed all India embeddings and 440,000 inference rows before the
notebook service restarted and terminated it and its supervisor. Completed shards
and the next shard's saved candidates/features survived. Current PID 1328767
resumed these artifacts; no holdout evaluation or model training was repeated.

The notebook service is `jupyter-1nt23cb058.service`, with `KillMode=control-group`.
`nohup` and `setsid` survive terminal disconnection but not termination of that
service. A separate `systemd-run` service was attempted and **denied by OS
authentication**, not started. No admin settings were changed. This account has
no passwordless sudo or user systemd manager.

`logs/report_encoder_activity.py` now reports genuine ongoing pipeline activity
through the current server's existing `JUPYTERHUB_ACTIVITY_URL` using its existing
environment token. It stops when the pipeline exits or API access is denied. It
does not change idle-culler policy. **Never print, copy into a file, or commit that
token.** The current reports return 200. Forced service restarts still require a
resume; do not promise survival of machine/service shutdown.

If and only if host inspection confirms the encoder is stopped:

```sh
cd /home/jupyter-1nt23cb058/Amazon-ML-Challenge
nohup setsid sh logs/run_encoder_durable.sh >> logs/encoder_launcher.log 2>&1 < /dev/null &
echo $! > logs/encoder_launcher.pid
sleep 7
cat logs/encoder_pipeline.pid
tail -10 logs/encoder_pipeline.log
```

Use host/GPU-capable execution. Verify that the PID survives and the log advances.
The pipeline also holds an advisory lock in `cache/encoder_native_v1/pipeline.lock`.
Do not delete the lock/checkpoints or start duplicates to work around a live lock.

The launcher sets OMP/MKL/OpenBLAS/NumExpr threads to 8 and feature workers to 4.
Hardware: 32 CPU affinity cores, V100 32 GiB, roughly 125 GiB system RAM. Recent
India shards took ~53 seconds per 5,000 Source1 (20s retrieval, 14s features, 19s
scoring), with main-process peak RSS around 44 GiB and torch peak GPU allocation
around 8 GiB. Resource samples fluctuate with pipeline stages. More than 160 GiB
disk remained. Detect current resources again before any new workload.

Log wall timestamps changed timezone after the notebook restart (UTC versus IST).
Use `elapsed_seconds`, stage durations and epoch timestamps for runtime accounting;
do not subtract naive log date strings across sessions.

## Remaining work and completion gates

1. Supervise the **existing** process through remaining India and all US rows.
2. Inspect failures, repair the affected stage, and reuse valid checkpoints.
3. Assemble exactly **1,732,544** Source1 rows: France 259,452; India 809,986;
   US 663,106. Feed totals: Source2 4,887,273; Source3 5,082,316.
4. Produce `output/matching_results.tsv` with `source1_entity_id, matched_entity_ids`
   and `output/candidate_pairs.tsv` with `source1_entity_id, candidate_entity_ids`
   (tab-separated, one row per Source1, comma-separated ID lists).
5. Run the official commands below. The pipeline invokes both automatically after
   assembly, but failure raises an exception: diagnose, fix and rerun until PASS.
   ID/subset warnings must not be dismissed merely because exit status is zero.
6. Inspect official packaging requirements in `student_resource/README.md` and its
   `Documentation_template.md`; both were already read and are committed here.
7. Verify the final archive, its contents, output hashes and size. Exclude raw data,
   embeddings and caches. Include source, pinned requirements, README, frozen model
   artifacts and the methodology. Team name/members were requested but not provided;
   do not invent identities. Optional `submission_identity.json` supplies
   `team_name` and `team_members`; otherwise the technical filename is
   `encoder_submission.zip` and the document says identity is not provided.
8. Update status and write an honest final report. No completion claim before the
   outputs, validator passes and archive exist.

```sh
python student_resource/utils/validate_submission.py \
  --matching output/matching_results.tsv \
  --candidate output/candidate_pairs.tsv \
  --test-dir student_resource/dataset/test

python student_resource/utils/validate_submission.py \
  --matching output/matching_results.tsv \
  --candidate output/candidate_pairs.tsv \
  --test-dir student_resource/dataset/test --check-ids
```

`encoder_pipeline.py --stage all` reuses frozen validation and continues production.
`--stage full` explicitly runs frozen inference; `--stage package` assembles,
validates and packages existing shards. All currently initialize CUDA and acquire
the production lock, so do not run these alongside the live job.

`encoder_delivery.py` adds the filled methodology, complete report, output/model
metadata and audited archive, then sets `delivery_complete: true`. It does not
train or alter predictions. The pipeline calls it after both validators; the
supervisor can finish it if an older pipeline version omitted the call. Do not
treat the earlier `stage: COMPLETE` alone as proof of audited delivery: inspect
`delivery_complete`, the final report and archive. This delivery path has been
compiled but **has not yet run end-to-end**, because full inference is incomplete.
Future debugging may be needed; no validator PASS or final artifact is claimed now.

Final report paths: `logs/ENCODER_FINAL_REPORT.json` and `.md`.
Validator logs: `logs/encoder_validator_standard.log` and `encoder_validator_ids.log`.
Archive listing: `logs/encoder_archive_contents.txt`.
Report clean/calibration scores, baseline/target flags, architecture, retrieval
recall/oracle, country scores, decoder, row/pair/link/empty counts, runtime/resource
measurements and their coverage limits, validator commands/results, output paths,
archive path and size. Original France shards have combined inference timing only;
separate retrieval/scoring totals cover resumed India/US, not the whole run.

## Changes preserved in this branch

This branch preserves the previously uncommitted work, not just this handoff:
encoder pipeline/tests, production normalization/runner/tests, retrieval/blocking/
feature parallelization changes, requirements, README and notebook experiments.
Recovery added completed-country skipping, a production lock, detached launch and
resource/activity helpers, audited delivery/reporting, documentation and log snapshots.
Existing raw data and caches were preserved, not reset or deleted.

Verification: the repository suite had **61 passing tests**; a fresh suite result
is recorded in `handoff/encoder/tests.txt`. Tests cover existing matching,
retrieval, split and decoder behavior. GPU production itself has successfully
processed France and hundreds of thousands of India entities. The historical
tokenizer emits a regex warning when loading the saved XLM-R tokenizer; it was
already present during this frozen run. Do not silently change tokenization and
mix incompatible predictions across checkpoints.
