"""Complete the encoder audit report and official submission package.

This does not train, select, score, or modify predictions. It operates only on
completed outputs which have passed both official validator modes.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import re
import time
import zipfile
from pathlib import Path

from .config import REPO_ROOT


def write_json(path, value):
    tmp = path.with_suffix(path.suffix + '.tmp')
    tmp.write_text(json.dumps(value, indent=2) + '\n')
    tmp.replace(path)


def sha256(path):
    digest = hashlib.sha256()
    with path.open('rb') as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b''):
            digest.update(block)
    return digest.hexdigest()


def measurements(root):
    events = []
    for line in (root / 'logs/encoder_experiments.jsonl').read_text().splitlines():
        try:
            events.append(json.loads(line))
        except json.JSONDecodeError:
            continue
    shards = [json.loads(p.read_text()) for p in
              (root / 'cache/encoder_native_v1/full').glob('*/shard_*.json')]
    assert shards and all(s.get('complete') for s in shards)
    totals = {key: sum(s[key] for s in shards) for key in
              ['source1_rows', 'candidate_pairs', 'predicted_pairs', 'empty_predictions']}
    assert totals['source1_rows'] == 1732544
    # Each progress rate measures the current encoding attempt cumulatively.
    # Split attempts on a path change, counter reset, or a reused checkpoint gap.
    attempts = []
    current = None
    for e in events:
        if e['event'] != 'embedding_progress' or '/full/' not in e.get('path', ''):
            continue
        if (current is None or e['path'] != current['path']
                or e['elapsed_seconds'] < current['elapsed_seconds']
                or e['rows'] <= current['rows']):
            if current is not None:
                attempts.append(current)
            current = dict(e, offset=max(0, e['rows'] - 8192))
        else:
            current.update(e)
    if current is not None:
        attempts.append(current)
    sessions = []
    elapsed = 0
    for e in events:
        value = e.get('elapsed_seconds', elapsed)
        if e['event'] == 'hardware' and elapsed:
            sessions.append(elapsed)
            elapsed = 0
        elapsed = max(elapsed, value)
    sessions.append(elapsed)
    embedding_seconds = sum((e['rows'] - e['offset']) / e['rows_per_sec'] for e in attempts)
    detailed = [s for s in shards if 'retrieval_seconds' in s]
    totals.update({
        'full_test_embedding_seconds_estimate': embedding_seconds,
        'embedding_timing_note': 'Summed encoding-attempt progress rates; excludes downtime and model loading.',
        'full_test_inference_seconds': sum(s['seconds'] for s in shards),
        'total_active_logged_seconds': sum(sessions),
        'total_runtime_note': 'Active runtime sums logged process sessions. Absolute historical log timestamps use different host timezones after a notebook restart, so wall-clock duration is not inferred from them. Package finishing time follows these measurements.',
        'retrieval_seconds_instrumented': sum(s['retrieval_seconds'] for s in detailed),
        'feature_seconds_instrumented': sum(s['feature_seconds'] for s in detailed),
        'scoring_seconds_instrumented': sum(s['inference_seconds'] for s in detailed),
        'timing_coverage_source1': sum(s['source1_rows'] for s in detailed),
        'timing_note': 'Original France shards record combined inference time only; separate retrieval/scoring totals cover resumed India and US.',
        'peak_main_process_ram_gib': max((s.get('peak_ram_gib', 0) for s in shards), default=0),
        'peak_torch_vram_gib': max([s.get('peak_vram_gib', 0) for s in shards] +
                                  [e.get('vram', e.get('peak_vram_gib', 0)) for e in events]),
    })
    resource_path = root / 'logs/encoder_resources.jsonl'
    samples = []
    if resource_path.exists():
        for line in resource_path.read_text().splitlines():
            try:
                samples.append(json.loads(line))
            except json.JSONDecodeError:
                continue
    totals['sampled_peak_process_tree_ram_gib'] = max((s.get('rss_gib', 0) for s in samples), default=None)
    totals['sampled_peak_device_vram_mib'] = max((s.get('gpu_memory_mib', 0) for s in samples), default=None)
    totals['resource_note'] = 'Resource samples cover resumed production; historical whole-process peaks were not recorded.'
    return totals


def deliver(root=REPO_ROOT):
    root = Path(root)
    work, logs = root / 'cache/encoder_native_v1', root / 'logs'
    metrics = json.loads((work / 'clean_holdout_metrics.json').read_text())
    config = json.loads((work / 'selection.json').read_text())
    previous = json.loads((logs / 'ENCODER_FINAL_REPORT.json').read_text())
    assert previous['validator'] == previous['id_check'] == 'PASS'
    validator_logs = {kind: (logs / f'encoder_validator_{kind}.log').read_text()
                      for kind in ['standard', 'ids']}
    assert all('PASS' in value for value in validator_logs.values())
    # Official ID and subset problems can be warnings, so do not hide them.
    assert not any(term in validator_logs['ids'].lower() for term in
                   ['unknown target', 'not found in', 'not in the candidate', 'not a subset',
                    'matched ids not present', 'skipping', 'check is off'])
    stats = measurements(root)
    recovery_file = logs / 'encoder_recovery_events.jsonl'
    recoveries = [json.loads(line) for line in recovery_file.read_text().splitlines()] if recovery_file.exists() else []
    assert stats['candidate_pairs'] == previous['candidate_pairs']
    assert stats['predicted_pairs'] == previous['predicted_links']
    identity_path = root / 'submission_identity.json'
    identity = json.loads(identity_path.read_text()) if identity_path.exists() else {}
    team = identity.get('team_name', 'Not provided')
    members = identity.get('team_members', 'Not provided')
    stem = re.sub(r'[^A-Za-z0-9_-]+', '_', identity.get('team_name', 'encoder')).strip('_') or 'encoder'
    archive = root / f'{stem}_submission.zip'
    command = ('python student_resource/utils/validate_submission.py '
               '--matching output/matching_results.tsv --candidate output/candidate_pairs.tsv '
               '--test-dir student_resource/dataset/test')
    hashes = {name: sha256(root / 'output' / name) for name in
              ['matching_results.tsv', 'candidate_pairs.tsv']}
    report = {
        **previous, **stats,
        'status': 'ENCODER SUBMISSION STATUS',
        'calibration_macro_f0_5': config['calibration_score'],
        'clean_macro_f0_5': metrics['macro_f0_5'], 'baseline': .965614, 'target': .98,
        'beat_baseline': metrics['macro_f0_5'] > .965614,
        'reached_0_98': metrics['macro_f0_5'] >= .98,
        'architecture': config['architecture'], 'bi_encoder': config['model'],
        'cross_encoder': 'intfloat/multilingual-e5-base fine-tuned paired classifier, 278044417 parameters',
        'candidate_retrieval': 'same-country structured E5 top40 + address E5 top20 + bounded exact full/core name, address and rare-address recovery',
        'holdout_candidate_recall': metrics['retrieval']['pair_candidate_recall'],
        'holdout_oracle_macro_f0_5': metrics['retrieval']['oracle_macro_f0_5'],
        'india_macro_f0_5': metrics['by_country']['India']['macro_f0_5'],
        'us_macro_f0_5': metrics['by_country']['US']['macro_f0_5'],
        'thresholds': config['thresholds'], 'xgb_weight': config['xgb_weight'],
        'cross_encoder_gate': config['gate'], 'holdout_entities': metrics['entities'],
        'holdout_TP': metrics['true_positive_pairs'], 'holdout_FP': metrics['FP'], 'holdout_FN': metrics['FN'],
        'holdout_pair_precision': metrics['pair_micro_precision'],
        'holdout_pair_recall': metrics['pair_micro_recall'],
        'validator_command': command, 'id_check_command': command + ' --check-ids',
        'validator_summary': validator_logs, 'output_sha256': hashes,
        'recovery_events': recoveries,
        'archive': str(archive), 'team_name': team, 'team_members': members,
        'identity_note': 'Team identity must be supplied by the submitter; no identity was invented.' if not identity else '',
        'validation_note': 'One frozen Source1 entity-level holdout evaluation on a 10000-entity sampled training universe; no France labels or hidden-test score.',
    }
    # Archive size/hash can only be recorded externally after the ZIP is closed.
    report.pop('archive_bytes', None)
    report.pop('archive_sha256', None)
    report['latest_job_reported_seconds'] = report.get('total_seconds')
    report['total_seconds'] = stats['total_active_logged_seconds']
    documentation = f'''# ML Challenge 2026: Business Entity Resolution Solution

**Team Name:** {team}

**Team Members:** {members}

**Submission Date:** {time.strftime('%Y-%m-%d')}

## 1. Executive Summary
The frozen encoder-first system combines multilingual semantic retrieval, structural recovery, GPU XGBoost, and a fine-tuned pair classifier. Clean entity-level holdout macro F0.5 is {metrics['macro_f0_5']:.9f}; the 0.980000 target was not reached. Every test Source1, including France, is represented in both output files.

## 2. Methodology
### 2.1 Problem Analysis
Names and addresses contain abbreviations, missing tokens, number conflicts and multilingual variants. Multiple locations can share business names or buildings. Precision-heavy decoding therefore combines semantic similarity with lexical and numeric consistency.

### 2.2 Solution Strategy
Normalize the official records, encode structured query/passage texts, retrieve within dynamically discovered countries, compute 53 sparse lexical/semantic/competition features, score candidates and threshold the blended probabilities. No external business records or entity lookup is used.

## 3. Candidate Generation (Blocking)
Structured multilingual E5 retrieves top40 by blocked exact FP16 cosine. An independent address-only E5 view adds top20. Exact normalized full/core names and full addresses use country-restricted posting lists capped at 100; rare-address token recovery adds up to 20. Pair duplicates are removed before features. Full-test candidate pairs: {stats['candidate_pairs']:,}.

Frozen holdout candidate pair recall: {metrics['retrieval']['pair_candidate_recall']:.9f}; oracle macro F0.5: {metrics['retrieval']['oracle_macro_f0_5']:.9f}. These retrieval measures are not the final model score.

## 4. Matching Model
The bi-encoder is `intfloat/multilingual-e5-base`, revision `d128750597153bb5987e10b1c3493a34e5a4502a`, MIT licensed. Texts use `query:` and `passage:` prefixes with business name, address and country. Sequence length is 128; normalized 768-dimensional embeddings are checkpointed as float16.

GPU XGBoost uses name/address exactness, token Jaccard/overlap, hashed character cosine, length ratios, number/postcode conflicts, source flags, retrieval provenance, E5 ranks/gaps and candidate competition. The paired XLM-R classifier initialized from E5 has 278,044,417 parameters (below 8B); its first eight transformer layers are frozen. It trains on positive pairs and per-entity hard negatives for two epochs, selected on calibration. Inference cross-encodes candidates whose XGBoost probability is at least {config['gate']}. Final score is 0.5 XGBoost + 0.5 cross-encoder; the cross-encoder contribution is zero below the gate. No stacking model uses in-sample cross-encoder predictions.

Source1 ID hashing with seed 260926 creates 6,033 training, 1,987 calibration, and 1,980 untouched holdout entities. All pairs for one Source1 stay together. Calibration alone selects retrieval, the model, epoch, blend and thresholds. Clean holdout is evaluated once after freezing; no subsequent model selection is performed. Thresholds: India {config['thresholds']['countries']['india']:.12f}, US/global {config['thresholds']['global']:.12f}. France has no labeled validation and uses the pooled global threshold. Zero and multiple matches are allowed; no top-k prediction cap is imposed.

## 5. Results & Error Analysis
- Calibration macro F0.5: {config['calibration_score']:.9f}
- Clean holdout macro F0.5: {metrics['macro_f0_5']:.9f}
- India holdout macro F0.5: {metrics['by_country']['India']['macro_f0_5']:.9f}
- US holdout macro F0.5: {metrics['by_country']['US']['macro_f0_5']:.9f}
- Holdout TP / FP / FN: {metrics['true_positive_pairs']} / {metrics['FP']} / {metrics['FN']}
- Pair precision / recall: {metrics['pair_micro_precision']:.9f} / {metrics['pair_micro_recall']:.9f}
- Full-test Source1: {stats['source1_rows']:,}; predicted links: {stats['predicted_pairs']:,}; empty predictions: {stats['empty_predictions']:,}

Validation uses a bounded training-feed universe (10,000 Source1 and 1,034,502 feed records), so it is not an estimate of the hidden leaderboard at full feed scale. Of 418 holdout missed links, 99 were absent from candidates; the rest were rejected by scoring/decoding. The 97 false positives and country metrics quantify remaining scorer limitations; no unmeasured case-category rates are claimed. Both official validation modes PASS. Validation checks format and ID consistency, not hidden-test accuracy.

## 6. Conclusion
The clean score exceeds the historical 0.965614 baseline and remains below 0.98. The strongest already-frozen encoder configuration was preserved during recovery so that submission completion did not compromise holdout integrity.

## Appendix
### A. Code Artefacts
`code/business_entity_resolution/src/business_entity_resolution/encoder_pipeline.py` is the entry point. The README documents installation, official dataset placement and the `--stage full` command. Frozen XGBoost, cross-encoder, feature/config metadata and holdout metrics ship under `artifacts/`; large training caches, embeddings, raw datasets and logs are excluded. The base E5 encoder downloads at its pinned revision if absent from the standard model cache.

### B. Additional Results
`artifacts/ENCODER_FINAL_REPORT.json` records validation, output counts, hashes and measured timing/resource limitations. The historical France inference did not separately record retrieval/scoring runtimes. Interrupted encoding resumes after the last atomic 8,192-row marker; inference resumes per 5,000 Source1 shard. Run `{command}` and repeat with `--check-ids`.
'''
    (work / 'Documentation_template.md').write_text(documentation)
    (root / 'Documentation_template.md').write_text(documentation)
    write_json(logs / 'ENCODER_FINAL_REPORT.json', report)
    temporary = archive.with_suffix('.zip.tmp')
    with zipfile.ZipFile(temporary, 'w', compression=zipfile.ZIP_DEFLATED, compresslevel=3) as z:
        for name in hashes:
            z.write(root / 'output' / name, 'output/' + name)
        base = root / 'code/business_entity_resolution'
        for p in sorted((base / 'src').rglob('*.py')):
            z.write(p, str(p.relative_to(root)))
        for name in ['README.md', 'requirements.txt']:
            z.write(base / name, str((base / name).relative_to(root)))
        z.write(work / 'Documentation_template.md', 'Documentation_template.md')
        artifact_names = ['selection.json', config['model_file'],
                          str(Path(config['model_file']).with_suffix('.metadata.json')),
                          'clean_holdout_metrics.json', 'production_configuration.json']
        for name in artifact_names:
            z.write(work / name, 'code/business_entity_resolution/artifacts/' + name)
        for p in sorted((work / 'cross_encoder/best').iterdir()):
            if p.is_file():
                z.write(p, 'code/business_entity_resolution/artifacts/cross_encoder/' + p.name)
        for name in ['ENCODER_FINAL_REPORT.json', 'encoder_model_audit.json']:
            source = logs / name if (logs / name).exists() else work / name
            if source.exists():
                z.write(source, 'code/business_entity_resolution/artifacts/' + name)
    with zipfile.ZipFile(temporary) as z:
        assert z.testzip() is None, 'Archive CRC check failed'
        names = z.namelist()
        assert len(names) == len(set(names))
        assert not any(n.startswith(('cache/', 'logs/', 'student_resource/')) for n in names)
        (logs / 'encoder_archive_contents.txt').write_text('\n'.join(names) + '\n')
    temporary.replace(archive)
    report['archive_bytes'] = archive.stat().st_size
    report['archive_sha256'] = sha256(archive)
    report['delivery_complete'] = True
    write_json(logs / 'ENCODER_FINAL_REPORT.json', report)
    lines = ['ENCODER SUBMISSION STATUS', '']
    for key, value in report.items():
        lines.append(f'{key}: {json.dumps(value) if isinstance(value, (dict, list)) else value}')
    (logs / 'ENCODER_FINAL_REPORT.md').write_text('\n\n'.join(lines) + '\n')
    state = json.loads((work / 'state.json').read_text())
    state['recovery_events'] = recoveries
    state.update(stage='COMPLETE', validator='PASS', id_check='PASS',
                 submission_archive=str(archive), archive_bytes=report['archive_bytes'],
                 current_bottleneck='none', checkpoints_completed='all countries, outputs, official validation and audited archive',
                 next_step='Supply team identity if absent, then upload the final files',
                 updated=time.strftime('%Y-%m-%d %H:%M:%S'), delivery_complete=True)
    write_json(work / 'state.json', state)
    (logs / 'ASTRA_STATUS.md').write_text('# Encoder-first submission status\n\n' +
        '\n'.join(f'- **{k}**: {v}' for k, v in state.items()) + '\n')
    print(json.dumps({'delivery_complete': True, 'archive': str(archive),
                      'archive_bytes': report['archive_bytes']}), flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--root', type=Path, default=REPO_ROOT)
    deliver(parser.parse_args().root)
