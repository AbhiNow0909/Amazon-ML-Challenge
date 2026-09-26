# Business Entity Resolution — Tier 0

This package implements the CPU-first E0 pipeline for the Amazon ML Challenge
2026. It reads the official TSV files without modifying them, builds bounded
multi-channel candidates, creates a compact numeric feature table, tunes a small
deterministic rule matcher on Source-1-level validation data, and writes the two
required submission tables.

## Environment

Tier 0 uses only packages already present in the supplied environment. The
audited versions are pinned in `requirements.txt`; no package installation or
upgrade is required for the current workspace.

Run commands from the repository root:

```bash
export PYTHONPATH="$PWD/code/business_entity_resolution/src"
python3 -m business_entity_resolution.run_pipeline --stage audit-lite
python3 -m pytest code/business_entity_resolution/tests
python3 -m business_entity_resolution.run_pipeline --stage smoke
```

The first bounded validation experiment is:

```bash
python3 -m business_entity_resolution.run_pipeline \
  --stage evaluate \
  --validation-s1-limit 2000 \
  --distractors-per-source 50000 \
  --persist
```

It evaluates candidates at K=10, 20, 30, and 50, selects the smallest K whose
pair recall is within 0.0005 of the best observed recall, builds features at that
K, tunes E0 thresholds only on the selected validation entities, and stores a
JSON report plus configuration-fingerprinted Parquet caches.

The optional E1 name character TF-IDF experiment is disabled by default and can
be run on the same deterministic universe with:

```bash
python3 -m business_entity_resolution.run_pipeline \
  --stage e1 \
  --validation-s1-limit 2000 \
  --distractors-per-source 50000 \
  --persist
```

E1 fits `char_wb` 3–4 gram float32 TF-IDF indexes separately for every country,
queries Source 1 in bounded chunks, performs sparse matrix multiplication, and
retains only row-wise top-K neighbors. It compares the unchanged E0 blockers to
their union with TF-IDF at final caps 20/30/50/75/100 and reports TF-IDF-only
recall at top-K 10/20/30/50. No dense similarity matrix is constructed.

E1.1 keeps the two retrieval systems unchanged and compares quota, reciprocal
rank, normalized-score, and protected quota-plus-rank fusion on the same
validation universe:

```bash
python3 -m business_entity_resolution.run_pipeline \
  --stage e1.1 \
  --validation-s1-limit 2000 \
  --distractors-per-source 50000 \
  --persist
```

The E1.1 search space is centralized in `config.py`. It evaluates TF-IDF input
depths 20/30/50 and final candidate caps 30/40/50/75, retains both systems'
rank, score, and provenance fields, and re-runs E0 at its fixed 0.70 threshold.

E2 trains the first pair classifier on a separate deterministic S1-level split:

```bash
python3 -m business_entity_resolution.run_pipeline --stage e2 --persist
```

It reconstructs the reviewed RRF configuration (legacy weight 2, TF-IDF weight
1, constant 20, TF-IDF depth 30, final K=50), adds label-free candidate-relative
features, compares all generated negatives with a bounded 5:1 hard-negative
sample, and tunes one global probability threshold on held-out S1 entities.
E2 requires LightGBM; this workspace uses version 4.6.0. Model artifacts are
written under `models/lgbm/`, validation probabilities under
`cache/predictions/`, and metrics under `experiments/`.

E2.1 holds the selected E2 retrieval, feature, and LightGBM configurations
fixed while testing a larger deterministic universe:

```bash
python3 -m business_entity_resolution.run_pipeline --stage e2.1 --persist
```

It selects 10,000 S1 entities, retains all of their truth-linked feed rows, and
uses one hash-ranked distractor sample to create nested 100,000, 500,000, and
1,000,000-distractor retrieval tests. The largest universe supplies the frozen
entity-level training/validation run. E2.1 reports both the previous 0.759
decision threshold and a threshold retuned on the new held-out entities. It
does not run test inference.

E2.2 reuses those normalized caches for retrieval-density diagnosis and an
address character TF-IDF experiment:

```bash
python3 -m business_entity_resolution.run_pipeline --stage e2.2 --persist
```

It measures every legacy blocker at the three cached distractor densities,
tests opt-in relative/hybrid/ranked rare-token policies, and evaluates sparse
country-sharded address TF-IDF at top-K 5/10/20. Three-channel fusion retains
legacy, name-TF-IDF, and address-TF-IDF ranks and scores while enforcing final
K=50. The E2.1 LightGBM model is not loaded or retrained by this stage.

E2.3 keeps the E2.2 full-country address retriever as a reference and tests
sparse pre-blocked address retrieval on the same cached universe:

```bash
python3 -m business_entity_resolution.run_pipeline --stage e2.3 --persist
```

The optimized path unions postcode, useful numeric-token, and rare lexical
address postings before applying the unchanged character TF-IDF similarity.
It records candidate route provenance and uses an explicit country fallback
for addresses without a usable structural key. No classifier is loaded or
trained by this stage. The selected E2.3 defaults use postcode postings,
numeric-token DF up to 5,000, the three rarest lexical address tokens with DF
up to 500, a 15,000-row pool cap, address top-K 20, and the unchanged E2.2
protected fusion at final K=50.

Individual bounded stages are also available:

```bash
python3 -m business_entity_resolution.run_pipeline --stage split
python3 -m business_entity_resolution.run_pipeline --stage normalize
python3 -m business_entity_resolution.run_pipeline --stage block
python3 -m business_entity_resolution.run_pipeline --stage features
python3 -m business_entity_resolution.run_pipeline --stage baseline
```

There is deliberately no unguarded `all` command. Full test inference is not
performed until validation results have been reviewed. The `submit` stage
requires `--confirm-full-test`, and the official validator can be run after
outputs exist:

```bash
python3 -m business_entity_resolution.run_pipeline --stage validate --check-ids
```

## Data and split behavior

Raw paths are anchored to the repository root in `config.py`:

- `student_resource/dataset/train`
- `student_resource/dataset/test`
- `student_resource/utils/validate_submission.py`

TSVs are read with an explicit tab delimiter, `dtype=str`, and empty-string
preservation. The validation split hashes whole Source-1 IDs; candidate pairs
are never randomly split. The bounded experiment includes every true S2/S3 link
for selected S1 entities plus deterministic feed distractors.

## Candidate generation

Candidate channels are exact normalized full/core/sorted names, rare name
tokens, exact normalized addresses, and rare address tokens. Primary postings
are country-sharded using dynamically discovered country strings. Exact full
names also have a cross-country fallback. Posting lists and final candidates are
bounded, and each pair retains blocker provenance, a cheap rank, and context
features.

## Outputs and caches

Generated data is placed only under `cache/`, `experiments/`, and `output/`.
Parquet artifacts have JSON sidecars containing the configuration fingerprint.
The raw challenge files and official validator are read-only inputs.

`matching_results.tsv` and `candidate_pairs.tsv` are built with one row for every
test S1 entity. Internal validation treats matched-without-candidate and unknown
target IDs as errors even though some checks are warnings in the official
validator.
