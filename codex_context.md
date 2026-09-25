# Amazon ML Challenge 2026 — Business Entity Resolution
## Unified Implementation Specification for Codex / VS Code

**Document purpose:** This file is the primary technical context/specification for implementing our Amazon ML Challenge 2026 solution in VS Code using Codex.

**Implementation strategy:** Build a strong, reproducible competition pipeline with a working baseline first, then add advanced components behind feature flags only when local validation proves they help.

**Core philosophy:**

> Build a simple, always-submittable pipeline first. Then add higher-value components one at a time and keep only changes that improve local validation F0.5 or materially improve candidate recall without creating unacceptable runtime or complexity.

---

# 1. Source Problem Summary

The challenge is a **Business Entity Resolution** problem.

There are three independent business-record sources:

- **Source 1** — deduplicated reference source
- **Source 2** — noisy feed
- **Source 3** — noisy feed

Each record contains:

```text
entity_id
business_name
business_address
country
```

The task is:

> For every Source 1 entity in the test set, identify all Source 2 and Source 3 records that refer to the same real-world business.

A Source 1 entity may have:

```text
0 matches
1 match
multiple matches
```

The records are deliberately noisy. Names may contain abbreviations, legal-suffix differences, punctuation changes, typos, word-order changes, transliterations, and DBA/trade names. Addresses may be partial, reordered, abbreviated, missing components, or landmark-based.

---

# 2. Non-Negotiable Challenge Constraints

These constraints must be reflected directly in the implementation.

## 2.1 TSV input

All challenge files are tab-separated.

Always load with:

```python
pd.read_csv(path, sep="\t", dtype=str, keep_default_na=False)
```

Do not parse them as comma-separated CSV.

---

## 2.2 Source IDs

There is no explicit source column.

The source is identified by:

```text
S1-... → Source 1
S2-... → Source 2
S3-... → Source 3
```

The files themselves also identify the source.

---

## 2.3 Open-set country

Training covers:

```text
US
India
```

while the test set additionally contains:

```text
France
```

Therefore:

- never hard-code the set `{US, India}`
- never filter the test set to known training countries
- never require a country to have appeared during training
- treat `country` as an open-set string attribute

Every test Source 1 entity, including France records, must receive a prediction row.

---

## 2.4 Output files

The final submission must contain:

```text
output/
├── matching_results.tsv
└── candidate_pairs.tsv
```

### `matching_results.tsv`

Columns:

```text
source1_entity_id
matched_entity_ids
```

Rules:

- exactly one row for every Source 1 test entity
- empty `matched_entity_ids` is valid
- only S2/S3 IDs may be matched
- IDs must exist in the test data
- no duplicates within an ID list

### `candidate_pairs.tsv`

Columns:

```text
source1_entity_id
candidate_entity_ids
```

This must be the **final candidate set actually passed to the matching model**.

Every matched ID in `matching_results.tsv` must also appear in the corresponding candidate list.

---

## 2.5 Submission validation

Before every submission:

```bash
python3 utils/validate_submission.py \
    --matching output/matching_results.tsv \
    --candidate output/candidate_pairs.tsv \
    --test-dir dataset/test
```

The validator checks formatting and consistency, not the challenge score.

---

## 2.6 Metric

The challenge uses macro-averaged per-Source-1 **F0.5**:

```text
F0.5 = (1.25 * Precision * Recall) /
       (0.25 * Precision + Recall)
```

F0.5 is precision-heavy.

Singleton behavior:

```text
truth = empty
prediction = empty
→ score = 1.0

truth = empty
prediction = non-empty
→ score = 0.0
```

Therefore, the final decision layer must explicitly support:

```text
MATCH
NO MATCH
```

and must not force every Source 1 entity to match something.

---

## 2.7 Model license / size

The final model must satisfy the challenge rule:

- MIT or Apache 2.0 license
- no more than 8 billion parameters

Before adding any pretrained or external model, explicitly verify the intended license and parameter count and document the decision.

---

## 2.8 External lookup is prohibited

Do not use:

- commercial entity-resolution APIs
- business-registration lookup
- government business databases
- geocoding APIs
- Google Maps / business search
- external business data
- internet-based business identity augmentation

All entity resolution must use the provided challenge data.

AWS may be used for:

```text
compute
storage
training
experimentation
inference
```

but not for acquiring external business identity information.

---

# 3. Unified Solution

The solution combines the simpler original plan with the strongest ideas from the second proposed approach.

Target architecture:

```text
Raw TSVs
   ↓
Data validation
   ↓
Normalization / canonicalization
   ↓
Candidate generation / blocking
   ├── exact name
   ├── rare-token/name keys
   ├── character TF-IDF
   ├── address TF-IDF
   ├── optional multilingual embeddings
   └── optional reverse retrieval
   ↓
Candidate union + deduplication + Top-K
   ↓
Pairwise features
   ├── name
   ├── address
   ├── country
   ├── structural/context
   └── competition features
   ↓
LightGBM
   +
optional cross-encoder score
   ↓
Probability calibration
   ↓
Optional graph consistency
   ↓
Entity-level expected-F0.5 decoder
   ↓
matching_results.tsv
candidate_pairs.tsv
   ↓
official validator
```

The **core target** should be:

```text
better normalization
+
high-recall multi-blocking
+
hard negatives
+
rich pairwise features
+
competition features
+
LightGBM
+
calibration
+
expected-F0.5 decoding
```

The following are optional:

```text
multilingual embedding retrieval
cross-encoder
graph consistency
France pseudo-labeling
LLM tie-breaker
```

Each optional stage must be feature-flagged and validated independently.

---

# 4. Progressive Implementation Strategy

Do not implement the complete advanced system in one pass.

## Tier 0 — Always-submittable baseline

```text
load
→ normalize
→ basic blocking
→ similarity features
→ LightGBM/rule baseline
→ threshold
→ output
→ validator
```

## Tier 1 — Strong competition version

Add:

```text
better canonicalization
multi-channel blocking
hard negatives
competition features
calibration
expected-F0.5 decoding
```

## Tier 2 — Advanced retrieval/modeling

Add:

```text
multilingual embeddings
cross-encoder stacking
```

## Tier 3 — Experimental

Add only if justified:

```text
graph consistency
France pseudo-labeling
LLM tie-breaking
```

A working baseline must never depend on an experimental component.

---

# 5. Repository Structure

Suggested development repository:

```text
project-root/
│
├── dataset/
│   ├── train/
│   │   ├── train_source1.tsv
│   │   ├── train_source2.tsv
│   │   ├── train_source3.tsv
│   │   └── train_ground_truth.tsv
│   │
│   └── test/
│       ├── test_source1.tsv
│       ├── test_source2.tsv
│       └── test_source3.tsv
│
├── src/
│   └── business_entity_resolution/
│       ├── __init__.py
│       ├── config.py
│       ├── io_utils.py
│       ├── normalize.py
│       ├── noise_mining.py
│       ├── split.py
│       ├── tfidf_retrieval.py
│       ├── embedding_retrieval.py
│       ├── blocking.py
│       ├── features.py
│       ├── labels.py
│       ├── train_lgbm.py
│       ├── train_cross_encoder.py
│       ├── calibrate.py
│       ├── graph_consistency.py
│       ├── decode.py
│       ├── france_adaptation.py
│       ├── llm_tiebreaker.py
│       ├── evaluation.py
│       ├── submission.py
│       └── run_pipeline.py
│
├── notebooks/
│   ├── 01_eda.ipynb
│   ├── 02_normalization_analysis.ipynb
│   ├── 03_blocking_analysis.ipynb
│   ├── 04_features.ipynb
│   ├── 05_model.ipynb
│   └── 06_decoder.ipynb
│
├── cache/
│   ├── normalized/
│   ├── candidates/
│   ├── features/
│   └── predictions/
│
├── models/
│   ├── lgbm/
│   ├── cross_encoder/
│   └── calibration/
│
├── output/
│   ├── matching_results.tsv
│   └── candidate_pairs.tsv
│
├── dictionaries/
│   ├── legal_forms.tsv
│   ├── street_types.tsv
│   └── mined_substitutions.tsv
│
├── experiments/
│   └── results.csv
│
├── tests/
│   ├── test_io.py
│   ├── test_normalize.py
│   ├── test_blocking.py
│   ├── test_features.py
│   ├── test_metric.py
│   └── test_submission.py
│
├── requirements.txt
├── README.md
├── Documentation_template.md
└── .gitignore
```

At final packaging time, make sure the required challenge structure is respected:

```text
<team_name>_submission.zip
├── output/
├── code/
│   └── business_entity_resolution/
│       ├── src/
│       ├── README.md
│       └── requirements.txt
└── Documentation_template.md
```

---

# 6. Configuration

Keep critical parameters centralized.

Suggested initial `config.py`:

```python
SEED = 42

TRAIN_DIR = "dataset/train"
TEST_DIR = "dataset/test"

CACHE_DIR = "cache"
MODEL_DIR = "models"
OUTPUT_DIR = "output"

# Blocking
TFIDF_NAME_TOP_K = 50
TFIDF_ADDRESS_TOP_K = 20
EMBEDDING_TOP_K = 30
REVERSE_TOP_K = 3
FINAL_CANDIDATE_K = 30

# Optional components
USE_EMBEDDINGS = False
USE_CROSS_ENCODER = False
USE_GRAPH = False
USE_PSEUDO_LABELS = False
USE_LLM_TIEBREAKER = False

# Features
USE_PHONETIC = True
USE_ALIASES = True
USE_COMPETITION_FEATURES = True

# Decision
USE_EXPECTED_F05_DECODER = True
DECODER_SAMPLES = 4000
DECODER_MAX_K = 10

RANDOM_STATE = 42
```

Do not scatter magic numbers through the codebase.

Every experiment should save the effective configuration.

---

# 7. Data Loading

Implement robust loaders in `io_utils.py`.

Suggested functions:

```python
load_tsv(path)
load_source1(path)
load_source2(path)
load_source3(path)
load_ground_truth(path)
validate_dataframe_schema(...)
```

Recommended generic loader:

```python
def load_tsv(path):
    return pd.read_csv(
        path,
        sep="\t",
        dtype=str,
        keep_default_na=False,
    )
```

Requirements:

- preserve entity IDs as strings
- preserve empty strings
- validate required columns
- validate ID prefixes
- detect duplicate IDs
- preserve row counts
- fail loudly on malformed input

---

# 8. EDA Before Modeling

Run EDA before building a sophisticated model.

Calculate:

## Dataset sizes

```text
n_source1
n_source2
n_source3
```

## Missingness

For each source:

```text
business_name missing %
business_address missing %
country missing %
```

## Country distribution

For each source:

```text
country
count
percentage
```

## String statistics

For name/address:

```text
mean length
median length
p25
p75
p95
max
token counts
```

## Ground-truth distribution

Calculate for Source 1:

```text
0 matches
1 match
2 matches
3+ matches
```

Also:

```text
singleton %
mean matches
median matches
maximum matches
S2-only %
S3-only %
S2+S3 %
```

## Important structural tests

Check:

1. Does any S2/S3 ID appear under multiple S1 entities?
2. What percentage of S2/S3 records match nothing?
3. Do true matches ever have country mismatches?
4. How duplicated are normalized names?
5. How duplicated are normalized addresses?
6. How dense are likely candidate sets?

Do **not** assume the one-owner property before checking the ground truth.

---

# 9. Ground-Truth Parsing

Convert:

```text
S1-00001 → S2-00047,S3-00812
```

into:

```python
{
    "S1-00001": {"S2-00047", "S3-00812"}
}
```

An empty list becomes:

```python
set()
```

Create a positive pair representation:

```text
source1_entity_id
candidate_entity_id
label=1
```

and build negatives from candidate pairs not present in ground truth.

---

# 10. Normalization / Canonicalization

Create multiple representations instead of one aggressively cleaned field.

## Name representations

```text
name_raw
name_full
name_core
name_sorted
name_tokens
aliases
legal_type
acronym
phonetic_key
```

## Address representations

```text
address_raw
addr_full
addr_core
postcode
numbers
landmark
city
state
address_tokens
```

Multiple views let different model features use different levels of normalization.

---

# 11. Name Normalization

Recommended operations:

## Unicode

Use Unicode normalization and case folding.

Example:

```text
Société Générale
→ societe generale
```

## Whitespace / punctuation

Normalize punctuation and whitespace while preserving useful alphanumeric structure.

## Connectors

Potentially normalize:

```text
&
and
et
+
```

when context makes them equivalent.

## Common wrappers

Potential examples:

```text
M/s
Shri
Sri
Shree
```

Use deterministic transformations and validate them.

## Legal forms

Maintain a configurable dictionary for forms such as:

```text
pvt
private
ltd
limited
inc
incorporated
corp
corporation
co
company
llc
llp
sarl
sas
sa
eurl
snc
cie
```

Store the detected legal type separately.

Create:

```text
name_full
name_core
```

Do not rely only on legal-suffix-stripped names.

---

# 12. DBA / Trade Names

Recognize simple patterns:

```text
dba
d/b/a
doing business as
t/a
trading as
aka
```

Example:

```text
Acme Inc dba QuickFix
```

could produce:

```text
name_core = acme
aliases = [quickfix]
```

Keep alias parsing conservative.

---

# 13. Word-Order and Acronym Views

Create:

```text
name_sorted
```

so:

```text
Sharma Traders
Traders Sharma
```

become similar under a sorted-token representation.

Create an optional acronym:

```text
International Business Machines
→ ibm
```

These are features, not hard match rules.

---

# 14. Phonetic View

Optionally create a phonetic representation, e.g. Double Metaphone.

Useful for noisy transliteration/spelling:

```text
laxmi
lakshmi
```

Keep this optional until validation proves useful.

---

# 15. Address Normalization

Retain both:

```text
addr_full
addr_core
```

and extract:

```text
postcode
numbers
landmark
city
state
```

---

## Postcode

Use conservative regex extraction.

Normalize variants like:

```text
560 001
560001
```

Do not treat every 5/6 digit number as a postal code without context.

---

## Numbers

Store normalized numbers from forms such as:

```text
12
12/3
12-3
Plot 12
No. 12
```

Retain enough structure to compare house/building numbers.

---

## Street types

A configurable normalization table can include:

```text
street → st
road → rd
avenue → ave
boulevard → blvd
lane → ln
drive → dr
suite → ste
building → bldg
floor → fl
```

Be conservative around ambiguous abbreviations.

---

## City/state aliases

A small deterministic normalization dictionary may include well-known spelling variants such as:

```text
Bangalore ↔ Bengaluru
Bombay ↔ Mumbai
Madras ↔ Chennai
Calcutta ↔ Kolkata
Gurgaon ↔ Gurugram
```

Do not use external geocoding or lookup services.

---

# 16. Noise-Table Mining

High-value enhancement.

From known positive pairs:

```text
ground-truth pairs
       ↓
light normalization
       ↓
token alignment
       ↓
recurring substitutions
       ↓
manual review
       ↓
normalization dictionary
```

Potential discoveries:

```text
pvt ↔ private
ltd ↔ limited
rd ↔ road
laxmi ↔ lakshmi
```

Store reviewed mappings in:

```text
dictionaries/mined_substitutions.tsv
```

Do not blindly promote every frequent replacement.

---

# 17. Candidate Generation

Blocking answers:

> Which S2/S3 records are plausible candidates for this S1?

It must prioritize recall.

A missing true match at blocking time can never be recovered later.

Use the union of several blocking channels.

---

# 18. Blocking Channel A — Exact Keys

Candidate keys can include:

```text
name_full exact
name_core exact
name_sorted exact
```

Potential compound keys:

```text
postcode + name prefix
```

only where validated.

Do not make country a mandatory filter before testing whether country mismatches exist among true pairs.

---

# 19. Blocking Channel B — Rare Name Tokens

Build an inverted index over informative tokens.

Downweight generic tokens such as:

```text
restaurant
services
company
trading
```

Prefer rare tokens.

Initial idea:

```text
document_frequency <= 20
```

as a starting threshold.

Tune this empirically.

---

# 20. Blocking Channel C — Character TF-IDF

Use character-level retrieval for typos, abbreviations and punctuation changes.

Suggested starting point:

```python
TfidfVectorizer(
    analyzer="char_wb",
    ngram_range=(3, 4),
    sublinear_tf=True,
)
```

Index:

```text
name_core
```

Retrieve top candidates by cosine similarity.

Start with:

```text
top_k_name = 50
```

then tune based on candidate recall.

---

# 21. Blocking Channel D — Address TF-IDF

Build a separate index on:

```text
addr_core
```

Initial:

```text
top_k_address = 20
```

Especially useful for:

```text
same address
different business-name representation
```

and rebranding/noisy names.

---

# 22. Blocking Channel E — Multilingual Embedding Retrieval

Optional advanced channel.

Potential model:

```text
intfloat/multilingual-e5-base
```

Encode:

```text
name + address + country
```

or another consistent structured text format.

Use FAISS retrieval.

Purpose:

- transliteration
- word reordering
- French/unseen-country robustness
- semantic similarity beyond character overlap

Start disabled.

Measure:

```text
candidate recall improvement
candidate volume
runtime
```

before making it part of the final pipeline.

---

# 23. Reverse Blocking

Normal retrieval:

```text
S1 → S2/S3
```

Also perform:

```text
S2/S3 → S1
```

Add top reverse candidates back into the union.

Initial reverse K:

```text
3
```

This can recover candidates buried by crowded forward results.

---

# 24. Candidate Union

For each S1:

```python
candidates = (
    exact_candidates
    | rare_token_candidates
    | name_tfidf_candidates
    | address_tfidf_candidates
    | embedding_candidates
    | reverse_candidates
)
```

Then:

1. deduplicate
2. remove invalid IDs
3. keep only S2/S3 candidates
4. optionally rank with a cheap score
5. cap to final K

---

# 25. Choosing Final Candidate K

Measure candidate recall for:

```text
K = 5
10
20
30
40
50
```

and track:

```text
candidate recall
average candidate count
p95 candidate count
maximum candidate count
runtime
oracle F0.5
```

Choose the smallest K where recall has essentially saturated.

Do not assume a fixed value such as 30 without validation.

---

# 26. Candidate Metrics

For validation, report:

```text
candidate recall / pair completeness
reduction ratio
oracle F0.5
average candidate count
p95 candidate count
```

Oracle F0.5 means:

> What macro F0.5 could be achieved if the final matcher were perfect within the candidate set?

This tells us whether the bottleneck is retrieval or classification.

---

# 27. Pairwise Feature Engineering

For each candidate:

```text
S1 record ↔ S2/S3 record
```

create one numeric feature row.

Target initial feature range:

```text
40–80 features
```

but do not add features only to reach a count.

---

# 28. Name Features

Recommended:

```text
name_exact_full
name_exact_core
name_exact_sorted
name_jaro_winkler
name_levenshtein_similarity
name_token_jaccard
name_token_set_ratio
name_token_sort_ratio
name_partial_ratio
name_tfidf_cosine
name_embedding_cosine          # optional
name_shared_token_count
name_shared_idf_weight
name_length_ratio
name_token_count_difference
name_acronym_match
name_phonetic_overlap
legal_type_equal
legal_type_conflict
alias_max_similarity
```

Start with the strongest/simple features and expand only when validation supports it.

---

# 29. Address Features

Recommended:

```text
address_jaro_winkler
address_levenshtein_similarity
address_token_jaccard
address_token_set_ratio
address_tfidf_cosine
address_embedding_cosine        # optional
postcode_equal
postcode_conflict
postcode_missing
house_number_equal
numbers_jaccard
city_equal
state_equal
landmark_overlap
address_length_ratio
address_token_count_difference
```

Use explicit missingness.

Do not automatically convert missing to mismatch.

---

# 30. Country Features

Use open-set-safe features:

```text
same_country
country_conflict
country_missing_left
country_missing_right
both_country_present
```

Avoid closed-world country encoding as the main mechanism.

---

# 31. Context Features

Useful context signals:

```text
name_core_frequency
name_token_document_frequency
shared_token_idf_sum
candidate_count_for_s1
source_is_s2
source_is_s3
```

Rare names should generally carry more information than extremely common names.

---

# 32. Competition Features

Strongly recommended.

For each S1 candidate set calculate:

```text
rank_by_name_similarity
rank_by_address_similarity
rank_by_embedding_similarity
rank_by_cheap_combined_score
score_margin_to_runner_up
reciprocal_best_flag
number_of_close_competitors
```

Also calculate reverse-direction context:

```text
rank_of_s1_for_candidate
reverse_reciprocal_best
reverse_score_margin
```

Example:

```text
S1-A ↔ S2-X = 0.95
S1-B ↔ S2-X = 0.71
```

The model should learn that S2-X has stronger support for S1-A.

Competition features are especially valuable against look-alike false merges.

Compute them within the current train/validation/test universe to avoid leakage.

---

# 33. Training Pair Construction

The training set for the pairwise model should be built from realistic candidates.

Positive examples:

```text
candidate pair exists in ground truth
```

Negative examples:

```text
candidate pair generated by blocking
but not in ground truth
```

This is superior to using only random negatives.

---

# 34. Hard-Negative Sampling

Prefer negatives such as:

```text
same/similar name, different address
same/similar address, different name
same city, similar name
same postcode, similar name
very high fuzzy similarity but known non-match
```

Start with approximately:

```text
4 hard negatives per positive
```

and experiment.

Avoid a negative set dominated by trivially unrelated businesses.

---

# 35. Baseline Models

Implement a deterministic baseline:

```text
normalized exact name
```

and optionally a simple ML baseline:

```text
Logistic Regression
```

Then establish:

```text
candidate recall
precision
recall
F0.5
runtime
```

The baseline must remain runnable even after advanced models are added.

---

# 36. Primary Model — LightGBM

Use LightGBM as the core classifier.

Why:

- excellent for tabular feature interactions
- CPU-friendly
- fast to retrain
- handles nonlinear relationships
- easy to debug
- efficient for repeated experiments

Starting parameters:

```text
num_leaves = 63
learning_rate = 0.05
feature_fraction = 0.8
bagging_fraction = 0.8
min_data_in_leaf = 50
```

These are initial values, not final hyperparameters.

Avoid huge hyperparameter searches during the challenge.

---

# 37. Optional Cross-Encoder

Optional advanced model:

```text
microsoft/mdeberta-v3-base
```

Role:

> Add a learned semantic pairwise score to the handcrafted features.

Input:

```text
record_A:
{name} | {address} | {country}

record_B:
{name} | {address} | {country}
```

Use hard negatives.

Initial configuration may be approximately:

```text
max_length = 128
learning_rate ≈ 2e-5
epochs = 2
```

Generate out-of-fold predictions for stacking.

Do not feed in-sample cross-encoder predictions into the stacker.

---

# 38. Cross-Encoder Stacking

If the cross-encoder is enabled:

```text
handcrafted features
+
cross_encoder_OOF_probability
```

→ LightGBM.

This lets LightGBM learn interactions such as:

```text
semantic similarity high
+
postcode conflict
→ lower final probability
```

or:

```text
high name similarity
+
same city
+
same postcode
→ stronger final probability
```

The cross-encoder is a signal, not necessarily the final decision maker.

---

# 39. Probability Calibration

Expected-F0.5 decoding depends on probability quality.

Evaluate:

```text
raw probabilities
```

versus:

```text
calibrated probabilities
```

Potential method:

```python
sklearn.isotonic.IsotonicRegression
```

Fit calibration on out-of-fold predictions.

Keep calibration only if it improves downstream validation F0.5 or decision quality.

---

# 40. Simple Entity-Level Decision

First implement:

```python
prediction = probability >= threshold
```

Tune the threshold on validation.

Do not assume 0.5.

Report:

```text
threshold
precision
recall
F0.5
singleton behavior
```

---

# 41. Expected-F0.5 Decoder

After calibrated probabilities are reliable, test the more advanced decision layer.

For each S1:

```text
candidate A → pA
candidate B → pB
candidate C → pC
...
```

sort by descending probability and evaluate:

```text
predict nothing
predict top 1
predict top 2
predict top 3
...
predict top K
```

Choose the number of predictions with the highest expected F0.5.

A Monte Carlo implementation is acceptable as an initial approach.

Requirements:

- deterministic seed
- bounded samples
- bounded max K
- vectorized NumPy where possible
- direct comparison with best global threshold

---

# 42. Decoder Caveat

A simple Monte Carlo expected-F0.5 decoder assumes candidate truth events are approximately independent.

That may be imperfect if an S2/S3 record can have only one S1 owner.

Therefore:

1. verify the one-owner property using ground truth
2. only then experiment with competition/exclusivity-aware adjustments
3. never hard-code the assumption simply because Source 1 is deduplicated

---

# 43. One-Owner Constraint

First run the structural check:

```text
group candidate_entity_id
count unique source1_entity_id
```

If the maximum is:

```text
1
```

throughout ground truth, then it may be reasonable to test one-owner constraints.

If not, do not enforce them.

All owner/graph constraints must be ablation-tested against local macro F0.5.

---

# 44. Graph Consistency — Optional

Potential graph:

```text
S1 ↔ S2
S1 ↔ S3
S2 ↔ S3
```

Possible uses:

- reciprocal ownership
- one-owner consistency
- strong transitive support

Example:

```text
S1 ↔ S2 = very high confidence
S2 ↔ S3 = very high confidence
```

could strengthen:

```text
S1 ↔ S3
```

But graph logic can propagate errors.

Therefore:

- keep disabled by default
- validate every rule separately
- never let graph rules blindly override strong pairwise evidence

---

# 45. France / Unseen-Country Strategy

France is unseen during training.

Recommended:

## 45.1 Country-agnostic features

Prefer:

```text
same_country
country_missing
country_conflict
```

rather than assuming only US/India exist.

## 45.2 Leave-one-country-out validation

Approximate unseen-country robustness:

```text
train = India
validate = US
```

and:

```text
train = US
validate = India
```

This is only a proxy for France, but it can reveal over-reliance on country-specific patterns.

## 45.3 French normalization

Use only deterministic string normalization rules, such as:

```text
rue
av
bd
pl
ch
rte

sarl
sas
sa
eurl
snc
cie
```

No external business information.

## 45.4 Pseudo-labeling

Treat as optional.

Only add pseudo-labels when confidence is extremely high and several independent signals agree.

Example condition:

```text
very high calibrated p
+
reciprocal best
+
strong postcode/address agreement
```

Use one carefully controlled round at most unless validation clearly justifies more.

---

# 46. Optional LLM Tie-Breaker

Treat the LLM as a stretch experiment.

Scope it to:

```text
uncertain candidate pairs
```

for example:

```text
0.3 < calibrated_probability < 0.7
```

If an LLM is used:

- verify MIT/Apache license requirement
- verify parameter limit
- no external lookup
- run locally
- compare validation F0.5 against the non-LLM baseline

Do not make the LLM the primary matcher.

---

# 47. Validation Split Design

Do not split individual pairs randomly.

Use Source 1 entity/universe folds.

Recommended:

1. assign each S1 entity to a fold
2. all associated ground-truth feed records inherit that fold
3. distribute unmatched feed records appropriately
4. treat each fold as a self-contained mini test universe
5. compute blocking and competition features within the fold only

This better approximates the actual test scenario.

---

# 48. Validation Metrics

Every major experiment must report:

```text
macro F0.5
precision
recall
candidate recall
candidate reduction ratio
runtime
```

Also:

```text
F0.5 by country
F0.5 by true cluster size

0 matches
1 match
2 matches
3+ matches
```

and:

```text
number of predicted matches
false merges
empty predictions
```

---

# 49. Oracle Analysis

Always separate:

```text
retrieval failure
```

from:

```text
classification failure
```

For false negatives:

```text
true match missing from candidates
```

means blocking failure.

If it was a candidate but rejected:

```text
model/threshold/decoder failure
```

This diagnostic determines what to improve next.

---

# 50. False Positive Diagnostics

For validation false positives log:

```text
S1 ID
candidate ID
name similarity values
address similarity values
country
postcode
city
candidate rank
reverse rank
model probability
decoder decision
```

Group errors by:

```text
same name / different location
same address / different business
generic shared tokens
transliteration
legal suffix confusion
city-only overlap
postcode collision
```

Use these error categories to drive feature engineering.

---

# 51. False Negative Diagnostics

For every false negative:

```text
S1 ID
true candidate
was true candidate in blocking?
which blocker retrieved it?
name features
address features
model score
final decision
```

This tells the team whether to improve:

```text
blocking
features
classifier
threshold
decoder
```

---

# 52. Competition Features and Universe Isolation

Competition features are calculated relative to competitors.

Therefore, they must be computed separately for:

```text
training universe
validation universe
test universe
```

Do not accidentally rank a validation candidate against records that belong only to the training universe.

Never leak future/test relationships into labels.

---

# 53. Preventing Data Leakage

Critical rules:

## Pair splits

Never randomly split candidate pairs independently.

## Competition features

Compute ranks only within the current evaluation universe.

## Cross-encoder stacking

Use OOF scores.

## Calibration

Fit the calibrator on OOF predictions.

## Test

Do not use test ground truth — it does not exist.

Do not use external lookup.

---

# 54. Caching

Cache expensive artifacts.

Recommended:

```text
cache/
├── normalized/
│   └── records_norm.parquet
├── folds/
│   └── folds.parquet
├── candidates/
│   └── candidates.parquet
├── features/
│   └── features.parquet
└── predictions/
    ├── ce_oof.npy
    ├── ce_test.npy
    └── final_scores.parquet
```

Use Parquet for tables.

Use `.npy` for large numerical arrays where appropriate.

Every cache should be invalidated when relevant configuration/code versions change.

---

# 55. Runtime Optimization

Potential bottlenecks:

```text
TF-IDF candidate retrieval
fuzzy string scoring
embedding generation
cross-encoder inference
```

Optimization order:

1. control candidate count
2. use sparse vectorized retrieval
3. batch GPU inference
4. parallelize CPU similarity calculations
5. cache expensive outputs

Never compute expensive features for all possible S1 × S2/S3 pairs.

---

# 56. Tech Stack

## Core

```text
Python 3.10+
pandas
numpy
scipy
scikit-learn
rapidfuzz
lightgbm
pyarrow
joblib
```

## Optional

```text
metaphone
faiss-cpu / faiss-gpu
torch
transformers
sentence-transformers
accelerate
networkx
vllm
```

## Storage / infrastructure

```text
Amazon S3
EC2 GPU when needed
Git / GitHub
VS Code
Codex
```

---

# 57. AWS Architecture

Use AWS mainly for compute and artifact storage.

```text
                 AWS Account
                      │
            ┌─────────┴─────────┐
            │                   │
            ▼                   ▼
           S3                 EC2
            │                   │
            │            ┌──────┴──────┐
            │            │             │
            │            ▼             ▼
            │          CPU stages    GPU stages
            │            │             │
            └────────────┴─────────────┘
                         │
                         ▼
                   Models / Cache
                         │
                         ▼
                      Outputs
```

---

# 58. S3 Layout

Suggested:

```text
s3://<bucket>/
├── data/
├── cache/
├── models/
├── experiments/
└── output/
```

Store:

```text
normalized records
candidate tables
features
models
prediction artifacts
final TSVs
experiment logs
```

---

# 59. EC2 vs SageMaker

The solution does not require SageMaker.

For this type of competition workload:

```text
EC2
```

may be simpler when direct access to:

- CUDA
- PyTorch
- FAISS
- local filesystem
- SSH
- background jobs

is useful.

SageMaker is still valid if the team is already comfortable with it.

The code should remain portable.

---

# 60. CPU vs GPU

CPU stages:

```text
loading
normalization
exact blocking
TF-IDF retrieval
feature generation
LightGBM
calibration
decoding
submission
validation
```

GPU stages:

```text
multilingual embeddings
cross-encoder
optional local LLM
```

The baseline should remain CPU-capable.

---

# 61. AWS Cost Discipline

Implement:

- budget alerts
- stop/terminate unused GPU instances
- S3 synchronization
- no unnecessary overnight GPU usage
- record actual runtime/cost

Do not assume a fixed cost until actual dataset size and model workload are measured.

---

# 62. Command-Line Interface

Implement a single orchestrator:

```text
src/business_entity_resolution/run_pipeline.py
```

Suggested stages:

```text
eda
normalize
mine-noise
split
tfidf
embed
block
features
train
cross-encoder
calibrate
graph
decode
submit
validate
all
```

Examples:

```bash
python -m src.business_entity_resolution.run_pipeline --stage eda

python -m src.business_entity_resolution.run_pipeline --stage normalize

python -m src.business_entity_resolution.run_pipeline --stage block

python -m src.business_entity_resolution.run_pipeline --stage features

python -m src.business_entity_resolution.run_pipeline --stage train

python -m src.business_entity_resolution.run_pipeline --stage decode

python -m src.business_entity_resolution.run_pipeline --stage validate
```

Full run:

```bash
python -m src.business_entity_resolution.run_pipeline --stage all
```

The full pipeline should reuse valid cached artifacts.

---

# 63. Function Targets

## `io_utils.py`

```python
load_tsv(path)
load_source1(path)
load_source2(path)
load_source3(path)
load_ground_truth(path)
validate_dataframe_schema(...)
```

## `normalize.py`

```python
normalize_name(text)
normalize_address(text)
normalize_country(text)
extract_postcode(address)
extract_numbers(address)
extract_landmark(address)
extract_legal_type(name)
generate_name_views(name)
generate_address_views(address)
normalize_records(df)
```

## `noise_mining.py`

```python
mine_token_substitutions(...)
review_candidate_substitutions(...)
save_noise_dictionary(...)
```

## `split.py`

```python
make_universe_folds(...)
```

## `blocking.py`

```python
exact_name_candidates(...)
rare_token_candidates(...)
name_tfidf_candidates(...)
address_tfidf_candidates(...)
embedding_candidates(...)
reverse_candidates(...)
merge_candidate_channels(...)
cap_candidates(...)
```

## `features.py`

```python
compute_name_features(...)
compute_address_features(...)
compute_country_features(...)
compute_context_features(...)
compute_competition_features(...)
build_feature_matrix(...)
```

## `labels.py`

```python
build_pair_labels(...)
sample_hard_negatives(...)
```

## `train_lgbm.py`

```python
train_lgbm(...)
predict_lgbm(...)
```

## `train_cross_encoder.py`

```python
build_cross_encoder_dataset(...)
train_cross_encoder(...)
predict_cross_encoder(...)
```

## `calibrate.py`

```python
fit_isotonic(...)
apply_calibration(...)
```

## `graph_consistency.py`

```python
check_one_owner_property(...)
apply_owner_competition(...)
apply_transitive_support(...)
```

## `decode.py`

```python
decode_global_threshold(...)
decode_expected_f05(...)
decode_entities(...)
```

## `evaluation.py`

```python
calculate_entity_f05(...)
calculate_macro_f05(...)
calculate_candidate_recall(...)
calculate_oracle_f05(...)
evaluate_by_country(...)
evaluate_by_cluster_size(...)
```

## `submission.py`

```python
build_matching_results(...)
build_candidate_pairs_output(...)
validate_submission_constraints(...)
```

---

# 64. Unit Testing Requirements

At minimum:

## IO tests

- correct TSV parsing
- empty fields
- string IDs
- invalid prefixes
- duplicate IDs

## Normalization tests

Cover:

```text
Pvt. Ltd.
Private Limited
ABC & Sons
ABC and Sons
Bangalore/Bengaluru
560 001/560001
```

## Ground-truth tests

Cover:

```text
no matches
one match
multiple matches
```

## Metric tests

Reproduce the challenge example.

## Candidate tests

Assert:

```text
final_match_ids ⊆ candidate_ids
```

## Submission tests

Ensure:

```text
one row per S1
no duplicate S1 rows
no duplicate IDs
only S2/S3 IDs
empty lists preserved
```

---

# 65. Logging Requirements

Every stage should log:

```text
input rows
output rows
runtime
cache path
configuration
```

Blocking additionally:

```text
candidate count before dedup
candidate count after dedup
average per S1
p95 per S1
max per S1
candidate recall on validation
```

Matching:

```text
number of scored pairs
probability distribution
threshold
number predicted positive
```

Decoding:

```text
number of empty predictions
number of 1+ predictions
average matches predicted
```

---

# 66. Experiment Tracking

Maintain:

```text
experiments/results.csv
```

Columns:

```text
experiment_id
timestamp
git_commit
config_hash
blocking_version
feature_version
model_version
threshold
decoder_enabled
cross_encoder_enabled
graph_enabled
candidate_recall
precision
recall
f05
runtime_seconds
notes
```

Example:

| Experiment | Change | Candidate Recall | Precision | Recall | F0.5 |
|---|---|---:|---:|---:|---:|
| E0 | Exact baseline | TBD | TBD | TBD | TBD |
| E1 | + normalization | TBD | TBD | TBD | TBD |
| E2 | + TF-IDF | TBD | TBD | TBD | TBD |
| E3 | + address | TBD | TBD | TBD | TBD |
| E4 | + LightGBM | TBD | TBD | TBD | TBD |
| E5 | + competition | TBD | TBD | TBD | TBD |
| E6 | + hard negatives | TBD | TBD | TBD | TBD |
| E7 | + calibration | TBD | TBD | TBD | TBD |
| E8 | + decoder | TBD | TBD | TBD | TBD |
| E9 | + embeddings | TBD | TBD | TBD | TBD |
| E10 | + cross encoder | TBD | TBD | TBD | TBD |

---

# 67. Experimental Roadmap

Recommended progression:

## E0 — Exact normalized name

Purpose:

```text
baseline
```

## E1 — Name TF-IDF

Purpose:

```text
recover typo/abbreviation cases
```

## E2 — Address TF-IDF

Purpose:

```text
recover address-supported matches
```

## E3 — Better canonicalization

Purpose:

```text
reduce synthetic noise
```

## E4 — LightGBM

Purpose:

```text
learn interactions among name/address features
```

## E5 — Hard negatives

Purpose:

```text
improve precision on look-alikes
```

## E6 — Competition features

Purpose:

```text
use candidate context and reciprocal evidence
```

## E7 — Probability calibration

Purpose:

```text
make probabilities usable for metric-aware decoding
```

## E8 — Expected-F0.5 decoder

Purpose:

```text
select per-entity prediction count
```

## E9 — Embedding blocking

Purpose:

```text
increase recall on semantic/transliterated cases
```

## E10 — Cross-encoder stacking

Purpose:

```text
add learned semantic pair score
```

## E11 — Graph constraints

Purpose:

```text
reduce structurally inconsistent false merges
```

## E12 — France adaptation

Purpose:

```text
improve unseen-country robustness
```

## E13 — Pseudo-labeling

Purpose:

```text
investigate transductive adaptation
```

## E14 — LLM tie-breaker

Purpose:

```text
test ambiguous pairs only
```

---

# 68. Submission Strategy

There are 5 submissions per day over 3 days.

Treat submissions as controlled experiments.

Do not burn submissions on insignificant changes.

Suggested progression:

### Early submissions

```text
baseline
+
normalization
+
blocking
+
features
+
LightGBM
```

### Mid submissions

```text
competition
+
hard negatives
+
calibration
+
decoder
```

### Later submissions

```text
embeddings
+
cross encoder
+
France/graph experiments
```

### Final

Use the configuration selected by local validation and a clean reproducible run.

---

# 69. Clean-Final-Run Procedure

Before the final package:

1. Clone/checkout the exact final Git commit.
2. Create a clean environment.
3. Install pinned dependencies.
4. Verify model licenses.
5. Run from the provided dataset directories.
6. Recreate normalized/cache artifacts.
7. Generate final outputs.
8. Run the official validator.
9. Inspect candidate subset consistency.
10. Create the final ZIP.
11. Preserve the exact Git commit/configuration used.

---

# 70. What to Do If Blocking Is the Problem

If:

```text
candidate recall is low
```

focus on:

```text
normalization
name TF-IDF
address TF-IDF
rare tokens
reverse retrieval
embedding retrieval
top-K
```

Do not immediately change the classifier.

---

# 71. What to Do If Classification Is the Problem

If:

```text
candidate recall is high
but F0.5 is low
```

focus on:

```text
features
hard negatives
competition features
calibration
threshold
decoder
```

---

# 72. What to Do If Precision Is Poor

Investigate:

```text
same name / different location
same address / different business
generic names
postcode collisions
country inconsistencies
look-alike businesses
```

Likely fixes:

```text
competition features
hard negatives
address components
postcode conflict features
higher/metric-aware threshold
```

---

# 73. What to Do If Recall Is Poor

Investigate:

```text
blocking
over-aggressive normalization
too-small K
threshold
decoder rejecting true multi-match entities
```

Distinguish:

```text
blocked-out true pair
```

from:

```text
candidate rejected by model
```

---

# 74. France Diagnostics

If France underperforms:

1. compare character TF-IDF recall
2. compare embedding recall
3. inspect French legal-form normalization
4. inspect address normalization
5. run leave-one-country-out validation
6. inspect French false positives/negatives
7. only then consider pseudo-labeling

Do not use external French business data.

---

# 75. Model/Component Decision Rules

Use these rules:

## Keep a component if

```text
validation F0.5 improves
```

or:

```text
candidate recall materially improves
without unacceptable candidate growth/runtime
```

## Reject a component if

```text
score does not improve
```

or:

```text
score improvement is tiny compared with complexity/runtime
```

or:

```text
it creates unstable behavior across folds
```

Do not keep an advanced component just because it sounds sophisticated.

---

# 76. Cut List When Time Runs Short

Remove in this order:

```text
1. LLM tie-breaker
2. pseudo-labeling
3. graph transitive support
4. cross-encoder
5. embedding retrieval
```

Do not remove the essentials:

```text
metric replica
normalization
blocking
pairwise features
LightGBM
competition features
threshold/decoder evaluation
validator
submission generation
```

---

# 77. Recommended Core Architecture

```text
                         RAW DATA
                            │
                            ▼
                    Safe TSV Loader
                            │
                            ▼
                   Data Validation
                            │
                            ▼
                Canonicalization Layer
                            │
           ┌────────────────┼────────────────┐
           │                │                │
           ▼                ▼                ▼
       Name Views      Address Views     Metadata
           │                │                │
           └────────────────┼────────────────┘
                            │
                            ▼
                 MULTI-CHANNEL BLOCKING
                            │
       ┌──────────┬─────────┼────────┬───────────┐
       │          │         │        │           │
       ▼          ▼         ▼        ▼           ▼
     Exact      Rare      Name     Address    Embedding
     Keys       Tokens    TF-IDF   TF-IDF      optional
       │          │         │        │           │
       └──────────┴─────────┴────────┴───────────┘
                            │
                            ▼
                    Reverse Retrieval
                            │
                            ▼
                   Candidate Union + K
                            │
                            ▼
                      Candidate Pairs
                            │
          ┌─────────────────┼──────────────────┐
          │                 │                  │
          ▼                 ▼                  ▼
       Pair Features  Competition Features  Cross Encoder
                                             optional
          │                 │                  │
          └─────────────────┼──────────────────┘
                            ▼
                         LightGBM
                            │
                            ▼
                   Probability Calibration
                            │
                            ▼
                  Optional Graph Rules
                            │
                            ▼
                Expected-F0.5 Decoder
                            │
                            ▼
                    Final Predictions
                            │
                 ┌──────────┴──────────┐
                 ▼                     ▼
     matching_results.tsv   candidate_pairs.tsv
                 │                     │
                 └──────────┬──────────┘
                            ▼
                    Official Validator
```

---

# 78. Final Recommended Priority

## HIGH PRIORITY

```text
1. Data loader + validation
2. Exact macro F0.5
3. EDA
4. Robust normalization
5. Exact + TF-IDF blocking
6. Pairwise features
7. LightGBM
8. Hard negatives
9. Competition features
10. Threshold tuning
11. Expected-F0.5 decoder
12. Submission validation
```

## MEDIUM PRIORITY

```text
13. Noise-table mining
14. Reverse blocking
15. Probability calibration
16. Multilingual embedding blocking
17. Cross-encoder stacking
18. Leave-one-country-out validation
```

## EXPERIMENTAL

```text
19. Graph consistency
20. France pseudo-labeling
21. LLM tie-breaker
22. Set-level cross-encoder
```

---

# 79. Final Codex Instructions

Codex should implement incrementally.

### First inspect

- actual repository
- actual dataset paths
- available Python version
- available CPU/GPU
- installed libraries
- challenge validator location

Do not assume the project structure if the repository already has one.

### Then implement in this order

```text
1. Safe TSV loading.
2. Exact macro-F0.5 metric.
3. Ground-truth parsing.
4. EDA report.
5. Entity-level universe split.
6. Name normalization.
7. Address normalization.
8. Exact-key blocking.
9. Name TF-IDF blocking.
10. Address TF-IDF blocking.
11. Candidate union.
12. Candidate recall measurement.
13. Pairwise feature extraction.
14. Hard-negative dataset.
15. LightGBM baseline.
16. Global threshold tuning.
17. Submission generation.
18. Official validator.
19. Competition features.
20. Probability calibration.
21. Expected-F0.5 decoder.
22. Noise-table mining.
23. Reverse blocking.
24. Optional embedding retrieval.
25. Optional cross-encoder stacking.
26. Optional graph rules.
27. Optional France adaptation.
28. Optional pseudo-labeling.
29. Optional LLM tie-breaker.
30. Final reproducible clean run.
```

After every major stage:

```text
run tests
run validation
save metrics
save configuration
cache expensive artifacts
commit working code
```

Never allow a new experimental component to overwrite the best known configuration.

---

# 80. Final Design Principles

Codex must follow these rules throughout the implementation:

### Rule 1
Do not build the most complex system first.

### Rule 2
Every advanced component must be optional.

### Rule 3
Optimize using the actual validation metric.

### Rule 4
Never hard-code US/India as the only countries.

### Rule 5
Never use external business identity lookup.

### Rule 6
Do not assume one-owner feed records before verifying ground truth.

### Rule 7
Do not leak validation/test information into training labels.

### Rule 8
Keep a working submission pipeline at all times.

### Rule 9
Every feature should have an explainable purpose.

### Rule 10
Prefer measurable improvements over architectural complexity.

---

# 81. Bottom Line

The solution should not be thought of as just a classifier.

It is a three-part optimization system:

```text
PART A — RETRIEVAL
Can we put the true match into the candidate set?

PART B — PAIR CLASSIFICATION
Given a candidate, is it actually the same business?

PART C — ENTITY DECISION
How many candidates should this Source 1 entity output, including zero?
```

The corresponding technologies are:

```text
A:
normalization
blocking
TF-IDF
optional embeddings
reverse retrieval

B:
similarity features
competition features
hard negatives
LightGBM
optional cross encoder
calibration

C:
threshold tuning
expected-F0.5 decoding
singleton handling
multi-match handling
optional structural constraints
```

The strongest expected architecture is therefore:

```text
HIGH-RECALL RETRIEVAL
        +
RICH PAIR FEATURES
        +
PRECISION-ORIENTED ML
        +
COMPETITION CONTEXT
        +
CALIBRATED PROBABILITIES
        +
METRIC-AWARE DECODING
```

with advanced components added only when the actual dataset and local experiments demonstrate that they are worthwhile.

**Implementation principle for Codex: start with a correct, testable, CPU-capable baseline and progressively turn on advanced components.**
