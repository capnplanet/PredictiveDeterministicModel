# Corpus Validation Guide

This guide provides reproducible corpus-based validation for predictive analytics quality and determinism.

## Current Coverage

This validation guide aligns with the current repository capabilities:

- Deterministic model training and inference
- Query endpoint validation (`/query`) with intent-aware ordering
- Optional LLM augmentation checks for interpretation and narratives
- CI validation via Backend CI, Determinism Matrix CI, and E2E CI
- Telemetry aggregation using `performance-report`

## Goals

- Validate predictive quality across regression, classification, and ranking outputs.
- Validate deterministic behavior for repeat runs on fixed corpora.
- Use both synthetic and public corpora.

## Prerequisites

- Running Postgres reachable by `DATABASE_URL`.
- Python dependencies installed from `requirements.txt`.
- DB schema applied: `alembic upgrade head`.

## Baseline Synthetic Corpus

Generate and validate with the built-in synthetic generator.

```bash
PYTHONPATH=. python -m app.cli generate-synth data/corpora/synth_baseline 500 50000 20000 2000 42
PYTHONPATH=. python -m app.cli ingest-entities data/corpora/synth_baseline/entities.csv
PYTHONPATH=. python -m app.cli ingest-events data/corpora/synth_baseline/events.csv
PYTHONPATH=. python -m app.cli ingest-interactions data/corpora/synth_baseline/interactions.csv
PYTHONPATH=. python -m app.cli ingest-artifacts data/corpora/synth_baseline/artifacts_manifest.csv
PYTHONPATH=. python -m app.cli extract-features
PYTHONPATH=. python -m app.cli train --config-path data/api_train_config.json
```

Recommended acceptance targets for synthetic baseline:

- `reg_r2 >= 0.80`
- `cls_f1 >= 0.85`
- `rank_ndcg@10 >= 0.80`

Note: Exact values can vary by corpus profile size and target construction. Keep thresholds versioned with corpus metadata and revisit thresholds after data schema changes.

## Determinism Check (Synthetic)

Run deterministic end-to-end verification:

```bash
PYTHONPATH=. python -m app.cli determinism-check
```

Expected report fields to be true:

- `same_run_id`
- `same_metrics`
- `same_model_sha`
- `same_predictions`

## Public Corpus: MovieLens 100k

1. Download `ml-100k.zip` from GroupLens.
2. Use `u.data` as input.

Convert to ingestion-ready CSVs:

```bash
PYTHONPATH=. python -m app.cli prepare-movielens /path/to/ml-100k/u.data data/corpora/movielens_100k
```

Run ingestion/training:

```bash
PYTHONPATH=. python -m app.cli ingest-entities data/corpora/movielens_100k/entities.csv
PYTHONPATH=. python -m app.cli ingest-events data/corpora/movielens_100k/events.csv
PYTHONPATH=. python -m app.cli ingest-interactions data/corpora/movielens_100k/interactions.csv
PYTHONPATH=. python -m app.cli train
```

Recommended acceptance targets:

- `rank_ndcg@10 >= 0.65`
- `rank_spearman >= 0.55`

## Public Corpus: UCI Adult-Like CSV

Prepare a CSV containing a target column named one of `income`, `target`, or `label`.

Convert to ingestion-ready CSVs:

```bash
PYTHONPATH=. python -m app.cli prepare-uci-adult /path/to/adult.csv data/corpora/uci_adult
```

Run ingestion/training:

```bash
PYTHONPATH=. python -m app.cli ingest-entities data/corpora/uci_adult/entities.csv
PYTHONPATH=. python -m app.cli ingest-events data/corpora/uci_adult/events.csv
PYTHONPATH=. python -m app.cli ingest-interactions data/corpora/uci_adult/interactions.csv
PYTHONPATH=. python -m app.cli train
```

Recommended acceptance targets:

- `cls_f1 >= 0.75`
- `cls_precision >= 0.73`
- `cls_recall >= 0.70`

## CI Strategy

- Keep synthetic determinism checks as required PR gates.
- Run public corpus validations on a nightly schedule to avoid long PR feedback loops.
- Store threshold expectations in versioned config reviewed in pull requests.
- Ensure `Backend CI`, `Determinism Matrix CI`, and `E2E CI` all pass before promoting corpus or threshold updates.

## Query Validation (Recommended)

After training on a corpus, validate query behavior with intent-rich prompts:

```bash
curl -sS -X POST http://localhost:8000/query \
	-H "Content-Type: application/json" \
	-d '{"query":"Identify top entities by relationship strength with elevated probability and explain confidence limits.","limit":5}'
```

Check for:

- `interpreted_as` includes intent tags (match/order/probability)
- `llm_used` reflects actual LLM availability
- Stable ranking order for repeated calls against the same run/data
- Narrative text remains grounded in model outputs and ingested evidence

## Telemetry Validation

Generate and archive a performance summary after corpus validation:

```bash
PYTHONPATH=. python -m app.cli performance-report
```

Review report fields for p50/p95 latency, event counts, and error rates before signing off.

## Troubleshooting

- If health endpoint returns `503`, verify DB availability and migration state.
- If determinism checks fail, compare run IDs and `model_sha256.txt` in artifacts.
- If public corpus metrics drift, first verify conversion inputs and target column mapping.
