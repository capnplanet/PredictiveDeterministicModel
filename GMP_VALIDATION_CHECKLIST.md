# GMP Validation Checklist

This checklist defines a GMP-focused validation and promotion standard for model-assisted manufacturing oversight and continuous improvement workflows.

Use this document when validating data ingestion, model behavior, explainability, telemetry, and release readiness for GMP decision-support operations.

## Scope and Intended Use

- Applies to GMP-oriented batch, lot, line, and process monitoring use cases.
- Supports decision support for QA, manufacturing science, and compliance teams.
- Does not authorize autonomous product release decisions without human review.

## Validation Objectives

- Verify deterministic and reproducible model behavior under fixed inputs and runtime controls.
- Verify complete data lineage from ingestion through prediction and explanation outputs.
- Verify evidence quality for internal QA review and external audit/inspection readiness.
- Verify release gates are enforced before production promotion.

## Prerequisites

- Database migrations applied: `alembic upgrade head`.
- API, worker, Redis, and database running.
- Training and prediction configs stored in versioned files.
- Controlled test dataset available for GMP scenario replay.

## Checklist

### A. Data Governance and Integrity

- [ ] Confirm source data contract for entities, events, interactions, and artifacts is approved.
- [ ] Confirm required GMP fields are present (batch/lot identifiers, timestamps, unit/line context, operator or system source).
- [ ] Confirm time ordering is stable and timezone policy is consistently applied.
- [ ] Confirm ingestion rejects malformed records with explicit error accounting.
- [ ] Confirm artifact hashing and integrity checks are enabled and retained.
- [ ] Confirm checkpoint and resume behavior works for high-volume loads without duplication.

### B. Deterministic Ingestion and Feature Pipeline

- [ ] Confirm repeated ingestion of identical input produces equivalent persisted outcomes.
- [ ] Confirm feature extraction version and cache semantics are stable across repeated runs.
- [ ] Confirm stale feature caches are invalidated when extraction logic changes.
- [ ] Confirm run manifests capture config and feature lineage metadata.

### C. Training Run Lifecycle and Reproducibility

- [ ] Confirm training lifecycle transitions are persisted (`pending -> running -> success|failed`).
- [ ] Confirm async enqueue idempotency keys prevent duplicate training job creation.
- [ ] Confirm determinism-check passes for representative GMP datasets.
- [ ] Confirm model artifact identifiers and hashes are captured and retained.
- [ ] Confirm threshold policies and acceptance criteria are versioned and reviewable.

### D. Prediction and Explainability Controls

- [ ] Confirm prediction outputs include expected heads (regression/probability/ranking as configured).
- [ ] Confirm explanation payloads are present for high-impact decisions.
- [ ] Confirm repeated predictions on fixed run/data preserve deterministic ordering and values.
- [ ] Confirm ranking and probability thresholds map to documented QA action tiers.
- [ ] Confirm all high-risk alerts route to human review queues.

### E. Async Queue and Telemetry Validation

- [ ] Confirm queue health endpoint reports backlog, oldest pending age, and saturation metrics.
- [ ] Confirm telemetry includes async enqueued/success/failed events for training, feature extraction, and batch prediction.
- [ ] Confirm correlation ID propagation is present from request to async task and telemetry events.
- [ ] Confirm no unexplained gaps in event counts during stress or replay tests.

### F. Compliance, Auditability, and Change Control

- [ ] Confirm immutable audit trail includes who/what/when for ingest, train, and predict operations.
- [ ] Confirm run lineage can reconstruct each released model from source data and config state.
- [ ] Confirm deviations and CAPA-linked model updates are traceable to evidence and approvals.
- [ ] Confirm access control boundaries are enforced for model promotion and operational overrides.
- [ ] Confirm SOP references and validation records are attached to the release package.

### G. CI and Promotion Gates

- [ ] Confirm `Backend CI` passes on the candidate commit.
- [ ] Confirm `Determinism Matrix CI` passes on the candidate commit.
- [ ] Confirm `Phase 4 Release Gate` passes on the candidate commit.
- [ ] Confirm performance-report smoke includes expected event coverage before promotion.
- [ ] Confirm documented sign-off by QA/compliance owner and technical owner.

## Recommended Validation Commands

```bash
PYTHONPATH=. python -m app.cli determinism-check
PYTHONPATH=. python -m app.cli performance-report
curl -sS http://localhost:8000/health/queues
```

Example async training enqueue with correlation context:

```bash
curl -sS -X POST http://localhost:8000/train/async \
  -H "Content-Type: application/json" \
  -H "x-correlation-id: gmp-validation-001" \
  -d '{"idempotency_key":"gmp-train-001","config":{"epochs":1,"batch_size":8}}'
```

## Evidence Package Template

Store and version the following artifacts for each GMP validation cycle:

- Validation scope and dataset manifest
- Ingestion reports and error summaries
- Determinism-check report output
- Model run metadata and artifact hashes
- Prediction/explanation sampling report
- Queue health and telemetry summaries
- CI run references for required gates
- QA/compliance/technical sign-off record

## Release Decision

Promote only when all checklist items are complete and sign-off records are attached. Any failed or waived item must include documented rationale, owner, mitigation, and revalidation date.
