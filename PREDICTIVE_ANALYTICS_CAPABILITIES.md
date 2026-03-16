# Predictive Analytics Capabilities Guide

This document explains what predictive analytics this repository can do today, how the system works end to end, and what technical terms mean in plain language.

## Who This Is For

This guide is for engineers, analysts, product teams, and compliance reviewers who want to understand:

- What predictions the platform can produce
- How data moves from raw input to scored output
- How reliability and reproducibility are enforced
- What each technical term means in practical terms

## Executive Summary

The repository provides a full predictive analytics platform with these major capabilities:

- Data ingestion for structured records and media artifacts
- High-volume chunked ingestion with checkpoint/resume support
- Feature extraction from image, audio, and video files
- Deterministic model training for repeatable outcomes
- Async queue-backed execution for training, extraction, and batch inference
- Multi-task prediction per entity (regression, probability, and ranking)
- Explainability outputs that describe why a prediction was made
- Natural-language query over predictions with intent-aware ranking
- Agent-driven analytics workflows with approval, audit, and determinism gates
- Queue/worker health telemetry plus structured performance reporting
- CI reproducibility and release-gate checks

In plain terms: you can load data, train a model, score entities, ask questions in natural language, and trace how and why results were produced.

## Capability Map

### 1) Structured Data Ingestion

What it does:

- Accepts CSV uploads for entities, events, and interactions
- Stores normalized records in PostgreSQL
- Returns row-level ingestion summaries

Where in API:

- POST /ingest/entities
- POST /ingest/events
- POST /ingest/interactions

Jargon explained:

- Entity: the core item being analyzed (for example a customer, account, product, patient, or device).
- Event: a timestamped action or observation attached to an entity.
- Interaction: a relationship between two entities, often with a strength value.

Why this matters:

- Predictive models are only as useful as the data graph they can learn from.
- The entity event interaction structure gives the model both history and context.

### 1.1) Chunked Ingestion and Checkpoint Resume

What it does:

- Supports configurable chunk-size processing for large CSV/manifest loads.
- Persists checkpoint progress so interrupted ingest can resume deterministically.
- Exposes checkpoint controls through ingest API form fields.

Jargon explained:

- Chunking: processing data in smaller batches instead of one giant transaction.
- Checkpoint resume: continuing from the last confirmed processed row after interruption.

Why this matters:

- Improves throughput and reliability for high-volume datasets.
- Avoids restart-from-zero behavior after failures.

### 2) Artifact Ingestion and Multimodal Support

What it does:

- Ingests artifact manifests and single uploaded files
- Supports image, audio, and video artifact types
- Computes SHA256 hashes to ensure file identity and integrity
- Associates artifacts to entities and timestamps

Where in API:

- POST /ingest/artifacts
- POST /ingest/artifact

Jargon explained:

- Artifact: non-tabular evidence such as images, audio clips, or videos.
- SHA256 hash: a fingerprint of file bytes; if bytes change, the hash changes.
- Data integrity: confidence that stored data has not been altered unexpectedly.

Why this matters:

- Real-world predictions often require context beyond CSV columns.
- Hashing supports deduplication, traceability, and audit confidence.

### 3) Feature Extraction Pipeline

What it does:

- Processes pending artifacts into numerical vectors
- Caches extracted vectors in data/feature_cache
- Records feature dimensions and version hash in the database

Where in API:

- POST /features/extract

Jargon explained:

- Feature vector: a numeric representation a model can consume.
- Feature extraction: converting raw inputs (pixels, waveforms, frames) into numbers.
- Feature version hash: a fingerprint of feature code/config state to track compatibility.

Why this matters:

- Models cannot learn directly from files; they learn from numeric features.
- Versioned features reduce silent drift between training and inference behavior.

### 4) Deterministic Model Training

What it does:

- Trains a multi-task model that predicts three outputs per entity:
- Regression output (continuous numeric estimate)
- Classification probability output (likelihood estimate)
- Ranking score output (relative priority ordering)
- Persists model artifacts, config, metrics, and data manifest for each run
- Uses deterministic controls for repeatable outcomes

Where in API:

- POST /train
- GET /runs
- GET /runs/{run_id}

Key implementation behaviors:

- Seeds are fixed for Python, NumPy, and PyTorch
- Deterministic algorithms are enabled in PyTorch
- Thread counts are constrained to reduce nondeterministic scheduling effects
- Run identifiers are derived from hashed inputs (config + manifests + versions)

Jargon explained:

- Deterministic: same code + same data + same config produce the same result.
- Run: one complete training execution with a unique identifier.
- Model artifact: saved model files and metadata from a run.
- Data manifest: a record of what data and split settings were used.

Why this matters:

- Enables reproducibility for debugging, governance, and regulated use cases.
- Makes model behavior auditable over time.

### 5) Data Splitting and Threshold Policies

What it does:

- Splits data into train, validation, and test partitions
- Supports random or time-based splitting strategies
- Evaluates metrics per split
- Supports threshold policy checks and optional hard enforcement

Jargon explained:

- Train split: subset used to fit model parameters.
- Validation split: subset used to evaluate model tuning quality.
- Test split: holdout subset used for unbiased performance checks.
- Threshold policy: minimum acceptable metric values for a defined corpus.

Why this matters:

- Keeps evaluation honest and prevents overconfidence from train-only metrics.
- Supports quality gates before promoting or relying on a model.

### 6) Prediction and Inference

What it does:

- Scores one or more entity IDs using a specific run or latest available run
- Returns:
- Regression value
- Probability value
- Ranking score
- Embedding vector
- Narrative text (template and optionally LLM-augmented)
- Optional explanations per entity

Where in API:

- POST /predict

Jargon explained:

- Inference: using a trained model to generate predictions on input data.
- Embedding: compressed numeric representation of an entity in model space.
- Narrative mode: controls whether prediction text is deterministic template text, LLM-generated text, or both.

Why this matters:

- Gives actionable scores plus human-readable context.
- Supports both machine consumption (scores) and analyst workflows (narratives).

### 7) Built-In Explainability

What it does:

- Computes integrated-gradient style fused attributions
- Returns attention weights for sequence context
- Returns artifact contribution scores

Jargon explained:

- Explainability: methods that help a human understand why the model scored an item a certain way.
- Attribution: estimated contribution of an input component to an output.
- Attention weights: relative focus values across sequence inputs.
- Integrated gradients: a method that estimates feature influence by integrating gradients from a baseline to actual input.

Why this matters:

- Improves trust and troubleshooting.
- Helps teams answer what drove this prediction.

### 8) Natural-Language Query Over Predictions

What it does:

- Accepts plain-language query text
- Finds candidate entities using explicit, fuzzy, or broad matching
- Infers sorting intent (strongest, weakest, default)
- Infers probability intent (elevated or default)
- Produces ranked prediction results with interpretation metadata
- Optionally enriches narratives using an LLM endpoint

Where in API:

- POST /query

Jargon explained:

- Fuzzy match: approximate text match rather than exact ID match.
- Retrieval strategy: the method used to find candidates before ranking.
- Intent inference: guessing desired ordering/filtering from user wording.

Why this matters:

- Analysts can ask business questions directly, not only by IDs.
- The response still stays grounded in deterministic prediction outputs.

### 9) Agentic Analytics Workflows

What it does:

- Plans and executes tool-based analytics steps
- Supports tools for training, prediction, query, run metrics, and determinism verification
- Supports approval workflow before execution when required
- Tracks run and step status, retries, timeout behavior, and metrics
- Writes immutable audit events for governance traceability
- Exposes compliance summaries for approval and determinism outcomes

Where in API:

- POST /agents/runs
- GET /agents/runs/{agent_run_id}/plan
- POST /agents/runs/{agent_run_id}/approve
- POST /agents/runs/{agent_run_id}/steps/{step_index}/execute
- POST /agents/runs/{agent_run_id}/loop
- POST /agents/runs/{agent_run_id}/control
- GET /agents/runs/{agent_run_id}
- GET /agents/runs
- GET /agents/runs/{agent_run_id}/audit
- GET /agents/compliance/approval-summary
- GET /agents/compliance/determinism-audit

Jargon explained:

- Agent run: one orchestrated analytic workflow execution.
- Tool registry: controlled list of actions the agent can execute.
- Immutable audit event: a log entry that cannot be edited or deleted.
- Determinism gate: a rule that blocks successful completion when parity checks fail.

Why this matters:

- Automates common analytics tasks while preserving oversight and traceability.
- Combines speed of orchestration with governance controls.

### 10) Reproducibility and Determinism Verification

What it does:

- Reproduces a prior run using its saved configuration
- Compares run ID, model hash, metrics, and prediction parity
- Provides synthetic end-to-end determinism checks

Jargon explained:

- Parity: exact matching between two runs on a comparison dimension.
- Reproducibility check: rerun and compare to prove repeatability.

Why this matters:

- Detects hidden nondeterminism early.
- Supports CI gating and compliance evidence.

### 11) Async Orchestration and Queue Health

What it does:

- Enqueues long-running operations and tracks lifecycle status through task APIs.
- Isolates workload classes in separate queues (`training`, `extraction`, `batch_inference`).
- Provides queue health endpoint coverage with backlog, pending age, and saturation metadata.

Where in API:

- POST /train/async, GET /train/async/{task_id}
- POST /features/extract/async, GET /features/extract/async/{task_id}
- POST /predict/async, GET /predict/async/{task_id}
- GET /health/queues

Jargon explained:

- Queue backlog: pending + running work waiting to drain.
- Saturation: how close current running workload is to configured worker capacity.
- Idempotency key: stable client key that prevents duplicate effective execution.

Why this matters:

- Keeps heavy jobs off synchronous request paths.
- Improves system responsiveness under load while preserving deterministic controls.

### 12) Telemetry and Operational Observability

What it does:

- Emits structured timing and status events for ingestion, feature extraction, training, prediction, and determinism checks
- Propagates correlation IDs from request to async task to telemetry event payloads
- Stores events in JSON Lines format for post-run reporting
- Supports performance report generation from telemetry logs

Jargon explained:

- Telemetry: machine-generated operational measurements.
- Observability: ability to inspect what a system did, when, and how fast.
- JSON Lines: one JSON object per line, easy for streaming and analytics.

Why this matters:

- Helps teams spot regressions and bottlenecks.
- Provides evidence for service-level performance claims.

## End-to-End Predictive Workflow

A typical workflow from raw data to decision support:

1. Ingest entities, events, interactions, and artifacts.
2. Extract artifact features.
3. Train deterministic model run.
4. Retrieve run details and metrics.
5. Predict for target entities via /predict.
6. Query prediction space via /query for analyst exploration.
7. If using agent mode, run guided tool plans with approval and audit tracking.
8. Review compliance summaries and determinism audit endpoints.

## What Makes This Repository Distinct

Compared with many analytics repositories, this one combines:

- Multimodal ingestion and feature extraction
- Multi-head predictive outputs in one pipeline
- Deterministic training and reproducibility checks as first-class features
- Queue-backed heavy workload orchestration with lifecycle and idempotency controls
- Queue health and saturation telemetry for operations visibility
- Explainability outputs integrated into prediction responses
- Agentic workflows with approval, immutable audit events, and compliance summaries

In plain language: this is not only a model training script. It is an operational analytics system designed to be repeatable, inspectable, and usable by both engineers and analysts.

## Practical Notes and Limitations

- Determinism requires stable environment assumptions; CI checks help enforce this.
- LLM narrative text can vary when enabled, while core numeric prediction outputs remain deterministic under fixed run/data conditions.
- Prediction quality depends on data quality, feature coverage, and target definitions in attributes.
- Explainability outputs indicate contribution signals, not absolute causation proof.

Jargon explained:

- Causation: one factor directly causes another.
- Correlation: two factors move together, but one may not cause the other.

## Quick Glossary

- Regression: predicting a continuous number.
- Classification probability: predicting likelihood of class membership.
- Ranking score: predicting relative priority order.
- Feature engineering: preparing model-ready numeric inputs.
- Determinism: repeatable identical results under identical conditions.
- Reproducibility: ability to rerun and confirm prior outcomes.
- Explainability: techniques that make model decisions understandable.
- Lineage: recorded chain of what created what (data, features, model run, agent run).
- Audit trail: timestamped record of actions and decisions.
- Threshold policy: minimum quality requirements for model metrics.
- Backlog: count of queued/running work not yet complete.
- Saturation ratio: running tasks divided by queue concurrency capacity.
- Correlation ID: trace key linking one request across API, tasks, and telemetry.

## Suggested Next Documentation Additions

If you want to extend this guide later, useful follow-ups are:

- A full endpoint cookbook with request and response examples per route
- A metric dictionary defining each training and inference metric
- A governance playbook for approvals, escalation, and audit retention
- A model promotion checklist tied to threshold and determinism gates
