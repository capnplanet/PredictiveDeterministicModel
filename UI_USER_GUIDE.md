# UI User Guide - Defense-Grade Decision Console

This guide explains how to use the current frontend UI end-to-end, including what each tab does, what data is required, and how to troubleshoot common issues.

## Audience

Use this guide if you are:
- Operating the web console for ingest, training, and prediction
- Running demos for stakeholders
- Validating backend behavior through the UI

## Before You Start

1. Start the stack:

```bash
docker-compose up -d
```

2. Confirm backend health:

```bash
curl http://localhost:8000/health
```

3. Open the UI:
- http://localhost:5173

## UI Layout Overview

At the top of the page, you will see:
- Platform title: Defense-Grade Decision Console
- Mode badge: Development
- Active View badge: shows the currently selected tab

Primary navigation tabs:
- Data Intake
- Model Ops
- Run Ledger
- Inference

Global status banner behavior:
- Neutral: baseline/ready state
- Warning: operation started or input missing
- Success: operation completed
- Error: request failed (typically includes HTTP error code)

## Recommended Workflow (Happy Path)

Follow this sequence for reliable results:
1. Upload Entities CSV in Data Intake
2. Upload Events CSV in Data Intake
3. Upload Interactions CSV in Data Intake
4. Go to Model Ops and execute training
5. Go to Run Ledger and sync runs
6. Go to Inference and execute prediction

## Tab-by-Tab Guide

## 1) Data Intake

Purpose:
- Load the three CSV datasets used by training and prediction.

Controls:
- Entities Manifest file input
- Events Stream file input
- Interactions Graph file input

What happens on upload:
- Each input immediately calls a backend ingest endpoint when a file is selected.
- The status banner reports rows accepted and total rows processed.

Endpoint mapping:
- Entities Manifest -> POST /ingest/entities
- Events Stream -> POST /ingest/events
- Interactions Graph -> POST /ingest/interactions

Success message pattern:
- Upload complete: <kind> (<success_rows>/<total_rows> rows).

Example:
- Upload complete: entities (250/250 rows).

Important notes:
- The current UI does not include artifact upload controls.
- Artifact endpoints exist in backend API, but they are not exposed in this UI flow.

## 2) Model Ops

Purpose:
- Trigger deterministic model training.

Control:
- Execute Training Operation button

Endpoint mapping:
- POST /train

Success message pattern:
- Training complete. Run ID: <run_id>

What to expect:
- Run ID is a long hex string that identifies the model run and artifact directory.
- If training fails, the status banner shows an error and HTTP status details.

## 3) Run Ledger

Purpose:
- Review and refresh historical runs.

Controls:
- Sync Run Ledger button

Endpoint mapping:
- GET /runs

Run metrics cards:
- Latest Run: latest run id (shortened in card)
- Run Count: number of runs loaded
- Model Health: average of available reg_r2, cls_f1, rank_ndcg@10 from latest run

Run list row details:
- Full run id
- Localized run timestamp
- Metric chips:
  - r2 <value>
  - f1 <value>
  - ndcg <value>

Success message pattern:
- Run ledger refreshed. <count> records available.

## 4) Inference

Purpose:
- Generate predictions for one or more entity IDs.

Controls:
- Entity ID input: comma-separated ids
- Execute Inference button

Endpoint mapping:
- POST /predict

Input rules:
- Enter one or more entity IDs separated by commas.
- Whitespace is trimmed automatically.
- Empty input is blocked with a warning.

Success message pattern:
- Prediction complete from run <run_id>.

Predict metrics cards:
- Entities Predicted
- Avg Probability
- Avg Rank Score

Prediction list fields per entity:
- Entity ID
- Reg <regression>
- Prob <probability>
- Rank <ranking_score>

## CSV Expectations for UI Uploads

The backend validates CSV structure. Use these headers to avoid ingest failures.

Entities CSV (minimum expected columns):
- entity_id
- attributes
- created_at (recommended)

Events CSV (commonly used columns):
- timestamp
- entity_id
- event_type
- event_value
- event_metadata (optional)

Interactions CSV (commonly used columns):
- timestamp
- src_entity_id
- dst_entity_id
- interaction_type
- interaction_value
- metadata (optional)

Practical tip:
- Start from the synthetic data generator output for known-good format.

```bash
python -m app.training.synth_data
```

## Error Handling and Troubleshooting

## Upload appears successful but rows are 0/N

Cause:
- CSV header mismatch or malformed field values.

Action:
1. Verify headers exactly match expected names.
2. Check quoting/escaping for JSON-like fields such as attributes.
3. Re-upload and watch status banner response.

## Train fails

Cause:
- Missing required data or backend validation/runtime error.

Action:
1. Ensure all three CSV types were uploaded first.
2. Check backend container logs:

```bash
docker-compose logs backend --tail=200
```

3. Retry from Model Ops.

## Run Ledger is empty

Cause:
- No completed training run yet, or backend issue.

Action:
1. Execute training in Model Ops.
2. Return to Run Ledger and click Sync Run Ledger.

## Prediction fails or returns no entities

Cause:
- Invalid entity IDs, no trained run, or entity not represented in data.

Action:
1. Confirm a successful run exists in Run Ledger.
2. Use IDs that exist in uploaded entities data.
3. Retry with a single known entity ID first.

## Operational Best Practices

- Keep Data Intake -> Model Ops -> Run Ledger -> Inference order.
- Use small validation CSVs before large uploads.
- Refresh Run Ledger after each training operation.
- For demos, pre-stage known-good CSVs and sample entity IDs.

## UI-to-API Mapping Reference

- Data Intake
  - Entities Manifest -> POST /ingest/entities
  - Events Stream -> POST /ingest/events
  - Interactions Graph -> POST /ingest/interactions
- Model Ops
  - Execute Training Operation -> POST /train
- Run Ledger
  - Sync Run Ledger -> GET /runs
- Inference
  - Execute Inference -> POST /predict

## Known Scope of Current UI

Included in UI:
- CSV ingestion for entities, events, interactions
- Train trigger
- Run history and core metrics chips/cards
- Entity prediction and output metrics

Not currently included in UI:
- Artifact manifest upload and single artifact upload forms
- Advanced training config editing from the frontend
- Run detail drill-down view (/runs/{run_id})

## Quick Demo Script (UI-Oriented)

1. Open Data Intake and upload three CSV files.
2. Move to Model Ops and click Execute Training Operation.
3. Move to Run Ledger and click Sync Run Ledger.
4. Move to Inference, enter 1-3 known entity IDs, and click Execute Inference.
5. Narrate result chips: Reg, Prob, Rank for each entity.

This sequence is aligned with current tab names and control labels in the live UI.
