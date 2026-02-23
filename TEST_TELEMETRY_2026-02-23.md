# Test Telemetry Report — 2026-02-23

## Scope

This document captures telemetry and test outcomes observed on 2026-02-23 after introducing structured performance metrics and CI report generation.

- Repository: `capnplanet/PredictiveDeterministicModel`
- Branch: `main`
- Latest tested commit at report time: `11256882c455508d1f629c3e9803ff81e0e7d736`

## Local Telemetry Snapshot

Source files:

- `data/performance_metrics.jsonl`
- `data/performance_report.json`

Aggregated summary (`data/performance_report.json`):

- `generated_at`: `2026-02-23T22:48:16.895606+00:00`
- `total_events`: `2`
- `status_counts`: `{ "ok": 2 }`
- `event_counts`:
  - `api.request`: `1`
  - `ingestion.entities_csv`: `1`

Duration stats (ms):

| Event | Count | Avg | P95 | Max |
|---|---:|---:|---:|---:|
| `api.request` | 1 | 11.372 | 11.372 | 11.372 |
| `ingestion.entities_csv` | 1 | 9.375 | 9.375 | 9.375 |

Recent raw telemetry lines:

```json
{"duration_ms": 9.375, "event": "ingestion.entities_csv", "failed_rows": 2, "file_path": "data/uploads/entities/entities.csv", "status": "ok", "success_rows": 0, "throughput_rows_per_sec": 0.0, "timestamp": "2026-02-23T22:47:19.027587+00:00", "total_rows": 2}
{"duration_ms": 11.372, "event": "api.request", "method": "POST", "path": "/ingest/entities", "status": "ok", "status_code": 200, "timestamp": "2026-02-23T22:47:19.028127+00:00"}
```

## Test Execution Evidence (Today)

Local checks executed:

1. `python -m ruff check app --fix`  
   - Result: `18 fixed, 0 remaining`
2. `python -m ruff check app`  
   - Result: `All checks passed`
3. `PYTHONPATH=. pytest -q app/tests/test_performance_metrics.py`  
   - Result: `2 passed`

## GitHub Actions Snapshot (Today)

### Finalized runs for commit `11256882c455508d1f629c3e9803ff81e0e7d736`

- Backend CI — `failure`  
  https://github.com/capnplanet/PredictiveDeterministicModel/actions/runs/22328528874  
  Failed step: `Type check (mypy)`
- Determinism Matrix CI — `success`  
  https://github.com/capnplanet/PredictiveDeterministicModel/actions/runs/22328528871  
  Jobs passed: `determinism (3.11)`, `determinism (3.12)`, `compare`
- E2E CI — `failure`  
  https://github.com/capnplanet/PredictiveDeterministicModel/actions/runs/22328528863  
  Failed step: `Run E2E tests`

### Prior run context (same day)

For commit `6f20a7104e2e401d905b6a8c33060bfb492c816d`:

- Backend CI — `failure` (failed at Ruff lint stage)
- Determinism Matrix CI — `success`
- E2E CI — `failure`

## Notes

- The local telemetry dataset currently reflects a small test sample (`2` events). This is expected given targeted local validation commands.
- Determinism matrix checks are green for the monitored commit.
- Documentation-only commit `d40293ad898e394025c697c1fc2e4bfc2188cece` does not trigger these workflows because workflow path filters target backend/frontend/workflow files.
- This document was updated after workflow completion to reflect final statuses.
