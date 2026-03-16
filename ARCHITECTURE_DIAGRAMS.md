# Architecture Diagrams and Visual Reference

This document provides visual representations of the system architecture, data flows, and integration patterns.

## Current Architecture Status (March 2026)

The repository now includes deterministic high-volume MLOps operations through Phase 4:

- Async orchestration with Redis + Celery worker queues for training, feature extraction, and batch inference.
- Task lifecycle persistence (`pending`, `running`, `success`, `failed`) in dedicated task tables.
- API support for sync and async routes, including enqueue/status endpoints.
- Queue health telemetry with backlog, oldest pending age, and saturation indicators.
- Request-to-task correlation ID propagation via middleware and performance telemetry.
- Release governance workflow (`Phase 4 Release Gate`) combining determinism, integration, and performance smoke checks.

## System Architecture Overview

```
┌──────────────────────────────────────────────────────────────────────┐
│                           USER INTERFACE                             │
│                                                                      │
│  ┌────────────────────────────────────────────────────────────┐   │
│  │                 React Frontend (Port 5173)                  │   │
│  │                                                              │   │
│  │  ┌──────────┬──────────┬──────────┬──────────┬──────────┐ │   │
│  │  │ Dataset  │  Train   │   Runs   │ Predict  │  Query   │ │   │
│  │  │  Upload  │ Trigger  │  History │  + Explain│ NL Rank │ │   │
│  │  └──────────┴──────────┴──────────┴──────────┴──────────┘ │   │
│  └────────────────────┬─────────────────────────────────────────┘   │
└───────────────────────┼──────────────────────────────────────────────┘
                        │
                        │ HTTP/REST (JSON)
                        │
┌───────────────────────▼──────────────────────────────────────────────┐
│                      BACKEND API LAYER                               │
│                                                                      │
│  ┌────────────────────────────────────────────────────────────┐   │
│  │           FastAPI Application (Port 8000)                   │   │
│  │                                                              │   │
│  │  ┌─────────────────────────────────────────────────────┐  │   │
│  │  │         API Routes (Routers)                        │  │   │
│  │  │  • /health         - Health check                   │  │   │
│  │  │  • /health/queues  - Broker + queue telemetry       │  │   │
│  │  │  • /ingest/*       - Data upload endpoints          │  │   │
│  │  │  • /features/*     - Feature extraction             │  │   │
│  │  │  • /train          - Model training                 │  │   │
│  │  │  • /train/async/*  - Training enqueue + status      │  │   │
│  │  │  • /runs/*         - Training run management        │  │   │
│  │  │  • /predict        - Inference + Explainability     │  │   │
│  │  │  • /predict/async/*- Batch inference queue APIs     │  │   │
│  │  │  • /query          - Query + ranked retrieval       │  │   │
│  │  │  • /agents/*       - Governed agent workflows       │  │   │
│  │  │  • /demo/preload   - One-click synthetic bootstrap  │  │   │
│  │  └─────────────────────────────────────────────────────┘  │   │
│  └────────────────────┬─────────────────────────────────────────┘   │
└───────────────────────┼──────────────────────────────────────────────┘
                        │
                        │
┌───────────────────────▼──────────────────────────────────────────────┐
│                    SERVICE + ORCHESTRATION LAYER                     │
│                                                                      │
│  ┌──────────────────────────┬────────────────────────────────┐    │
│  │   Data Services          │   ML + Queue Services           │    │
│  │                          │                                 │    │
│  │  • csv_ingestion.py      │  • image_features.py           │    │
│  │  • artifact_ingestion.py │  • audio_features.py           │    │
│  │  • feature_extraction.py │  • video_features.py           │    │
│  │  • parquet_export.py     │  • model.py (FullModel)        │    │
│  │                          │  • train.py                     │    │
│  │                          │  • synth_data.py                │    │
│  │                          │  • training_tasks.py            │    │
│  │                          │  • feature_tasks.py             │    │
│  │                          │  • batch_inference_tasks.py     │    │
│  └──────────────────────────┴────────────────────────────────┘    │
│                                                                      │
│  ┌──────────────────────────────────────────────────────────────┐ │
│  │             Redis Broker + Celery Worker Queues              │ │
│  │             Queues: training, extraction, batch_inference    │ │
│  └──────────────────────────────────────────────────────────────┘ │
│                                                                      │
│  ┌──────────────────────────────────────────────────────────────┐ │
│  │             Database Session Management                       │ │
│  │             SQLAlchemy ORM + Alembic Migrations              │ │
│  └──────────────────────────────────────────────────────────────┘ │
└───────────────────────┬──────────────────────────────────────────────┘
                        │
                        │ SQL Queries
                        │
┌───────────────────────▼──────────────────────────────────────────────┐
│                  DATA PERSISTENCE LAYER                              │
│                                                                      │
│  ┌────────────────────────────────────────────────────────────┐   │
│  │           PostgreSQL 16 (Port 5432)                         │   │
│  │                                                              │   │
│  │  Tables:                                                     │   │
│  │  ├─ entities       (Core entity records)                    │   │
│  │  ├─ events         (Time-series data)                       │   │
│  │  ├─ interactions   (Entity relationships)                   │   │
│  │  ├─ artifacts      (Media file metadata)                    │   │
│  │  ├─ model_runs     (Training history + run state)           │   │
│  │  ├─ training_tasks (Async train lifecycle)                  │   │
│  │  ├─ feature_extraction_tasks (Async extraction lifecycle)    │   │
│  │  ├─ batch_inference_tasks (Async predict lifecycle)          │   │
│  │  ├─ agent_runs / agent_steps                                │   │
│  │  └─ agent_audit_events (Immutable governance audit trail)   │   │
│  └────────────────────────────────────────────────────────────┘   │
└──────────────────────────────────────────────────────────────────────┘

┌──────────────────────────────────────────────────────────────────────┐
│                      FILE STORAGE                                    │
│                                                                      │
│  📁 artifacts_store/       (Uploaded media files)                   │
│     ├─ images/                                                       │
│     ├─ audio/                                                        │
│     └─ video/                                                        │
│                                                                      │
│  📁 feature_cache/          (Extracted feature vectors)              │
│     └─ {sha256}.npy         (Cached numpy arrays)                   │
│                                                                      │
│  📁 models/                 (Trained model artifacts)                │
│     └─ {run_id}/                                                     │
│        ├─ model.pt          (PyTorch state dict)                    │
│        ├─ config.json       (Training config)                       │
│        ├─ metrics.json      (Performance metrics)                   │
│        ├─ data_manifest.json (Data snapshot)                        │
│        └─ training_log.jsonl (Training progress)                    │
│                                                                      │
│  📁 data/checkpoints/       (Deterministic ingest resume state)     │
│     ├─ entities/                                                     │
│     ├─ events/                                                       │
│     ├─ interactions/                                                 │
│     └─ artifacts/                                                    │
└──────────────────────────────────────────────────────────────────────┘
```

---

## Complete Data Flow Pipeline

```
┌─────────────────────────────────────────────────────────────────────┐
│                         PHASE 1: DATA INGESTION                     │
└─────────────────────────────────────────────────────────────────────┘

   ┌──────────────┐
   │  CSV Files   │  entities.csv, events.csv, interactions.csv
   └──────┬───────┘
          │
          ▼
   ┌──────────────────┐
   │ Parse & Validate │  Type checking, FK validation
   └──────┬───────────┘
          │
          ▼
   ┌──────────────────┐
   │  PostgreSQL DB   │  Insert into tables
   └──────────────────┘

   ┌──────────────┐
   │ Media Files  │  images, audio, video
   └──────┬───────┘
          │
          ▼
   ┌──────────────────┐
   │ Compute SHA256   │  Deduplication via hash
   └──────┬───────────┘
          │
          ▼
   ┌──────────────────┐
   │ artifacts_store/ │  Save files to disk
   └──────┬───────────┘
          │
          ▼
   ┌──────────────────┐
   │  PostgreSQL DB   │  Insert artifact metadata
   └──────────────────┘


┌─────────────────────────────────────────────────────────────────────┐
│                    PHASE 2: FEATURE EXTRACTION                      │
└─────────────────────────────────────────────────────────────────────┘

   ┌──────────────────┐
   │  Query Artifacts │  WHERE feature_status = 'pending'
   └──────┬───────────┘
          │
          ├──────────────┬──────────────┬──────────────┐
          │              │              │              │
          ▼              ▼              ▼              ▼
   ┌───────────┐  ┌───────────┐  ┌───────────┐  ┌───────────┐
   │  Image    │  │   Audio   │  │   Video   │  │   Other   │
   │ Features  │  │ Features  │  │ Features  │  │ Features  │
   └─────┬─────┘  └─────┬─────┘  └─────┬─────┘  └─────┬─────┘
         │              │              │              │
         │ Histogram    │ MFCC         │ Frame        │ Custom
         │ HOG          │ Mel-spec     │ Sampling     │ Extract
         │ Color        │ Spectral     │ Histogram    │
         │              │              │              │
         └──────────────┴──────────────┴──────────────┘
                        │
                        ▼
               ┌─────────────────┐
               │  Numpy Arrays   │  Float32 feature vectors
               └────────┬────────┘
                        │
                        ▼
               ┌─────────────────┐
               │ feature_cache/  │  Save as {sha256}.npy
               └────────┬────────┘
                        │
                        ▼
               ┌─────────────────┐
               │  Update DB      │  feature_status = 'done'
               └─────────────────┘


┌─────────────────────────────────────────────────────────────────────┐
│                      PHASE 3: MODEL TRAINING                        │
└─────────────────────────────────────────────────────────────────────┘

   ┌──────────────────┐
   │ Query Database   │  Load entities + attributes + targets
   └──────┬───────────┘
          │
          ▼
   ┌──────────────────┐
   │  Load Features   │  Events, interactions, artifacts
   └──────┬───────────┘
          │
          ▼
   ┌──────────────────┐
   │ Build Tensors    │  Convert to PyTorch tensors
   └──────┬───────────┘
          │
          ▼
   ┌─────────────────────────────────────────────────────────────────┐
   │                      FullModel Architecture                      │
   │                                                                  │
   │  ┌──────────────────────────────────────────────────────────┐  │
   │  │                  Input Data                               │  │
   │  │  ├─ Entity Attributes [x, y, z]                           │  │
   │  │  ├─ Event Sequences [(type, value, timestamp), ...]       │  │
   │  │  ├─ Interaction Graph (neighbor attributes)               │  │
   │  │  └─ Artifact Features [feature vectors...]                │  │
   │  └──────────────────────┬───────────────────────────────────────┘
   │                         │
   │  ┌──────────────────────▼───────────────────────────────────────┐
   │  │              4 SPECIALIZED ENCODERS                           │
   │  ├───────────────────────────────────────────────────────────────┤
   │  │ 1. AttrEncoder (16-dim)        ┌──────────────────────┐     │
   │  │    └─ Entity attributes         │  x, y, z → MLP       │     │
   │  │                                  └──────────────────────┘     │
   │  │                                                                │
   │  │ 2. EventSequenceEncoder (64-dim) ┌─────────────────────┐    │
   │  │    ├─ Event Type Embeddings      │ Embedding Layer     │    │
   │  │    ├─ Multi-head Attention       │ 4 heads             │    │
   │  │    └─ Temporal modeling          │ Time delta encoding │    │
   │  │                                  └─────────────────────┘     │
   │  │                                                                │
   │  │ 3. GraphEncoder (32-dim)         ┌─────────────────────┐    │
   │  │    ├─ Neighbor Aggregation       │ Mean pooling        │    │
   │  │    └─ Self + Neighbor Fusion     │ Concatenate + MLP   │    │
   │  │                                  └─────────────────────┘     │
   │  │                                                                │
   │  │ 4. ArtifactEncoder (32-dim)      ┌─────────────────────┐    │
   │  │    ├─ Mean + Max Pooling         │ Aggregate features  │    │
   │  │    └─ MLP Feature Fusion         │ Linear layers       │    │
   │  │                                  └─────────────────────┘     │
   │  └──────────────────────┬─────────────────────────────────────────┘
   │                         │
   │                         ▼
   │              Concatenate (144-dim)
   │                         │
   │                         ▼
   │              Linear Projection (64-dim)
   │                         │
   │                         ▼
   │  ┌──────────────────────────────────────────────────────────────┐
   │  │                 MULTI-TASK OUTPUT HEAD                        │
   │  ├───────────────────────────────────────────────────────────────┤
   │  │ 1. Regression Head → Continuous Value                         │
   │  │ 2. Classification Head → Binary Probability (sigmoid)         │
   │  │ 3. Ranking Head → Ranking Score                               │
   │  └───────────────────────────────────────────────────────────────┘
   └─────────────────────────────────────────────────────────────────┘
          │
          ▼
   ┌──────────────────┐
   │ Compute Loss     │  MSE + BCE + Ranking Loss
   └──────┬───────────┘
          │
          ▼
   ┌──────────────────┐
   │ Backpropagation  │  Adam optimizer
   └──────┬───────────┘
          │
          ▼
   ┌──────────────────┐
   │ Save Checkpoint  │  model.pt + config + metrics
   └──────┬───────────┘
          │
          ▼
   ┌──────────────────┐
   │  Store Run Info  │  Insert into model_runs table
   └──────────────────┘


┌─────────────────────────────────────────────────────────────────────┐
│                  PHASE 4: PREDICTION & EXPLANATION                  │
└─────────────────────────────────────────────────────────────────────┘

   ┌──────────────────┐
   │  Entity IDs      │  Input: ["ent_001", "ent_002", ...]
   └──────┬───────────┘
          │
          ▼
   ┌──────────────────┐
   │  Load Model      │  Fetch model.pt from models/{run_id}/
   └──────┬───────────┘
          │
          ▼
   ┌──────────────────┐
   │ Query Database   │  Get entity data + events + artifacts
   └──────┬───────────┘
          │
          ▼
   ┌──────────────────┐
   │  Build Batch     │  Convert to tensors
   └──────┬───────────┘
          │
          ▼
   ┌──────────────────┐
   │  Forward Pass    │  FullModel inference
   └──────┬───────────┘
          │
          ├──────────────────────────────────────┐
          │                                      │
          ▼                                      ▼
   ┌──────────────────┐              ┌──────────────────────┐
   │  Raw Outputs     │              │  Compute Explanations│
   │  • regression    │              │  (if requested)      │
   │  • probability   │              └──────────┬───────────┘
   │  • ranking_score │                         │
   │  • embedding     │              ┌──────────▼───────────┐
   └──────┬───────────┘              │ Integrated Gradients │
          │                          │ • Attribute attr.     │
          │                          │ • Event attention     │
          │                          │ • Artifact contrib.   │
          │                          └──────────┬───────────┘
          │                                     │
          └─────────────────┬───────────────────┘
                            │
                            ▼
                   ┌─────────────────┐
                   │ JSON Response   │  Return predictions + explanations
                   └─────────────────┘
```

---

## Natural-Language Query and Narrative Flow

```
┌─────────────────────────────────────────────────────────────────────┐
│                       PHASE 5: QUERY RETRIEVAL                      │
└─────────────────────────────────────────────────────────────────────┘

   Analyst Prompt
   (UI preset or free text)
          │
          ▼
   ┌──────────────────┐
   │ POST /query      │
   └──────┬───────────┘
          │
          ▼
   ┌──────────────────────────────────────────────────────────────┐
   │ Intent Inference                                             │
   │ • match strategy: explicit | fuzzy | broad_scan             │
   │ • order intent: strongest | weakest | default               │
   │ • probability intent: elevated | default                    │
   └──────┬───────────────────────────────────────────────────────┘
          │
          ▼
   ┌──────────────────────────────────────────────────────────────┐
   │ Candidate Scoring (deterministic)                           │
   │ • Uses /predict in template mode for candidate batch        │
   │ • Sorts by ranking_score / probability intent               │
   │ • Truncates to requested top-k                              │
   └──────┬───────────────────────────────────────────────────────┘
          │
          ▼
   ┌──────────────────────────────────────────────────────────────┐
   │ Narrative Enrichment (optional LLM)                         │
   │ • Long-form narrative only for final top-k                  │
   │ • Facts/scores constrained to deterministic outputs         │
   │ • Confidence limits requested in prompt                     │
   └──────┬───────────────────────────────────────────────────────┘
          │
          ▼
   ┌──────────────────────────────────────────────────────────────┐
   │ QueryResponse                                                │
   │ • interpreted_as includes match/order/probability tags      │
   │ • llm_used indicates whether enrichment succeeded            │
   │ • results contain stable scores and narratives              │
   └──────────────────────────────────────────────────────────────┘
```

---

## CI and Validation Architecture

```
                         Push to main / PR
                                │
        ┌───────────────────────┼────────────────────────┐
        │                       │                        │
        ▼                       ▼                        ▼
  Backend CI             Determinism Matrix CI          E2E CI
  - ruff/mypy/tests      - Python 3.11 + 3.12          - full stack boot
  - migration checks     - run/hash consistency         - UI and API journey
  - API contracts        - artifacts + report output    - query + predict paths
        │                       │                        │
        └───────────────────────┼────────────────────────┘
                                ▼
                       Optional Frontend CI
                       - vitest integration tests
                       - UI interaction coverage
```

Performance telemetry (`data/performance_metrics.jsonl`) is aggregated into
`data/performance_report.json` and uploaded by CI workflows as artifacts.

---

## Neural Network Architecture Details

```
┌───────────────────────────────────────────────────────────────────┐
│                         INPUT LAYER                               │
└───────────────────────────────────────────────────────────────────┘

   Entity Attributes       Event Sequence        Graph Structure      Artifacts
   ┌───────────┐          ┌───────────┐         ┌──────────┐        ┌─────────┐
   │ x: 0.5    │          │ events[0] │         │  self    │        │ feat[0] │
   │ y: 0.3    │          │ events[1] │         │ neighbor │        │ feat[1] │
   │ z: 0.8    │          │ events[2] │         │ neighbor │        │ feat[2] │
   └─────┬─────┘          │   ...     │         │   ...    │        │   ...   │
         │                └─────┬─────┘         └────┬─────┘        └────┬────┘
         │                      │                    │                   │
         ▼                      ▼                    ▼                   ▼
   ┌─────────────┐        ┌──────────────┐    ┌───────────────┐  ┌──────────────┐
   │AttrEncoder  │        │EventSeqEnc   │    │ GraphEncoder  │  │ArtifactEnc   │
   │             │        │              │    │               │  │              │
   │ Linear(3→16)│        │Embed(4→32)   │    │ MeanPool      │  │MeanMaxPool   │
   │ ReLU        │        │TimeEmbed     │    │ Concat        │  │Linear(→32)   │
   │ Linear(16→16)│       │MultiHeadAttn│    │ MLP(→32)      │  │ReLU          │
   │             │        │  (4 heads)   │    │               │  │              │
   └─────┬───────┘        └──────┬───────┘    └───────┬───────┘  └──────┬───────┘
         │                       │                    │                 │
         │ 16-dim                │ 64-dim             │ 32-dim          │ 32-dim
         │                       │                    │                 │
         └───────────────────────┴────────────────────┴─────────────────┘
                                       │
                                       ▼
                              ┌─────────────────┐
                              │  Concatenate    │
                              │  (144-dim)      │
                              └────────┬────────┘
                                       │
                                       ▼
                              ┌─────────────────┐
                              │ Linear(144→64)  │
                              │      ReLU       │
                              └────────┬────────┘
                                       │
                                       ▼
                              ┌─────────────────┐
                              │  Fused Embed    │
                              │   (64-dim)      │
                              └────────┬────────┘
                                       │
                        ┌──────────────┼──────────────┐
                        │              │              │
                        ▼              ▼              ▼
              ┌─────────────┐ ┌─────────────┐ ┌─────────────┐
              │Regression   │ │Classification│ │  Ranking    │
              │Linear(64→1) │ │Linear(64→1) │ │Linear(64→1) │
              │             │ │  Sigmoid     │ │             │
              └──────┬──────┘ └──────┬──────┘ └──────┬──────┘
                     │               │               │
                     ▼               ▼               ▼
              ┌─────────────┐ ┌─────────────┐ ┌─────────────┐
              │  y_reg      │ │  y_prob     │ │  y_rank     │
              │  (float)    │ │  (0-1)      │ │  (float)    │
              └─────────────┘ └─────────────┘ └─────────────┘
```

---

## Enterprise Integration Patterns

### Pattern 1: ETL Pipeline
```
┌────────────┐     ┌────────────┐     ┌──────────────┐
│   Data     │────▶│    ETL     │────▶│  ML Stack    │
│ Warehouse  │     │   Script   │     │  (REST API)  │
└────────────┘     └────────────┘     └──────┬───────┘
                                              │
                                              ▼
                                      ┌───────────────┐
                                      │  Predictions  │
                                      │  Write Back   │
                                      └───────────────┘
```

### Pattern 2: Microservices
```
┌──────────┐     ┌──────────┐     ┌──────────┐
│   User   │────▶│   API    │────▶│ Business │
│ Service  │     │ Gateway  │     │  Logic   │
└──────────┘     └──────────┘     └────┬─────┘
                                        │
                                        ▼
                                 ┌──────────────┐
                                 │ ML Prediction│
                                 │   Service    │
                                 └──────────────┘
```

### Pattern 3: Batch Processing
```
Scheduler (Cron/Airflow)
         │
         ▼
Extract Entity IDs → Batch → Parallel API Calls
         │
         ▼
Aggregate Results → Store → Report
```

### Pattern 4: Real-Time API
```
User Action → App Backend → ML Service → Prediction
                                   ↓
                          Decision Logic
                                   ↓
                            Take Action
```

---

## Determinism Guarantees

```
┌───────────────────────────────────────────────────────────────────┐
│                    DETERMINISM CONTROLS                           │
└───────────────────────────────────────────────────────────────────┘

   Random Seed Control
   ┌────────────────────────────────────────────────────────┐
   │ • Python hash seed:  PYTHONHASHSEED=0                  │
   │ • NumPy seed:        np.random.seed(1234)              │
   │ • PyTorch seed:      torch.manual_seed(1234)           │
   │ • Random module:     random.seed(1234)                 │
   └────────────────────────────────────────────────────────┘

   Algorithm Control
   ┌────────────────────────────────────────────────────────┐
   │ • Deterministic ops: torch.use_deterministic_algorithms│
   │ • Single-threaded:   OMP_NUM_THREADS=1                 │
   │ • No GPU:            CPU-only inference                │
   └────────────────────────────────────────────────────────┘

   Version Control
   ┌────────────────────────────────────────────────────────┐
   │ • Feature version:   SHA256(code + config + libs)      │
   │ • Model hash:        SHA256(model.pt bytes)            │
   │ • Run ID:            SHA256(config + data + features)  │
   └────────────────────────────────────────────────────────┘

                         ↓
   ┌────────────────────────────────────────────────────────┐
   │           GUARANTEE: Bit-Exact Reproducibility         │
   │                                                        │
   │  Same Input + Same Config → Identical Output           │
   │                                                        │
   │  Verified by: reproduce_run() & determinism checks     │
   └────────────────────────────────────────────────────────┘
```

---

## Explainability Methods

```
┌───────────────────────────────────────────────────────────────────┐
│                      EXPLAINABILITY STACK                         │
└───────────────────────────────────────────────────────────────────┘

   Prediction: y_prob = 0.85 (fraud likely)

   ┌──────────────────────────────────────────────────────────────┐
   │  1. Integrated Gradients                                     │
   │     └─ Attribute Contributions:                              │
   │        • x attribute: +0.35                                  │
   │        • y attribute: +0.28                                  │
   │        • z attribute: +0.15                                  │
   └──────────────────────────────────────────────────────────────┘

   ┌──────────────────────────────────────────────────────────────┐
   │  2. Attention Weights (Event Sequence)                       │
   │     └─ Most important events:                                │
   │        • Event 3 (large_transaction):   0.40                 │
   │        • Event 2 (login_unusual_time):  0.30                 │
   │        • Event 1 (account_created):     0.20                 │
   │        • Event 0 (first_login):         0.10                 │
   └──────────────────────────────────────────────────────────────┘

   ┌──────────────────────────────────────────────────────────────┐
   │  3. Artifact Contributions                                   │
   │     └─ Feature importance from media:                        │
   │        • receipt_image.jpg:     +0.22                        │
   │        • voice_call.mp3:        +0.18                        │
   │        • verification_photo.jpg: -0.15                       │
   └──────────────────────────────────────────────────────────────┘

                         ↓
   ┌──────────────────────────────────────────────────────────────┐
   │                    HUMAN-READABLE EXPLANATION                │
   │                                                              │
   │  "This transaction is flagged as fraud (85% confidence)     │
   │   because of:                                               │
   │   • Unusual location (x-coordinate: +35%)                   │
   │   • Large transaction amount (event focus: 40%)             │
   │   • Suspicious receipt image (artifact: +22%)"              │
   └──────────────────────────────────────────────────────────────┘
```

---

## Deployment Options

### Docker Compose (Development)
```
┌─────────────────────────────────────────┐
│  Docker Compose                         │
│  ├─ db (PostgreSQL)                     │
│  ├─ backend (FastAPI)                   │
│  └─ frontend (React/Vite)               │
└─────────────────────────────────────────┘
```

### Kubernetes (Production)
```
┌─────────────────────────────────────────┐
│  Kubernetes Cluster                     │
│  ├─ PostgreSQL StatefulSet              │
│  ├─ Backend Deployment (3 replicas)     │
│  ├─ Frontend Deployment (2 replicas)    │
│  ├─ Ingress (Load Balancer)             │
│  └─ PersistentVolumes (artifacts/models)│
└─────────────────────────────────────────┘
```

### Cloud Services (AWS Example)
```
┌─────────────────────────────────────────┐
│  AWS Architecture                       │
│  ├─ RDS PostgreSQL                      │
│  ├─ ECS/Fargate (Backend containers)    │
│  ├─ S3 (Artifacts & Models)             │
│  ├─ CloudFront (Frontend CDN)           │
│  └─ ALB (Application Load Balancer)     │
└─────────────────────────────────────────┘
```

---

## Security Architecture

```
┌───────────────────────────────────────────────────────────────────┐
│                        SECURITY LAYERS                            │
└───────────────────────────────────────────────────────────────────┘

   Layer 1: Network Security
   ┌────────────────────────────────────────────────────────┐
   │ • TLS/SSL for all API endpoints (HTTPS)                │
   │ • Firewall rules (restrict DB access)                  │
   │ • VPC/Network isolation                                │
   └────────────────────────────────────────────────────────┘

   Layer 2: Authentication & Authorization
   ┌────────────────────────────────────────────────────────┐
   │ • API Key authentication                               │
   │ • JWT tokens for sessions                              │
   │ • Role-based access control (RBAC)                     │
   └────────────────────────────────────────────────────────┘

   Layer 3: Data Encryption
   ┌────────────────────────────────────────────────────────┐
   │ • Database encryption (at rest)                        │
   │ • File encryption (artifacts_store)                    │
   │ • TLS (in transit)                                     │
   └────────────────────────────────────────────────────────┘

   Layer 4: Audit & Compliance
   ┌────────────────────────────────────────────────────────┐
   │ • Access logs (who accessed what, when)               │
   │ • Model versioning & hashing                           │
   │ • Data lineage tracking                                │
   │ • GDPR/HIPAA compliance features                       │
   └────────────────────────────────────────────────────────┘
```

---

*This visual reference complements the main documentation.*  
*See COMPREHENSIVE_REPOSITORY_GUIDE.md for detailed explanations.*
