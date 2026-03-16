# Comprehensive Repository Guide: Deterministic Multimodal Analytics Stack

## Table of Contents
1. [Executive Summary](#executive-summary)
2. [What This System Does (In Simple Terms)](#what-this-system-does-in-simple-terms)
3. [Core Concepts Explained](#core-concepts-explained)
4. [Technical Architecture](#technical-architecture)
5. [Key Capabilities](#key-capabilities)
6. [Enterprise Integration Patterns](#enterprise-integration-patterns)
7. [Use Cases and Applications](#use-cases-and-applications)
8. [Getting Started](#getting-started)
9. [API Reference](#api-reference)
10. [Security and Compliance Considerations](#security-and-compliance-considerations)
11. [Current Platform Snapshot (March 2026)](#current-platform-snapshot-march-2026)

---

## Executive Summary

The **Deterministic Multimodal Analytics Stack** is a production-ready, full-stack machine learning platform that enables enterprises to:

- **Ingest diverse data types**: CSV datasets (entities, events, interactions) and multimodal artifacts (images, audio, video)
- **Train reproducible models**: Deterministic training ensures identical results across runs for compliance and debugging
- **Make explainable predictions**: Multi-task predictions (regression, classification, ranking) with built-in explainability
- **Maintain complete audit trails**: SHA256 hashing, version tracking, and data lineage for regulatory compliance

**Technology Stack**: FastAPI + PostgreSQL + PyTorch + React, fully containerized with Docker Compose

**Key Differentiator**: Unlike typical ML platforms, this system guarantees bit-exact reproducibility through deterministic algorithms, making it ideal for regulated industries (finance, healthcare, legal) where model auditability is critical.

---

## Current Platform Snapshot (March 2026)

The repository currently includes the following production-facing capabilities:

- End-to-end ingestion for entities, events, interactions, artifact manifests, and single artifact uploads
- Chunked ingestion with deterministic checkpoint/resume controls
- Demo preload route (`POST /demo/preload`) that can bootstrap data, feature extraction, and starter training
- Feature cache version pinning and invalidation to prevent stale feature reuse
- Deterministic multi-head prediction (`/predict`) with regression, probability, and ranking outputs
- Async orchestration endpoints:
  - `POST /train/async`, `GET /train/async/{task_id}`
  - `POST /features/extract/async`, `GET /features/extract/async/{task_id}`
  - `POST /predict/async`, `GET /predict/async/{task_id}`
- Queue health endpoint (`GET /health/queues`) with broker status, backlog, latency-age, and saturation metrics
- Correlation ID propagation (`x-correlation-id`) from API request to async task and telemetry events
- Natural-language query route (`/query`) with intent-aware ranking:
  - relationship strength ordering
  - elevated-probability intent handling
  - retrieval strategy tagging (`explicit`, `fuzzy`, `broad_scan`)
- Optional Hugging Face endpoint-backed query interpretation and narrative enrichment
- Query UI presets for complex prompts (strongest relationships, risk investigation, operational prioritization, anomaly storyline)
- Structured performance telemetry with report generation (`python -m app.cli performance-report`)
- Active CI workflow gates:
  - Backend CI
  - Determinism Matrix CI
  - Phase 4 Release Gate
  - E2E CI
  - Frontend CI

---

## What This System Does (In Simple Terms)

Think of this system as a **"time machine for machine learning"** that can:

1. **Learn from your data patterns**: You give it information about things (entities), what happened to them over time (events), how they interact with each other (relationships), and supporting evidence (images, audio, video).

2. **Make three types of predictions**:
   - **Numbers**: "This customer will spend $X next month" (regression)
   - **Yes/No answers**: "Will this transaction be fraudulent?" (classification)
   - **Rankings**: "Which products should we recommend first?" (ranking)

3. **Explain why**: The system doesn't just give you an answer—it shows you which pieces of data led to that conclusion, making it trustworthy and debuggable.

4. **Guarantee consistency**: Run the same training twice, get the exact same model. This is like having a recipe that always produces the same cake, which is crucial when regulators ask "how did you arrive at this decision?"

### The "Feynman Explanation"

Imagine you're a detective solving a case:
- **Entities** are your suspects and witnesses
- **Events** are the timeline of what each person did
- **Interactions** are who talked to whom
- **Artifacts** are your evidence photos, audio recordings, and video footage
- **The Model** is your trained intuition after reviewing thousands of cases
- **Predictions** are your hypotheses about new cases, with evidence to back them up
- **Determinism** means if you review the same evidence twice, you'll always reach the same conclusion

---

## Core Concepts Explained

### 1. Entities
**What they are**: The fundamental "things" you're analyzing—customers, products, transactions, patients, sensors, etc.

**Structure**: Each entity has:
- A unique ID
- Attributes (properties like age, location, type)
- Associated events, interactions, and artifacts

**Example**: In an e-commerce system, an entity might be a customer with attributes `{age: 35, location: "NYC", member_since: "2023"}`.

### 2. Events
**What they are**: Time-stamped actions or observations about entities—purchases, clicks, sensor readings, medical visits.

**Structure**: 
- Timestamp
- Event type (e.g., "purchase", "login", "alert")
- Event value (amount, duration, severity)

**Example**: Customer "C123" had an event: `{type: "purchase", value: 150.00, timestamp: "2024-01-15"}`.

### 3. Interactions
**What they are**: Relationships between entities—who bought from whom, which products are frequently purchased together, patient referral networks.

**Structure**:
- Source entity
- Destination entity
- Interaction type and value

**Example**: Customer "C123" → Product "P456" with interaction type "purchased" and value "3" (quantity).

### 4. Artifacts
**What they are**: Rich media evidence—product images, customer service call recordings, medical scans, surveillance footage.

**Structure**:
- File path and type (image/audio/video)
- SHA256 hash (for deduplication and integrity)
- Extracted features (numerical representations)
- Association with entities and timestamps

**Example**: Product "P456" has an artifact: `{type: "image", path: "product_images/P456.jpg", features: [0.42, 0.87, ...]}`.

### 5. Deterministic Training
**What it means**: Every random element in the machine learning process is controlled by fixed "seeds"—like setting a random number generator to always start from the same point.

**Why it matters**:
- **Debugging**: If a model behaves unexpectedly, you can reproduce the exact training to investigate
- **Compliance**: Regulators can verify your model by re-running training and getting identical results
- **Testing**: Compare changes by running old vs. new code with the same data
- **Trust**: Stakeholders know the results aren't due to random chance

**How it works**:
- Fixed random seeds for Python, NumPy, PyTorch
- Single-threaded execution (no race conditions)
- Deterministic algorithms only (no GPU randomness)
- SHA256 hashing of configuration, data, and model states

### 6. Explainability
**What it means**: The system shows you which input features contributed most to each prediction.

**Methods used**:
- **Integrated Gradients**: Measures how much each input feature influenced the output
- **Attention Weights**: Shows which events in a timeline the model focused on
- **Artifact Contributions**: Identifies which images/audio/video files were most important

**Example**: For a fraud detection prediction, the system might show: "80% confidence this is fraud, primarily due to: unusual transaction time (35%), location mismatch (28%), and suspicious audio pattern in customer call (22%)."

---

## Technical Architecture

### System Components

```
┌─────────────────────────────────────────────────────────────┐
│                        FRONTEND (React)                     │
│  ┌──────────┬──────────┬──────────┬──────────┐            │
│  │ Dataset  │  Train   │   Runs   │ Predict  │            │
│  │  Upload  │ Trigger  │  History │  + Explain│            │
│  └──────────┴──────────┴──────────┴──────────┘            │
└────────────────────────┬────────────────────────────────────┘
                         │ REST API
┌────────────────────────▼────────────────────────────────────┐
│                    BACKEND (FastAPI)                        │
│  ┌──────────────────────────────────────────────────────┐  │
│  │ API Routes: /health /ingest /features /train /predict│  │
│  └──────────────┬───────────────────────────────────────┘  │
│                 │                                            │
│  ┌──────────────▼──────────────┬──────────────────────┐   │
│  │       Services              │   ML + Async Pipeline│   │
│  │  • CSV Ingestion            │  • Feature Extract   │   │
│  │  • Artifact Management      │  • Model Training    │   │
│  │  • Feature Extraction       │  • Prediction        │   │
│  │  • Parquet Export           │  • Explainability    │   │
│  │  • Queue Health             │  • Task Lifecycle    │   │
│  └──────────────┬──────────────┴──────────────────────┘   │
│                 │                                            │
│  Redis + Celery Broker/Worker (training, extraction, batch inference queues) │
└─────────────────┼───────────────────────────────────────────┘
                  │
┌─────────────────▼───────────────────────────────────────────┐
│                  DATABASE (PostgreSQL 16)                   │
│  ┌──────────┬─────────┬──────────────┬──────────┬────────┐│
│  │ entities │ events  │ interactions │ artifacts│ runs   ││
│  │ training_tasks │ feature_extraction_tasks │ batch_inference_tasks │
│  │ agent_runs │ agent_steps │ agent_audit_events │
│  └──────────┴─────────┴──────────────┴──────────┴────────┘│
└─────────────────────────────────────────────────────────────┘

┌─────────────────────────────────────────────────────────────┐
│                    FILE STORAGE                             │
│  • artifacts_store/  (uploaded media files)                 │
│  • feature_cache/    (extracted feature vectors)            │
│  • models/           (trained model checkpoints)            │
└─────────────────────────────────────────────────────────────┘
```

### Data Flow Pipeline

**Phase 1: Data Ingestion**
```
CSV Files → Parse & Validate → Database (entities/events/interactions)
Media Files → Compute SHA256 → Store in artifacts_store/ → Database (artifacts)
Checkpoint File → Resume ingest from last processed row (optional)
```

**Phase 2: Feature Extraction**
```
Artifacts (pending) → Feature Extractors → Numerical Vectors
  ├─ Images: Histogram + HOG + Color Features (240-dim)
  ├─ Audio: MFCC + Mel-Spectrograms + Spectral Features
  └─ Video: Frame Sampling + Histogram Features
→ Cache as .npy files → Update artifacts table (status="done")
```

**Phase 3: Model Training**
```
Database → Load Entities + Attributes + Targets
       → Build Tensors (attributes, events, interactions, artifacts)
       → FullModel (4 encoders + fusion head)
       → Multi-task Loss (regression + classification + ranking)
       → Optimize with Adam
       → Save model.pt + config + metrics + SHA256 hash
       → Store ModelRun record
```

**Phase 4: Prediction & Explanation**
```
Entity IDs → Query Database → Build Input Batch
          → Forward Pass through Model
          → Output (regression, probability, ranking score)
          → Compute Integrated Gradients (if explanations requested)
          → Return Predictions + Explanations
```

### Neural Network Architecture (FullModel)

The system uses a **multi-encoder fusion architecture**:

```
Input Data:
  ├─ Entity Attributes [x, y, z] 
  ├─ Event Sequences [(type, value, timestamp), ...]
  ├─ Interaction Graph (neighbor attributes)
  └─ Artifact Features [feature vectors...]

         │
         ▼
┌────────────────────────────────────────────────────┐
│              4 SPECIALIZED ENCODERS                │
├────────────────────────────────────────────────────┤
│ 1. AttrEncoder (16-dim)                           │
│    └─ Embeds entity attributes (x, y, z)          │
│                                                    │
│ 2. EventSequenceEncoder (64-dim)                  │
│    ├─ Event Type Embeddings                       │
│    ├─ Multi-head Attention (4 heads)              │
│    └─ Temporal modeling with time deltas          │
│                                                    │
│ 3. GraphEncoder (32-dim)                          │
│    ├─ Neighbor Aggregation (mean pooling)         │
│    └─ Self + Neighbor Fusion                      │
│                                                    │
│ 4. ArtifactEncoder (32-dim)                       │
│    ├─ Mean + Max Pooling across artifacts         │
│    └─ MLP Feature Fusion                          │
└────────────────┬───────────────────────────────────┘
                 │
                 ▼
         Concatenate (144-dim)
                 │
                 ▼
         Linear Projection (64-dim)
                 │
                 ▼
┌────────────────────────────────────────────────────┐
│           MULTI-TASK OUTPUT HEAD                   │
├────────────────────────────────────────────────────┤
│ 1. Regression Head → Continuous Value              │
│ 2. Classification Head → Binary Probability        │
│ 3. Ranking Head → Ranking Score                    │
└────────────────────────────────────────────────────┘
```

**Key Features**:
- **Attention Mechanism**: Automatically learns which events in a sequence are most important
- **Graph Modeling**: Captures network effects and relationships between entities
- **Multimodal Fusion**: Combines tabular, time-series, graph, and media data
- **Multi-Task Learning**: Shares representations across tasks for better generalization

---

## Key Capabilities

### 1. Data Ingestion
**What you can ingest**:
- **CSV files**: Entities, events, interactions (up to 1M rows, 50MB per file)
- **Media files**: Images (JPG, PNG), audio (WAV, MP3), video (MP4, AVI) up to 200MB each
- **Manifest files**: Bulk artifact registration via CSV

**Validation features**:
- Type checking (event types, interaction types)
- Foreign key validation (events must reference valid entities)
- Duplicate detection (SHA256-based for artifacts)
- Size limits and error reporting
- Chunk size controls and deterministic checkpoint/resume for high-volume loads

**API Endpoints**:
```
POST /ingest/entities
POST /ingest/events  
POST /ingest/interactions
POST /ingest/artifacts (manifest)
POST /ingest/artifact (single file)
```

### 2. Feature Extraction
**Supported formats**:
- **Images**: Extracts 240-dimensional features including:
  - Grayscale and RGB histograms
  - HOG (Histogram of Oriented Gradients) for shape detection
  - Edge density metrics
  
- **Audio**: Extracts acoustic features including:
  - MFCC (Mel-Frequency Cepstral Coefficients) - 13 coefficients
  - Mel-spectrograms for frequency analysis
  - Spectral features (centroid, bandwidth, rolloff)
  
- **Video**: Frame-based extraction:
  - Samples frames at regular intervals
  - Computes histogram features per frame
  - Aggregates across frames

**Deterministic guarantees**:
- Same file + same feature extraction version = identical features
- Version hash includes: code hash + config + library versions
- Cached in `.npy` files for fast re-access
- Cache reuse only when `feature_version_hash` matches current extractor version

**API Endpoint**:
```
POST /features/extract
```

### 3. Model Training
**Configuration options**:
```json
{
  "epochs": 100,
  "batch_size": 32,
  "learning_rate": 0.001,
  "seed": 1234,
  "max_neighbors": 10,
  "max_artifacts_per_entity": 5
}
```

**Determinism features**:
- Fixed random seeds (Python, NumPy, PyTorch, standard library)
- Single-threaded execution (`OMP_NUM_THREADS=1`)
- Deterministic algorithms only (`torch.use_deterministic_algorithms(True)`)
- GPU disabled (CPU-only for full reproducibility)
- Reproducible run IDs: `SHA256(config + data_manifest + feature_version)`
- Explicit run lifecycle transitions: `pending -> success` or `pending -> failed`
- Persistence failure safety so filesystem/database divergence is recoverable and explicit

### 3.1 Async Orchestration and Queue Controls

The stack now supports queue-backed execution for long-running workflows:

- Training async enqueue and status APIs
- Feature extraction async enqueue and status APIs
- Batch inference async enqueue and status APIs
- Idempotency keys to prevent duplicate effective executions
- Queue isolation by workload class (`training`, `extraction`, `batch_inference`)

Queue operations telemetry is exposed via `GET /health/queues` and includes:

- backlog depth
- oldest pending age
- queue saturation ratio and configured max concurrency

**Training metrics**:
- **Regression**: MAE, RMSE, R²
- **Classification**: F1, Precision, Recall, AUC-ROC
- **Ranking**: NDCG@10, Spearman correlation

**Outputs stored**:
- `model.pt`: PyTorch state dict + config
- `config.json`: Training configuration
- `metrics.json`: Performance metrics
- `data_manifest.json`: Data snapshot (table row counts, feature versions)
- `model_sha256.txt`: Hash for reproducibility verification
- `training_log.jsonl`: Line-by-line training progress

**API Endpoints**:
```
POST /train
GET /runs
GET /runs/{run_id}
```

### 4. Prediction & Explainability
**Input**:
```json
{
  "entity_ids": ["E001", "E002", "E003"],
  "run_id": "abc123...",  // optional, uses latest if omitted
  "explanations": true     // optional, enables explainability
}
```

**Output**:
```json
{
  "run_id": "abc123...",
  "predictions": [
    {
      "entity_id": "E001",
      "regression": 42.5,
      "probability": 0.85,
      "ranking_score": 0.92,
      "embedding": [0.12, 0.34, ...],  // 64-dim representation
      "explanation": {
        "attribute_attributions": {
          "x": 0.35,
          "y": 0.28,
          "z": 0.15
        },
        "event_attention_weights": [0.1, 0.3, 0.4, 0.2],
        "artifact_contributions": [
          {
            "artifact_id": "A001",
            "contribution": 0.22
          }
        ]
      }
    }
  ]
}
```

**Explanation methods**:
- **Integrated Gradients**: Industry-standard attribution method that computes how much each input feature contributed to the output
- **Attention Weights**: Shows which events in the sequence received the most "focus" from the model
- **Artifact Contributions**: Quantifies the importance of each image/audio/video file

**API Endpoint**:
```
POST /predict
```

### 5. Reproducibility & Audit Trail
**Version tracking**:
- Every model run gets a deterministic ID based on config + data + feature versions
- Model files are hashed (SHA256) for integrity verification
- Training logs include timestamps, loss curves, and configuration snapshots

**Reproducibility verification**:
```python
# Re-run training with the same config
original_run = get_run("abc123...")
reproduced_run = reproduce_run("abc123...")

# Verify predictions are bit-exact identical
assert original_run.predictions == reproduced_run.predictions
```

**Audit capabilities**:
- Query which data was used for a specific model run
- Trace predictions back to their source data
- Verify model integrity via hash comparison
- Export data manifests for regulatory review

### 6. UI Dashboard
**Five main tabs**:

1. **Data Intake**: Upload CSV files, artifact manifests, and single artifacts
2. **Model Ops**: Trigger deterministic training with current data
3. **Run Ledger**: View all training runs with timestamps and metrics
4. **Inference**: Run predictions on specific entities
5. **Query**: Run natural-language retrieval prompts and inspect ranked results

**Features**:
- Real-time status updates
- File upload with validation
- Query prompt presets for advanced analyst prompts
- Interpreted query metadata (`match`, `order`, `probability`)
- Results visualization
- Error reporting

### 7. Query and Narrative Layer

The query path is intentionally split into two phases for quality and latency:

1. Deterministic candidate scoring and sorting
2. Optional LLM enrichment only for the final top-k results

This design keeps score integrity stable while allowing richer narratives. In repeated runs with unchanged data and run IDs, entity ordering is expected to remain stable; narrative wording may vary when LLM temperature is non-zero.

---

## Enterprise Integration Patterns

### Pattern 1: ETL Pipeline Integration
**Use case**: Integrate with existing data warehouses (Snowflake, BigQuery, Redshift)

**Architecture**:
```
Data Warehouse → ETL Job → Export CSV + Media → Upload via API
                                                       ↓
                                            Deterministic ML Stack
                                                       ↓
                                           Predictions → Write back
```

**Implementation**:
```python
# Sample ETL script
import requests
import pandas as pd

# Extract from data warehouse
df_entities = snowflake.execute("SELECT * FROM customers")
df_events = snowflake.execute("SELECT * FROM transactions")

# Transform to CSV
df_entities.to_csv('entities.csv', index=False)
df_events.to_csv('events.csv', index=False)

# Load into ML stack
with open('entities.csv', 'rb') as f:
    response = requests.post('http://ml-stack/ingest/entities', files={'file': f})

with open('events.csv', 'rb') as f:
    response = requests.post('http://ml-stack/ingest/events', files={'file': f})

# Extract features and train
requests.post('http://ml-stack/features/extract')
response = requests.post('http://ml-stack/train')
run_id = response.json()['run_id']

# Make predictions
predictions = requests.post('http://ml-stack/predict', json={
    'entity_ids': df_entities['entity_id'].tolist(),
    'run_id': run_id,
    'explanations': True
}).json()

# Write back to warehouse
pred_df = pd.DataFrame(predictions['predictions'])
snowflake.write_dataframe(pred_df, 'ml_predictions')
```

### Pattern 2: Microservices Architecture
**Use case**: Deploy as a prediction service in a microservices ecosystem

**Architecture**:
```
┌─────────────┐    ┌─────────────┐    ┌─────────────┐
│   User      │───▶│  API        │───▶│  Business   │
│   Service   │    │  Gateway    │    │  Logic      │
└─────────────┘    └─────────────┘    └──────┬──────┘
                                              │
                                              ▼
                                   ┌──────────────────┐
                                   │  ML Prediction   │
                                   │  Service         │
                                   │ (This Stack)     │
                                   └──────────────────┘
```

**Docker Compose Integration**:
```yaml
# Add to existing docker-compose.yml
services:
  ml-prediction-service:
    image: deterministic-ml-stack:latest
    environment:
      - DATABASE_URL=postgresql://...
      - PYTHONHASHSEED=0
    ports:
      - "8000:8000"
    volumes:
      - ./models:/app/models
      - ./artifacts:/app/artifacts_store
    networks:
      - enterprise-network
```

### Pattern 3: Batch Prediction Pipeline
**Use case**: Nightly batch scoring of all customers/products/transactions

**Architecture**:
```
Scheduler (Airflow/Cron)
    ↓
Extract Entity IDs → Batch into chunks → Parallel API calls
    ↓
Aggregate Results → Store in Database → Generate Reports
```

**Implementation (Apache Airflow DAG)**:
```python
from airflow import DAG
from airflow.operators.python import PythonOperator
from datetime import datetime, timedelta

def batch_predict():
    import requests
    import pandas as pd
    
    # Get all entity IDs
    entity_ids = get_entity_ids_from_database()
    
    # Batch into chunks of 100
    chunks = [entity_ids[i:i+100] for i in range(0, len(entity_ids), 100)]
    
    results = []
    for chunk in chunks:
        response = requests.post('http://ml-stack/predict', json={
            'entity_ids': chunk,
            'explanations': False  # Faster without explanations
        })
        results.extend(response.json()['predictions'])
    
    # Store results
    df = pd.DataFrame(results)
    write_to_database(df, 'daily_predictions')

dag = DAG(
    'nightly_predictions',
    default_args={'owner': 'airflow'},
    schedule_interval='0 2 * * *',  # 2 AM daily
    start_date=datetime(2024, 1, 1)
)

predict_task = PythonOperator(
    task_id='batch_predict',
    python_callable=batch_predict,
    dag=dag
)
```

### Pattern 4: Real-Time API Integration
**Use case**: Embed predictions in user-facing applications (fraud detection, recommendations)

**Architecture**:
```
User Action → Web/Mobile App → Backend API → ML Prediction Service
                                                    ↓
                                            Return prediction
                                                    ↓
                            Decision Logic ← Parse Response
                                ↓
                          Take Action (approve/reject/recommend)
```

**Example (Node.js Backend)**:
```javascript
const express = require('express');
const axios = require('axios');

const app = express();

app.post('/api/check-transaction', async (req, res) => {
  const { transactionId } = req.body;
  
  // Get prediction from ML service
  const mlResponse = await axios.post('http://ml-stack/predict', {
    entity_ids: [transactionId],
    explanations: true
  });
  
  const prediction = mlResponse.data.predictions[0];
  
  // Business logic
  if (prediction.probability > 0.9) {
    // High fraud probability
    res.json({
      status: 'BLOCKED',
      reason: 'High fraud risk',
      confidence: prediction.probability,
      explanation: prediction.explanation
    });
  } else {
    res.json({
      status: 'APPROVED',
      confidence: 1 - prediction.probability
    });
  }
});

app.listen(3000);
```

### Pattern 5: MLOps Pipeline with CI/CD
**Use case**: Automated model retraining and deployment

**Architecture**:
```
Code Change → GitHub Actions → Run Tests → Build Container
                                                ↓
                                         Deploy to Staging
                                                ↓
                                    Validate Model Performance
                                                ↓
                                        Deploy to Production
```

**GitHub Actions Workflow**:
```yaml
name: ML Pipeline

on:
  schedule:
    - cron: '0 0 * * 0'  # Weekly retraining
  push:
    branches: [main]

jobs:
  train-and-deploy:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v2
      
      - name: Build Docker Image
        run: docker-compose build
      
      - name: Start Services
        run: docker-compose up -d
      
      - name: Ingest Latest Data
        run: |
          python scripts/etl_pipeline.py
      
      - name: Train Model
        run: |
          curl -X POST http://localhost:8000/train
      
      - name: Run Tests
        run: |
          PYTHONPATH=. pytest -q app/tests/test_determinism_check.py
          PYTHONPATH=. pytest -q app/tests/test_features_deterministic.py
      
      - name: Validate Performance
        run: |
          python scripts/validate_metrics.py
      
      - name: Deploy
        if: success()
        run: |
          docker tag ml-stack:latest registry.company.com/ml-stack:latest
          docker push registry.company.com/ml-stack:latest
```

### Pattern 6: Multi-Tenant SaaS Deployment
**Use case**: Serve multiple customers with isolated data

**Architecture**:
```
Tenant A → API with tenant_id header → Route to Tenant A DB
Tenant B → API with tenant_id header → Route to Tenant B DB
Tenant C → API with tenant_id header → Route to Tenant C DB
```

**Implementation**:
```python
# Modify app/db/session.py for multi-tenancy
from fastapi import Header

def get_tenant_db(tenant_id: str = Header(...)):
    database_url = f"postgresql://user:pass@host/db_{tenant_id}"
    engine = create_engine(database_url)
    SessionLocal = sessionmaker(bind=engine)
    return SessionLocal()

# Use in routes
@router.post("/ingest/entities")
async def ingest_entities(
    file: UploadFile,
    db: Session = Depends(get_tenant_db)
):
    # DB session is now tenant-specific
    pass
```

### Pattern 7: Hybrid Cloud Deployment
**Use case**: On-premises data processing with cloud prediction service

**Architecture**:
```
On-Premises:
  ├─ Data Collection
  ├─ Feature Extraction
  └─ Secure Upload → 

Cloud:
  ├─ Model Training
  ├─ Prediction Service
  └─ Model Management
```

**Security Considerations**:
- TLS encryption for data in transit
- API key authentication
- Data anonymization before cloud upload
- Model download for on-prem inference (if needed)

---

## Use Cases and Applications

### Financial Services

**1. Credit Risk Scoring**
- **Entities**: Loan applicants
- **Events**: Payment history, account activities
- **Interactions**: Co-signers, business partners
- **Artifacts**: Scanned documents, signature images
- **Prediction**: Default probability with explainable factors

**Why determinism matters**: Regulatory compliance (FCRA, GDPR) requires explaining credit decisions and proving consistency.

**2. Fraud Detection**
- **Entities**: Transactions
- **Events**: Transaction sequences over time
- **Interactions**: Account-to-account transfers
- **Artifacts**: Receipt images, voice recordings from customer service
- **Prediction**: Fraud probability with attribution to suspicious patterns

**Why determinism matters**: Legal defensibility when blocking transactions; ability to reproduce model behavior for investigations.

**3. Algorithmic Trading**
- **Entities**: Securities
- **Events**: Price movements, volume changes
- **Interactions**: Correlation networks between assets
- **Artifacts**: News article images, earnings call audio
- **Prediction**: Price movement predictions with feature importance

**Why determinism matters**: Backtesting must be reproducible; debugging trading strategies requires exact replay.

### Healthcare

**4. Patient Risk Stratification**
- **Entities**: Patients
- **Events**: Medical visits, lab results, prescriptions
- **Interactions**: Healthcare provider networks, patient referrals
- **Artifacts**: Medical imaging (X-rays, MRIs), voice notes
- **Prediction**: Readmission risk, disease progression likelihood

**Why determinism matters**: HIPAA compliance, medical liability, peer review of clinical decision support systems.

**5. Drug Discovery**
- **Entities**: Chemical compounds
- **Events**: Experimental results over time
- **Interactions**: Compound-protein interactions
- **Artifacts**: Molecular structure images, spectroscopy data
- **Prediction**: Efficacy and toxicity scores

**Why determinism matters**: FDA submissions require reproducible computational models; scientific rigor demands replicability.

### Retail & E-Commerce

**6. Customer Lifetime Value (CLV) Prediction**
- **Entities**: Customers
- **Events**: Purchases, browsing sessions, support tickets
- **Interactions**: Social network effects, referrals
- **Artifacts**: Product review images, video tutorials watched
- **Prediction**: Future revenue per customer with key value drivers

**Why determinism matters**: Marketing budget allocation requires consistent predictions; A/B testing needs stable models.

**7. Personalized Recommendations**
- **Entities**: Products
- **Events**: View, cart additions, purchases
- **Interactions**: Product co-purchase networks
- **Artifacts**: Product images, demo videos
- **Prediction**: Ranking of recommended products with explanation

**Why determinism matters**: Debugging recommendation issues requires reproducible runs; legal compliance for algorithmic recommendations.

### Manufacturing & IoT

**8. Predictive Maintenance**
- **Entities**: Industrial equipment
- **Events**: Sensor readings (temperature, vibration, pressure)
- **Interactions**: Dependencies between connected machines
- **Artifacts**: Thermal images, vibration audio recordings
- **Prediction**: Failure probability with contributing sensor signals

**Why determinism matters**: Safety-critical applications require validated models; debugging false alarms needs reproducibility.

**9. Quality Control**
- **Entities**: Manufactured parts
- **Events**: Production steps, measurements
- **Interactions**: Batch relationships, supplier networks
- **Artifacts**: Inspection images, ultrasound scans
- **Prediction**: Defect classification with visual attribution

**Why determinism matters**: ISO certification requires documented, reproducible processes; liability claims need audit trails.

**10. GMP Manufacturing Oversight and Continuous Improvement**
- **Entities**: Batches, lots, lines, and process units
- **Events**: Electronic batch records, deviations, CAPA actions, environmental readings
- **Interactions**: Supplier-material-line dependencies and shift/hand-off relationships
- **Artifacts**: SOP revisions, calibration certificates, in-process images, QA documents
- **Prediction**: Deviation risk, yield drift, and quality trend signals with explainable contributors

**Why determinism matters**: GMP operations require traceable, reproducible evidence for audit readiness, deviation investigations, and validated continuous improvement actions.

### Telecommunications

**11. Churn Prediction**
- **Entities**: Subscribers
- **Events**: Usage patterns, billing history, support calls
- **Interactions**: Social network effects
- **Artifacts**: Call recordings, chat transcripts
- **Prediction**: Churn probability with retention strategies

**Why determinism matters**: Retention campaigns need stable targeting; debugging model drift requires comparing runs.

### Government & Public Sector

**12. Social Program Eligibility**
- **Entities**: Applicants
- **Events**: Employment history, benefit usage
- **Interactions**: Family networks, referrals
- **Artifacts**: Document scans, interview recordings
- **Prediction**: Eligibility scores with explainable criteria

**Why determinism matters**: Fairness and equity require auditable decisions; legal challenges need model reproducibility.

**13. Infrastructure Monitoring**
- **Entities**: Bridges, roads, public facilities
- **Events**: Inspection reports, usage metrics
- **Interactions**: Infrastructure networks
- **Artifacts**: Drone images, sensor data
- **Prediction**: Maintenance priority rankings

**Why determinism matters**: Budget accountability requires transparent prioritization; disaster prevention needs reliable predictions.

## Detailed Implementation Workflows by Use Case

The following workflows provide an execution-ready blueprint for each use case listed above. Each workflow follows a common operating model:

1. Data contract and schema mapping
2. Deterministic ingestion and checkpoint controls
3. Feature extraction and cache/version validation
4. Train (sync or async) and record run lineage
5. Validate quality, explainability, and determinism
6. Promote to production with telemetry and governance checks

### 1. Credit Risk Scoring Workflow

1. Define a canonical applicant entity schema with numeric attributes and target fields (`target_regression`, `target_binary`, `target_ranking`).
2. Build event feeds for payment history, delinquency transitions, and account actions in `events.csv` format.
3. Build interaction feeds for co-signers and entity linkages in `interactions.csv` format.
4. Ingest entities/events/interactions through `/ingest/*` using chunked form controls (`chunk_size`, `checkpoint_key`, `resume_from_checkpoint`) for large batches.
5. Ingest supporting document artifacts through `/ingest/artifacts` or `/ingest/artifact` and run `/features/extract` or `/features/extract/async`.
6. Start training with `/train/async` and an idempotency key; poll `/train/async/{task_id}` to completion.
7. Validate run outputs with `/runs/{run_id}` and run deterministic parity checks (`python -m app.cli determinism-check`).
8. Score applicant cohorts via `/predict` or `/predict/async`; persist probability and ranking outputs in the decision platform.
9. Attach explanation payloads to adverse-action and approval workflows for compliance review.
10. Monitor queue and latency health via `/health/queues` and performance report artifacts before expanding traffic.

### 2. Fraud Detection Workflow

1. Model transactions as entities or entity-linked events depending on operational granularity.
2. Capture sequential behavior (velocity, amount spikes, geo shifts) in timestamped events.
3. Capture transfer graph edges in interactions to expose suspicious network motifs.
4. Ingest receipt imagery, IVR/call artifacts, or chat transcript artifacts for multimodal evidence.
5. Run feature extraction with versioned cache validation and ensure stale features are recomputed.
6. Train asynchronously with small epochs first for quick calibration cycles, then full training jobs.
7. Validate false-positive and true-positive tradeoffs using saved run metrics and threshold policies.
8. Deploy synchronous `/predict` for real-time scoring and async `/predict/async` for retrospective sweeps.
9. Route high-risk scores to hold/review queues with explanation fragments attached for investigator context.
10. Use correlation IDs to trace each flagged decision from API request to queued task and telemetry event.

### 3. Algorithmic Trading Workflow

1. Define securities or strategy units as entities with normalized state attributes.
2. Stream market micro-events into deterministic event records ordered by timestamp plus stable identifiers.
3. Encode cross-asset relationships in interactions (correlation, sector linkage, spread dynamics).
4. Ingest supporting news/audio artifacts where available for multimodal context.
5. Use checkpoint-enabled ingestion to recover from feed interruption without data duplication.
6. Train run candidates asynchronously and compare run-level metrics and hash outputs.
7. Execute deterministic replay checks to confirm same-run reproducibility under pinned runtime.
8. Publish scored outputs to strategy simulation environments before production use.
9. Apply query workflows (`/query`) for analyst exploration of strongest/weakest signals.
10. Gate production deployment on CI determinism matrix and release-gate workflow success.

### 4. Patient Risk Stratification Workflow

1. Define patient entity schema with strict governance over sensitive attributes.
2. Record encounter, lab, treatment, and utilization timelines as events.
3. Map provider referral and care-team dependencies as interactions.
4. Ingest imaging/audio artifacts and enforce SHA256 integrity controls.
5. Run deterministic ingestion and checkpointing to support long-running healthcare data loads.
6. Execute feature extraction and verify feature version consistency across train/infer paths.
7. Train with explicit run-state lifecycle tracking to preserve recoverability evidence.
8. Score populations with `/predict`; include explanations for clinician review and second-level audit.
9. Use approval-controlled agent workflows for recurring cohort analysis and policy-sensitive tasks.
10. Archive run metadata, audit events, and determinism reports for regulatory and peer-review requirements.

### 5. Drug Discovery Workflow

1. Define compounds as entities and encode assay outcomes as events.
2. Model compound-protein or compound-pathway links as interactions.
3. Ingest molecular or spectroscopy artifacts and attach to relevant entities.
4. Batch ingest datasets with deterministic checkpointing to handle large experimental corpora.
5. Extract multimodal features and lock feature-version hashes for experiment reproducibility.
6. Train multiple candidate runs asynchronously using distinct config payloads and idempotency keys.
7. Compare regression, probability, and ranking metrics across runs and document threshold policy decisions.
8. Reproduce top run(s) and verify parity before reporting efficacy/toxicity outputs.
9. Query high-potential compounds using natural-language ranking prompts through `/query`.
10. Store model/run artifacts and telemetry snapshots to support downstream submission packages.

### 6. Customer Lifetime Value Workflow

1. Define customer entities with standardized profile and target attributes.
2. Ingest purchase/support/engagement histories as ordered events.
3. Encode referral and household relationships as interaction graphs.
4. Add relevant artifact context (reviews, call audio, media engagement) where available.
5. Execute chunked ingest for daily updates and resume from checkpoints on interruption.
6. Train asynchronously to avoid blocking API paths during periodic retraining windows.
7. Score full customer base via async batch prediction and persist outputs in CRM marts.
8. Use explanation outputs to identify dominant value drivers by segment.
9. Use `/query` prompts for campaign planning (high value, elevated risk, strongest relationship signals).
10. Monitor queue saturation and backlog to keep retraining/scoring jobs within SLA windows.

### 7. Personalized Recommendations Workflow

1. Define catalog items and users as entities according to recommendation objective.
2. Capture views/carts/purchases as events with strict timestamp ordering.
3. Encode co-purchase and affinity edges as interactions.
4. Ingest product images/video/audio artifacts to enrich item understanding.
5. Run feature extraction and confirm cache freshness after extractor changes.
6. Train ranking-oriented runs and evaluate rank-focused metrics from run outputs.
7. Serve near-real-time recommendations through synchronous `/predict` for request path use.
8. Serve periodic bulk recommendation refreshes through `/predict/async`.
9. Expose narrative and explanation details to merchandisers and support teams.
10. Validate recommendation stability with repeated deterministic checks before seasonal rollout.

### 8. Predictive Maintenance Workflow

1. Define machines/assets as entities with equipment metadata attributes.
2. Stream sensor observations as events, preserving deterministic order.
3. Encode machine-line dependencies and upstream/downstream links as interactions.
4. Ingest thermal/vibration/acoustic artifacts from inspections and sensors.
5. Use checkpointed ingestion to process high-volume telemetry reliably.
6. Extract features and validate modality coverage for each critical asset class.
7. Train asynchronously and monitor task lifecycle transitions to completion.
8. Score assets and prioritize failures via ranking output and probability thresholds.
9. Attach explanation signals to maintenance work orders for technician context.
10. Track queue health and telemetry to ensure backlog drains inside maintenance planning windows.

### 9. Quality Control Workflow

1. Define manufactured units or batches as entities.
2. Capture process station readings and QA checkpoints as events.
3. Encode supplier, batch, and production-line relationships as interactions.
4. Ingest inspection images/audio/ultrasound artifacts and verify hash integrity.
5. Ingest and extract in deterministic chunks to sustain production-scale throughput.
6. Train defect scoring models with explicit pending/success/failed run states.
7. Score production output through sync or async APIs depending on latency requirements.
8. Use explanations to localize likely defect drivers (event sequence, artifact signals, interaction context).
9. Automate quarantine/escalation rules using probability and ranking thresholds.
10. Retain audit-ready run manifests and telemetry reports for certification evidence.

### 10. GMP Manufacturing Oversight and Continuous Improvement Workflow

1. Define GMP-relevant entities for batches, lots, production lines, and equipment states.
2. Ingest electronic batch records, deviations, environmental monitoring, and process step telemetry as ordered events.
3. Encode supplier-material-line and hand-off dependencies as interactions to expose systemic quality risk paths.
4. Ingest controlled artifacts (SOP revisions, calibration records, QA documentation, inspection media) with integrity checks.
5. Use checkpointed ingestion for large historical quality datasets and for reliable daily incremental ingestion.
6. Extract features and verify feature-version stability so validated model behavior remains reproducible across cycles.
7. Train asynchronously with idempotency keys and preserve full run lineage for validation and quality-unit review.
8. Score lots and in-process checkpoints for deviation/yield risk; attach explanation payloads for root-cause triage.
9. Route high-risk signals to human review workflows (QA, manufacturing science, compliance) rather than autonomous release decisions.
10. Monitor queue and telemetry health, and enforce determinism and CI release gates before promoting model updates into GMP decision-support operations.

### 11. Churn Prediction Workflow

1. Define subscriber entities and align retention-related targets.
2. Ingest billing, usage, support, and engagement events.
3. Encode social or account-link relationships as interactions where policy allows.
4. Ingest call/chat artifacts and extract multimodal features for risk context.
5. Run asynchronous retraining on rolling cadence with idempotency-protected jobs.
6. Validate model quality and determinism before campaign activation.
7. Score all subscribers in async batches; persist results to retention tooling.
8. Drive intervention prioritization from ranking scores and churn probabilities.
9. Use `/query` for analyst workflows (elevated risk segments, strongest relationship patterns).
10. Monitor queue telemetry and model drift to adjust retraining frequency.

### 12. Social Program Eligibility Workflow

1. Define applicant entity schema and policy-aligned target labels.
2. Capture application and verification timelines as events.
3. Encode legally permitted household/referral relationships as interactions.
4. Ingest document and interview artifacts with strict integrity tracking.
5. Process all data through deterministic checkpointed ingestion for evidentiary traceability.
6. Train with run lifecycle persistence and capture all config/manifest metadata.
7. Generate predictions and explanations for reviewer decision support, not auto-adjudication alone.
8. Use agent workflows with approval-required controls for higher-risk automation operations.
9. Produce compliance summaries and immutable audit records for governance review.
10. Enforce promotion gates (determinism + integration + performance smoke) before policy deployment.

### 13. Infrastructure Monitoring Workflow

1. Define infrastructure assets as entities with geo/type/criticality attributes.
2. Ingest inspection cycles and telemetry readings as events.
3. Encode network dependencies and cascading-impact relationships as interactions.
4. Ingest drone imagery and sensor artifacts for multimodal condition evidence.
5. Run high-volume checkpointed ingestion for recurring inspection datasets.
6. Extract features and retrain asynchronously on maintenance cadence.
7. Score assets for maintenance priority using ranking and risk probabilities.
8. Attach explanations to budget and maintenance board decisions for transparency.
9. Query high-risk cohorts via natural-language prompts for planning briefings.
10. Track queue saturation and end-to-end latency telemetry to maintain operational SLAs.

### Shared Validation and Release Checklist for All Use Cases

1. Confirm ingest reports indicate expected totals, successes, and failures.
2. Confirm queue health is acceptable before and during async workloads.
3. Confirm run status transitions and artifact persistence are consistent.
4. Confirm deterministic parity checks pass on representative data slices.
5. Confirm explanation payloads are present for regulated/high-impact workflows.
6. Confirm CI gates (`Backend CI`, `Determinism Matrix CI`, `Phase 4 Release Gate`) pass before promotion.
7. Confirm telemetry reports include request, async task, and queue health events.

---

## Getting Started

### Prerequisites
- Docker and Docker Compose installed
- At least 4GB RAM available
- 10GB disk space for models and artifacts

### Quick Start (5 Minutes)

**1. Clone the repository**
```bash
git clone https://github.com/capnplanet/PredictiveDeterministicModel.git
cd PredictiveDeterministicModel
```

**2. Start all services**
```bash
docker-compose up -d
```

This starts three services:
- PostgreSQL database on port 5432
- FastAPI backend on port 8000
- React frontend on port 5173

**3. Verify the system is running**
```bash
curl http://localhost:8000/health
# Should return: {"status":"ok"}
```

**4. Open the UI**
Navigate to `http://localhost:5173` in your browser.

**5. Run a complete workflow**

a. **Generate synthetic data**
```bash
docker-compose exec backend python -m app.training.synth_data
```

b. **Extract features**
```bash
curl -X POST http://localhost:8000/features/extract
```

c. **Train a model**
```bash
curl -X POST http://localhost:8000/train
```

d. **Make predictions**
```bash
curl -X POST http://localhost:8000/predict \
  -H "Content-Type: application/json" \
  -d '{"entity_ids": ["ent_000", "ent_001"], "explanations": true}'
```

### Custom Data Integration

**1. Prepare your CSV files**

**entities.csv**:
```csv
entity_id,attributes
customer_001,"{""age"": 35, ""location"": ""NYC"", ""x"": 0.5, ""y"": 0.3, ""z"": 0.8, ""target_regression"": 42.0, ""target_binary"": 1, ""target_ranking"": 0.75}"
customer_002,"{""age"": 28, ""location"": ""LA"", ""x"": 0.3, ""y"": 0.6, ""z"": 0.4, ""target_regression"": 37.5, ""target_binary"": 0, ""target_ranking"": 0.55}"
```

**events.csv**:
```csv
entity_id,timestamp,event_type,event_value
customer_001,2024-01-15T10:30:00,purchase,150.0
customer_001,2024-02-20T14:20:00,support_call,1.0
customer_002,2024-01-18T09:15:00,purchase,75.0
```

**interactions.csv**:
```csv
src_entity_id,dst_entity_id,interaction_type,interaction_value,timestamp
customer_001,product_456,purchased,3,2024-01-15T10:30:00
customer_002,product_789,purchased,1,2024-01-18T09:15:00
customer_001,customer_002,referred,1,2024-02-01T00:00:00
```

**2. Upload via API**
```bash
curl -X POST http://localhost:8000/ingest/entities \
  -F "file=@entities.csv"

curl -X POST http://localhost:8000/ingest/events \
  -F "file=@events.csv"

curl -X POST http://localhost:8000/ingest/interactions \
  -F "file=@interactions.csv"
```

**3. Upload artifacts (optional)**
```bash
curl -X POST http://localhost:8000/ingest/artifact \
  -F "file=@customer_photo.jpg" \
  -F "artifact_type=image" \
  -F "entity_id=customer_001"
```

**4. Extract features and train**
```bash
curl -X POST http://localhost:8000/features/extract
curl -X POST http://localhost:8000/train \
  -H "Content-Type: application/json" \
  -d '{"epochs": 50, "batch_size": 16, "learning_rate": 0.001}'
```

### Development Setup

**1. Install Python dependencies**
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
pip install -r requirements.txt
```

**2. Set up database**
```bash
# Start PostgreSQL
docker-compose up -d db

# Run migrations
alembic upgrade head
```

**3. Run backend locally**
```bash
uvicorn app.main:app --reload --port 8000
```

**4. Run frontend locally**
```bash
cd frontend
npm install
npm run dev
```

### Configuration

**Environment Variables** (`.env` file):
```
DATABASE_URL=postgresql://postgres:postgres@localhost:5432/mlstack
PYTHONHASHSEED=0
OMP_NUM_THREADS=1
MKL_NUM_THREADS=1
CUBLAS_WORKSPACE_CONFIG=:4096:8
```

**Training Configuration** (via API):
```json
{
  "epochs": 100,
  "batch_size": 32,
  "learning_rate": 0.001,
  "seed": 1234,
  "max_neighbors": 10,
  "max_artifacts_per_entity": 5
}
```

---

## API Reference

### Health Check

**Endpoint**: `GET /health`

**Response**:
```json
{
  "status": "ok"
}
```

### Data Ingestion

**Demo preload (synthetic bootstrap)**
```
POST /demo/preload
Content-Type: application/json

{
  "profile": "small",
  "reset_existing": true,
  "extract_features": true,
  "train_model": true
}
```

**Upload Entities**
```
POST /ingest/entities
Content-Type: multipart/form-data

file: entities.csv
```

**Upload Events**
```
POST /ingest/events
Content-Type: multipart/form-data

file: events.csv
```

**Upload Interactions**
```
POST /ingest/interactions
Content-Type: multipart/form-data

file: interactions.csv
```

**Upload Single Artifact**
```
POST /ingest/artifact
Content-Type: multipart/form-data

file: image.jpg
artifact_type: "image" | "audio" | "video"
entity_id: "optional_entity_id"
timestamp: "optional_timestamp"
metadata: "optional_json_metadata"
```

**Response Format**:
```json
{
  "success_rows": 150,
  "failed_rows": 2,
  "errors": [
    {"row": 25, "error": "Invalid entity_id"},
    {"row": 48, "error": "Missing required field"}
  ]
}
```

### Feature Extraction

**Extract Features for Pending Artifacts**
```
POST /features/extract
```

**Response**:
```json
{
  "updated_artifacts": 42
}
```

### Model Training

**Train New Model**
```
POST /train
Content-Type: application/json

{
  "epochs": 100,
  "batch_size": 32,
  "learning_rate": 0.001,
  "seed": 1234
}
```

**Response**:
```json
{
  "run_id": "a3f7b2c1...",
  "status": "completed",
  "metrics": {
    "regression": {
      "mae": 2.34,
      "rmse": 3.12,
      "r2": 0.87
    },
    "classification": {
      "f1": 0.92,
      "precision": 0.89,
      "recall": 0.95
    },
    "ranking": {
      "ndcg": 0.88,
      "spearman": 0.81
    }
  },
  "config": {...},
  "created_at": "2024-01-15T10:30:00Z",
  "model_sha256": "e3b0c44298fc1c149afb..."
}
```

**List All Training Runs**
```
GET /runs
```

**Response**:
```json
[
  {
    "run_id": "a3f7b2c1...",
    "created_at": "2024-01-15T10:30:00Z",
    "status": "completed",
    "metrics": {...}
  },
  ...
]
```

**Get Specific Run Details**
```
GET /runs/{run_id}
```

### Predictions

**Make Predictions**
```
POST /predict
Content-Type: application/json

{
  "entity_ids": ["ent_001", "ent_002", "ent_003"],
  "run_id": "optional_run_id",
  "explanations": true
}
```

### Natural-Language Query

**Run Query Retrieval**
```
POST /query
Content-Type: application/json

{
  "query": "Identify top entities by relationship strength with elevated probability.",
  "limit": 5,
  "run_id": "optional_run_id"
}
```

**Response**:
```json
{
  "run_id": "abc123...",
  "query": "Identify top entities by relationship strength with elevated probability.",
  "interpreted_as": "... [match=broad_scan; order=strongest; probability=elevated]",
  "llm_used": true,
  "results": [
    {
      "entity_id": "E00027",
      "regression": -0.3466,
      "probability": 0.4326,
      "ranking_score": 0.6073,
      "narrative": "Data-grounded narrative text"
    }
  ]
}
```

**Response**:
```json
{
  "run_id": "a3f7b2c1...",
  "predictions": [
    {
      "entity_id": "ent_001",
      "regression": 42.5,
      "probability": 0.85,
      "ranking_score": 0.92,
      "embedding": [0.12, 0.34, -0.15, ...],
      "explanation": {
        "attribute_attributions": {
          "x": 0.35,
          "y": 0.28,
          "z": 0.15
        },
        "event_attention_weights": [0.1, 0.3, 0.4, 0.2],
        "artifact_contributions": [
          {
            "artifact_id": "art_001",
            "contribution": 0.22,
            "artifact_type": "image"
          }
        ]
      }
    },
    ...
  ]
}
```

---

## Security and Compliance Considerations

### Data Security

**1. Data at Rest**
- Database encryption: Use PostgreSQL encryption extensions (pgcrypto)
- File encryption: Encrypt artifact storage with filesystem-level encryption
- Secrets management: Store credentials in environment variables or secret managers (AWS Secrets Manager, HashiCorp Vault)

**2. Data in Transit**
- Enable TLS/SSL for all API endpoints
- Use HTTPS for frontend-backend communication
- Configure PostgreSQL to require SSL connections

**3. Access Control**
- Implement API key authentication for all endpoints
- Add role-based access control (RBAC) for multi-user deployments
- Use JWT tokens for session management

**Example API Key Middleware**:
```python
from fastapi import Security, HTTPException
from fastapi.security import APIKeyHeader

api_key_header = APIKeyHeader(name="X-API-Key")

async def verify_api_key(api_key: str = Security(api_key_header)):
    if api_key != os.getenv("API_KEY"):
        raise HTTPException(status_code=403, detail="Invalid API Key")
    return api_key

# Apply to routes
@router.post("/predict", dependencies=[Depends(verify_api_key)])
async def predict(...):
    ...
```

### Privacy & Compliance

**1. GDPR Compliance**
- **Right to be forgotten**: Implement entity deletion endpoints
- **Data portability**: Use Parquet export for data extraction
- **Consent tracking**: Add consent flags to entity attributes
- **Audit logs**: Log all data access and predictions

**2. HIPAA Compliance (Healthcare)**
- **PHI protection**: Encrypt all personal health information
- **Access logs**: Track who accessed which patient data
- **De-identification**: Support anonymization before model training
- **Business Associate Agreement (BAA)**: Required for cloud deployments

**3. SOC 2 Type II**
- **Availability**: Deploy with high availability (load balancers, replicas)
- **Confidentiality**: Implement encryption and access controls
- **Processing integrity**: Deterministic training ensures consistency
- **Privacy**: Data minimization and retention policies

### Audit Trail Implementation

**Add audit logging**:
```python
# app/middleware/audit.py
import logging
from datetime import datetime

audit_logger = logging.getLogger("audit")

@app.middleware("http")
async def audit_middleware(request: Request, call_next):
    start_time = datetime.now()
    response = await call_next(request)
    duration = (datetime.now() - start_time).total_seconds()
    
    audit_logger.info({
        "timestamp": start_time.isoformat(),
        "method": request.method,
        "path": request.url.path,
        "status": response.status_code,
        "duration": duration,
        "user": request.headers.get("X-User-ID"),
        "ip": request.client.host
    })
    
    return response
```

### Model Governance

**1. Model Versioning**
- All models are versioned with deterministic run IDs
- Config and data manifests are stored for each run
- Model hashes enable integrity verification

**2. Model Monitoring**
- Track prediction distributions over time
- Alert on data drift (input distribution changes)
- Monitor prediction latency and throughput

**3. Model Reproducibility**
- Deterministic training ensures bit-exact reproduction
- Use `reproduce_run()` function to verify model integrity
- Store training logs for forensic analysis

**4. Model Explainability**
- Integrated gradients provide feature attributions
- Attention weights show which events matter
- Documentation required for high-stakes decisions

### Regulatory Mapping

| Regulation | Key Requirements | How This System Addresses It |
|------------|------------------|------------------------------|
| **GDPR (EU)** | Right to explanation, data portability, right to be forgotten | Explainability features, Parquet export, entity deletion API |
| **CCPA (California)** | Disclose data sources, allow data deletion | Data manifest tracking, entity deletion API |
| **FCRA (Credit Reporting)** | Adverse action notices, model transparency | Explanation API provides reasons for decisions |
| **HIPAA (Healthcare)** | PHI protection, access logs, encryption | Database encryption, audit logs, access controls |
| **SR 11-7 (Banking)** | Model risk management, validation, documentation | Reproducibility checks, metrics tracking, documentation |
| **EU AI Act** | High-risk system requirements, transparency | Explainability, determinism, audit trails |

---

## Summary

The **Deterministic Multimodal Analytics Stack** is a production-ready, enterprise-grade machine learning platform that brings reproducibility, explainability, and auditability to predictive analytics. Its unique deterministic guarantees make it ideal for regulated industries where model consistency and transparency are critical.

**Key Differentiators**:
✅ **True Reproducibility**: Bit-exact identical results across training runs  
✅ **Multi-Task Learning**: Regression, classification, and ranking in one model  
✅ **Multimodal Fusion**: Combines tabular, time-series, graph, and media data  
✅ **Built-in Explainability**: Understand what drives every prediction  
✅ **Enterprise-Ready**: Docker Compose deployment, RESTful APIs, audit trails  
✅ **Compliance-Focused**: Designed for GDPR, HIPAA, SR 11-7, and EU AI Act

**Quick Integration**: Deploy in 5 minutes with Docker Compose, integrate with existing systems via REST APIs, scale to production with Kubernetes or cloud services.

**Get Started**: Visit the [GitHub repository](https://github.com/capnplanet/PredictiveDeterministicModel) for code, examples, and documentation.

---

## Additional Resources

- **API Documentation**: Available at `http://localhost:8000/docs` (Swagger UI)
- **Sample Datasets**: Included in `app/training/synth_data.py`
- **Test Suite**: Run `PYTHONPATH=. pytest -q app/tests` for comprehensive test coverage
- **Performance Benchmarks**: See `app/tests/test_determinism_check.py` for reproducibility verification

For questions, issues, or feature requests, please open an issue on GitHub.

---

*Document Version: 1.0*  
*Last Updated: January 2026*  
*Maintained by: PredictiveDeterministicModel Team*
