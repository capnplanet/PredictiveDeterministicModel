# Deterministic Multimodal Analytics Stack

This repository implements a deterministic, domain-agnostic analytics stack with a FastAPI backend, Postgres storage, and a React/Vite frontend. The backend provides CSV and artifact ingestion, multimodal feature extraction, deterministic training and inference, and reproducibility tooling. The frontend exposes dataset management, training, runs, and prediction workflows.

## 📚 Documentation

For comprehensive information about this repository and how to use it in enterprise environments:

- **[UI User Guide](UI_USER_GUIDE.md)** - Detailed walkthrough of the current Defense-Grade Decision Console UI, including tab-by-tab workflows, status messaging, CSV expectations, troubleshooting, and UI-to-API mapping

- **[Quick Start Guide](QUICK_START_GUIDE.md)** - Get up and running in 5 minutes with curl examples, CSV formats, and troubleshooting

- **[Comprehensive Repository Guide](COMPREHENSIVE_REPOSITORY_GUIDE.md)** - Complete 46KB guide with:
  - Feynman-style explanations of core concepts (entities, events, artifacts, determinism)
  - Complete technical architecture and data flow
  - 7 enterprise integration patterns with working code (ETL, microservices, batch, real-time, MLOps, multi-tenant, hybrid cloud)
  - 12 industry-specific use cases (Finance, Healthcare, Retail, Manufacturing, Government)
  - Full API reference with request/response examples
  - Security and compliance mapping (GDPR, HIPAA, SOC 2, FCRA, EU AI Act)

- **[Architecture Diagrams](ARCHITECTURE_DIAGRAMS.md)** - Visual reference with ASCII diagrams showing:
  - System architecture and component interactions
  - Complete data flow pipeline (ingestion → features → training → prediction)
  - Neural network architecture breakdown
  - Enterprise integration patterns
  - Determinism guarantees and explainability methods

## 🚀 Quick Start

```bash
# Clone and start
git clone https://github.com/capnplanet/PredictiveDeterministicModel.git
cd PredictiveDeterministicModel
docker-compose up -d

# Verify
curl http://localhost:8000/health

# Open UI
open http://localhost:5173
```

## ✨ Key Features

- ✅ **Deterministic Training** - Bit-exact reproducibility across runs
- ✅ **Determinism Matrix CI** - Cross-environment reproducibility gate with artifact/hash comparison
- ✅ **Performance Telemetry** - Structured latency/throughput metrics with CI report artifacts
- ✅ **Multi-Task Learning** - Regression, Classification, and Ranking in one model
- ✅ **Multimodal Support** - Images, Audio, Video, and Tabular data
- ✅ **Built-in Explainability** - Understand what drives every prediction
- ✅ **LLM Narrative Augmentation (Optional)** - Generate plain-text long-form narratives from deterministic outputs
- ✅ **Natural-Language Query Endpoint** - Ask retrieval-style questions over entity predictions via `/query`
- ✅ **Enterprise Ready** - Docker Compose, REST APIs, complete audit trails
- ✅ **Compliance Focused** - Designed for GDPR, HIPAA, and regulated industries

## 🤖 Optional LLM Settings

Set these environment variables to enable Hugging Face endpoint-based narrative/query augmentation:

```bash
LLM_ENABLED=true
LLM_PROVIDER=huggingface
LLM_ENDPOINT_URL=<your_hf_inference_endpoint>
LLM_API_TOKEN=<your_hf_token>
LLM_MODEL_ID=meta-llama/Llama-3.1-8B-Instruct
LLM_TIMEOUT_SECONDS=5
LLM_MAX_TOKENS=500
LLM_TEMPERATURE=0.2
```

If disabled or unavailable, the platform automatically falls back to deterministic template narratives.

## 📈 Performance Metrics

The backend now emits structured performance events to `data/performance_metrics.jsonl` for API requests, ingestion, training, and determinism stages.

Generate a summarized report locally:

```bash
PYTHONPATH=. python -m app.cli performance-report
```

CI uploads these as artifacts in backend and determinism-matrix workflows (report-only mode).
