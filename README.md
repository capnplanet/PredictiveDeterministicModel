# Deterministic Multimodal Analytics Stack

This repository implements a deterministic, domain-agnostic analytics stack with a FastAPI backend, Postgres storage, and a React/Vite frontend. The backend provides CSV and artifact ingestion, multimodal feature extraction, deterministic training and inference, and reproducibility tooling. The frontend exposes dataset management, training, runs, and prediction workflows.

## 📚 Documentation

For comprehensive information about this repository and how to use it in enterprise environments:

- **[Documentation Index](DOCUMENTATION_INDEX.md)** - Central map of all guides and reference docs

- **[UI User Guide](UI_USER_GUIDE.md)** - Detailed walkthrough of the current Defense-Grade Decision Console UI, including tab-by-tab workflows, status messaging, CSV expectations, troubleshooting, and UI-to-API mapping

- **[Quick Start Guide](QUICK_START_GUIDE.md)** - Get up and running in 5 minutes with curl examples, CSV formats, and troubleshooting

- **[Corpus Validation Guide](CORPUS_VALIDATION_GUIDE.md)** - Reproducible validation playbooks for synthetic and public corpora, determinism checks, and acceptance thresholds

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

- **[Predictive Analytics Capabilities Guide](PREDICTIVE_ANALYTICS_CAPABILITIES.md)** - Plain-language deep dive into predictive capabilities, with in-context explanations of technical jargon and practical workflow guidance

- **[Telemetry Snapshot](TEST_TELEMETRY_2026-02-23.md)** - Historical CI run telemetry and performance summary example

## 🚀 Quick Start

```bash
# Clone and start
git clone https://github.com/capnplanet/PredictiveDeterministicModel.git
cd PredictiveDeterministicModel
docker compose up -d

# Verify
curl http://localhost:8000/health

# Open UI (Linux)
xdg-open http://localhost:5173
```

## ✨ Key Features

- ✅ **Deterministic Training** - Bit-exact reproducibility across runs
- ✅ **Determinism Matrix CI** - Cross-environment reproducibility gate with artifact/hash comparison
- ✅ **Performance Telemetry** - Structured latency/throughput metrics with CI report artifacts
- ✅ **Multi-Task Learning** - Regression, Classification, and Ranking in one model
- ✅ **Multimodal Support** - Images, Audio, Video, and Tabular data
- ✅ **Built-in Explainability** - Understand what drives every prediction
- ✅ **LLM Narrative Augmentation (Optional)** - Generate plain-text long-form narratives from deterministic outputs
- ✅ **Natural-Language Query Endpoint** - Ask retrieval-style questions over entity predictions via `/query` with intent tags (match/order/probability)
- ✅ **Query Prompt Presets in UI** - Pre-populated prompts for strongest relationships, risk investigation, prioritization, and anomaly storyline
- ✅ **Query Ranking Controls** - Relationship-driven sorting with elevated-probability intent support
- ✅ **Enterprise Ready** - Docker Compose, REST APIs, complete audit trails
- ✅ **Compliance Focused** - Designed for GDPR, HIPAA, and regulated industries

## 🤖 Optional LLM Settings

Set these environment variables to enable Hugging Face endpoint-based narrative/query augmentation:

```bash
LLM_ENABLED=true
LLM_PROVIDER=huggingface
HF_ENDPOINT_URL=<your_hf_inference_endpoint>
HF_TOKEN=<your_hf_token>
LLM_MODEL_ID=meta-llama/Llama-3.1-8B-Instruct
LLM_TIMEOUT_SECONDS=5
LLM_MAX_TOKENS=500
LLM_TEMPERATURE=0.2
```

`LLM_ENDPOINT_URL` / `LLM_API_TOKEN` aliases are also supported.

If disabled or unavailable, the platform automatically falls back to deterministic template narratives.

## 🔁 CI Workflows

Main branch changes are validated by:

- `Backend CI` (lint, type checks, backend tests)
- `Determinism Matrix CI` (cross-Python reproducibility gate)
- `E2E CI` (full-stack UI/API behavior)
- `Frontend CI` (frontend tests and quality checks)

## 📈 Performance Metrics

The backend now emits structured performance events to `data/performance_metrics.jsonl` for API requests, ingestion, training, and determinism stages.

Generate a summarized report locally:

```bash
PYTHONPATH=. python -m app.cli performance-report
```

CI uploads these as artifacts in backend and determinism-matrix workflows (report-only mode).

## 🧠 Query Behavior Notes

- Stable entity ordering across repeated queries is expected when run ID and data are unchanged (deterministic ranking outputs).
- Narrative wording can vary slightly between requests when LLM is enabled (`LLM_TEMPERATURE > 0`).
- Query responses include interpretation metadata in `interpreted_as`, including retrieval strategy and ordering intent.
