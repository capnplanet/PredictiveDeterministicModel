# Deterministic Multimodal Analytics Stack

This repository implements a deterministic, domain-agnostic analytics stack with a FastAPI backend, Postgres storage, and a React/Vite frontend. The backend provides CSV and artifact ingestion, multimodal feature extraction, deterministic training and inference, and reproducibility tooling. The frontend exposes dataset management, training, runs, and prediction workflows.

## 📚 Documentation

For comprehensive information about this repository and how to use it in enterprise environments:

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
- ✅ **Multi-Task Learning** - Regression, Classification, and Ranking in one model
- ✅ **Multimodal Support** - Images, Audio, Video, and Tabular data
- ✅ **Built-in Explainability** - Understand what drives every prediction
- ✅ **Enterprise Ready** - Docker Compose, REST APIs, complete audit trails
- ✅ **Compliance Focused** - Designed for GDPR, HIPAA, and regulated industries
