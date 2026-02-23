# Quick Start Guide - Deterministic Multimodal Analytics Stack

## What Is This?

A production-ready machine learning platform that:
- ✅ Ingests CSV data and media files (images/audio/video)
- ✅ Trains reproducible, explainable AI models
- ✅ Makes predictions with transparent reasoning
- ✅ Provides complete audit trails for compliance

**Perfect for**: Finance, Healthcare, Legal, and any regulated industry requiring model accountability.

---

## 5-Minute Quick Start

### 1. Start the System
```bash
git clone https://github.com/capnplanet/PredictiveDeterministicModel.git
cd PredictiveDeterministicModel
docker-compose up -d
```

**Services Started:**
- 🔷 PostgreSQL Database (port 5432)
- 🔶 FastAPI Backend (port 8000)
- 🔷 React Frontend (port 5173)

### 2. Verify Health
```bash
curl http://localhost:8000/health
# Should return: {"status":"ok"}
```

### 3. View the UI
Open browser: `http://localhost:5173`

### 4. Run Complete Demo Workflow
```bash
# Generate synthetic demo data
docker-compose exec backend python -m app.training.synth_data

# Extract features from artifacts
curl -X POST http://localhost:8000/features/extract

# Train a model
curl -X POST http://localhost:8000/train

# Make predictions with explanations
curl -X POST http://localhost:8000/predict \
  -H "Content-Type: application/json" \
  -d '{"entity_ids": ["ent_000", "ent_001"], "explanations": true}'
```

---

## Key Concepts (Simple)

### Entities
**What**: The things you want to analyze (customers, products, patients, sensors)  
**Example**: `customer_001` with attributes like age, location

### Events
**What**: Time-stamped actions about entities  
**Example**: Customer made a purchase on 2024-01-15 for $150

### Interactions
**What**: Relationships between entities  
**Example**: Customer purchased Product, User referred Friend

### Artifacts
**What**: Supporting media files  
**Example**: Product images, customer service call recordings, medical scans

### Predictions
**What**: Three types of outputs:
- **Regression**: Continuous numbers (revenue prediction)
- **Classification**: Yes/No (fraud detection)
- **Ranking**: Order of items (recommendations)

### Determinism
**What**: Same input → Always same output  
**Why**: Compliance, debugging, scientific rigor

---

## API Cheat Sheet

### Upload Data
```bash
# Upload entities CSV
curl -X POST http://localhost:8000/ingest/entities -F "file=@entities.csv"

# Upload events CSV
curl -X POST http://localhost:8000/ingest/events -F "file=@events.csv"

# Upload interactions CSV
curl -X POST http://localhost:8000/ingest/interactions -F "file=@interactions.csv"

# Upload single image
curl -X POST http://localhost:8000/ingest/artifact \
  -F "file=@image.jpg" \
  -F "artifact_type=image" \
  -F "entity_id=customer_001"
```

### Extract Features
```bash
curl -X POST http://localhost:8000/features/extract
```

### Train Model
```bash
# Default config
curl -X POST http://localhost:8000/train

# Custom config
curl -X POST http://localhost:8000/train \
  -H "Content-Type: application/json" \
  -d '{
    "epochs": 50,
    "batch_size": 16,
    "learning_rate": 0.001
  }'
```

### List Training Runs
```bash
curl http://localhost:8000/runs
```

### Make Predictions
```bash
# Simple predictions
curl -X POST http://localhost:8000/predict \
  -H "Content-Type: application/json" \
  -d '{"entity_ids": ["ent_001", "ent_002"]}'

# With explanations
curl -X POST http://localhost:8000/predict \
  -H "Content-Type: application/json" \
  -d '{
    "entity_ids": ["ent_001", "ent_002"],
    "explanations": true
  }'
```

---

## CSV Data Format

### entities.csv
```csv
entity_id,attributes
customer_001,"{""x"": 0.5, ""y"": 0.3, ""z"": 0.8, ""target_regression"": 42.0, ""target_binary"": 1, ""target_ranking"": 0.75}"
```

**Required fields in attributes JSON**:
- `x`, `y`, `z`: Entity coordinates/features
- `target_regression`: Target for regression task (float)
- `target_binary`: Target for classification task (0 or 1)
- `target_ranking`: Target for ranking task (0.0 to 1.0)

### events.csv
```csv
entity_id,timestamp,event_type,event_value
customer_001,2024-01-15T10:30:00,purchase,150.0
customer_001,2024-02-20T14:20:00,login,1.0
```

**Event types**: Custom strings (e.g., "purchase", "login", "support_call")

### interactions.csv
```csv
src_entity_id,dst_entity_id,interaction_type,interaction_value,timestamp
customer_001,product_456,purchased,3,2024-01-15T10:30:00
customer_001,customer_002,referred,1,2024-02-01T00:00:00
```

---

## Enterprise Integration Patterns

### Pattern 1: REST API Integration
```python
import requests

# Upload data
with open('data.csv', 'rb') as f:
    requests.post('http://ml-stack:8000/ingest/entities', files={'file': f})

# Train
response = requests.post('http://ml-stack:8000/train')
run_id = response.json()['run_id']

# Predict
predictions = requests.post('http://ml-stack:8000/predict', json={
    'entity_ids': ['customer_001', 'customer_002'],
    'run_id': run_id
}).json()
```

### Pattern 2: Batch Scoring (Airflow)
```python
from airflow import DAG
from airflow.operators.python import PythonOperator

def nightly_predictions():
    import requests
    entity_ids = get_all_entity_ids()
    predictions = requests.post('http://ml-stack:8000/predict', 
                               json={'entity_ids': entity_ids}).json()
    store_predictions(predictions)

dag = DAG('ml_predictions', schedule_interval='0 2 * * *')
PythonOperator(task_id='predict', python_callable=nightly_predictions, dag=dag)
```

### Pattern 3: Real-Time API
```javascript
// Node.js/Express example
app.post('/check-fraud', async (req, res) => {
  const prediction = await axios.post('http://ml-stack:8000/predict', {
    entity_ids: [req.body.transaction_id],
    explanations: true
  });
  
  const result = prediction.data.predictions[0];
  res.json({
    isFraud: result.probability > 0.9,
    confidence: result.probability,
    reasons: result.explanation
  });
});
```

---

## Common Use Cases

### Financial Services
- **Credit Scoring**: Predict default risk with explainable factors
- **Fraud Detection**: Real-time transaction monitoring with attribution
- **Churn Prediction**: Identify at-risk customers with retention strategies

### Healthcare
- **Patient Risk**: Predict readmission or complications
- **Drug Discovery**: Predict compound efficacy and toxicity
- **Resource Planning**: Forecast bed utilization and staffing needs

### Retail & E-Commerce
- **Customer Lifetime Value**: Predict future revenue per customer
- **Recommendations**: Rank products with explanations
- **Inventory Optimization**: Forecast demand by product

### Manufacturing
- **Predictive Maintenance**: Predict equipment failures before they happen
- **Quality Control**: Detect defects with visual explanations
- **Supply Chain**: Optimize routing and scheduling

---

## Troubleshooting

### Issue: "Database connection failed"
**Solution**: Ensure PostgreSQL container is running
```bash
docker-compose ps
docker-compose up -d db
```

### Issue: "Model not found"
**Solution**: Train a model first
```bash
curl -X POST http://localhost:8000/train
```

### Issue: "No features for artifacts"
**Solution**: Extract features before training
```bash
curl -X POST http://localhost:8000/features/extract
```

### Issue: "Port already in use"
**Solution**: Change ports in docker-compose.yml or stop conflicting services
```bash
docker-compose down
sudo lsof -i :8000  # Find what's using port 8000
```

---

## Configuration

### Environment Variables
```bash
# .env file
DATABASE_URL=postgresql://user:pass@host/db
PYTHONHASHSEED=0              # Deterministic hashing
OMP_NUM_THREADS=1             # Single-threaded
MKL_NUM_THREADS=1
CUBLAS_WORKSPACE_CONFIG=:4096:8
```

### Training Config
```json
{
  "epochs": 100,              // Training iterations
  "batch_size": 32,           // Samples per batch
  "learning_rate": 0.001,     // Optimizer step size
  "seed": 1234,               // Random seed
  "max_neighbors": 10,        // Graph neighbor limit
  "max_artifacts_per_entity": 5  // Artifact limit
}
```

---

## Next Steps

1. **Read the Full Guide**: See `COMPREHENSIVE_REPOSITORY_GUIDE.md` for detailed documentation
2. **Try Custom Data**: Prepare your CSV files and upload them
3. **Explore the API**: Visit `http://localhost:8000/docs` for interactive API documentation
4. **Monitor Training**: Check logs with `docker-compose logs -f backend`
5. **Scale to Production**: Deploy to Kubernetes or cloud platforms

---

## Key Features Summary

✅ **Deterministic Training** - Bit-exact reproducibility  
✅ **Multi-Task Learning** - Regression + Classification + Ranking  
✅ **Multimodal Support** - Images + Audio + Video + Tabular data  
✅ **Built-in Explainability** - Understand every prediction  
✅ **Enterprise Ready** - Docker, REST APIs, audit trails  
✅ **Compliance Focused** - GDPR, HIPAA, SOC 2 compatible  

---

## Support & Resources

- **Full Documentation**: `COMPREHENSIVE_REPOSITORY_GUIDE.md`
- **API Docs**: `http://localhost:8000/docs`
- **Sample Data**: `python -m app.training.synth_data`
- **Tests**: `PYTHONPATH=. pytest -q app/tests`
- **GitHub**: [capnplanet/PredictiveDeterministicModel](https://github.com/capnplanet/PredictiveDeterministicModel)

---

*Quick Start Version: 1.0*  
*For detailed explanations, see COMPREHENSIVE_REPOSITORY_GUIDE.md*
