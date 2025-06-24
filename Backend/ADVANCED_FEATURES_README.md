# 🚀 Canis AI Advanced Backend Features

This document describes the advanced production-ready features added to the Canis AI Backend.

## 📋 Table of Contents

1. [Real-Time Inference](#real-time-inference)
2. [Model Versioning System](#model-versioning-system)
3. [Logging & Monitoring](#logging--monitoring)
4. [Async Task Queue](#async-task-queue)
5. [API Endpoints](#api-endpoints)
6. [Installation & Setup](#installation--setup)
7. [Testing](#testing)

---

## 🔴 1. Real-Time Inference

### Overview
Real-time prediction endpoints that accept JSON data and return predictions with probabilities.

### Features
- **Single Prediction**: `/predict/` - Predict on single data point
- **Batch Prediction**: `/predict-batch/` - Predict on multiple data points
- **Model Info**: `/model-info/` - Get current model information
- **Automatic Preprocessing**: Handles data preprocessing automatically
- **Probability Support**: Returns class probabilities for classification tasks
- **Target Decoding**: Automatically decodes predictions using target encoders

### Usage Examples

#### Single Prediction
```bash
curl -X POST "http://localhost:8000/api/v1/predict/" \
  -H "Content-Type: application/json" \
  -d '{
    "feature1": 1.5,
    "feature2": 2.3,
    "feature3": 0.8,
    "feature4": 1.2
  }'
```

#### Batch Prediction
```bash
curl -X POST "http://localhost:8000/api/v1/predict-batch/" \
  -H "Content-Type: application/json" \
  -d '[
    {"feature1": 1.5, "feature2": 2.3, "feature3": 0.8, "feature4": 1.2},
    {"feature1": 2.1, "feature2": 1.8, "feature3": 1.1, "feature4": 0.9}
  ]'
```

#### Response Format
```json
{
  "predictions": [25.6, 18.9],
  "probabilities": [
    {"class_0": 0.2, "class_1": 0.8},
    {"class_0": 0.7, "class_1": 0.3}
  ],
  "input_shape": [2, 4],
  "model_type": "RandomForestRegressor",
  "task_type": "regression"
}
```

---

## 🔴 2. Model Versioning System

### Overview
Complete model lifecycle management with versioning, metadata tracking, and SQLite database storage.

### Features
- **Version Management**: Automatic versioning with timestamps
- **Metadata Tracking**: Store accuracy, task type, features, etc.
- **SQLite Database**: Persistent storage of model metadata
- **Model Loading**: Load specific model versions
- **Model Deletion**: Remove old model versions
- **Tagging System**: Add tags and descriptions to models

### API Endpoints

#### Save Model Version
```bash
curl -X POST "http://localhost:8000/api/v1/save-model-version/" \
  -G \
  -d "model_name=my_model" \
  -d "description=Best performing model" \
  -d "tags=production,high_accuracy"
```

#### List Models
```bash
curl -X GET "http://localhost:8000/api/v1/list-models/"
```

#### Load Model
```bash
curl -X POST "http://localhost:8000/api/v1/load-model/" \
  -G \
  -d "model_id=1"
```

#### Delete Model
```bash
curl -X DELETE "http://localhost:8000/api/v1/delete-model/" \
  -G \
  -d "model_id=1"
```

#### Model Details
```bash
curl -X GET "http://localhost:8000/api/v1/model-details/" \
  -G \
  -d "model_id=1"
```

### Database Schema
```sql
CREATE TABLE model_registry (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    model_name TEXT NOT NULL,
    version TEXT NOT NULL,
    file_path TEXT NOT NULL,
    task_type TEXT,
    target_column TEXT,
    accuracy REAL,
    f1_score REAL,
    r2_score REAL,
    n_samples INTEGER,
    n_features INTEGER,
    model_type TEXT,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    description TEXT,
    tags TEXT,
    is_active BOOLEAN DEFAULT 1,
    UNIQUE(model_name, version)
);
```

---

## 🟡 3. Logging & Monitoring

### Overview
Comprehensive logging system with loguru and system health monitoring.

### Features
- **Structured Logging**: JSON-formatted logs with rotation
- **Multiple Log Levels**: DEBUG, INFO, WARNING, ERROR
- **Log Rotation**: Automatic log file rotation (10MB)
- **Error Tracking**: Separate error log file
- **System Monitoring**: CPU, memory, disk usage
- **Health Check**: `/healthcheck/` endpoint
- **Uptime Tracking**: System uptime monitoring

### Log Files
- `logs/backend.log` - General application logs
- `logs/errors.log` - Error-only logs
- Console output - Colored logs for development

### Health Check Response
```json
{
  "status": "healthy",
  "timestamp": "2024-01-15T10:30:00",
  "system": {
    "uptime_formatted": "2h 15m 30s",
    "cpu": {
      "usage_percent": 25.5,
      "count": 8
    },
    "memory": {
      "total": 16777216000,
      "used": 8388608000,
      "percent": 50.0
    },
    "disk": {
      "total": 1000000000000,
      "used": 500000000000,
      "percent": 50.0
    }
  },
  "model": {
    "model_loaded": true,
    "model_type": "RandomForestRegressor",
    "task_type": "regression"
  },
  "dataset": {
    "n_samples": 1000,
    "n_features": 4
  },
  "version": "1.0.0"
}
```

---

## 🟡 4. Async Task Queue

### Overview
Background task processing for long-running operations like model training and benchmarking.

### Features
- **Background Processing**: Non-blocking task execution
- **Task Status Tracking**: Monitor task progress
- **Persistent Storage**: Tasks survive server restarts
- **Task Cancellation**: Cancel running tasks
- **Progress Tracking**: Real-time progress updates
- **Multiple Workers**: Configurable thread pool

### Supported Task Types
- `train_model` - Model training
- `benchmark_models` - Model benchmarking
- `hyperparameter_tuning` - Hyperparameter optimization
- `data_processing` - Data preprocessing

### API Endpoints

#### Start Async Training
```bash
curl -X POST "http://localhost:8000/api/v1/async-train/" \
  -G \
  -d "model_name=RandomForest"
```

#### Start Async Benchmark
```bash
curl -X POST "http://localhost:8000/api/v1/async-benchmark/"
```

#### Check Task Status
```bash
curl -X GET "http://localhost:8000/api/v1/task-status/" \
  -G \
  -d "task_id=uuid-here"
```

#### List All Tasks
```bash
curl -X GET "http://localhost:8000/api/v1/all-tasks/"
```

#### Cancel Task
```bash
curl -X DELETE "http://localhost:8000/api/v1/cancel-task/" \
  -G \
  -d "task_id=uuid-here"
```

### Task Status Response
```json
{
  "id": "uuid-here",
  "type": "train_model",
  "status": "running",
  "params": {"model_name": "RandomForest"},
  "created_at": "2024-01-15T10:30:00",
  "started_at": "2024-01-15T10:30:05",
  "progress": 45,
  "result": null,
  "error": null
}
```

---

## 📡 5. API Endpoints

### Complete API Reference

#### Core Endpoints
- `POST /api/v1/upload-csv/` - Upload CSV dataset
- `POST /api/v1/upload-url/` - Upload dataset from URL
- `POST /api/v1/train-model/` - Train model
- `POST /api/v1/evaluate-model/` - Evaluate model
- `POST /api/v1/explain-model/` - Explain model

#### New Advanced Endpoints

##### Real-Time Inference
- `POST /api/v1/predict/` - Single prediction
- `POST /api/v1/predict-batch/` - Batch prediction
- `GET /api/v1/model-info/` - Model information

##### Model Versioning
- `POST /api/v1/save-model-version/` - Save model version
- `GET /api/v1/list-models/` - List all models
- `POST /api/v1/load-model/` - Load specific model
- `DELETE /api/v1/delete-model/` - Delete model
- `GET /api/v1/model-details/` - Model details

##### Async Tasks
- `POST /api/v1/async-train/` - Start async training
- `POST /api/v1/async-benchmark/` - Start async benchmark
- `GET /api/v1/task-status/` - Check task status
- `GET /api/v1/all-tasks/` - List all tasks
- `DELETE /api/v1/cancel-task/` - Cancel task

##### Monitoring
- `GET /api/v1/healthcheck/` - System health
- `GET /api/v1/status/` - Detailed status

##### Benchmarking
- `POST /api/v1/benchmark-models/` - Run benchmark
- `GET /api/v1/benchmark-summary/` - Benchmark summary
- `GET /api/v1/compare-models/` - Compare models

---

## 🛠️ 6. Installation & Setup

### Prerequisites
- Python 3.8+
- pip

### Installation
```bash
# Clone the repository
git clone <repository-url>
cd Canis/Backend

# Install dependencies
pip install -r requirements.txt

# Create necessary directories
mkdir -p logs models tmp

# Set environment variables (optional)
export GEMINI_API_KEY="your-api-key-here"
```

### New Dependencies Added
```
loguru==0.7.2          # Advanced logging
psutil==5.9.6          # System monitoring
```

### Directory Structure
```
Canis AI Backend/
├── app/
│   ├── api/
│   │   └── routes.py          # All API endpoints
│   │   └── ... (existing modules)
│   └── main.py               # FastAPI app
├── logs/                     # Log files
├── models/                   # Model storage
├── tmp/                      # Temporary data
└── requirements.txt          # Dependencies
```

### Starting the Server
```bash
# From the Backend directory
python -m uvicorn app.main:app --reload --host 127.0.0.1 --port 8000
```

### Environment Variables
```bash
# Optional: Set in .env file or environment
GEMINI_API_KEY=your-gemini-api-key
LOG_LEVEL=INFO
MAX_WORKERS=4
```

---

## 🧪 7. Testing

### Run All Tests
```bash
# Make sure server is running first
python test_advanced_features.py
```

### Test Individual Features

#### Health Check
```bash
curl -X GET "http://localhost:8000/api/v1/healthcheck/"
```

#### Real-time Prediction
```bash
curl -X POST "http://localhost:8000/api/v1/predict/" \
  -H "Content-Type: application/json" \
  -d '{"feature1": 1.5, "feature2": 2.3}'
```

#### Model Versioning
```bash
# Save model
curl -X POST "http://localhost:8000/api/v1/save-model-version/" \
  -G -d "model_name=test_model"

# List models
curl -X GET "http://localhost:8000/api/v1/list-models/"
```

#### Async Tasks
```bash
# Start async training
curl -X POST "http://localhost:8000/api/v1/async-train/" \
  -G -d "model_name=RandomForest"

# Check status (replace with actual task_id)
curl -X GET "http://localhost:8000/api/v1/task-status/" \
  -G -d "task_id=your-task-id"
```

---

## 🎯 Next Steps

### Immediate Actions
1. **Install Dependencies**: `pip install -r requirements.txt`
2. **Start Server**: `python -m uvicorn app.main:app --reload`
3. **Run Tests**: `python test_advanced_features.py`
4. **Upload Dataset**: Use `/upload-csv/` endpoint
5. **Train Model**: Use `/train-model/` or `/async-train/`
6. **Make Predictions**: Use `/predict/` endpoint

### Future Enhancements
- **Authentication**: Add JWT-based authentication
- **Database**: Migrate to PostgreSQL for production
- **Redis**: Add Redis for task queue in production
- **Docker**: Containerize the application
- **Kubernetes**: Add K8s deployment configs
- **Monitoring**: Add Prometheus/Grafana integration
- **API Rate Limiting**: Add rate limiting middleware
- **Model Serving**: Add model serving with TensorFlow Serving

---

## 📞 Support

For issues or questions:
1. Check the logs in `logs/` directory
2. Use the `/healthcheck/` endpoint for system status
3. Review the test script for usage examples
4. Check the API documentation at `http://localhost:8000/docs`

---

**🎉 Congratulations! Your Canis AI Backend now has production-ready advanced features!** 