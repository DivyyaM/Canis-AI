# 🚀 Canis AI Backend - Advanced Features

This document describes all the advanced features implemented in the Canis AI Backend.

## 📋 Table of Contents

1. [Upload Dataset by URL/Other Types](#1-upload-dataset-by-urlother-types)
2. [Model Download Endpoint](#2-model-download-endpoint)
3. [Cross-Validation](#3-cross-validation)
4. [Hyperparameter Tuning](#4-hyperparameter-tuning)
5. [Advanced Explainability](#5-advanced-explainability)
6. [Chat Memory](#6-chat-memory)

---

## 1. Upload Dataset by URL/Other Types

### Overview
Enhanced file upload functionality that supports multiple file formats and URL-based dataset uploads.

### Supported Formats
- **CSV** (`.csv`)
- **Excel** (`.xlsx`, `.xls`)
- **TSV** (`.tsv`, `.txt`)
- **JSON** (`.json`)

### Endpoints

#### Upload File
```http
POST /upload-csv/
Content-Type: multipart/form-data

file: [file upload]
```

#### Upload from URL
```http
POST /upload-url/?url=https://example.com/dataset.csv
```

### Example Usage
```python
import requests

# Upload from URL
response = requests.post("http://127.0.0.1:8000/upload-url/", 
                        params={"url": "https://raw.githubusercontent.com/datasciencedojo/datasets/master/breast-cancer-wisconsin.csv"})

# Upload file
with open("dataset.csv", "rb") as f:
    response = requests.post("http://127.0.0.1:8000/upload-csv/", 
                           files={"file": f})
```

### Response
```json
{
  "status": "success",
  "rows": 683,
  "cols": ["Sample code number", "Clump Thickness", ...],
  "target_column": "Class",
  "task_type": "binary_classification",
  "message": "Dataset uploaded and analyzed successfully"
}
```

---

## 2. Model Download Endpoint

### Overview
Download trained models for deployment or sharing.

### Endpoints

#### Download Current Model
```http
GET /download-model/
```

#### Download Best Benchmark Model
```http
GET /download-best-benchmark-model/
```

### Example Usage
```python
import requests

# Download current model
response = requests.get("http://127.0.0.1:8000/download-model/")
with open("trained_model.joblib", "wb") as f:
    f.write(response.content)

# Download best benchmark model
response = requests.get("http://127.0.0.1:8000/download-best-benchmark-model/")
with open("best_benchmark_model.joblib", "wb") as f:
    f.write(response.content)
```

---

## 3. Cross-Validation

### Overview
Enhanced model training with k-fold cross-validation for more robust performance evaluation.

### Features
- **Automatic CV**: Integrated into training process
- **Configurable Folds**: Default 5-fold CV
- **Task-Aware Scoring**: Appropriate metrics for classification/regression
- **Comprehensive Results**: Mean, standard deviation, and individual fold scores

### Training Response (Enhanced)
```json
{
  "status": "model_trained",
  "model": "LogisticRegression",
  "target_column": "Class",
  "feature_columns": [...],
  "train_score": 0.9707,
  "test_score": 0.9562,
  "cv_mean": 0.9534,
  "cv_std": 0.0123,
  "cv_scores": [0.9452, 0.9589, 0.9523, 0.9612, 0.9512],
  "train_samples": 546,
  "test_samples": 137,
  "model_params": {...},
  "benchmark_auto_run": true
}
```

---

## 4. Hyperparameter Tuning

### Overview
Automated hyperparameter optimization using grid search or random search.

### Endpoints

#### Hyperparameter Tuning
```http
POST /tune-hyperparameters/?search_type=grid&cv_folds=5
```

### Parameters
- `search_type`: `"grid"` or `"random"`
- `cv_folds`: Number of cross-validation folds (default: 5)

### Supported Models & Parameters

#### Classification Models
- **RandomForestClassifier**: `n_estimators`, `max_depth`, `min_samples_split`, `min_samples_leaf`
- **LogisticRegression**: `C`, `max_iter`
- **SVC**: `C`, `gamma`, `kernel`
- **KNeighborsClassifier**: `n_neighbors`, `weights`
- **DecisionTreeClassifier**: `max_depth`, `min_samples_split`, `min_samples_leaf`
- **GaussianNB**: No hyperparameters

#### Regression Models
- **RandomForestRegressor**: `n_estimators`, `max_depth`, `min_samples_split`, `min_samples_leaf`
- **LinearRegression**: No hyperparameters
- **Ridge**: `alpha`
- **SVR**: `C`, `gamma`, `kernel`
- **KNeighborsRegressor**: `n_neighbors`, `weights`
- **DecisionTreeRegressor**: `max_depth`, `min_samples_split`, `min_samples_leaf`

### Example Usage
```python
import requests

# Grid search
response = requests.post("http://127.0.0.1:8000/tune-hyperparameters/",
                        params={"search_type": "grid", "cv_folds": 5})

# Random search
response = requests.post("http://127.0.0.1:8000/tune-hyperparameters/",
                        params={"search_type": "random", "cv_folds": 3})
```

### Response
```json
{
  "status": "hyperparameter_tuning_completed",
  "model": "RandomForestClassifier",
  "search_type": "grid",
  "best_params": {
    "n_estimators": 200,
    "max_depth": 20,
    "min_samples_split": 5
  },
  "best_cv_score": 0.9634,
  "test_score": 0.9589,
  "cv_folds": 5,
  "scoring": "accuracy",
  "total_combinations": 48,
  "message": "Best RandomForestClassifier found with grid search"
}
```

---

## 5. Advanced Explainability

### Overview
Advanced model interpretability using SHAP and LIME explanations.

### Endpoints

#### Feature Importance Analysis
```http
GET /feature-importance/
```

#### SHAP Explanations
```http
GET /shap-explanations/?sample_index=0&num_samples=100
```

#### LIME Explanations
```http
GET /lime-explanations/?sample_index=0&num_features=10
```

### Parameters
- `sample_index`: Index of sample to explain (optional for SHAP)
- `num_samples`: Number of samples for global SHAP analysis
- `num_features`: Number of features to include in LIME explanation

### Example Usage
```python
import requests

# Feature importance
response = requests.get("http://127.0.0.1:8000/feature-importance/")

# SHAP explanations
response = requests.get("http://127.0.0.1:8000/shap-explanations/",
                       params={"sample_index": 0, "num_samples": 50})

# LIME explanations
response = requests.get("http://127.0.0.1:8000/lime-explanations/",
                       params={"sample_index": 0, "num_features": 5})
```

### Response Examples

#### Feature Importance
```json
{
  "status": "success",
  "analysis_type": "feature_importance",
  "analysis": {
    "feature_importance": {
      "Clump Thickness": 0.234,
      "Uniformity of Cell Size": 0.189,
      ...
    },
    "top_features": [
      ["Clump Thickness", 0.234],
      ["Uniformity of Cell Size", 0.189],
      ...
    ],
    "statistics": {
      "mean_importance": 0.083,
      "std_importance": 0.067,
      "max_importance": 0.234,
      "min_importance": 0.012
    },
    "num_features": 10
  }
}
```

#### SHAP Explanations
```json
{
  "status": "success",
  "explanation_type": "shap",
  "explanation": {
    "sample_index": 0,
    "actual_value": 2,
    "predicted_value": 2,
    "feature_importance": {
      "Clump Thickness": 0.123,
      "Uniformity of Cell Size": -0.045,
      ...
    },
    "base_value": 0.456
  }
}
```

### Installation Requirements
For SHAP and LIME functionality:
```bash
pip install shap lime
```

---

## 6. Chat Memory

### Overview
Enhanced chat functionality with conversation memory and context awareness.

### Features
- **Conversation History**: Remembers previous exchanges
- **Context Awareness**: Maintains session context
- **Memory Management**: View and clear conversation history
- **Persistent Context**: Remembers dataset and model information

### Endpoints

#### Chat with Memory
```http
POST /chat/?query=What is the target column?
```

#### Get Chat History
```http
GET /chat-history/?num_messages=10
```

#### Clear Chat Memory
```http
POST /clear-chat-memory/
```

### Example Usage
```python
import requests

# Send chat messages
response = requests.post("http://127.0.0.1:8000/chat/",
                        params={"query": "What is the target column?"})

# Get chat history
response = requests.get("http://127.0.0.1:8000/chat-history/",
                       params={"num_messages": 5})

# Clear memory
response = requests.post("http://127.0.0.1:8000/clear-chat-memory/")
```

### Response Examples

#### Chat Response
```json
{
  "response": "The target column is 'Class' with 2 unique values...",
  "context": {
    "task_type": "binary_classification",
    "target_column": "Class",
    "n_samples": 683,
    "n_features": 10,
    "model": "LogisticRegression",
    "memory_size": 5
  }
}
```

#### Chat History
```json
{
  "status": "success",
  "history": [
    {
      "user_query": "What is the target column?",
      "ai_response": "The target column is 'Class'...",
      "context": {...},
      "timestamp": "2024-01-15T10:30:00"
    }
  ],
  "total_messages": 5
}
```

---

## 🧪 Testing All Features

### Comprehensive Test Script
Run the complete test suite to verify all features:

```bash
python test_all_features.py
```

This script tests:
- ✅ URL upload functionality
- ✅ Complete ML workflow
- ✅ Hyperparameter tuning
- ✅ Model download
- ✅ Advanced explainability
- ✅ Chat memory

### Individual Testing
You can also test features individually using the provided endpoints and examples above.

---

## 📊 Feature Summary

| Feature | Status | Endpoints | Description |
|---------|--------|-----------|-------------|
| URL Upload | ✅ | `/upload-url/` | Upload datasets from URLs |
| Multi-Format | ✅ | `/upload-csv/` | Support for CSV, XLSX, TSV, JSON |
| Model Download | ✅ | `/download-model/`, `/download-best-benchmark-model/` | Download trained models |
| Cross-Validation | ✅ | Enhanced `/train-model/` | K-fold CV with detailed metrics |
| Hyperparameter Tuning | ✅ | `/tune-hyperparameters/` | Grid/Random search optimization |
| SHAP Explanations | ✅ | `/shap-explanations/` | Model interpretability |
| LIME Explanations | ✅ | `/lime-explanations/` | Local explanations |
| Feature Importance | ✅ | `/feature-importance/` | Global feature analysis |
| Chat Memory | ✅ | `/chat/`, `/chat-history/`, `/clear-chat-memory/` | Conversation persistence |

---

## 🚀 Getting Started

1. **Start the server**:
   ```bash
   uvicorn app.main:app --reload
   ```

2. **Upload a dataset**:
   ```bash
   curl -X POST "http://127.0.0.1:8000/upload-url/?url=https://raw.githubusercontent.com/datasciencedojo/datasets/master/breast-cancer-wisconsin.csv"
   ```

3. **Run the complete workflow**:
   ```bash
   python test_all_features.py
   ```

4. **Explore the API**:
   Visit `http://127.0.0.1:8000/docs` for interactive API documentation.

---

## 🔧 Dependencies

### Required
- `fastapi`
- `pandas`
- `numpy`
- `scikit-learn`
- `joblib`
- `requests`

### Optional (for advanced features)
- `shap` (for SHAP explanations)
- `lime` (for LIME explanations)
- `openpyxl` (for Excel file support)

### Installation
```bash
pip install fastapi pandas numpy scikit-learn joblib requests
pip install shap lime openpyxl  # Optional
```

---

## 📝 Notes

- **SHAP/LIME**: These features require additional packages. If not installed, the endpoints will return helpful error messages.
- **Memory**: Chat memory is currently in-memory only. For production, consider persistent storage.
- **File Formats**: Excel support requires `openpyxl` package.
- **Performance**: Hyperparameter tuning can be computationally intensive. Consider using fewer CV folds for large datasets.

---

## 🎯 Next Steps

Potential future enhancements:
- Persistent chat memory storage
- Advanced model deployment features
- Real-time model monitoring
- Automated feature engineering
- Model versioning and management
- Integration with cloud platforms 