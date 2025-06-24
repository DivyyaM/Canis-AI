# 🚀 Model Benchmark Module

## Overview

The `model_benchmark.py` module provides comprehensive benchmarking capabilities for your Canis AI Backend. It automatically detects the task type and tests multiple machine learning models to find the best performer for your dataset.

## 🎯 Features

### ✅ **Automatic Task Detection**
- **Binary Classification**: 2 unique classes
- **Multiclass Classification**: >2 classes  
- **Regression**: Continuous numeric values
- **Clustering**: Unsupervised learning (ignores target)

### ✅ **Comprehensive Model Testing**

#### Classification Models
- `LogisticRegression` - Fast linear classifier
- `RandomForestClassifier` - Robust ensemble method
- `KNeighborsClassifier` - Distance-based classifier
- `DecisionTreeClassifier` - Interpretable tree model
- `GaussianNB` - Probabilistic classifier
- `SVC` - Support Vector Classifier

#### Regression Models
- `LinearRegression` - Simple linear model
- `RandomForestRegressor` - Ensemble regressor
- `Ridge` - Regularized linear regression
- `SVR` - Support Vector Regression
- `DecisionTreeRegressor` - Tree-based regression
- `KNeighborsRegressor` - Distance-based regression

#### Clustering Models
- `KMeans` - Centroid-based clustering
- `AgglomerativeClustering` - Hierarchical clustering

### ✅ **Smart Evaluation Metrics**

#### Classification
- **Accuracy**: Overall correct predictions
- **F1 Score**: Balanced precision-recall
- **Precision**: True positives / (True + False positives)
- **Recall**: True positives / (True + False negatives)

#### Regression
- **R² Score**: Explained variance (0-1, higher is better)
- **RMSE**: Root Mean Square Error (lower is better)
- **MSE**: Mean Square Error
- **MAE**: Mean Absolute Error

#### Clustering
- **Silhouette Score**: Cluster quality (-1 to 1, higher is better)
- **Calinski-Harabasz Score**: Cluster separation (higher is better)

## 🛠️ API Endpoints

### 1. **POST /benchmark-models/**
Run comprehensive model benchmarking

**Response:**
```json
{
  "task_type": "binary_classification",
  "best_model": "RandomForestClassifier",
  "best_score": 0.92,
  "best_metric": "accuracy",
  "all_results": {
    "LogisticRegression": {
      "accuracy": 0.85,
      "f1_score": 0.84,
      "precision": 0.86,
      "recall": 0.82,
      "training_time": 0.123
    },
    "RandomForestClassifier": {
      "accuracy": 0.92,
      "f1_score": 0.91,
      "precision": 0.93,
      "recall": 0.89,
      "training_time": 0.456
    }
  },
  "n_samples": 1000,
  "n_features": 10,
  "n_classes": 2
}
```

### 2. **GET /benchmark-summary/**
Get a summary of benchmark results

**Response:**
```json
{
  "task_type": "binary_classification",
  "best_model": "RandomForestClassifier",
  "best_score": 0.92,
  "best_metric": "accuracy",
  "total_models_tested": 6,
  "successful_models": 6,
  "failed_models": 0,
  "data_info": {
    "n_samples": 1000,
    "n_features": 10
  },
  "top_models": [
    {"name": "RandomForestClassifier", "score": 0.92},
    {"name": "SVC", "score": 0.89},
    {"name": "LogisticRegression", "score": 0.85}
  ]
}
```

### 3. **GET /compare-models/**
Compare current model with benchmark results

**Response:**
```json
{
  "current_model": "LogisticRegression",
  "benchmark_task": "binary_classification",
  "benchmark_best_model": "RandomForestClassifier",
  "benchmark_best_score": 0.92,
  "current_performance": {
    "accuracy": 0.85
  },
  "recommendation": "Consider switching to RandomForestClassifier for 0.07 improvement"
}
```

## 🔧 Usage Examples

### Python Usage
```python
from app.core.model_benchmark import benchmark_models, get_benchmark_summary

# Run full benchmark
results = benchmark_models()
print(f"Best model: {results['best_model']}")
print(f"Best score: {results['best_score']}")

# Get summary
summary = get_benchmark_summary()
print(f"Top 3 models: {summary['top_models']}")
```

### cURL Examples
```bash
# Run benchmark
curl -X POST "http://localhost:8000/benchmark-models/"

# Get summary
curl -X GET "http://localhost:8000/benchmark-summary/"

# Compare with current model
curl -X GET "http://localhost:8000/compare-models/"
```

## 🎯 Workflow Integration

### Prerequisites
1. **Upload CSV**: `POST /upload-csv/`
2. **Train Model**: `POST /train-model/` (optional, for comparison)

### Recommended Workflow
1. **Upload your dataset**
2. **Run benchmark**: `POST /benchmark-models/`
3. **Review results**: `GET /benchmark-summary/`
4. **Compare with current**: `GET /compare-models/`
5. **Train best model** (if different from current)

## 🔍 Task Detection Logic

### Classification Detection
```python
if y.dtype in ['int64', 'float64']:
    unique_ratio = len(np.unique(y)) / len(y)
    if unique_ratio < 0.1:  # Less than 10% unique values
        n_classes = len(np.unique(y))
        if n_classes == 2:
            return "binary_classification"
        else:
            return "multiclass_classification"
    else:
        return "regression"
else:
    # String/object type - must be classification
    n_classes = len(np.unique(y))
    if n_classes == 2:
        return "binary_classification"
    else:
        return "multiclass_classification"
```

## 🚀 Performance Features

### ✅ **Parallel Processing Ready**
- Each model is trained independently
- Easy to add multiprocessing for faster benchmarks

### ✅ **Error Handling**
- Graceful handling of model failures
- Detailed error messages for debugging
- Continues testing even if some models fail

### ✅ **Timing Information**
- Training time for each model
- Helps identify fast vs. slow models

### ✅ **Memory Efficient**
- Models are trained sequentially
- Memory is freed after each model

## 🔧 Customization

### Adding New Models
```python
# In benchmark_classification function
models = {
    "YourNewModel": YourNewModelClass(param1=value1),
    # ... existing models
}
```

### Custom Metrics
```python
# Add custom evaluation logic
def custom_metric(y_true, y_pred):
    # Your custom metric
    return score

# Use in results
results[name]["custom_metric"] = custom_metric(y_test, y_pred)
```

## 🎯 Best Practices

### ✅ **For Small Datasets (< 1000 samples)**
- Focus on simple models (LogisticRegression, DecisionTree)
- Avoid complex models that may overfit

### ✅ **For Large Datasets (> 10000 samples)**
- Try ensemble methods (RandomForest, XGBoost)
- Consider deep learning models

### ✅ **For High-Dimensional Data**
- Use regularization (Ridge, Lasso)
- Consider dimensionality reduction

### ✅ **For Imbalanced Classes**
- Focus on F1 score rather than accuracy
- Consider SMOTE or other balancing techniques

## 🐛 Troubleshooting

### Common Issues

#### "Training data not found"
- Ensure you've run `POST /train-model/` first
- Check that `tmp/X_train.pkl` and `tmp/y_train.pkl` exist

#### "Model failed to train"
- Check data quality (no infinite values, proper dtypes)
- Ensure sufficient samples for the model type
- Verify feature scaling for sensitive models

#### "Memory error"
- Reduce number of models tested
- Use smaller datasets for initial testing
- Consider using lighter models

## 🎉 Success Metrics

Your benchmark is successful when:
- ✅ All models train without errors
- ✅ Best model achieves >80% accuracy (classification) or >0.7 R² (regression)
- ✅ Training times are reasonable (<30 seconds per model)
- ✅ Clear performance differences between models

---

**🎯 Ready to find the best model for your data? Start benchmarking!** 