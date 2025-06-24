import pandas as pd
import numpy as np
import joblib
import os
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.linear_model import LogisticRegression, LinearRegression, Ridge
from sklearn.svm import SVC, SVR
from sklearn.neighbors import KNeighborsClassifier, KNeighborsRegressor
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor
from sklearn.cluster import KMeans, AgglomerativeClustering
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import (
    accuracy_score, f1_score, precision_score, recall_score,
    r2_score, mean_squared_error, silhouette_score, calinski_harabasz_score
)
import time
from .gemini_brain import gemini
from .benchmark_manager import BenchmarkManager
from sklearn.pipeline import Pipeline

# Define tmp directory
TMP_DIR = "tmp"

# Create benchmark manager instance
_benchmark_manager = BenchmarkManager(TMP_DIR)

def benchmark_models():
    """Comprehensive model benchmarking across all task types"""
    try:
        return _benchmark_manager.run_benchmark()
    except Exception as e:
        return {"error": f"Benchmarking error: {str(e)}"}

def get_benchmark_summary():
    """Get a summary of the benchmark results"""
    try:
        return _benchmark_manager.get_summary()
    except Exception as e:
        return {"error": f"Summary error: {str(e)}"}

def compare_with_current_model():
    """Compare benchmark results with the current model in Gemini Brain"""
    try:
        metadata = gemini.get_metadata()
        current_model_name = getattr(gemini.model, '__class__', None)
        current_model_name = current_model_name.__name__ if current_model_name else "None"
        benchmark_results = benchmark_models()
        if "error" in benchmark_results:
            return {"error": benchmark_results["error"]}
        current_performance = None
        if hasattr(gemini, 'evaluation_results') and gemini.evaluation_results:
            current_performance = gemini.evaluation_results
        elif hasattr(gemini, 'training_results') and gemini.training_results:
            current_performance = gemini.training_results
        best_model_name = benchmark_results.get("best_model_name", "Unknown")
        comparison = {
            "current_model": current_model_name,
            "benchmark_task": benchmark_results["task_type"],
            "benchmark_best_model": best_model_name,
            "benchmark_best_score": benchmark_results["best_score"],
            "current_performance": current_performance,
            "recommendation": ""
        }
        if current_performance and benchmark_results["best_score"]:
            current_score = None
            if "accuracy" in current_performance:
                current_score = current_performance["accuracy"]
            elif "r2_score" in current_performance:
                current_score = current_performance["r2_score"]
            if current_score:
                if benchmark_results["best_score"] > current_score:
                    improvement = benchmark_results["best_score"] - current_score
                    comparison["recommendation"] = f"Consider switching to {best_model_name} for {improvement:.3f} improvement"
                else:
                    comparison["recommendation"] = f"Current model performs well. Best benchmark score: {benchmark_results['best_score']:.3f}"
        return comparison
    except Exception as e:
        return {"error": f"Comparison error: {str(e)}"}

def save_best_benchmark_model():
    """Save the best performing model from benchmark results"""
    try:
        results = benchmark_models()
        if "error" in results:
            return {"error": results["error"]}
        success = _benchmark_manager.save_best_model("")
        if success:
            return {
                "status": "success",
                "message": "Best benchmark model saved successfully",
                "model_path": f"{TMP_DIR}/best_benchmark_model.joblib"
            }
        else:
            return {"error": "Failed to save best model"}
    except Exception as e:
        return {"error": f"Save error: {str(e)}"}

# Legacy functions for backward compatibility
def detect_task_type(y):
    """Detect the task type based on target variable characteristics"""
    return _benchmark_manager.detect_task_type(y)

def benchmark_classification(X_train, X_test, y_train, y_test, classification_type):
    """Benchmark classification models"""
    return _benchmark_manager.benchmark_classification(X_train, X_test, y_train, y_test, classification_type)

def benchmark_regression(X_train, X_test, y_train, y_test):
    """Benchmark regression models"""
    return _benchmark_manager.benchmark_regression(X_train, X_test, y_train, y_test)

def benchmark_clustering(X_train, X_test):
    """Benchmark clustering models"""
    return _benchmark_manager.benchmark_clustering(X_train, X_test)