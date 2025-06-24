"""
Modular Benchmark Manager for ML Model Comparison
Handles preprocessing, model training, and evaluation in a clean, reusable way.
"""

import pandas as pd
import numpy as np
import joblib
import time
import os
from typing import Dict, List, Tuple, Any, Optional
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
from sklearn.pipeline import Pipeline

class BenchmarkManager:
    """Manages model benchmarking with preprocessing pipeline integration"""
    
    def __init__(self, tmp_dir: str = "tmp"):
        self.tmp_dir = tmp_dir
        self.results = {}
        self.best_model = None
        self.best_score = 0
        
        # Model configurations
        self.classification_models = {
            "LogisticRegression": LogisticRegression(max_iter=1000, random_state=42),
            "RandomForestClassifier": RandomForestClassifier(n_estimators=100, random_state=42),
            "KNeighborsClassifier": KNeighborsClassifier(n_neighbors=5),
            "DecisionTreeClassifier": DecisionTreeClassifier(random_state=42),
            "GaussianNB": GaussianNB(),
            "SVC": SVC(random_state=42)
        }
        
        self.regression_models = {
            "LinearRegression": LinearRegression(),
            "RandomForestRegressor": RandomForestRegressor(n_estimators=100, random_state=42),
            "Ridge": Ridge(random_state=42),
            "SVR": SVR(),
            "DecisionTreeRegressor": DecisionTreeRegressor(random_state=42),
            "KNeighborsRegressor": KNeighborsRegressor(n_neighbors=5)
        }
        
        self.clustering_models = {
            "KMeans": KMeans(n_clusters=3, random_state=42),
            "AgglomerativeClustering": AgglomerativeClustering(n_clusters=3)
        }
    
    def load_data(self) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """Load raw data for benchmarking"""
        try:
            # Load raw data
            X_train, X_test, y_train, y_test = joblib.load(f"{self.tmp_dir}/benchmark_data.pkl")
            
            # Handle target encoding if needed
            y_train_encoded, y_test_encoded = self._encode_targets(y_train, y_test)
            
            return X_train, X_test, y_train_encoded, y_test_encoded
            
        except Exception as e:
            raise ValueError(f"Failed to load data: {str(e)}")
    
    def _encode_targets(self, y_train: np.ndarray, y_test: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """Encode target variables if they are categorical"""
        try:
            if y_train.dtype == 'object':
                target_encoder = joblib.load(f"{self.tmp_dir}/target_encoder.pkl")
                y_train_encoded = target_encoder.transform(y_train)
                y_test_encoded = target_encoder.transform(y_test)
            else:
                y_train_encoded = y_train.values if hasattr(y_train, 'values') else y_train
                y_test_encoded = y_test.values if hasattr(y_test, 'values') else y_test
            
            return y_train_encoded, y_test_encoded
            
        except Exception:
            # Fallback to original values
            y_train_encoded = y_train.values if hasattr(y_train, 'values') else y_train
            y_test_encoded = y_test.values if hasattr(y_test, 'values') else y_test
            return y_train_encoded, y_test_encoded
    
    def detect_task_type(self, y: np.ndarray) -> str:
        """Detect the ML task type based on target variable"""
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
    
    def compute_metrics(self, y_true: np.ndarray, y_pred: np.ndarray, task_type: str) -> Dict[str, float]:
        """Compute appropriate metrics based on task type"""
        print(f"Debug compute_metrics: task_type = '{task_type}'")
        print(f"Debug compute_metrics: task_type.startswith('classification') = {task_type.startswith('classification')}")
        print(f"Debug compute_metrics: 'classification' in task_type = {'classification' in task_type}")
        print(f"Debug compute_metrics: task_type == 'binary_classification' = {task_type == 'binary_classification'}")
        
        if 'classification' in task_type:
            # For binary classification, dynamically determine positive label
            if task_type == "binary_classification":
                unique_labels = sorted(np.unique(y_true))
                if len(unique_labels) == 2:
                    pos_label = max(unique_labels)  # Automatically choose higher class as positive
                else:
                    pos_label = 1  # fallback
                print(f"Debug compute_metrics: binary classification, pos_label = {pos_label}")
                
                metrics = {
                    "accuracy": accuracy_score(y_true, y_pred),
                    "f1_score": f1_score(y_true, y_pred, average='binary', pos_label=pos_label, zero_division=0),
                    "precision": precision_score(y_true, y_pred, average='binary', pos_label=pos_label, zero_division=0),
                    "recall": recall_score(y_true, y_pred, average='binary', pos_label=pos_label, zero_division=0)
                }
            else:
                # Multiclass classification
                metrics = {
                    "accuracy": accuracy_score(y_true, y_pred),
                    "f1_score": f1_score(y_true, y_pred, average='weighted', zero_division=0),
                    "precision": precision_score(y_true, y_pred, average='weighted', zero_division=0),
                    "recall": recall_score(y_true, y_pred, average='weighted', zero_division=0)
                }
            
            print(f"Debug compute_metrics: returning classification metrics = {metrics}")
            return metrics
        elif task_type == "regression":
            metrics = {
                "r2_score": r2_score(y_true, y_pred),
                "rmse": np.sqrt(mean_squared_error(y_true, y_pred)),
                "mse": mean_squared_error(y_true, y_pred),
                "mae": np.mean(np.abs(y_true - y_pred))
            }
            print(f"Debug compute_metrics: returning regression metrics = {metrics}")
            return metrics
        else:
            print(f"Debug compute_metrics: unknown task_type, returning empty dict")
            return {}
    
    def benchmark_classification(self, X_train: np.ndarray, X_test: np.ndarray, 
                               y_train: np.ndarray, y_test: np.ndarray, 
                               classification_type: str) -> Dict[str, Any]:
        """Benchmark classification models using fitted pipeline"""
        try:
            # Load the fitted preprocessor (ColumnTransformer)
            preprocessor = joblib.load(f"{self.tmp_dir}/preprocessor.pkl")
            
            metrics = {}
            best_score = -1
            best_model = None
            
            for name, model in self.classification_models.items():
                try:
                    start_time = time.time()
                    # Always wrap preprocessor and model in a new Pipeline
                    pipeline = Pipeline([
                        ('preprocessor', preprocessor),
                        ('model', model)
                    ])
                    pipeline.fit(X_train, y_train)
                    y_pred = pipeline.predict(X_test)
                    
                    # Debug: Check data types and shapes
                    print(f"Debug {name}: y_test shape: {y_test.shape}, y_pred shape: {y_pred.shape}")
                    print(f"Debug {name}: y_test dtype: {y_test.dtype}, y_pred dtype: {y_pred.dtype}")
                    print(f"Debug {name}: y_test unique: {np.unique(y_test)}, y_pred unique: {np.unique(y_pred)}")
                    
                    result = self.compute_metrics(y_test, y_pred, f"{classification_type}_classification")
                    print(f"Debug {name}: result from compute_metrics = {result}")
                    training_time = time.time() - start_time
                    metrics[name] = {
                        **{k: round(v, 4) for k, v in result.items()},
                        "training_time": round(training_time, 3)
                    }
                    print(f"Debug {name}: checking if result['accuracy'] exists: {'accuracy' in result}")
                    if result["accuracy"] > best_score:
                        best_score = result["accuracy"]
                        best_model = pipeline
                except Exception as e:
                    print(f"Error in {name}: {str(e)}")
                    import traceback
                    traceback.print_exc()
                    metrics[name] = {"error": str(e)}
            return {
                "task_type": f"{classification_type}_classification",
                "best_model_name": type(best_model.named_steps['model']).__name__ if best_model else None,
                "best_score": round(best_score, 4),
                "best_metric": "accuracy",
                "all_results": metrics,
                "n_samples": len(X_train),
                "n_features": X_train.shape[1],
                "n_classes": len(np.unique(y_train))
            }
        except Exception as e:
            return {"error": f"Classification benchmarking failed: {str(e)}"}
    
    def benchmark_regression(self, X_train: np.ndarray, X_test: np.ndarray, 
                           y_train: np.ndarray, y_test: np.ndarray) -> Dict[str, Any]:
        """Benchmark regression models using fitted pipeline"""
        try:
            preprocessor = joblib.load(f"{self.tmp_dir}/preprocessor.pkl")
            metrics = {}
            best_score = -float('inf')
            best_model = None
            for name, model in self.regression_models.items():
                try:
                    start_time = time.time()
                    pipeline = Pipeline([
                        ('preprocessor', preprocessor),
                        ('model', model)
                    ])
                    pipeline.fit(X_train, y_train)
                    y_pred = pipeline.predict(X_test)
                    result = self.compute_metrics(y_test, y_pred, "regression")
                    training_time = time.time() - start_time
                    metrics[name] = {
                        **{k: round(v, 4) for k, v in result.items()},
                        "training_time": round(training_time, 3)
                    }
                    if result["r2_score"] > best_score:
                        best_score = result["r2_score"]
                        best_model = pipeline
                except Exception as e:
                    metrics[name] = {"error": str(e)}
            return {
                "task_type": "regression",
                "best_model_name": type(best_model.named_steps['model']).__name__ if best_model else None,
                "best_score": round(best_score, 4),
                "best_metric": "r2_score",
                "all_results": metrics,
                "n_samples": len(X_train),
                "n_features": X_train.shape[1]
            }
        except Exception as e:
            return {"error": f"Regression benchmarking failed: {str(e)}"}
    
    def benchmark_clustering(self, X_train: np.ndarray, X_test: np.ndarray) -> Dict[str, Any]:
        """Benchmark clustering models"""
        try:
            preprocessor = joblib.load(f"{self.tmp_dir}/preprocessor.pkl")
            X_combined = np.vstack([X_train, X_test])
            X_combined_processed = preprocessor.transform(X_combined)
            metrics = {}
            best_score = -1
            best_model = None
            for name, model in self.clustering_models.items():
                try:
                    start_time = time.time()
                    labels = model.fit_predict(X_combined_processed)
                    silhouette = silhouette_score(X_combined_processed, labels)
                    calinski_harabasz = calinski_harabasz_score(X_combined_processed, labels)
                    training_time = time.time() - start_time
                    metrics[name] = {
                        "silhouette_score": round(silhouette, 4),
                        "calinski_harabasz_score": round(calinski_harabasz, 4),
                        "n_clusters": len(np.unique(labels)),
                        "training_time": round(training_time, 3)
                    }
                    if silhouette > best_score:
                        best_score = silhouette
                        best_model = model
                except Exception as e:
                    metrics[name] = {"error": str(e)}
            return {
                "task_type": "clustering",
                "best_model_name": type(best_model).__name__ if best_model else None,
                "best_score": round(best_score, 4),
                "best_metric": "silhouette_score",
                "all_results": metrics,
                "n_samples": len(X_combined),
                "n_features": X_combined.shape[1]
            }
        except Exception as e:
            return {"error": f"Clustering benchmarking failed: {str(e)}"}
    
    def run_benchmark(self) -> Dict[str, Any]:
        """Run comprehensive model benchmarking"""
        try:
            # Load raw data
            X_train, X_test, y_train, y_test = self.load_data()
            
            # Detect task type
            task_type = self.detect_task_type(y_train)
            
            # Run appropriate benchmark
            if task_type == "binary_classification":
                return self.benchmark_classification(X_train, X_test, y_train, y_test, "binary")
            elif task_type == "multiclass_classification":
                return self.benchmark_classification(X_train, X_test, y_train, y_test, "multiclass")
            elif task_type == "regression":
                return self.benchmark_regression(X_train, X_test, y_train, y_test)
            elif task_type == "clustering":
                return self.benchmark_clustering(X_train, X_test)
            else:
                return {"error": f"Unknown task type: {task_type}"}
                
        except Exception as e:
            return {"error": f"Benchmarking error: {str(e)}"}
    
    def get_summary(self) -> Dict[str, Any]:
        """Get a summary of benchmark results"""
        try:
            results = self.run_benchmark()
            
            if "error" in results:
                return results
            
            summary = {
                "task_type": results["task_type"],
                "best_model_name": results["best_model_name"],
                "best_score": results["best_score"],
                "best_metric": results["best_metric"],
                "total_models_tested": len(results["all_results"]),
                "successful_models": len([r for r in results["all_results"].values() if "error" not in r]),
                "failed_models": len([r for r in results["all_results"].values() if "error" in r]),
                "data_info": {
                    "n_samples": results.get("n_samples", 0),
                    "n_features": results.get("n_features", 0)
                }
            }
            
            # Add top 3 models
            if results["all_results"]:
                metric = results["best_metric"]
                sorted_models = []
                
                for name, result in results["all_results"].items():
                    if "error" not in result and metric in result:
                        sorted_models.append((name, result[metric]))
                
                sorted_models.sort(key=lambda x: x[1], reverse=True)
                
                summary["top_models"] = [
                    {"name": name, "score": score} 
                    for name, score in sorted_models[:3]
                ]
            
            return summary
            
        except Exception as e:
            return {"error": f"Summary error: {str(e)}"}
    
    def save_best_model(self, model_name: str) -> bool:
        """Save the best performing model"""
        try:
            results = self.run_benchmark()
            if "error" in results:
                return False
            
            best_model = results["best_model"]
            if best_model is None:
                return False
            
            # ✅ Save the best model
            joblib.dump(best_model, f"{self.tmp_dir}/best_benchmark_model.joblib")
            return True
            
        except Exception as e:
            print(f"Failed to save best model: {str(e)}")
            return False 