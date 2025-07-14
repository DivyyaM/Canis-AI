"""
Trainer for Canis AI AutoML backend.
- Model training, cross-validation, and integration with Gemini Brain.
"""
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, cross_val_score, KFold
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.linear_model import LogisticRegression, LinearRegression, Ridge
from sklearn.svm import SVC, SVR
from sklearn.neighbors import KNeighborsClassifier, KNeighborsRegressor
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor
from sklearn.cluster import KMeans
from sklearn.naive_bayes import MultinomialNB
import xgboost as xgb
import joblib
import os
from .gemini_brain import gemini
import logging

# Create tmp directory for benchmark data
TMP_DIR = "tmp"
os.makedirs(TMP_DIR, exist_ok=True)

def train_model() -> dict:
    """
    Train the selected model using Gemini Brain metadata and user-selected model.

    Returns:
        dict: Training status, scores, and evaluation results.
    """
    try:
        # Get metadata from Gemini Brain
        metadata = gemini.get_metadata()
        
        if not metadata.get("target_column"):
            return {"error": "No target column identified. Please upload and analyze a dataset first."}
        
        # Get model selection
        from .preprocessor import select_model
        model_info = select_model()
        selected_model = model_info.get("selected_model")
        model_params = model_info.get("model_params", {})
        
        if not selected_model:
            return {"error": "No model selected"}
        
        # Load dataset
        df = pd.read_csv("tmp/dataset.csv")
        
        # Use Gemini Brain metadata for feature/target selection
        target_col = metadata["target_column"]
        feature_cols = metadata["feature_columns"]
        
        # Verify columns exist in dataframe
        if target_col not in df.columns:
            return {"error": f"Target column '{target_col}' not found in dataset"}
        
        missing_features = [col for col in feature_cols if col not in df.columns]
        if missing_features:
            return {"error": f"Feature columns not found: {missing_features}"}
        
        X = df[feature_cols]
        y = df[target_col]
        
        # Split data
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42, stratify=y if y.dtype == 'object' else None
        )
        
        # Save preprocessed training data for benchmarking
        joblib.dump((X_train, X_test, y_train, y_test), f"{TMP_DIR}/benchmark_data.pkl")
        
        # Create preprocessing pipeline
        from .preprocessor import create_preprocessing_pipeline
        preprocessor_info = create_preprocessing_pipeline()
        
        if "error" in preprocessor_info:
            return {"error": preprocessor_info["error"]}
        
        # Load preprocessor
        preprocessor = joblib.load("tmp/preprocessor.pkl")
        
        # Transform features
        X_train_transformed = preprocessor.fit_transform(X_train)
        X_test_transformed = preprocessor.transform(X_test)
        
        # Handle target encoding if needed
        if y.dtype == 'object':
            try:
                target_encoder = joblib.load("tmp/target_encoder.pkl")
                y_train_encoded = target_encoder.transform(y_train)
                y_test_encoded = target_encoder.transform(y_test)
            except:
                # Create new encoder if not found
                from sklearn.preprocessing import LabelEncoder
                target_encoder = LabelEncoder()
                y_train_encoded = target_encoder.fit_transform(y_train)
                y_test_encoded = target_encoder.transform(y_test)
                joblib.dump(target_encoder, "tmp/target_encoder.pkl")
        else:
            y_train_encoded = y_train.values
            y_test_encoded = y_test.values
        
        # Create and train model
        model = create_model(selected_model, model_params)
        model.fit(X_train_transformed, y_train_encoded)
        
        # Perform cross-validation
        cv_scores = perform_cross_validation(model, X_train_transformed, y_train_encoded, metadata.get("task_type", "classification"))
        
        # Save model and test data
        joblib.dump(model, "tmp/model.pkl")
        joblib.dump(X_test_transformed, "tmp/X_test.pkl")
        joblib.dump(y_test_encoded, "tmp/y_test.pkl")
        
        # Store in Gemini Brain
        gemini.model = model
        gemini.model_params = model_params
        gemini.training_results = {
            "train_score": model.score(X_train_transformed, y_train_encoded),
            "test_score": model.score(X_test_transformed, y_test_encoded),
            "cv_scores": cv_scores
        }
        
        # Auto-run benchmark after training
        try:
            from .benchmark_manager import BenchmarkManager
            benchmark_manager = BenchmarkManager()
            benchmark_results = benchmark_manager.run_benchmark()
            if "error" not in benchmark_results:
                gemini.benchmark_results = benchmark_results
        except Exception as e:
            logging.warning(f"Benchmark auto-run failed: {str(e)}")
        
        return {
            "status": "model_trained",
            "model": selected_model,
            "target_column": target_col,
            "feature_columns": feature_cols,
            "train_score": round(gemini.training_results["train_score"], 4),
            "test_score": round(gemini.training_results["test_score"], 4),
            "cv_mean": round(cv_scores["mean"], 4),
            "cv_std": round(cv_scores["std"], 4),
            "cv_scores": [round(score, 4) for score in cv_scores["scores"]],
            "train_samples": len(X_train),
            "test_samples": len(X_test),
            "model_params": model_params,
            "benchmark_auto_run": "error" not in benchmark_results if 'benchmark_results' in locals() else False
        }
        
    except Exception as e:
        logging.error(f"Model training failed: {str(e)}")
        return {"error": f"Model training failed: {str(e)}"}

def perform_cross_validation(model, X, y, task_type="classification", cv_folds=5) -> dict:
    """
    Perform k-fold cross-validation for the given model and data.

    Args:
        model: The ML model instance.
        X: Features (array-like).
        y: Target (array-like).
        task_type (str): Type of ML task ('classification', 'regression', etc.).
        cv_folds (int): Number of cross-validation folds.

    Returns:
        dict: Cross-validation scores and statistics.
    """
    try:
        # Determine scoring metric based on task type
        if task_type == "regression":
            scoring = 'r2'
        elif task_type == "binary_classification":
            scoring = 'accuracy'
        elif task_type == "multiclass_classification":
            scoring = 'accuracy'
        else:
            scoring = 'accuracy'  # default
        
        # Perform cross-validation
        cv = KFold(n_splits=cv_folds, shuffle=True, random_state=42)
        scores = cross_val_score(model, X, y, cv=cv, scoring=scoring)
        
        return {
            "scores": scores.tolist(),
            "mean": scores.mean(),
            "std": scores.std(),
            "cv_folds": cv_folds,
            "scoring": scoring
        }
        
    except Exception as e:
        logging.error(f"Cross-validation failed: {str(e)}")
        return {
            "scores": [],
            "mean": 0.0,
            "std": 0.0,
            "cv_folds": cv_folds,
            "scoring": "unknown",
            "error": str(e)
        }

def create_model(model_name: str, params: dict):
    """
    Create a model instance based on the model name and parameters.

    Args:
        model_name (str): Name of the ML model.
        params (dict): Model parameters.

    Returns:
        Model instance.
    """
    model_map = {
        "RandomForestClassifier": RandomForestClassifier,
        "RandomForestRegressor": RandomForestRegressor,
        "LogisticRegression": LogisticRegression,
        "LinearRegression": LinearRegression,
        "Ridge": Ridge,
        "SVC": SVC,
        "SVR": SVR,
        "KNeighborsClassifier": KNeighborsClassifier,
        "KNeighborsRegressor": KNeighborsRegressor,
        "DecisionTreeClassifier": DecisionTreeClassifier,
        "DecisionTreeRegressor": DecisionTreeRegressor,
        "KMeans": KMeans,
        "MultinomialNB": MultinomialNB,
        "XGBClassifier": xgb.XGBClassifier,
        "XGBRegressor": xgb.XGBRegressor
    }
    model_class = model_map.get(model_name, RandomForestClassifier)
    return model_class(**params)