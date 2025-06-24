import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor, GradientBoostingClassifier, GradientBoostingRegressor
from sklearn.linear_model import LogisticRegression, LinearRegression, Ridge, Lasso
from sklearn.svm import SVC, SVR
from sklearn.neighbors import KNeighborsClassifier, KNeighborsRegressor
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor
from sklearn.cluster import KMeans
from sklearn.naive_bayes import MultinomialNB
import xgboost as xgb

def select_model():
    """Intelligently select the best model based on task and data characteristics"""
    try:
        df = pd.read_csv("tmp/dataset.csv")
        
        # Get task classification
        from .task_classifier import classify_task
        task_info = classify_task()
        task = task_info.get("task", "unknown")
        
        # Get data characteristics
        n_samples, n_features = df.shape
        n_samples = n_samples - 1  # Exclude target column
        
        # Get target column
        from .target_identifier import find_target
        target_info = find_target()
        target_col = target_info.get("suggested_target")
        
        n_classes = None
        if target_col:
            y = df[target_col]
            if task.startswith("classification"):
                n_classes = y.nunique()
        
        # Model selection logic
        if task == "binary_classification":
            if n_samples < 1000:
                model = "LogisticRegression"
                params = {"C": 1.0, "max_iter": 1000}
            elif n_samples < 10000:
                model = "RandomForestClassifier"
                params = {"n_estimators": 100, "max_depth": 10}
            else:
                model = "XGBClassifier"
                params = {"n_estimators": 200, "max_depth": 6, "learning_rate": 0.1}
                
        elif task == "multiclass_classification":
            if n_classes and n_classes <= 5:
                if n_samples < 1000:
                    model = "LogisticRegression"
                    params = {"C": 1.0, "max_iter": 1000, "multi_class": "ovr"}
                else:
                    model = "RandomForestClassifier"
                    params = {"n_estimators": 100, "max_depth": 10}
            else:
                model = "XGBClassifier"
                params = {"n_estimators": 200, "max_depth": 6, "learning_rate": 0.1}
                
        elif task == "regression":
            if n_features <= 10:
                model = "LinearRegression"
                params = {}
            elif n_samples < 1000:
                model = "Ridge"
                params = {"alpha": 1.0}
            else:
                model = "XGBRegressor"
                params = {"n_estimators": 200, "max_depth": 6, "learning_rate": 0.1}
                
        elif task == "clustering":
            model = "KMeans"
            params = {"n_clusters": min(10, n_samples // 10)}
            
        elif task == "nlp":
            model = "MultinomialNB"
            params = {"alpha": 1.0}
            
        else:
            # Fallback
            model = "RandomForestClassifier"
            params = {"n_estimators": 100}
        
        # Additional model recommendations
        alternatives = get_alternative_models(task, n_samples, n_features)
        
        return {
            "selected_model": model,
            "model_params": params,
            "task": task,
            "data_characteristics": {
                "n_samples": n_samples,
                "n_features": n_features,
                "n_classes": n_classes
            },
            "alternatives": alternatives,
            "reasoning": f"Selected {model} for {task} task with {n_samples} samples and {n_features} features"
        }
        
    except Exception as e:
        return {"error": str(e)}

def get_alternative_models(task, n_samples, n_features):
    """Get alternative model suggestions"""
    alternatives = []
    
    if task.startswith("classification"):
        alternatives = [
            {"model": "SVC", "params": {"C": 1.0, "kernel": "rbf"}},
            {"model": "KNeighborsClassifier", "params": {"n_neighbors": 5}},
            {"model": "DecisionTreeClassifier", "params": {"max_depth": 10}}
        ]
    elif task == "regression":
        alternatives = [
            {"model": "SVR", "params": {"C": 1.0, "kernel": "rbf"}},
            {"model": "KNeighborsRegressor", "params": {"n_neighbors": 5}},
            {"model": "DecisionTreeRegressor", "params": {"max_depth": 10}}
        ]
    elif task == "clustering":
        alternatives = [
            {"model": "DBSCAN", "params": {"eps": 0.5, "min_samples": 5}},
            {"model": "AgglomerativeClustering", "params": {"n_clusters": 3}}
        ]
    
    return alternatives
