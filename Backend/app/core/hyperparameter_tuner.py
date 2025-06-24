import pandas as pd
import numpy as np
from sklearn.model_selection import GridSearchCV, RandomizedSearchCV
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.linear_model import LogisticRegression, LinearRegression, Ridge
from sklearn.svm import SVC, SVR
from sklearn.neighbors import KNeighborsClassifier, KNeighborsRegressor
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor
from sklearn.naive_bayes import GaussianNB
import joblib
import os
from .gemini_brain import gemini

# Create tmp directory
TMP_DIR = "tmp"
os.makedirs(TMP_DIR, exist_ok=True)

def tune_hyperparameters(search_type="grid", cv_folds=5):
    """Perform hyperparameter tuning using grid search or random search"""
    try:
        # Get metadata from Gemini Brain
        metadata = gemini.get_metadata()
        
        if not metadata.get("target_column"):
            return {"error": "No target column identified. Please upload and analyze a dataset first."}
        
        # Load preprocessed data
        try:
            X_train, X_test, y_train, y_test = joblib.load(f"{TMP_DIR}/benchmark_data.pkl")
        except FileNotFoundError:
            return {"error": "No training data found. Please train a model first."}
        
        # Get task type and model
        task_type = metadata.get("task_type", "classification")
        model_name = metadata.get("selected_model", "RandomForestClassifier")
        
        # Create base model and parameter grid
        model, param_grid = get_model_and_params(model_name, task_type)
        
        # Create a complete pipeline with preprocessing
        from sklearn.pipeline import Pipeline
        from sklearn.compose import ColumnTransformer
        from sklearn.preprocessing import StandardScaler, OneHotEncoder
        
        # Get feature information
        numeric_features = metadata.get("numeric_features", [])
        categorical_features = metadata.get("categorical_features", [])
        
        # Create preprocessor
        preprocessor = ColumnTransformer(
            transformers=[
                ('num', StandardScaler(), numeric_features) if numeric_features else ('num', 'passthrough', []),
                ('cat', OneHotEncoder(handle_unknown='ignore'), categorical_features) if categorical_features else ('cat', 'passthrough', [])
            ],
            remainder='passthrough'
        )
        
        # Create full pipeline
        pipeline = Pipeline([
            ('preprocessor', preprocessor),
            ('classifier', model)
        ])
        
        # Update parameter grid to include pipeline prefix
        if param_grid:
            param_grid_with_prefix = {}
            for param, values in param_grid.items():
                param_grid_with_prefix[f'classifier__{param}'] = values
        else:
            param_grid_with_prefix = {}
        
        # Determine scoring metric
        if task_type == "regression":
            scoring = 'r2'
        else:
            scoring = 'accuracy'
        
        # Perform hyperparameter tuning
        if search_type.lower() == "grid":
            search = GridSearchCV(
                pipeline, param_grid_with_prefix, cv=cv_folds, scoring=scoring, 
                n_jobs=-1, verbose=1
            )
        else:  # random search
            search = RandomizedSearchCV(
                pipeline, param_grid_with_prefix, n_iter=20, cv=cv_folds, scoring=scoring,
                n_jobs=-1, verbose=1, random_state=42
            )
        
        # Fit the search
        search.fit(X_train, y_train)
        
        # Get best model and results
        best_model = search.best_estimator_
        best_params = search.best_params_
        best_score = search.best_score_
        
        # Evaluate on test set
        test_score = best_model.score(X_test, y_test)
        
        # Save tuned model
        joblib.dump(best_model, f"{TMP_DIR}/tuned_model.joblib")
        
        # Update Gemini Brain
        gemini.model = best_model
        # Extract classifier parameters (remove pipeline prefix)
        classifier_params = {}
        for param, value in best_params.items():
            if param.startswith('classifier__'):
                classifier_params[param.replace('classifier__', '')] = value
        gemini.model_params = classifier_params
        gemini.training_results = {
            "train_score": best_score,
            "test_score": test_score,
            "best_params": classifier_params,
            "search_type": search_type,
            "cv_folds": cv_folds
        }
        
        return {
            "status": "hyperparameter_tuning_completed",
            "model": model_name,
            "search_type": search_type,
            "best_params": classifier_params,
            "best_cv_score": round(best_score, 4),
            "test_score": round(test_score, 4),
            "cv_folds": cv_folds,
            "scoring": scoring,
            "total_combinations": len(search.cv_results_['params']),
            "message": f"Best {model_name} found with {search_type} search"
        }
        
    except Exception as e:
        return {"error": f"Hyperparameter tuning failed: {str(e)}"}

def get_model_and_params(model_name, task_type):
    """Get model instance and parameter grid based on model name and task type"""
    
    if task_type == "regression":
        model_map = {
            "RandomForestRegressor": RandomForestRegressor,
            "LinearRegression": LinearRegression,
            "Ridge": Ridge,
            "SVR": SVR,
            "KNeighborsRegressor": KNeighborsRegressor,
            "DecisionTreeRegressor": DecisionTreeRegressor
        }
        
        param_grids = {
            "RandomForestRegressor": {
                "n_estimators": [50, 100, 200],
                "max_depth": [10, 20, None],
                "min_samples_split": [2, 5, 10],
                "min_samples_leaf": [1, 2, 4]
            },
            "LinearRegression": {
                # LinearRegression has no hyperparameters to tune
            },
            "Ridge": {
                "alpha": [0.1, 1.0, 10.0, 100.0]
            },
            "SVR": {
                "C": [0.1, 1, 10],
                "gamma": ['scale', 'auto'],
                "kernel": ['rbf', 'linear']
            },
            "KNeighborsRegressor": {
                "n_neighbors": [3, 5, 7, 9],
                "weights": ['uniform', 'distance']
            },
            "DecisionTreeRegressor": {
                "max_depth": [3, 5, 10, None],
                "min_samples_split": [2, 5, 10],
                "min_samples_leaf": [1, 2, 4]
            }
        }
    else:  # classification
        model_map = {
            "RandomForestClassifier": RandomForestClassifier,
            "LogisticRegression": LogisticRegression,
            "SVC": SVC,
            "KNeighborsClassifier": KNeighborsClassifier,
            "DecisionTreeClassifier": DecisionTreeClassifier,
            "GaussianNB": GaussianNB
        }
        
        param_grids = {
            "RandomForestClassifier": {
                "n_estimators": [50, 100, 200],
                "max_depth": [10, 20, None],
                "min_samples_split": [2, 5, 10],
                "min_samples_leaf": [1, 2, 4]
            },
            "LogisticRegression": {
                "C": [0.1, 1.0, 10.0],
                "max_iter": [1000]
            },
            "SVC": {
                "C": [0.1, 1, 10],
                "gamma": ['scale', 'auto'],
                "kernel": ['rbf', 'linear']
            },
            "KNeighborsClassifier": {
                "n_neighbors": [3, 5, 7, 9],
                "weights": ['uniform', 'distance']
            },
            "DecisionTreeClassifier": {
                "max_depth": [3, 5, 10, None],
                "min_samples_split": [2, 5, 10],
                "min_samples_leaf": [1, 2, 4]
            },
            "GaussianNB": {
                # GaussianNB has no hyperparameters to tune
            }
        }
    
    # Get model class
    model_class = model_map.get(model_name, RandomForestClassifier)
    model = model_class(random_state=42)
    
    # Get parameter grid
    param_grid = param_grids.get(model_name, {})
    
    return model, param_grid 