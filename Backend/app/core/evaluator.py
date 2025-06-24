import pandas as pd
import numpy as np
import joblib
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    classification_report, confusion_matrix, roc_auc_score,
    mean_squared_error, mean_absolute_error, r2_score
)
from .gemini_brain import gemini

def evaluate():
    """Evaluate the trained model using appropriate metrics"""
    try:
        # Get metadata from Gemini Brain
        metadata = gemini.get_metadata()
        
        if not metadata.get("target_column"):
            return {"error": "No dataset analyzed yet. Please upload a CSV file first."}
        
        # Check if model exists
        if not gemini.model:
            return {"error": "No model trained yet. Please train a model first."}
        
        # Load test data
        try:
            X_test = joblib.load("tmp/X_test.pkl")
            y_test = joblib.load("tmp/y_test.pkl")
        except FileNotFoundError:
            return {"error": "Test data not found. Please train the model first."}
        
        # Make predictions
        y_pred = gemini.model.predict(X_test)
        
        # Get task type
        task = metadata.get("task_type", "unknown")
        
        # Evaluate based on task type
        if task == "binary_classification":
            return evaluate_binary_classification(y_test, y_pred, metadata)
        elif task == "multiclass_classification":
            return evaluate_multiclass_classification(y_test, y_pred, metadata)
        elif task == "regression":
            return evaluate_regression(y_test, y_pred, metadata)
        elif task == "clustering":
            return evaluate_clustering(X_test, y_pred, metadata)
        else:
            return {"error": f"Unknown task type: {task}"}
            
    except Exception as e:
        return {"error": str(e)}

def evaluate_binary_classification(y_test, y_pred, metadata):
    """Evaluate binary classification model"""
    try:
        # Load test data for ROC AUC calculation
        X_test = joblib.load("tmp/X_test.pkl")
        
        # Dynamically determine positive label
        unique_labels = sorted(np.unique(y_test))
        if len(unique_labels) == 2:
            pos_label = max(unique_labels)  # Automatically choose higher class as positive
        else:
            pos_label = 1  # fallback
        
        # Basic metrics with correct pos_label
        accuracy = accuracy_score(y_test, y_pred)
        precision = precision_score(y_test, y_pred, pos_label=pos_label, average='binary', zero_division=0)
        recall = recall_score(y_test, y_pred, pos_label=pos_label, average='binary', zero_division=0)
        f1 = f1_score(y_test, y_pred, pos_label=pos_label, average='binary', zero_division=0)
        
        # Confusion matrix
        cm = confusion_matrix(y_test, y_pred)
        
        # Classification report
        class_report = classification_report(y_test, y_pred, output_dict=True, zero_division=0)
        
        # ROC AUC (if probabilities available)
        try:
            if hasattr(gemini.model, 'predict_proba'):
                y_proba = gemini.model.predict_proba(X_test)[:, 1]
                roc_auc = roc_auc_score(y_test, y_proba)
            else:
                roc_auc = None
        except:
            roc_auc = None
        
        # Store results in Gemini Brain
        gemini.evaluation_results = {
            "accuracy": float(accuracy),
            "precision": float(precision),
            "recall": float(recall),
            "f1_score": float(f1),
            "roc_auc": float(roc_auc) if roc_auc else None,
            "pos_label": int(pos_label)
        }
        
        return {
            "task": "binary_classification",
            "target_column": metadata.get("target_column"),
            "pos_label": int(pos_label),
            "unique_labels": np.array(unique_labels).tolist(),
            "metrics": {
                "accuracy": round(accuracy, 4),
                "precision": round(precision, 4),
                "recall": round(recall, 4),
                "f1_score": round(f1, 4),
                "roc_auc": round(roc_auc, 4) if roc_auc else None
            },
            "confusion_matrix": {
                "matrix": np.array(cm).tolist(),
                "labels": ["Negative", "Positive"]
            },
            "classification_report": class_report,
            "interpretation": {
                "accuracy": f"The model correctly predicts {accuracy*100:.1f}% of all samples",
                "precision": f"Of samples predicted as positive, {precision*100:.1f}% are actually positive",
                "recall": f"The model correctly identifies {recall*100:.1f}% of all positive samples",
                "f1_score": f"F1 score of {f1:.4f} provides balanced precision-recall measure",
                "pos_label_info": f"Using label {pos_label} as positive class",
                "overall": "Good performance" if accuracy > 0.8 else "Needs improvement"
            }
        }
        
    except Exception as e:
        return {"error": f"Error in binary classification evaluation: {str(e)}"}

def evaluate_multiclass_classification(y_test, y_pred, metadata):
    """Evaluate multiclass classification model"""
    try:
        # Basic metrics
        accuracy = accuracy_score(y_test, y_pred)
        precision = precision_score(y_test, y_pred, average='weighted', zero_division=0)
        recall = recall_score(y_test, y_pred, average='weighted', zero_division=0)
        f1 = f1_score(y_test, y_pred, average='weighted', zero_division=0)
        
        # Confusion matrix
        cm = confusion_matrix(y_test, y_pred)
        
        # Classification report
        class_report = classification_report(y_test, y_pred, output_dict=True, zero_division=0)
        
        # Store results in Gemini Brain
        gemini.evaluation_results = {
            "accuracy": float(accuracy),
            "precision": float(precision),
            "recall": float(recall),
            "f1_score": float(f1)
        }
        
        return {
            "task": "multiclass_classification",
            "target_column": metadata.get("target_column"),
            "n_classes": metadata.get("n_classes"),
            "metrics": {
                "accuracy": round(accuracy, 4),
                "precision": round(precision, 4),
                "recall": round(recall, 4),
                "f1_score": round(f1, 4)
            },
            "confusion_matrix": {
                "matrix": np.array(cm).tolist(),
                "shape": cm.shape
            },
            "classification_report": class_report,
            "interpretation": {
                "accuracy": f"The model correctly predicts {accuracy*100:.1f}% of all samples",
                "precision": f"Weighted precision of {precision*100:.1f}% across all classes",
                "recall": f"Weighted recall of {recall*100:.1f}% across all classes",
                "f1_score": f"Weighted F1 score of {f1:.4f} across all classes",
                "overall": "Good performance" if accuracy > 0.8 else "Needs improvement"
            }
        }
        
    except Exception as e:
        return {"error": f"Error in multiclass classification evaluation: {str(e)}"}

def evaluate_regression(y_test, y_pred, metadata):
    """Evaluate regression model"""
    try:
        # Calculate metrics
        mse = mean_squared_error(y_test, y_pred)
        rmse = np.sqrt(mse)
        mae = mean_absolute_error(y_test, y_pred)
        r2 = r2_score(y_test, y_pred)
        
        # Store results in Gemini Brain
        gemini.evaluation_results = {
            "mse": float(mse),
            "rmse": float(rmse),
            "mae": float(mae),
            "r2_score": float(r2)
        }
        
        return {
            "task": "regression",
            "target_column": metadata.get("target_column"),
            "metrics": {
                "mse": round(mse, 4),
                "rmse": round(rmse, 4),
                "mae": round(mae, 4),
                "r2_score": round(r2, 4)
            },
            "interpretation": {
                "r2_score": f"The model explains {r2*100:.1f}% of the variance in the target",
                "rmse": f"Average prediction error is {rmse:.4f} units",
                "mae": f"Mean absolute error is {mae:.4f} units",
                "overall": "Good fit" if r2 > 0.7 else "Needs improvement"
            }
        }
        
    except Exception as e:
        return {"error": f"Error in regression evaluation: {str(e)}"}

def evaluate_clustering(X_test, y_pred, metadata):
    """Evaluate clustering model"""
    try:
        from sklearn.metrics import silhouette_score, calinski_harabasz_score
        
        # Calculate clustering metrics
        silhouette = silhouette_score(X_test, y_pred)
        calinski_harabasz = calinski_harabasz_score(X_test, y_pred)
        
        # Count clusters
        n_clusters = len(np.unique(y_pred))
        
        # Store results in Gemini Brain
        gemini.evaluation_results = {
            "silhouette_score": float(silhouette),
            "calinski_harabasz_score": float(calinski_harabasz),
            "n_clusters": int(n_clusters)
        }
        
        return {
            "task": "clustering",
            "n_clusters": n_clusters,
            "metrics": {
                "silhouette_score": round(silhouette, 4),
                "calinski_harabasz_score": round(calinski_harabasz, 4)
            },
            "interpretation": {
                "silhouette_score": f"Silhouette score of {silhouette:.4f} indicates cluster quality",
                "calinski_harabasz": f"Calinski-Harabasz score of {calinski_harabasz:.4f} indicates cluster separation",
                "n_clusters": f"Model identified {n_clusters} clusters",
                "overall": "Good clustering" if silhouette > 0.5 else "Clusters may need refinement"
            }
        }
        
    except Exception as e:
        return {"error": f"Error in clustering evaluation: {str(e)}"}
