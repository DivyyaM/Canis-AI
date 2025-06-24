import pandas as pd
import numpy as np
import joblib
from sklearn.inspection import permutation_importance
from .gemini_brain import gemini


def explain():
    """Provide comprehensive explanations for model decisions and performance"""
    try:
        # Get metadata from Gemini Brain
        metadata = gemini.get_metadata()

        if not metadata.get("target_column"):
            return {"error": "No dataset analyzed yet. Please upload a CSV file first."}

        if not gemini.model:
            return {"error": "No model trained yet. Please train a model first."}

        # Load dataset
        try:
            df = pd.read_csv("tmp/dataset.csv")
        except Exception as e:
            return {"error": f"Could not load dataset: {str(e)}"}

        explanations = {
            "target_detection": explain_target_detection(metadata),
            "task_classification": explain_task_classification(metadata),
            "model_selection": explain_model_selection(gemini.model_params, metadata),
            "preprocessing": explain_preprocessing(metadata),
            "model_performance": explain_model_performance(gemini.training_results, metadata),
        }

        # Add feature importance
        try:
            feature_importance = explain_feature_importance(gemini.model, df, metadata)
            explanations["feature_importance"] = feature_importance
        except Exception as e:
            explanations["feature_importance"] = {"error": f"Could not calculate feature importance: {str(e)}"}

        return explanations

    except Exception as e:
        return {"error": str(e)}


def explain_target_detection(metadata):
    target_col = metadata.get("target_column")
    confidence_score = metadata.get("confidence_score", 0.0)

    explanations = {
        "target_column": target_col,
        "detection_method": "gemini_brain_analysis",
        "confidence_score": confidence_score,
        "explanation": f"The target column '{target_col}' was identified with {confidence_score:.1f}% confidence using intelligent analysis."
    }

    if confidence_score >= 8:
        explanations["reasoning"] = "This column has strong characteristics of a target variable (binary or categorical with few classes)."
    elif confidence_score >= 4:
        explanations["reasoning"] = "This column has moderate characteristics of a target variable."
    else:
        explanations["reasoning"] = "This column was selected as a fallback option."

    return explanations


def explain_task_classification(metadata):
    task = metadata.get("task_type")
    reason = metadata.get("task_reason", "unknown")

    explanations = {
        "task": task,
        "reason": reason,
        "explanation": f"The dataset was classified as a {task} task because {reason}."
    }

    task_details = {
        "binary_classification": "This is a binary classification problem where the target has exactly two classes.",
        "multiclass_classification": "This is a multiclass classification problem where the target has more than two classes.",
        "regression": "This is a regression problem where the target is a continuous numerical value.",
        "clustering": "This is a clustering problem where no target variable was identified.",
        "nlp": "This is a natural language processing task with text data."
    }

    if task in task_details:
        explanations["details"] = task_details[task]

    return explanations


def explain_model_selection(model_params, metadata):
    model_name = (
        list(gemini.model.__class__.__bases__)[0].__name__
        if gemini.model else "Unknown"
    )

    explanations = {
        "selected_model": model_name,
        "model_parameters": model_params,
        "task": metadata.get("task_type"),
        "data_characteristics": {
            "n_samples": metadata.get("n_samples"),
            "n_features": metadata.get("n_features"),
            "n_classes": metadata.get("n_classes")
        }
    }

    model_benefits_map = {
        "RandomForest": [
            "Handles both numerical and categorical features well",
            "Robust to outliers and non-linear relationships",
            "Provides feature importance rankings",
            "Good for medium to large datasets"
        ],
        "Logistic": [
            "Fast training and prediction",
            "Provides interpretable coefficients",
            "Good baseline for classification tasks",
            "Works well with small datasets"
        ],
        "XGB": [
            "Excellent performance on structured data",
            "Handles missing values automatically",
            "Provides feature importance",
            "Good for large datasets"
        ],
        "Linear": [
            "Simple and interpretable",
            "Fast training and prediction",
            "Good baseline for regression tasks",
            "Works well with linear relationships"
        ]
    }

    for keyword, benefits in model_benefits_map.items():
        if keyword in model_name:
            explanations["model_benefits"] = benefits
            break

    return explanations


def explain_preprocessing(metadata):
    numeric_features = metadata.get("numeric_features", [])
    categorical_features = metadata.get("categorical_features", [])
    missing_values = metadata.get("missing_values", {})

    explanations = {
        "numeric_features": numeric_features,
        "categorical_features": categorical_features,
        "missing_values": missing_values,
        "explanations": []
    }

    if any(missing_values.values()):
        explanations["explanations"].append({
            "step": "Missing Value Imputation",
            "method": "Median for numeric, Most Frequent for categorical",
            "reason": "Missing values were detected and will be handled appropriately."
        })

    if numeric_features:
        explanations["explanations"].append({
            "step": "Numeric Scaling",
            "method": "StandardScaler",
            "reason": "Numeric features were standardized to have zero mean and unit variance."
        })

    if categorical_features:
        explanations["explanations"].append({
            "step": "Categorical Encoding",
            "method": "OneHotEncoder",
            "reason": "Categorical features were one-hot encoded to convert them into numerical format."
        })

    return explanations


def explain_feature_importance(model, df, metadata):
    if hasattr(model, 'feature_importances_'):
        feature_names = metadata.get("feature_columns", [])
        feature_importances = model.feature_importances_

        importance_data = [
            {"feature": name, "importance": float(imp), "rank": i + 1}
            for i, (name, imp) in enumerate(zip(feature_names, feature_importances))
        ]
        importance_data.sort(key=lambda x: x["importance"], reverse=True)

        return {
            "method": "Tree-based feature importance",
            "explanation": "Feature importance was calculated using the model's built-in feature importance method.",
            "top_features": importance_data[:5],
            "total_features": len(feature_names)
        }

    elif hasattr(model, 'coef_'):
        feature_names = metadata.get("feature_columns", [])
        coefficients = model.coef_.flatten()

        importance_data = [
            {
                "feature": name,
                "coefficient": float(coef),
                "abs_coefficient": float(abs(coef)),
                "rank": i + 1
            }
            for i, (name, coef) in enumerate(zip(feature_names, coefficients))
        ]
        importance_data.sort(key=lambda x: x["abs_coefficient"], reverse=True)

        return {
            "method": "Linear model coefficients",
            "explanation": "Feature importance was calculated using the model's coefficients (absolute values).",
            "top_features": importance_data[:5],
            "total_features": len(feature_names)
        }

    else:
        return {
            "method": "Not available",
            "explanation": "This model type doesn't provide direct feature importance.",
            "suggestion": "Use permutation importance for a model-agnostic approach."
        }


def explain_model_performance(training_results, metadata):
    try:
        if not training_results:
            return {"error": "No training results available. Please train the model first."}

        train_score = training_results.get("train_score", 0.0)
        test_score = training_results.get("test_score", 0.0)
        task = metadata.get("task_type")

        explanation = {
            "task": task,
            "train_score": float(train_score),
            "test_score": float(test_score),
            "interpretation": {}
        }

        if task.startswith("classification"):
            explanation["interpretation"] = {
                "train_score": f"Training accuracy: {train_score:.1%}",
                "test_score": f"Test accuracy: {test_score:.1%}",
                "overfitting": "Model shows overfitting" if train_score > test_score + 0.1 else "Model generalizes well"
            }
        elif task == "regression":
            explanation["interpretation"] = {
                "train_score": f"Training R² score: {train_score:.4f}",
                "test_score": f"Test R² score: {test_score:.4f}",
                "overfitting": "Model shows overfitting" if train_score > test_score + 0.1 else "Model generalizes well"
            }

        return explanation

    except Exception as e:
        return {"error": f"Could not evaluate model performance: {str(e)}"}
