"""
Explainability for Canis AI AutoML backend.
- Explains model decisions, integrates SHAP/LIME, and supports business/ML transparency.
"""
import pandas as pd
import numpy as np
import joblib
from sklearn.inspection import permutation_importance
from .gemini_brain import gemini


def explain() -> dict:
    """
    Provide comprehensive explanations for model decisions and performance.
    Returns:
        dict: Explanations for target detection, task classification, model selection, preprocessing, performance, and feature importance.
    """
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
        return {"error": f"Explainability failed: {str(e)}"}


def explain_target_detection(metadata: dict) -> dict:
    """
    Explain how the target column was detected.
    Args:
        metadata (dict): Dataset metadata.
    Returns:
        dict: Explanation of target detection.
    """
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


def explain_task_classification(metadata: dict) -> dict:
    """
    Explain how the ML task was classified.
    Args:
        metadata (dict): Dataset metadata.
    Returns:
        dict: Explanation of task classification.
    """
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


def explain_model_selection(model_params: dict, metadata: dict) -> dict:
    """
    Explain why a particular model was selected.
    Args:
        model_params (dict): Model parameters.
        metadata (dict): Dataset metadata.
    Returns:
        dict: Explanation of model selection.
    """
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


def explain_preprocessing(metadata: dict) -> dict:
    """
    Explain the preprocessing steps applied to the data.
    Args:
        metadata (dict): Dataset metadata.
    Returns:
        dict: Explanation of preprocessing steps.
    """
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


def explain_feature_importance(model, df, metadata: dict) -> dict:
    """
    Explain feature importance using model-specific or model-agnostic methods.
    Args:
        model: Trained ML model.
        df: Input DataFrame.
        metadata (dict): Dataset metadata.
    Returns:
        dict: Explanation of feature importance.
    """
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


def explain_model_performance(training_results: dict, metadata: dict) -> dict:
    """
    Explain model performance metrics and interpretation.
    Args:
        training_results (dict): Training results and scores.
        metadata (dict): Dataset metadata.
    Returns:
        dict: Explanation of model performance.
    """
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
        return {"error": f"Model performance explanation failed: {str(e)}"}

# --- Advanced Explainability Tools (SHAP & LIME) ---
# These functions provide advanced model interpretability using SHAP and LIME.
# Originally from advanced_explainability.py, now unified for clarity.

import os

TMP_DIR = "tmp"
os.makedirs(TMP_DIR, exist_ok=True)

def generate_shap_explanations(sample_index=None, num_samples=100):
    """Generate SHAP explanations for model predictions"""
    try:
        if not gemini.model:
            return {"error": "No model trained yet. Please train a model first."}
        try:
            X_test = joblib.load(f"{TMP_DIR}/X_test.pkl")
            y_test = joblib.load(f"{TMP_DIR}/y_test.pkl")
        except FileNotFoundError:
            return {"error": "Test data not found. Please train the model first."}
        metadata = gemini.get_metadata()
        feature_names = metadata.get("feature_columns", [])
        try:
            import shap
        except ImportError:
            return {"error": "SHAP not installed. Please install with: pip install shap"}
        if hasattr(gemini.model, 'steps'):
            final_model = gemini.model.steps[-1][1]
            preprocessor = gemini.model.steps[0][1] if len(gemini.model.steps) > 1 else None
            if preprocessor:
                X_test_transformed = preprocessor.transform(X_test)
                if hasattr(preprocessor, 'get_feature_names_out'):
                    transformed_feature_names = preprocessor.get_feature_names_out().tolist()
                else:
                    transformed_feature_names = [f"feature_{i}" for i in range(X_test_transformed.shape[1])]
            else:
                X_test_transformed = X_test
                transformed_feature_names = feature_names
        else:
            final_model = gemini.model
            X_test_transformed = X_test
            transformed_feature_names = feature_names
        if hasattr(final_model, 'predict_proba'):
            if hasattr(final_model, 'feature_importances_'):
                explainer = shap.TreeExplainer(final_model)
            else:
                background_data = pd.DataFrame(X_test_transformed[:min(num_samples, len(X_test_transformed))], columns=transformed_feature_names)
                explainer = shap.KernelExplainer(final_model.predict_proba, background_data)
        else:
            if hasattr(final_model, 'feature_importances_'):
                explainer = shap.TreeExplainer(final_model)
            else:
                background_data = pd.DataFrame(X_test_transformed[:min(num_samples, len(X_test_transformed))], columns=transformed_feature_names)
                explainer = shap.KernelExplainer(final_model.predict, background_data)
        if sample_index is not None:
            if sample_index >= len(X_test_transformed):
                return {"error": f"Sample index {sample_index} out of range. Max index: {len(X_test_transformed)-1}"}
            sample = X_test_transformed[sample_index:sample_index+1]
            shap_values = explainer.shap_values(sample)
            if isinstance(shap_values, list):
                shap_values = shap_values[0]
            explanation = {
                "sample_index": sample_index,
                "actual_value": int(y_test[sample_index]),
                "predicted_value": int(gemini.model.predict(X_test[sample_index:sample_index+1])[0]),
                "feature_importance": dict(zip(transformed_feature_names, shap_values.tolist())),
                "base_value": float(explainer.expected_value) if hasattr(explainer, 'expected_value') else 0.0
            }
        else:
            if hasattr(final_model, 'feature_importances_'):
                importance = final_model.feature_importances_
            else:
                shap_values = explainer.shap_values(X_test_transformed[:num_samples])
                if isinstance(shap_values, list):
                    shap_values = np.abs(shap_values[0])
                importance = np.mean(np.abs(shap_values), axis=0)
            explanation = {
                "global_feature_importance": dict(zip(transformed_feature_names, importance.tolist())),
                "num_samples_analyzed": min(num_samples, len(X_test_transformed))
            }
        return {
            "status": "success",
            "explanation_type": "shap",
            "explanation": explanation
        }
    except Exception as e:
        return {"error": f"SHAP explanation failed: {str(e)}"}

def generate_lime_explanations(sample_index=0, num_features=10):
    """Generate LIME explanations for model predictions"""
    try:
        if not gemini.model:
            return {"error": "No model trained yet. Please train the model first."}
        try:
            X_test = joblib.load(f"{TMP_DIR}/X_test.pkl")
            y_test = joblib.load(f"{TMP_DIR}/y_test.pkl")
        except FileNotFoundError:
            return {"error": "Test data not found. Please train the model first."}
        metadata = gemini.get_metadata()
        feature_names = metadata.get("feature_columns", [])
        try:
            from lime import lime_tabular
        except ImportError:
            return {"error": "LIME not installed. Please install with: pip install lime"}
        if sample_index >= len(X_test):
            return {"error": f"Sample index {sample_index} out of range. Max index: {len(X_test)-1}"}
        explainer = lime_tabular.LimeTabularExplainer(
            X_test,
            feature_names=feature_names,
            class_names=['Class 0', 'Class 1'] if len(np.unique(y_test)) == 2 else None,
            mode='classification' if hasattr(gemini.model, 'predict_proba') else 'regression'
        )
        sample = X_test[sample_index]
        exp = explainer.explain_instance(
            sample, 
            gemini.model.predict_proba if hasattr(gemini.model, 'predict_proba') else gemini.model.predict,
            num_features=num_features
        )
        explanation_data = []
        for feature, weight in exp.as_list():
            explanation_data.append({
                "feature": feature,
                "weight": float(weight)
            })
        explanation = {
            "sample_index": sample_index,
            "actual_value": int(y_test[sample_index]),
            "predicted_value": int(gemini.model.predict([sample])[0]),
            "feature_contributions": explanation_data,
            "num_features": num_features
        }
        return {
            "status": "success",
            "explanation_type": "lime",
            "explanation": explanation
        }
    except Exception as e:
        return {"error": f"LIME explanation failed: {str(e)}"}

def generate_feature_importance_analysis():
    """Generate comprehensive feature importance analysis"""
    try:
        if not gemini.model:
            return {"error": "No model trained yet. Please train a model first."}
        metadata = gemini.get_metadata()
        feature_names = metadata.get("feature_columns", [])
        if hasattr(gemini.model, 'feature_importances_'):
            importances = gemini.model.feature_importances_
            importance_data = [
                {"feature": name, "importance": float(imp), "rank": i + 1}
                for i, (name, imp) in enumerate(zip(feature_names, importances))
            ]
            importance_data.sort(key=lambda x: x["importance"], reverse=True)
            return {
                "method": "Tree-based feature importance",
                "explanation": "Feature importance was calculated using the model's built-in feature importance method.",
                "top_features": importance_data[:5],
                "total_features": len(feature_names)
            }
        elif hasattr(gemini.model, 'coef_'):
            coefficients = gemini.model.coef_.flatten()
            importance_data = [
                {"feature": name, "coefficient": float(coef), "rank": i + 1}
                for i, (name, coef) in enumerate(zip(feature_names, coefficients))
            ]
            importance_data.sort(key=lambda x: abs(x["coefficient"]), reverse=True)
            return {
                "method": "Coefficient-based feature importance",
                "explanation": "Feature importance was calculated using model coefficients.",
                "top_features": importance_data[:5],
                "total_features": len(feature_names)
            }
        else:
            return {"error": "Model does not support feature importance analysis."}
    except Exception as e:
        return {"error": f"Feature importance analysis failed: {str(e)}"}
