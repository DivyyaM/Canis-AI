import pandas as pd
import numpy as np
import joblib
import os
from .gemini_brain import gemini

# Create tmp directory
TMP_DIR = "tmp"
os.makedirs(TMP_DIR, exist_ok=True)

def generate_shap_explanations(sample_index=None, num_samples=100):
    """Generate SHAP explanations for model predictions"""
    try:
        # Check if model exists
        if not gemini.model:
            return {"error": "No model trained yet. Please train a model first."}
        
        # Load test data
        try:
            X_test = joblib.load(f"{TMP_DIR}/X_test.pkl")
            y_test = joblib.load(f"{TMP_DIR}/y_test.pkl")
        except FileNotFoundError:
            return {"error": "Test data not found. Please train the model first."}
        
        # Get metadata
        metadata = gemini.get_metadata()
        feature_names = metadata.get("feature_columns", [])
        
        # Try to import SHAP
        try:
            import shap
        except ImportError:
            return {"error": "SHAP not installed. Please install with: pip install shap"}
        
        # Handle pipeline models
        if hasattr(gemini.model, 'steps'):
            # It's a pipeline - extract the final model and transform data
            final_model = gemini.model.steps[-1][1]  # Get the last step (classifier)
            preprocessor = gemini.model.steps[0][1] if len(gemini.model.steps) > 1 else None
            
            # Transform the data if we have a preprocessor
            if preprocessor:
                X_test_transformed = preprocessor.transform(X_test)
                # Get feature names after transformation
                if hasattr(preprocessor, 'get_feature_names_out'):
                    transformed_feature_names = preprocessor.get_feature_names_out().tolist()
                else:
                    transformed_feature_names = [f"feature_{i}" for i in range(X_test_transformed.shape[1])]
            else:
                X_test_transformed = X_test
                transformed_feature_names = feature_names
        else:
            # It's a single model
            final_model = gemini.model
            X_test_transformed = X_test
            transformed_feature_names = feature_names
        
        # Create explainer based on model type
        if hasattr(final_model, 'predict_proba'):
            # Classification model
            if hasattr(final_model, 'feature_importances_'):
                explainer = shap.TreeExplainer(final_model)
            else:
                # Ensure data is a pandas DataFrame for KernelExplainer
                background_data = pd.DataFrame(X_test_transformed[:min(num_samples, len(X_test_transformed))], 
                                             columns=transformed_feature_names)
                explainer = shap.KernelExplainer(final_model.predict_proba, background_data)
        else:
            # Regression model
            if hasattr(final_model, 'feature_importances_'):
                explainer = shap.TreeExplainer(final_model)
            else:
                # Ensure data is a pandas DataFrame for KernelExplainer
                background_data = pd.DataFrame(X_test_transformed[:min(num_samples, len(X_test_transformed))], 
                                             columns=transformed_feature_names)
                explainer = shap.KernelExplainer(final_model.predict, background_data)
        
        # Generate explanations
        if sample_index is not None:
            # Single sample explanation
            if sample_index >= len(X_test_transformed):
                return {"error": f"Sample index {sample_index} out of range. Max index: {len(X_test_transformed)-1}"}
            
            sample = X_test_transformed[sample_index:sample_index+1]
            shap_values = explainer.shap_values(sample)
            
            # Convert to list for JSON serialization
            if isinstance(shap_values, list):
                shap_values = shap_values[0]  # For classification, take first class
            
            explanation = {
                "sample_index": sample_index,
                "actual_value": int(y_test[sample_index]),
                "predicted_value": int(gemini.model.predict(X_test[sample_index:sample_index+1])[0]),
                "feature_importance": dict(zip(transformed_feature_names, shap_values.tolist())),
                "base_value": float(explainer.expected_value) if hasattr(explainer, 'expected_value') else 0.0
            }
        else:
            # Global feature importance
            if hasattr(final_model, 'feature_importances_'):
                # Tree-based models
                importance = final_model.feature_importances_
            else:
                # Other models - use SHAP values
                shap_values = explainer.shap_values(X_test_transformed[:num_samples])
                if isinstance(shap_values, list):
                    shap_values = np.abs(shap_values[0])  # For classification
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
        # Check if model exists
        if not gemini.model:
            return {"error": "No model trained yet. Please train a model first."}
        
        # Load test data
        try:
            X_test = joblib.load(f"{TMP_DIR}/X_test.pkl")
            y_test = joblib.load(f"{TMP_DIR}/y_test.pkl")
        except FileNotFoundError:
            return {"error": "Test data not found. Please train the model first."}
        
        # Get metadata
        metadata = gemini.get_metadata()
        feature_names = metadata.get("feature_columns", [])
        
        # Try to import LIME
        try:
            from lime import lime_tabular
        except ImportError:
            return {"error": "LIME not installed. Please install with: pip install lime"}
        
        # Check sample index
        if sample_index >= len(X_test):
            return {"error": f"Sample index {sample_index} out of range. Max index: {len(X_test)-1}"}
        
        # Create LIME explainer
        explainer = lime_tabular.LimeTabularExplainer(
            X_test,
            feature_names=feature_names,
            class_names=['Class 0', 'Class 1'] if len(np.unique(y_test)) == 2 else None,
            mode='classification' if hasattr(gemini.model, 'predict_proba') else 'regression'
        )
        
        # Generate explanation
        sample = X_test[sample_index]
        exp = explainer.explain_instance(
            sample, 
            gemini.model.predict_proba if hasattr(gemini.model, 'predict_proba') else gemini.model.predict,
            num_features=num_features
        )
        
        # Extract explanation data
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
        # Check if model exists
        if not gemini.model:
            return {"error": "No model trained yet. Please train a model first."}
        
        # Get metadata
        metadata = gemini.get_metadata()
        feature_names = metadata.get("feature_columns", [])
        
        # Handle pipeline models
        if hasattr(gemini.model, 'steps'):
            # It's a pipeline - extract the final model
            final_model = gemini.model.steps[-1][1]  # Get the last step (classifier)
            preprocessor = gemini.model.steps[0][1] if len(gemini.model.steps) > 1 else None
            
            # Get feature names after transformation if we have a preprocessor
            if preprocessor and hasattr(preprocessor, 'get_feature_names_out'):
                transformed_feature_names = preprocessor.get_feature_names_out().tolist()
            else:
                transformed_feature_names = feature_names
        else:
            # It's a single model
            final_model = gemini.model
            transformed_feature_names = feature_names
        
        # Get feature importance based on model type
        if hasattr(final_model, 'feature_importances_'):
            # Tree-based models (RandomForest, DecisionTree, etc.)
            importance = final_model.feature_importances_
        elif hasattr(final_model, 'coef_'):
            # Linear models (LogisticRegression, LinearRegression, etc.)
            importance = np.abs(final_model.coef_)
            if len(importance.shape) > 1:
                importance = np.mean(importance, axis=0)  # Average across classes
        else:
            # Try to get feature importance from the model if it exists
            model_type = type(final_model).__name__
            return {
                "error": f"Model type '{model_type}' does not support feature importance analysis. "
                        f"Supported models: RandomForest, DecisionTree, LogisticRegression, LinearRegression, etc."
            }
        
        # Create feature importance ranking
        feature_importance = dict(zip(transformed_feature_names, importance.tolist()))
        sorted_features = sorted(feature_importance.items(), key=lambda x: x[1], reverse=True)
        
        # Calculate statistics
        importance_values = list(feature_importance.values())
        
        analysis = {
            "feature_importance": feature_importance,
            "top_features": sorted_features[:10],  # Top 10 features
            "statistics": {
                "mean_importance": float(np.mean(importance_values)),
                "std_importance": float(np.std(importance_values)),
                "max_importance": float(np.max(importance_values)),
                "min_importance": float(np.min(importance_values))
            },
            "num_features": len(transformed_feature_names),
            "model_type": type(final_model).__name__
        }
        
        return {
            "status": "success",
            "analysis_type": "feature_importance",
            "analysis": analysis
        }
        
    except Exception as e:
        return {"error": f"Feature importance analysis failed: {str(e)}"} 