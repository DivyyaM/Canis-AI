"""
Inference for Canis AI AutoML backend.
- Real-time, production-ready prediction and model/encoder loading.
"""

import joblib
import pandas as pd
import numpy as np
from typing import Dict, List, Union, Any, Optional
import os
from .gemini_brain import gemini
import logging

logger = logging.getLogger(__name__)

class InferenceEngine:
    """
    Handles real-time model inference for the Canis AI platform.
    Loads models, preprocessors, and encoders, and provides prediction APIs.
    """
    def __init__(self, model_path: str = None):
        """
        Initialize the inference engine and load model artifacts.
        Args:
            model_path (str, optional): Path to the model file. Defaults to None.
        """
        self.model = None
        self.preprocessor = None
        self.target_encoder = None
        self.model_path = model_path or "tmp/current_model.joblib"
        self._load_model()
    
    def _load_model(self):
        """
        Load the trained model, preprocessor, and target encoder from disk or Gemini Brain.
        """
        try:
            # Try to load from Gemini Brain first
            if gemini.model:
                self.model = gemini.model
                logger.info("Loaded model from Gemini Brain")
            else:
                # Load from file
                if os.path.exists(self.model_path):
                    self.model = joblib.load(self.model_path)
                    logger.info(f"Loaded model from {self.model_path}")
                else:
                    logger.warning(f"Model not found at {self.model_path}")
                    return
            
            # Load preprocessor
            preprocessor_path = "tmp/preprocessor.pkl"
            if os.path.exists(preprocessor_path):
                self.preprocessor = joblib.load(preprocessor_path)
                logger.info("Loaded preprocessor")
            
            # Load target encoder if exists
            target_encoder_path = "tmp/target_encoder.pkl"
            if os.path.exists(target_encoder_path):
                self.target_encoder = joblib.load(target_encoder_path)
                logger.info("Loaded target encoder")
                
        except Exception as e:
            logger.error(f"Failed to load model: {str(e)}")
            raise
    
    def predict(self, data: Union[Dict, List[Dict], pd.DataFrame]) -> Dict[str, Any]:
        """
        Make predictions on input data.
        Args:
            data: Input data as dict, list of dicts, or DataFrame.
        Returns:
            dict: Predictions, probabilities (if available), and metadata.
        """
        try:
            if not self.model:
                return {"error": "No model loaded. Please train a model first."}
            
            # Convert input to DataFrame
            if isinstance(data, dict):
                df = pd.DataFrame([data])
            elif isinstance(data, list):
                df = pd.DataFrame(data)
            elif isinstance(data, pd.DataFrame):
                df = data.copy()
            else:
                return {"error": "Invalid input format. Expected dict, list of dicts, or DataFrame"}
            
            # Get target column info
            target_column = gemini.metadata.target_column
            if target_column and target_column in df.columns:
                # Remove target column if present
                df = df.drop(columns=[target_column])
            
            # Preprocess data
            if self.preprocessor:
                try:
                    # Handle pipeline vs standalone preprocessor
                    if hasattr(self.preprocessor, 'transform'):
                        df_processed = self.preprocessor.transform(df)
                    else:
                        df_processed = df
                except Exception as e:
                    logger.error(f"Preprocessing failed: {str(e)}")
                    return {"error": f"Preprocessing failed: {str(e)}"}
            else:
                df_processed = df
            
            # Make predictions
            predictions = self.model.predict(df_processed)
            
            # Get probabilities if available
            probabilities = None
            if hasattr(self.model, 'predict_proba'):
                try:
                    proba = self.model.predict_proba(df_processed)
                    # Convert to list of dicts for each class
                    if len(proba.shape) == 2:
                        class_names = getattr(self.model, 'classes_', [f"class_{i}" for i in range(proba.shape[1])])
                        probabilities = []
                        for i in range(len(proba)):
                            prob_dict = {class_names[j]: float(proba[i][j]) for j in range(len(class_names))}
                            probabilities.append(prob_dict)
                except Exception as e:
                    logger.warning(f"Could not get probabilities: {str(e)}")
            
            # Convert predictions to list
            predictions_list = predictions.tolist() if hasattr(predictions, 'tolist') else list(predictions)
            
            # Decode predictions if target encoder exists
            if self.target_encoder and hasattr(self.target_encoder, 'inverse_transform'):
                try:
                    decoded_predictions = self.target_encoder.inverse_transform(predictions)
                    predictions_list = decoded_predictions.tolist() if hasattr(decoded_predictions, 'tolist') else list(decoded_predictions)
                except Exception as e:
                    logger.warning(f"Could not decode predictions: {str(e)}")
            
            result = {
                "predictions": predictions_list,
                "input_shape": df.shape,
                "model_type": type(self.model).__name__,
                "task_type": gemini.metadata.task_type
            }
            
            if probabilities:
                result["probabilities"] = probabilities
            
            return result
            
        except Exception as e:
            logger.error(f"Prediction failed: {str(e)}")
            return {"error": f"Prediction failed: {str(e)}"}
    
    def get_model_info(self) -> Dict[str, Any]:
        """
        Get information about the loaded model and its configuration.
        Returns:
            dict: Model type, task type, target column, features, and parameters.
        """
        if not self.model:
            return {"error": "No model loaded"}
        
        info = {
            "model_type": type(self.model).__name__,
            "task_type": gemini.metadata.task_type,
            "target_column": gemini.metadata.target_column,
            "features": gemini.metadata.features,
            "model_params": getattr(self.model, 'get_params', lambda: {})()
        }
        
        if hasattr(self.model, 'n_features_in_'):
            info["n_features"] = self.model.n_features_in_
        
        return info

# Global inference engine instance
inference_engine = InferenceEngine()

def predict_single(data: Dict) -> Dict[str, Any]:
    """
    Make prediction on a single data point.
    Args:
        data (dict): Input data.
    Returns:
        dict: Prediction result.
    """
    return inference_engine.predict(data)

def predict_batch(data: List[Dict]) -> Dict[str, Any]:
    """
    Make predictions on a batch of data.
    Args:
        data (List[dict]): List of input data points.
    Returns:
        dict: Batch prediction results.
    """
    return inference_engine.predict(data)

def get_model_info() -> Dict[str, Any]:
    """
    Get current model information.
    Returns:
        dict: Model info and configuration.
    """
    return inference_engine.get_model_info() 