import pandas as pd
import numpy as np
from pandas.api.types import is_numeric_dtype, is_string_dtype, is_object_dtype
from .gemini_brain import gemini

def classify_task():
    """Get task classification from Gemini Brain metadata"""
    try:
        metadata = gemini.get_metadata()
        
        if not metadata.get("task_type"):
            return {"task": "unknown", "error": "No dataset analyzed yet. Please upload a CSV file first."}
        
        return {
            "task": metadata["task_type"],
            "reason": metadata["task_reason"],
            "target_column": metadata["target_column"],
            "n_classes": metadata["n_classes"],
            "confidence_score": metadata["confidence_score"]
        }
        
    except Exception as e:
        return {"task": "unknown", "error": str(e)}

def find_target():
    """Helper function to get target column"""
    from .target_identifier import find_target as find_target_func
    return find_target_func()
