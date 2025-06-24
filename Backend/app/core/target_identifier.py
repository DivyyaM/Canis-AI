import pandas as pd
import numpy as np
from .gemini_brain import gemini

def find_target():
    """Get target column from Gemini Brain metadata"""
    try:
        metadata = gemini.get_metadata()
        
        if not metadata.get("target_column"):
            return {"suggested_target": None, "error": "No dataset analyzed yet. Please upload a CSV file first."}
        
        return {
            "suggested_target": metadata["target_column"],
            "method": "gemini_brain_analysis",
            "confidence_score": metadata["confidence_score"],
            "task_type": metadata["task_type"],
            "n_classes": metadata["n_classes"],
            "target_dtype": metadata["target_dtype"]
        }
        
    except Exception as e:
        return {"suggested_target": None, "error": str(e)}