import pandas as pd
import numpy as np
from .gemini_brain import gemini

def profile_data():
    """Profile data using Gemini Brain metadata"""
    try:
        # Get metadata from Gemini Brain
        metadata = gemini.get_metadata()
        
        if not metadata.get("target_column"):
            return {"error": "No dataset analyzed yet. Please upload a CSV file first."}
        
        # Load dataset for additional profiling
        df = pd.read_csv("tmp/dataset.csv")
        
        # Get detailed profiling
        profiling = {
            "columns": list(df.columns),
            "types": df.dtypes.astype(str).to_dict(),
            "nulls": df.isnull().sum().to_dict(),
            "shape": df.shape,
            "target_column": metadata["target_column"],
            "task_type": metadata["task_type"],
            "task_reason": metadata["task_reason"],
            "n_classes": metadata["n_classes"],
            "confidence_score": metadata["confidence_score"],
            "feature_columns": metadata["feature_columns"],
            "numeric_features": metadata["numeric_features"],
            "categorical_features": metadata["categorical_features"],
            "missing_values": metadata["missing_values"]
        }
        
        # Add statistical summaries for numeric features
        if metadata["numeric_features"]:
            numeric_stats = df[metadata["numeric_features"]].describe().to_dict()
            profiling["numeric_statistics"] = numeric_stats
        
        # Add value counts for categorical features
        if metadata["categorical_features"]:
            categorical_counts = {}
            for col in metadata["categorical_features"]:
                categorical_counts[col] = df[col].value_counts().to_dict()
            profiling["categorical_counts"] = categorical_counts
        
        return profiling
        
    except Exception as e:
        return {"error": str(e)}