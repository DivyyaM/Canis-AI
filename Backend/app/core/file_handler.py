import pandas as pd
import os
import requests
from urllib.parse import urlparse
from fastapi import UploadFile, HTTPException
from .gemini_brain import gemini

def handle_file(file: UploadFile):
    """Handle file upload and trigger comprehensive analysis"""
    try:
        # Determine file type and read accordingly
        df = read_file_by_type(file)
        
        # Save to tmp directory
        df.to_csv("tmp/dataset.csv", index=False)
        
        # Reset Gemini Brain for new dataset
        gemini.reset()
        
        # Trigger comprehensive analysis
        analysis_results = gemini.analyze_dataset(df)
        
        # Create preprocessing pipeline
        preprocessing_info = gemini.create_preprocessing_pipeline(df)
        
        # Ensure all values are JSON-serializable
        def make_json_serializable(obj):
            if isinstance(obj, dict):
                return {k: make_json_serializable(v) for k, v in obj.items()}
            elif isinstance(obj, list):
                return [make_json_serializable(item) for item in obj]
            elif hasattr(obj, 'item'):  # numpy types
                return obj.item()
            elif hasattr(obj, 'tolist'):  # numpy arrays
                return obj.tolist()
            else:
                return obj
        
        # Convert analysis results to JSON-serializable format
        analysis_results = make_json_serializable(analysis_results)
        preprocessing_info = make_json_serializable(preprocessing_info)
        
        # Return simplified response to avoid serialization issues
        return {
            "status": "success",
            "rows": int(len(df)),
            "cols": list(df.columns),
            "target_column": analysis_results.get("target_column"),
            "task_type": analysis_results.get("task_type"),
            "message": "Dataset uploaded and analyzed successfully"
        }
        
    except Exception as e:
        return {"status": "error", "error": str(e)}

def read_file_by_type(file: UploadFile):
    """Read file based on its type/extension"""
    filename = file.filename.lower()
    
    try:
        if filename.endswith('.csv'):
            return pd.read_csv(file.file)
        elif filename.endswith('.xlsx') or filename.endswith('.xls'):
            return pd.read_excel(file.file)
        elif filename.endswith('.tsv') or filename.endswith('.txt'):
            return pd.read_csv(file.file, sep='\t')
        elif filename.endswith('.json'):
            return pd.read_json(file.file)
        else:
            # Try to read as CSV by default
            return pd.read_csv(file.file)
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Failed to read file: {str(e)}")

def handle_url_upload(url: str):
    """Handle dataset upload from URL"""
    try:
        # Validate URL
        parsed_url = urlparse(url)
        if not parsed_url.scheme or not parsed_url.netloc:
            return {"status": "error", "error": "Invalid URL provided"}
        
        # Download the file
        response = requests.get(url, timeout=30)
        response.raise_for_status()
        
        # Determine file type from URL or content
        filename = parsed_url.path.lower()
        
        # Create a temporary file-like object
        import io
        file_content = io.BytesIO(response.content)
        
        # Read based on file type
        if filename.endswith('.csv'):
            df = pd.read_csv(file_content)
        elif filename.endswith('.xlsx') or filename.endswith('.xls'):
            df = pd.read_excel(file_content)
        elif filename.endswith('.tsv') or filename.endswith('.txt'):
            df = pd.read_csv(file_content, sep='\t')
        elif filename.endswith('.json'):
            df = pd.read_json(file_content)
        else:
            # Try to read as CSV by default
            df = pd.read_csv(file_content)
        
        # Save to tmp directory
        df.to_csv("tmp/dataset.csv", index=False)
        
        # Reset Gemini Brain for new dataset
        gemini.reset()
        
        # Trigger comprehensive analysis
        analysis_results = gemini.analyze_dataset(df)
        
        # Create preprocessing pipeline
        preprocessing_info = gemini.create_preprocessing_pipeline(df)
        
        # Ensure all values are JSON-serializable
        def make_json_serializable(obj):
            if isinstance(obj, dict):
                return {k: make_json_serializable(v) for k, v in obj.items()}
            elif isinstance(obj, list):
                return [make_json_serializable(item) for item in obj]
            elif hasattr(obj, 'item'):  # numpy types
                return obj.item()
            elif hasattr(obj, 'tolist'):  # numpy arrays
                return obj.tolist()
            else:
                return obj
        
        # Convert analysis results to JSON-serializable format
        analysis_results = make_json_serializable(analysis_results)
        preprocessing_info = make_json_serializable(preprocessing_info)
        
        return {
            "status": "success",
            "rows": int(len(df)),
            "cols": list(df.columns),
            "target_column": analysis_results.get("target_column"),
            "task_type": analysis_results.get("task_type"),
            "message": f"Dataset uploaded from URL and analyzed successfully",
            "source_url": url
        }
        
    except requests.exceptions.RequestException as e:
        return {"status": "error", "error": f"Failed to download from URL: {str(e)}"}
    except Exception as e:
        return {"status": "error", "error": f"Failed to process URL data: {str(e)}"}