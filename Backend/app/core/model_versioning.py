"""
Model versioning system for Canis AI
Manages model versions, metadata, and lifecycle
"""

import sqlite3
import json
import os
import shutil
from datetime import datetime
from typing import Dict, List, Optional, Any
import joblib
from .gemini_brain import gemini
import logging

logger = logging.getLogger(__name__)

class ModelVersioning:
    """Manages model versions and metadata"""
    
    def __init__(self, db_path: str = "models/model_registry.db", models_dir: str = "models"):
        self.db_path = db_path
        self.models_dir = models_dir
        self._ensure_directories()
        self._init_database()
    
    def _ensure_directories(self):
        """Create necessary directories"""
        os.makedirs(self.models_dir, exist_ok=True)
        os.makedirs(os.path.dirname(self.db_path), exist_ok=True)
    
    def _init_database(self):
        """Initialize SQLite database with model registry table"""
        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            
            cursor.execute('''
                CREATE TABLE IF NOT EXISTS model_registry (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    model_name TEXT NOT NULL,
                    version TEXT NOT NULL,
                    file_path TEXT NOT NULL,
                    task_type TEXT,
                    target_column TEXT,
                    accuracy REAL,
                    f1_score REAL,
                    r2_score REAL,
                    n_samples INTEGER,
                    n_features INTEGER,
                    model_type TEXT,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    description TEXT,
                    tags TEXT,
                    is_active BOOLEAN DEFAULT 1,
                    UNIQUE(model_name, version)
                )
            ''')
            
            conn.commit()
            conn.close()
            logger.info("Model registry database initialized")
            
        except Exception as e:
            logger.error(f"Failed to initialize database: {str(e)}")
            raise
    
    def save_model_version(self, model_name: str, version: str = None, 
                          description: str = "", tags: List[str] = None) -> Dict[str, Any]:
        """
        Save current model as a new version
        
        Args:
            model_name: Name for the model
            version: Version string (auto-generated if None)
            description: Model description
            tags: List of tags
            
        Returns:
            Dictionary with save result
        """
        try:
            if not gemini.model:
                return {"error": "No model to save. Please train a model first."}
            
            # Generate version if not provided
            if not version:
                timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                version = f"v{timestamp}"
            
            # Create filename
            filename = f"{model_name}_{version}.joblib"
            file_path = os.path.join(self.models_dir, filename)
            
            # Save model
            joblib.dump(gemini.model, file_path)
            
            # Get model metadata
            metadata = gemini.get_metadata()
            training_results = getattr(gemini, 'training_results', {})
            evaluation_results = getattr(gemini, 'evaluation_results', {})
            
            # Extract metrics
            accuracy = training_results.get('accuracy') or evaluation_results.get('accuracy')
            f1_score = training_results.get('f1_score') or evaluation_results.get('f1_score')
            r2_score = training_results.get('r2_score') or evaluation_results.get('r2_score')
            
            # Save to database
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            
            cursor.execute('''
                INSERT INTO model_registry 
                (model_name, version, file_path, task_type, target_column, 
                 accuracy, f1_score, r2_score, n_samples, n_features, 
                 model_type, description, tags)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            ''', (
                model_name, version, file_path, metadata.task_type, metadata.target_column,
                accuracy, f1_score, r2_score, metadata.n_samples, metadata.n_features,
                type(gemini.model).__name__, description, json.dumps(tags or [])
            ))
            
            conn.commit()
            conn.close()
            
            logger.info(f"Model version saved: {model_name}_{version}")
            
            return {
                "status": "success",
                "model_name": model_name,
                "version": version,
                "file_path": file_path,
                "message": f"Model {model_name}_{version} saved successfully"
            }
            
        except Exception as e:
            logger.error(f"Failed to save model version: {str(e)}")
            return {"error": f"Failed to save model version: {str(e)}"}
    
    def list_models(self) -> Dict[str, Any]:
        """List all available model versions"""
        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            
            cursor.execute('''
                SELECT id, model_name, version, task_type, target_column,
                       accuracy, f1_score, r2_score, n_samples, n_features,
                       model_type, created_at, description, tags, is_active
                FROM model_registry
                ORDER BY created_at DESC
            ''')
            
            rows = cursor.fetchall()
            conn.close()
            
            models = []
            for row in rows:
                model = {
                    "id": row[0],
                    "model_name": row[1],
                    "version": row[2],
                    "task_type": row[3],
                    "target_column": row[4],
                    "accuracy": row[5],
                    "f1_score": row[6],
                    "r2_score": row[7],
                    "n_samples": row[8],
                    "n_features": row[9],
                    "model_type": row[10],
                    "created_at": row[11],
                    "description": row[12],
                    "tags": json.loads(row[13]) if row[13] else [],
                    "is_active": bool(row[14])
                }
                models.append(model)
            
            return {
                "status": "success",
                "models": models,
                "total_models": len(models)
            }
            
        except Exception as e:
            logger.error(f"Failed to list models: {str(e)}")
            return {"error": f"Failed to list models: {str(e)}"}
    
    def load_model(self, model_id: int) -> Dict[str, Any]:
        """Load a specific model version"""
        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            
            cursor.execute('''
                SELECT file_path, model_name, version, task_type, target_column
                FROM model_registry
                WHERE id = ?
            ''', (model_id,))
            
            row = cursor.fetchone()
            conn.close()
            
            if not row:
                return {"error": f"Model with ID {model_id} not found"}
            
            file_path, model_name, version, task_type, target_column = row
            
            if not os.path.exists(file_path):
                return {"error": f"Model file not found: {file_path}"}
            
            # Load model into Gemini Brain
            model = joblib.load(file_path)
            gemini.model = model
            gemini.metadata.task_type = task_type
            gemini.metadata.target_column = target_column
            
            # Update active status
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            cursor.execute('UPDATE model_registry SET is_active = 0')
            cursor.execute('UPDATE model_registry SET is_active = 1 WHERE id = ?', (model_id,))
            conn.commit()
            conn.close()
            
            logger.info(f"Model loaded: {model_name}_{version}")
            
            return {
                "status": "success",
                "model_name": model_name,
                "version": version,
                "task_type": task_type,
                "target_column": target_column,
                "message": f"Model {model_name}_{version} loaded successfully"
            }
            
        except Exception as e:
            logger.error(f"Failed to load model: {str(e)}")
            return {"error": f"Failed to load model: {str(e)}"}
    
    def delete_model(self, model_id: int) -> Dict[str, Any]:
        """Delete a specific model version"""
        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            
            cursor.execute('SELECT file_path, model_name, version FROM model_registry WHERE id = ?', (model_id,))
            row = cursor.fetchone()
            
            if not row:
                conn.close()
                return {"error": f"Model with ID {model_id} not found"}
            
            file_path, model_name, version = row
            
            # Delete file
            if os.path.exists(file_path):
                os.remove(file_path)
            
            # Delete from database
            cursor.execute('DELETE FROM model_registry WHERE id = ?', (model_id,))
            conn.commit()
            conn.close()
            
            logger.info(f"Model deleted: {model_name}_{version}")
            
            return {
                "status": "success",
                "message": f"Model {model_name}_{version} deleted successfully"
            }
            
        except Exception as e:
            logger.error(f"Failed to delete model: {str(e)}")
            return {"error": f"Failed to delete model: {str(e)}"}
    
    def get_model_details(self, model_id: int) -> Dict[str, Any]:
        """Get detailed information about a specific model"""
        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            
            cursor.execute('''
                SELECT * FROM model_registry WHERE id = ?
            ''', (model_id,))
            
            row = cursor.fetchone()
            conn.close()
            
            if not row:
                return {"error": f"Model with ID {model_id} not found"}
            
            columns = [description[0] for description in cursor.description]
            model_data = dict(zip(columns, row))
            
            # Parse JSON fields
            if model_data.get('tags'):
                model_data['tags'] = json.loads(model_data['tags'])
            
            return {
                "status": "success",
                "model": model_data
            }
            
        except Exception as e:
            logger.error(f"Failed to get model details: {str(e)}")
            return {"error": f"Failed to get model details: {str(e)}"}

# Global model versioning instance
model_versioning = ModelVersioning()

def save_model_version(model_name: str, version: str = None, 
                      description: str = "", tags: List[str] = None) -> Dict[str, Any]:
    """Save current model as a new version"""
    return model_versioning.save_model_version(model_name, version, description, tags)

def list_models() -> Dict[str, Any]:
    """List all available model versions"""
    return model_versioning.list_models()

def load_model(model_id: int) -> Dict[str, Any]:
    """Load a specific model version"""
    return model_versioning.load_model(model_id)

def delete_model(model_id: int) -> Dict[str, Any]:
    """Delete a specific model version"""
    return model_versioning.delete_model(model_id)

def get_model_details(model_id: int) -> Dict[str, Any]:
    """Get detailed information about a specific model"""
    return model_versioning.get_model_details(model_id) 