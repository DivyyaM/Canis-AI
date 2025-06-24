"""
Async task queue for Canis AI Backend
Handles long-running tasks like model training
"""

import asyncio
import uuid
import time
from datetime import datetime
from typing import Dict, Any, Optional, Callable
from enum import Enum
import threading
from concurrent.futures import ThreadPoolExecutor
import json
import os
from .gemini_brain import gemini
from .trainer import train_model
from .model_benchmark import benchmark_models
import logging

logger = logging.getLogger(__name__)

class TaskStatus(Enum):
    PENDING = "pending"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"

class TaskType(Enum):
    TRAIN_MODEL = "train_model"
    BENCHMARK_MODELS = "benchmark_models"
    HYPERPARAMETER_TUNING = "hyperparameter_tuning"
    DATA_PROCESSING = "data_processing"

class AsyncTaskQueue:
    """Manages async task execution"""
    
    def __init__(self, max_workers: int = 4):
        self.tasks: Dict[str, Dict[str, Any]] = {}
        self.executor = ThreadPoolExecutor(max_workers=max_workers)
        self.lock = threading.Lock()
        self._load_tasks()
    
    def _load_tasks(self):
        """Load tasks from persistent storage"""
        try:
            if os.path.exists("tmp/tasks.json"):
                with open("tmp/tasks.json", "r") as f:
                    saved_tasks = json.load(f)
                    for task_id, task_data in saved_tasks.items():
                        if task_data["status"] in [TaskStatus.PENDING.value, TaskStatus.RUNNING.value]:
                            # Reset running tasks to pending
                            task_data["status"] = TaskStatus.PENDING.value
                        self.tasks[task_id] = task_data
                logger.info(f"Loaded {len(saved_tasks)} tasks from storage")
        except Exception as e:
            logger.error(f"Failed to load tasks: {str(e)}")
    
    def _save_tasks(self):
        """Save tasks to persistent storage"""
        try:
            os.makedirs("tmp", exist_ok=True)
            with open("tmp/tasks.json", "w") as f:
                json.dump(self.tasks, f, indent=2, default=str)
        except Exception as e:
            logger.error(f"Failed to save tasks: {str(e)}")
    
    def create_task(self, task_type: TaskType, params: Dict[str, Any] = None) -> str:
        """Create a new task"""
        task_id = str(uuid.uuid4())
        
        with self.lock:
            self.tasks[task_id] = {
                "id": task_id,
                "type": task_type.value,
                "status": TaskStatus.PENDING.value,
                "params": params or {},
                "created_at": datetime.now().isoformat(),
                "started_at": None,
                "completed_at": None,
                "result": None,
                "error": None,
                "progress": 0
            }
            self._save_tasks()
        
        logger.info(f"Created task {task_id} of type {task_type.value}")
        return task_id
    
    def get_task(self, task_id: str) -> Optional[Dict[str, Any]]:
        """Get task by ID"""
        return self.tasks.get(task_id)
    
    def get_all_tasks(self) -> Dict[str, Dict[str, Any]]:
        """Get all tasks"""
        return self.tasks.copy()
    
    def cancel_task(self, task_id: str) -> bool:
        """Cancel a task"""
        with self.lock:
            if task_id in self.tasks:
                task = self.tasks[task_id]
                if task["status"] in [TaskStatus.PENDING.value, TaskStatus.RUNNING.value]:
                    task["status"] = TaskStatus.CANCELLED.value
                    task["completed_at"] = datetime.now().isoformat()
                    self._save_tasks()
                    logger.info(f"Cancelled task {task_id}")
                    return True
        return False
    
    def _update_task_status(self, task_id: str, status: TaskStatus, 
                           result: Any = None, error: str = None, progress: int = None):
        """Update task status"""
        with self.lock:
            if task_id in self.tasks:
                task = self.tasks[task_id]
                task["status"] = status.value
                
                if status == TaskStatus.RUNNING and not task["started_at"]:
                    task["started_at"] = datetime.now().isoformat()
                elif status in [TaskStatus.COMPLETED, TaskStatus.FAILED, TaskStatus.CANCELLED]:
                    task["completed_at"] = datetime.now().isoformat()
                
                if result is not None:
                    task["result"] = result
                if error is not None:
                    task["error"] = error
                if progress is not None:
                    task["progress"] = progress
                
                self._save_tasks()
    
    def _train_model_task(self, task_id: str, params: Dict[str, Any]):
        """Execute model training task"""
        try:
            self._update_task_status(task_id, TaskStatus.RUNNING, progress=10)
            
            # Extract parameters
            model_name = params.get("model_name", "RandomForest")
            
            # Load dataset
            import pandas as pd
            df = pd.read_csv("tmp/dataset.csv")
            
            self._update_task_status(task_id, TaskStatus.RUNNING, progress=30)
            
            # Train model
            result = gemini.auto_train_model(df, model_name)
            
            if "error" in result:
                self._update_task_status(task_id, TaskStatus.FAILED, error=result["error"])
            else:
                self._update_task_status(task_id, TaskStatus.COMPLETED, result=result, progress=100)
                
        except Exception as e:
            logger.error(f"Training task {task_id} failed: {str(e)}")
            self._update_task_status(task_id, TaskStatus.FAILED, error=str(e))
    
    def _benchmark_models_task(self, task_id: str, params: Dict[str, Any]):
        """Execute model benchmarking task"""
        try:
            self._update_task_status(task_id, TaskStatus.RUNNING, progress=10)
            
            # Run benchmark
            result = benchmark_models()
            
            if "error" in result:
                self._update_task_status(task_id, TaskStatus.FAILED, error=result["error"])
            else:
                self._update_task_status(task_id, TaskStatus.COMPLETED, result=result, progress=100)
                
        except Exception as e:
            logger.error(f"Benchmark task {task_id} failed: {str(e)}")
            self._update_task_status(task_id, TaskStatus.FAILED, error=str(e))
    
    def execute_task(self, task_id: str):
        """Execute a task in a separate thread"""
        task = self.get_task(task_id)
        if not task:
            logger.error(f"Task {task_id} not found")
            return
        
        task_type = TaskType(task["type"])
        params = task["params"]
        
        try:
            if task_type == TaskType.TRAIN_MODEL:
                self._train_model_task(task_id, params)
            elif task_type == TaskType.BENCHMARK_MODELS:
                self._benchmark_models_task(task_id, params)
            else:
                self._update_task_status(task_id, TaskStatus.FAILED, 
                                       error=f"Unknown task type: {task_type}")
                
        except Exception as e:
            logger.error(f"Task execution failed: {str(e)}")
            self._update_task_status(task_id, TaskStatus.FAILED, error=str(e))
    
    def submit_task(self, task_id: str):
        """Submit task for execution"""
        self.executor.submit(self.execute_task, task_id)
        logger.info(f"Submitted task {task_id} for execution")
    
    def cleanup_completed_tasks(self, max_age_hours: int = 24):
        """Clean up old completed tasks"""
        cutoff_time = datetime.now().timestamp() - (max_age_hours * 3600)
        
        with self.lock:
            tasks_to_remove = []
            for task_id, task in self.tasks.items():
                if task["status"] in [TaskStatus.COMPLETED.value, TaskStatus.FAILED.value, TaskStatus.CANCELLED.value]:
                    completed_at = task.get("completed_at")
                    if completed_at:
                        try:
                            task_time = datetime.fromisoformat(completed_at).timestamp()
                            if task_time < cutoff_time:
                                tasks_to_remove.append(task_id)
                        except:
                            pass
            
            for task_id in tasks_to_remove:
                del self.tasks[task_id]
            
            if tasks_to_remove:
                self._save_tasks()
                logger.info(f"Cleaned up {len(tasks_to_remove)} old tasks")

# Global task queue instance
task_queue = AsyncTaskQueue()

def create_async_task(task_type: TaskType, params: Dict[str, Any] = None) -> str:
    """Create and submit an async task"""
    task_id = task_queue.create_task(task_type, params)
    task_queue.submit_task(task_id)
    return task_id

def get_task_status(task_id: str) -> Optional[Dict[str, Any]]:
    """Get task status"""
    return task_queue.get_task(task_id)

def get_all_tasks() -> Dict[str, Dict[str, Any]]:
    """Get all tasks"""
    return task_queue.get_all_tasks()

def cancel_task(task_id: str) -> bool:
    """Cancel a task"""
    return task_queue.cancel_task(task_id)

def cleanup_tasks(max_age_hours: int = 24):
    """Clean up old tasks"""
    task_queue.cleanup_completed_tasks(max_age_hours) 