"""
Celery Application for Distributed Async Task Processing
Provides production-ready distributed task processing for Canis AI Backend
"""

import os
from celery import Celery
from celery.schedules import crontab
import logging
from datetime import datetime

logger = logging.getLogger(__name__)

# Celery configuration
CELERY_BROKER_URL = os.getenv("REDIS_URL", "redis://localhost:6379/0")
CELERY_RESULT_BACKEND = os.getenv("REDIS_URL", "redis://localhost:6379/0")

# Create Celery app
celery_app = Celery(
    "canis_ai",
    broker=CELERY_BROKER_URL,
    backend=CELERY_RESULT_BACKEND,
    include=[
        "app.core.celery_tasks"
    ]
)

# Celery configuration
celery_app.conf.update(
    # Task routing
    task_routes={
        "app.core.celery_tasks.train_model_task": {"queue": "training"},
        "app.core.celery_tasks.benchmark_models_task": {"queue": "benchmark"},
        "app.core.celery_tasks.hyperparameter_tuning_task": {"queue": "tuning"},
        "app.core.celery_tasks.data_processing_task": {"queue": "data"},
        "app.core.celery_tasks.model_deployment_task": {"queue": "deployment"},
    },
    
    # Task serialization
    task_serializer="json",
    accept_content=["json"],
    result_serializer="json",
    timezone="UTC",
    enable_utc=True,
    
    # Task execution
    task_always_eager=False,
    task_eager_propagates=True,
    task_ignore_result=False,
    task_store_errors_even_if_ignored=True,
    
    # Worker configuration
    worker_prefetch_multiplier=1,
    worker_max_tasks_per_child=1000,
    worker_disable_rate_limits=False,
    
    # Result backend configuration
    result_expires=3600,  # 1 hour
    result_persistent=True,
    
    # Beat schedule for periodic tasks
    beat_schedule={
        "cleanup-old-models": {
            "task": "app.core.celery_tasks.cleanup_old_models_task",
            "schedule": crontab(hour=2, minute=0),  # Daily at 2 AM
        },
        "model-performance-monitoring": {
            "task": "app.core.celery_tasks.monitor_model_performance_task",
            "schedule": crontab(minute="*/30"),  # Every 30 minutes
        },
        "backup-model-registry": {
            "task": "app.core.celery_tasks.backup_model_registry_task",
            "schedule": crontab(hour=1, minute=0),  # Daily at 1 AM
        },
        "health-check": {
            "task": "app.core.celery_tasks.health_check_task",
            "schedule": crontab(minute="*/5"),  # Every 5 minutes
        },
    },
    
    # Task retry configuration
    task_acks_late=True,
    task_reject_on_worker_lost=True,
    task_remote_tracebacks=True,
    
    # Security
    security_key=os.getenv("CELERY_SECURITY_KEY", "your-security-key"),
    security_certificate=os.getenv("CELERY_CERT_PATH"),
    security_cert_store=os.getenv("CELERY_CERT_STORE_PATH"),
)

# Task routing for different environments
if os.getenv("ENVIRONMENT") == "production":
    celery_app.conf.update(
        # Production-specific settings
        worker_max_memory_per_child=200000,  # 200MB
        worker_max_tasks_per_child=500,
        task_time_limit=3600,  # 1 hour
        task_soft_time_limit=3000,  # 50 minutes
        result_expires=7200,  # 2 hours
    )
elif os.getenv("ENVIRONMENT") == "development":
    celery_app.conf.update(
        # Development-specific settings
        task_always_eager=True,  # Execute tasks synchronously
        worker_max_memory_per_child=100000,  # 100MB
        task_time_limit=1800,  # 30 minutes
        result_expires=1800,  # 30 minutes
    )

# Task error handling
@celery_app.task(bind=True)
def debug_task(self):
    """Debug task for testing Celery setup"""
    logger.info(f"Request: {self.request!r}")
    return "Debug task completed"

# Health check task
@celery_app.task
def health_check_task():
    """Periodic health check task"""
    try:
        # Check database connectivity
        from .model_versioning import model_versioning
        conn = model_versioning._get_connection()
        conn.close()
        
        # Check Redis connectivity
        from redis import Redis
        redis_client = Redis.from_url(CELERY_BROKER_URL)
        redis_client.ping()
        
        logger.info("Health check passed")
        return {"status": "healthy", "timestamp": "2024-01-01T00:00:00Z"}
        
    except Exception as e:
        logger.error(f"Health check failed: {str(e)}")
        return {"status": "unhealthy", "error": str(e)}

# Task monitoring
@celery_app.task
def monitor_task_progress(task_id: str):
    """Monitor task progress and send notifications"""
    try:
        from .task_queue import get_task_status
        task_status = get_task_status(task_id)
        
        if task_status and task_status.get("status") == "completed":
            # Send notification
            logger.info(f"Task {task_id} completed successfully")
            return {"task_id": task_id, "status": "completed"}
        elif task_status and task_status.get("status") == "failed":
            # Send error notification
            logger.error(f"Task {task_id} failed: {task_status.get('error')}")
            return {"task_id": task_id, "status": "failed", "error": task_status.get("error")}
        
        return {"task_id": task_id, "status": "monitoring"}
        
    except Exception as e:
        logger.error(f"Task monitoring failed: {str(e)}")
        return {"error": str(e)}

# Task cleanup
@celery_app.task
def cleanup_completed_tasks_task():
    """Clean up old completed tasks"""
    try:
        from .task_queue import cleanup_tasks
        cleanup_tasks(max_age_hours=24)
        logger.info("Completed tasks cleanup finished")
        return {"status": "success", "message": "Tasks cleaned up"}
    except Exception as e:
        logger.error(f"Task cleanup failed: {str(e)}")
        return {"error": str(e)}

# Model registry backup
@celery_app.task
def backup_model_registry_task():
    """Backup model registry to external storage"""
    try:
        from .model_versioning import model_versioning
        backup_path = f"backups/model_registry_{datetime.now().strftime('%Y%m%d_%H%M%S')}.db"
        
        # Create backup
        import shutil
        shutil.copy2(model_versioning.db_path, backup_path)
        
        logger.info(f"Model registry backed up to {backup_path}")
        return {"status": "success", "backup_path": backup_path}
        
    except Exception as e:
        logger.error(f"Model registry backup failed: {str(e)}")
        return {"error": str(e)}

# Performance monitoring
@celery_app.task
def monitor_model_performance_task():
    """Monitor model performance and trigger alerts"""
    try:
        from .gemini_brain import gemini
        
        if gemini.model and gemini.training_results:
            # Check if performance is degrading
            test_score = gemini.training_results.get("test_score", 0)
            
            if test_score < 0.7:  # Performance threshold
                logger.warning(f"Model performance below threshold: {test_score}")
                return {
                    "status": "warning",
                    "message": f"Model performance below threshold: {test_score}",
                    "test_score": test_score
                }
        
        return {"status": "healthy", "message": "Model performance OK"}
        
    except Exception as e:
        logger.error(f"Performance monitoring failed: {str(e)}")
        return {"error": str(e)}

# Old model cleanup
@celery_app.task
def cleanup_old_models_task():
    """Clean up old model versions"""
    try:
        from .model_versioning import model_versioning
        
        # Get old models (older than 30 days)
        old_models = model_versioning.get_old_models(days=30)
        
        cleaned_count = 0
        for model in old_models:
            try:
                model_versioning.delete_model(model["id"])
                cleaned_count += 1
            except Exception as e:
                logger.error(f"Failed to delete old model {model['id']}: {str(e)}")
        
        logger.info(f"Cleaned up {cleaned_count} old models")
        return {"status": "success", "cleaned_count": cleaned_count}
        
    except Exception as e:
        logger.error(f"Old model cleanup failed: {str(e)}")
        return {"error": str(e)}

if __name__ == "__main__":
    celery_app.start() 