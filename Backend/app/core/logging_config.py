"""
Logging configuration for Canis AI Backend
"""

import logging
import os
from datetime import datetime
from loguru import logger
import sys
import psutil
import time

class InterceptHandler(logging.Handler):
    """Intercept standard logging and redirect to loguru"""
    
    def emit(self, record):
        try:
            level = logger.level(record.levelname).name
        except ValueError:
            level = record.levelno
        
        frame, depth = logging.currentframe(), 2
        while frame.f_code.co_filename == logging.__file__:
            frame = frame.f_back
            depth += 1
        
        logger.opt(depth=depth, exception=record.exc_info).log(level, record.getMessage())

def setup_logging():
    """Setup logging configuration"""
    # Create logs directory
    os.makedirs("logs", exist_ok=True)
    
    # Remove default loguru handler
    logger.remove()
    
    # Add console handler with color
    logger.add(
        sys.stdout,
        format="<green>{time:YYYY-MM-DD HH:mm:ss}</green> | <level>{level: <8}</level> | <cyan>{name}</cyan>:<cyan>{function}</cyan>:<cyan>{line}</cyan> - <level>{message}</level>",
        level="INFO",
        colorize=True
    )
    
    # Add file handler with rotation
    logger.add(
        "logs/backend.log",
        format="{time:YYYY-MM-DD HH:mm:ss} | {level: <8} | {name}:{function}:{line} - {message}",
        level="DEBUG",
        rotation="10 MB",
        retention="30 days",
        compression="zip"
    )
    
    # Add error file handler
    logger.add(
        "logs/errors.log",
        format="{time:YYYY-MM-DD HH:mm:ss} | {level: <8} | {name}:{function}:{line} - {message}",
        level="ERROR",
        rotation="5 MB",
        retention="60 days",
        compression="zip"
    )
    
    # Intercept standard logging
    logging.basicConfig(handlers=[InterceptHandler()], level=0, force=True)
    
    # Set loguru as the default logger for all modules
    for name in logging.root.manager.loggerDict.keys():
        logging.getLogger(name).handlers = []
        logging.getLogger(name).propagate = True

class SystemMonitor:
    """Monitor system health and performance"""
    
    def __init__(self):
        self.start_time = time.time()
    
    def get_system_health(self) -> dict:
        """Get comprehensive system health information"""
        try:
            # CPU usage
            cpu_percent = psutil.cpu_percent(interval=1)
            cpu_count = psutil.cpu_count()
            
            # Memory usage
            memory = psutil.virtual_memory()
            memory_info = {
                "total": memory.total,
                "available": memory.available,
                "used": memory.used,
                "percent": memory.percent
            }
            
            # Disk usage
            disk = psutil.disk_usage('/')
            disk_info = {
                "total": disk.total,
                "used": disk.used,
                "free": disk.free,
                "percent": (disk.used / disk.total) * 100
            }
            
            # Uptime
            uptime = time.time() - self.start_time
            
            return {
                "status": "healthy",
                "timestamp": datetime.now().isoformat(),
                "uptime_seconds": uptime,
                "uptime_formatted": self._format_uptime(uptime),
                "cpu": {
                    "usage_percent": cpu_percent,
                    "count": cpu_count
                },
                "memory": memory_info,
                "disk": disk_info
            }
            
        except Exception as e:
            logger.error(f"Failed to get system health: {str(e)}")
            return {
                "status": "error",
                "error": str(e),
                "timestamp": datetime.now().isoformat()
            }
    
    def _format_uptime(self, seconds: float) -> str:
        """Format uptime in human readable format"""
        days = int(seconds // 86400)
        hours = int((seconds % 86400) // 3600)
        minutes = int((seconds % 3600) // 60)
        secs = int(seconds % 60)
        
        if days > 0:
            return f"{days}d {hours}h {minutes}m {secs}s"
        elif hours > 0:
            return f"{hours}h {minutes}m {secs}s"
        elif minutes > 0:
            return f"{minutes}m {secs}s"
        else:
            return f"{secs}s"

# Global system monitor instance
system_monitor = SystemMonitor()

def get_health_status() -> dict:
    """Get overall system health status"""
    try:
        from .gemini_brain import gemini
        
        # Get system health
        system_health = system_monitor.get_system_health()
        
        # Get model status
        model_status = {
            "model_loaded": gemini.model is not None,
            "model_type": type(gemini.model).__name__ if gemini.model else None,
            "task_type": gemini.metadata.task_type if hasattr(gemini, 'metadata') else None,
            "target_column": gemini.metadata.target_column if hasattr(gemini, 'metadata') else None
        }
        
        # Get dataset info
        dataset_info = {}
        if hasattr(gemini, 'metadata'):
            dataset_info = {
                "n_samples": getattr(gemini.metadata, 'n_samples', 0),
                "n_features": getattr(gemini.metadata, 'n_features', 0),
                "features": getattr(gemini.metadata, 'features', [])
            }
        
        # Overall status
        overall_status = "healthy"
        if system_health.get("status") == "error":
            overall_status = "degraded"
        
        return {
            "status": overall_status,
            "timestamp": datetime.now().isoformat(),
            "system": system_health,
            "model": model_status,
            "dataset": dataset_info,
            "version": "1.0.0"
        }
        
    except Exception as e:
        logger.error(f"Failed to get health status: {str(e)}")
        return {
            "status": "error",
            "error": str(e),
            "timestamp": datetime.now().isoformat()
        } 