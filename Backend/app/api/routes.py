from fastapi import APIRouter, UploadFile, File, Query
from fastapi.responses import FileResponse
from ..core.file_handler import handle_file, handle_url_upload
from ..core.data_profile import profile_data
from ..core.target_identifier import find_target
from ..core.task_classifier import classify_task
from ..core.model_selector import select_model
from ..core.trainer import train_model
from ..core.evaluator import evaluate
from ..core.explainability import explain
from ..core.code_genrator import generate_pipeline_code
from ..core.llm_twin import chat, get_conversation_history, clear_memory
from ..core.model_benchmark import benchmark_models, get_benchmark_summary, compare_with_current_model, save_best_benchmark_model
from ..core.hyperparameter_tuner import tune_hyperparameters
from ..core.advanced_explainability import generate_shap_explanations, generate_lime_explanations, generate_feature_importance_analysis
from ..core.gemini_brain import gemini
from ..core.inference import predict_single, predict_batch, get_model_info
from ..core.model_versioning import save_model_version, list_models, load_model, delete_model, get_model_details
from ..core.logging_config import get_health_status
from ..core.task_queue import create_async_task, get_task_status, get_all_tasks, cancel_task, TaskType
import joblib
import os
import pandas as pd
from typing import Dict, List

router = APIRouter()

# Define tmp directory
TMP_DIR = "tmp"

@router.post("/upload-csv/")
async def upload_csv(file: UploadFile = File(...)):
    """Upload CSV file and trigger comprehensive analysis"""
    return handle_file(file)

@router.post("/upload-url/")
async def upload_url(url: str = Query(..., description="URL to the dataset file")):
    """Upload dataset from URL and trigger comprehensive analysis"""
    return handle_url_upload(url)

@router.post("/analyze-data/")
async def analyze_data_endpoint():
    """Analyze uploaded dataset"""
    return profile_data()

@router.post("/detect-target/")
async def detect_target_endpoint():
    """Detect target column in dataset"""
    return find_target()

@router.post("/classify-task/")
async def classify_task_endpoint():
    """Classify ML task type"""
    return classify_task()

@router.post("/suggest-model/")
async def suggest_model_endpoint():
    """Suggest appropriate model for the task"""
    return select_model()

@router.post("/train-model/")
async def train_model_endpoint():
    """Train the selected model"""
    return train_model()

@router.post("/evaluate-model/")
async def evaluate_model_endpoint():
    """Evaluate the trained model"""
    return evaluate()

@router.post("/explain-model/")
async def explain_model_endpoint():
    """Explain model decisions and performance"""
    return explain()

@router.post("/generate-code/")
async def generate_code_endpoint():
    """Generate code for the ML pipeline"""
    return generate_pipeline_code()

@router.post("/chat/")
async def chat_endpoint(query: str = Query(...)):
    """Chat with the AI assistant"""
    return chat(query)

@router.post("/benchmark-models/")
async def benchmark_models_endpoint():
    """Benchmark multiple models on the dataset"""
    try:
        result = benchmark_models()
        return result
    except Exception as e:
        return {"error": f"Benchmark failed: {str(e)}"}

@router.get("/benchmark-summary/")
async def benchmark_summary_endpoint():
    """Get a summary of benchmark results"""
    try:
        result = get_benchmark_summary()
        return result
    except Exception as e:
        return {"error": f"Failed to get benchmark summary: {str(e)}"}

@router.get("/compare-models/")
async def compare_models_endpoint():
    """Compare current model with benchmark results"""
    try:
        result = compare_with_current_model()
        return result
    except Exception as e:
        return {"error": f"Failed to compare models: {str(e)}"}

@router.post("/save-best-benchmark-model/")
async def save_best_benchmark_model_endpoint():
    """Save the best performing model from benchmark results"""
    try:
        result = save_best_benchmark_model()
        return result
    except Exception as e:
        return {"error": f"Failed to save best benchmark model: {str(e)}"}

@router.post("/auto-train-best-model/")
async def auto_train_best_model():
    """Automatically run benchmarking, save the best model, and load it into Gemini Brain"""
    try:
        # Run benchmark
        benchmark_result = benchmark_models()
        if "error" in benchmark_result:
            return {"error": benchmark_result["error"]}

        # Save best model
        success = save_best_benchmark_model()
        if "error" in success:
            return {"error": f"Benchmark succeeded, but saving best model failed: {success['error']}"}

        # Load the best model into Gemini Brain
        best_model_path = f"{TMP_DIR}/best_benchmark_model.joblib"
        if not os.path.exists(best_model_path):
            return {"error": "Best model file not found after saving"}
        
        # Load model into Gemini Brain
        load_result = gemini.load_model(best_model_path)
        if "error" in load_result:
            return {"error": f"Failed to load model: {load_result['error']}"}
        
        # Update Gemini Brain with benchmark results
        if isinstance(benchmark_result.get("all_results"), dict):
            gemini.training_results = benchmark_result.get("all_results", {})
        
        if gemini.model:
            gemini.model_params = gemini.model.get_params() if hasattr(gemini.model, 'get_params') else {}
        
        # Extract best model name from pipeline
        best_model_name = "Unknown"
        best_model = benchmark_result.get("best_model")
        if best_model and hasattr(best_model, 'steps'):
            try:
                # Get the model name from the pipeline
                model_step = best_model.steps[-1]
                best_model_name = model_step[0] if isinstance(model_step, tuple) else "Pipeline"
            except:
                best_model_name = "Pipeline"
        
        return {
            "status": "success",
            "message": f"Best model '{best_model_name}' loaded and ready for use",
            "best_score": benchmark_result.get("best_score"),
            "task_type": benchmark_result.get("task_type"),
            "best_model_name": best_model_name,
            "models_tested": len(benchmark_result.get("all_results", {})),
            "load_result": load_result
        }

    except Exception as e:
        return {"error": str(e)}

@router.get("/download-model/")
async def download_model():
    """Download the trained model"""
    try:
        # Check if model exists
        if not gemini.model:
            return {"error": "No model trained yet. Please train a model first."}
        
        # Save current model if not already saved
        model_path = f"{TMP_DIR}/current_model.joblib"
        joblib.dump(gemini.model, model_path)
        
        # Return the file
        return FileResponse(
            path=model_path,
            filename="trained_model.joblib",
            media_type="application/octet-stream"
        )
        
    except Exception as e:
        return {"error": f"Failed to download model: {str(e)}"}

@router.get("/download-best-benchmark-model/")
async def download_best_benchmark_model():
    """Download the best model from benchmark results"""
    try:
        # Check if best benchmark model exists
        best_model_path = f"{TMP_DIR}/best_benchmark_model.joblib"
        if not os.path.exists(best_model_path):
            return {"error": "No benchmark model found. Please run benchmarking first."}
        
        # Return the file
        return FileResponse(
            path=best_model_path,
            filename="best_benchmark_model.joblib",
            media_type="application/octet-stream"
        )
        
    except Exception as e:
        return {"error": f"Failed to download best benchmark model: {str(e)}"}

@router.post("/tune-hyperparameters/")
async def tune_hyperparameters_endpoint(
    search_type: str = Query("grid", description="Search type: 'grid' or 'random'"),
    cv_folds: int = Query(5, description="Number of cross-validation folds")
):
    """Perform hyperparameter tuning on the selected model"""
    return tune_hyperparameters(search_type, cv_folds)

@router.get("/shap-explanations/")
async def shap_explanations_endpoint(
    sample_index: int = Query(None, description="Sample index for individual explanation (optional)"),
    num_samples: int = Query(100, description="Number of samples for global explanation")
):
    """Generate SHAP explanations for model predictions"""
    return generate_shap_explanations(sample_index, num_samples)

@router.get("/lime-explanations/")
async def lime_explanations_endpoint(
    sample_index: int = Query(0, description="Sample index for explanation"),
    num_features: int = Query(10, description="Number of features to include in explanation")
):
    """Generate LIME explanations for model predictions"""
    return generate_lime_explanations(sample_index, num_features)

@router.get("/feature-importance/")
async def feature_importance_endpoint():
    """Generate comprehensive feature importance analysis"""
    return generate_feature_importance_analysis()

@router.get("/chat-history/")
async def chat_history_endpoint(num_messages: int = Query(10, description="Number of recent messages to retrieve")):
    """Get recent chat conversation history"""
    try:
        history = get_conversation_history(num_messages)
        return {
            "status": "success",
            "history": history,
            "total_messages": len(history)
        }
    except Exception as e:
        return {"error": f"Failed to retrieve chat history: {str(e)}"}

@router.post("/clear-chat-memory/")
async def clear_chat_memory_endpoint():
    """Clear chat conversation memory"""
    try:
        clear_memory()
        return {
            "status": "success",
            "message": "Chat memory has been cleared"
        }
    except Exception as e:
        return {"error": f"Failed to clear chat memory: {str(e)}"}

@router.post("/auto-train")
async def auto_train_model(model_name: str = "RandomForest"):
    """Auto-train a model with the specified name"""
    try:
        if not gemini.metadata.target_column:
            return {"error": "No target column set. Please upload a dataset first."}
        
        # Load the dataset
        df = pd.read_csv("tmp/dataset.csv")
        
        # Auto-train the model
        result = gemini.auto_train_model(df, model_name)
        
        if "error" in result:
            return result
        
        return {
            "status": "success",
            "message": f"Model '{model_name}' trained successfully!",
            "data": result
        }
        
    except Exception as e:
        return {"error": f"Auto-training failed: {str(e)}"}

@router.get("/suggest-models")
async def suggest_models():
    """Get model suggestions based on task type"""
    try:
        if not gemini.metadata.task_type:
            return {"error": "No task type detected. Please upload a dataset first."}
        
        models = gemini.suggest_models()
        
        return {
            "status": "success",
            "task_type": gemini.metadata.task_type,
            "suggested_models": models,
            "message": f"Suggested models for {gemini.metadata.task_type}"
        }
        
    except Exception as e:
        return {"error": f"Failed to suggest models: {str(e)}"}

@router.get("/evaluation-report")
async def get_evaluation_report():
    """Get the current evaluation report"""
    try:
        result = gemini.get_evaluation_report()
        return result
        
    except Exception as e:
        return {"error": f"Failed to get evaluation report: {str(e)}"}

@router.post("/generate-training-code")
async def generate_training_code():
    """Generate training code using Gemini LLM with existing API key configuration"""
    try:
        if not gemini.metadata.target_column:
            return {"error": "No dataset loaded. Please upload a dataset first."}
        
        # Load the dataset
        df = pd.read_csv("tmp/dataset.csv")
        
        # Generate code using LLM (no API key parameter needed)
        code = gemini.generate_training_code_llm(df)
        
        return {
            "status": "success",
            "message": "Training code generated successfully!",
            "code": code,
            "task_type": gemini.metadata.task_type,
            "target_column": gemini.metadata.target_column
        }
        
    except Exception as e:
        return {"error": f"Failed to generate training code: {str(e)}"}

@router.post("/train-with-suggestions")
async def train_with_suggestions():
    """Train models with all suggestions and return best one"""
    try:
        if not gemini.metadata.target_column:
            return {"error": "No target column set. Please upload a dataset first."}
        
        # Get model suggestions
        suggested_models = gemini.suggest_models()
        
        if not suggested_models:
            return {"error": "No models suggested for this task type."}
        
        # Load the dataset
        df = pd.read_csv("tmp/dataset.csv")
        
        results = []
        best_model = None
        best_score = -1
        
        # Train each suggested model
        for model_name in suggested_models[:3]:  # Limit to top 3 for speed
            try:
                result = gemini.auto_train_model(df, model_name)
                if "error" not in result:
                    results.append({
                        "model_name": model_name,
                        "test_score": result.get("test_score", 0),
                        "evaluation": result.get("evaluation", {})
                    })
                    
                    # Track best model
                    current_score = result.get("test_score", 0)
                    if current_score > best_score:
                        best_score = current_score
                        best_model = model_name
                        
            except Exception as e:
                results.append({
                    "model_name": model_name,
                    "error": str(e)
                })
        
        return {
            "status": "success",
            "message": f"Trained {len(results)} models successfully!",
            "best_model": best_model,
            "best_score": best_score,
            "all_results": results
        }
        
    except Exception as e:
        return {"error": f"Training with suggestions failed: {str(e)}"}

@router.post("/predict/")
async def predict_endpoint(data: Dict):
    """Make real-time predictions on input data"""
    try:
        result = predict_single(data)
        return result
    except Exception as e:
        return {"error": f"Prediction failed: {str(e)}"}

@router.post("/predict-batch/")
async def predict_batch_endpoint(data: List[Dict]):
    """Make predictions on batch of data"""
    try:
        result = predict_batch(data)
        return result
    except Exception as e:
        return {"error": f"Batch prediction failed: {str(e)}"}

@router.get("/model-info/")
async def model_info_endpoint():
    """Get information about the current model"""
    try:
        result = get_model_info()
        return result
    except Exception as e:
        return {"error": f"Failed to get model info: {str(e)}"}

@router.post("/save-model-version/")
async def save_model_version_endpoint(
    model_name: str = Query(..., description="Name for the model"),
    version: str = Query(None, description="Version string (auto-generated if not provided)"),
    description: str = Query("", description="Model description"),
    tags: str = Query("", description="Comma-separated tags")
):
    """Save current model as a new version"""
    try:
        tag_list = [tag.strip() for tag in tags.split(",")] if tags else []
        result = save_model_version(model_name, version, description, tag_list)
        return result
    except Exception as e:
        return {"error": f"Failed to save model version: {str(e)}"}

@router.get("/list-models/")
async def list_models_endpoint():
    """List all available model versions"""
    try:
        result = list_models()
        return result
    except Exception as e:
        return {"error": f"Failed to list models: {str(e)}"}

@router.post("/load-model/")
async def load_model_endpoint(model_id: int = Query(..., description="Model ID to load")):
    """Load a specific model version"""
    try:
        result = load_model(model_id)
        return result
    except Exception as e:
        return {"error": f"Failed to load model: {str(e)}"}

@router.delete("/delete-model/")
async def delete_model_endpoint(model_id: int = Query(..., description="Model ID to delete")):
    """Delete a specific model version"""
    try:
        result = delete_model(model_id)
        return result
    except Exception as e:
        return {"error": f"Failed to delete model: {str(e)}"}

@router.get("/model-details/")
async def model_details_endpoint(model_id: int = Query(..., description="Model ID to get details for")):
    """Get detailed information about a specific model"""
    try:
        result = get_model_details(model_id)
        return result
    except Exception as e:
        return {"error": f"Failed to get model details: {str(e)}"}

@router.get("/healthcheck/")
async def healthcheck_endpoint():
    """Get system health status"""
    try:
        result = get_health_status()
        return result
    except Exception as e:
        return {"error": f"Failed to get health status: {str(e)}"}

@router.get("/status/")
async def status_endpoint():
    """Get detailed system status"""
    try:
        result = get_health_status()
        return result
    except Exception as e:
        return {"error": f"Failed to get status: {str(e)}"}

@router.post("/async-train/")
async def async_train_endpoint(model_name: str = Query("RandomForest", description="Model name to train")):
    """Start async model training"""
    try:
        task_id = create_async_task(TaskType.TRAIN_MODEL, {"model_name": model_name})
        return {
            "status": "success",
            "task_id": task_id,
            "message": f"Training task started with ID: {task_id}"
        }
    except Exception as e:
        return {"error": f"Failed to start training task: {str(e)}"}

@router.post("/async-benchmark/")
async def async_benchmark_endpoint():
    """Start async model benchmarking"""
    try:
        task_id = create_async_task(TaskType.BENCHMARK_MODELS)
        return {
            "status": "success",
            "task_id": task_id,
            "message": f"Benchmark task started with ID: {task_id}"
        }
    except Exception as e:
        return {"error": f"Failed to start benchmark task: {str(e)}"}

@router.get("/task-status/")
async def task_status_endpoint(task_id: str = Query(..., description="Task ID to check")):
    """Get task status"""
    try:
        result = get_task_status(task_id)
        if result:
            return result
        else:
            return {"error": f"Task {task_id} not found"}
    except Exception as e:
        return {"error": f"Failed to get task status: {str(e)}"}

@router.get("/all-tasks/")
async def all_tasks_endpoint():
    """Get all tasks"""
    try:
        result = get_all_tasks()
        return result
    except Exception as e:
        return {"error": f"Failed to get tasks: {str(e)}"}

@router.delete("/cancel-task/")
async def cancel_task_endpoint(task_id: str = Query(..., description="Task ID to cancel")):
    """Cancel a task"""
    try:
        success = cancel_task(task_id)
        if success:
            return {
                "status": "success",
                "message": f"Task {task_id} cancelled successfully"
            }
        else:
            return {"error": f"Failed to cancel task {task_id}"}
    except Exception as e:
        return {"error": f"Failed to cancel task: {str(e)}"}
