from fastapi import APIRouter, UploadFile, File, Query, HTTPException, Depends
from fastapi.responses import FileResponse
from ..core.data_profile import profile_data, find_target, classify_task
from ..core.preprocessor import select_model
from ..core.trainer import train_model
from ..core.evaluator import evaluate
from ..core.explainability import explain

from ..core.gemini_brain import gemini
from ..core.inference import predict_single, predict_batch, get_model_info
from ..core.model_versioning import save_model_version, list_models, load_model, delete_model, get_model_details

from ..core.rbac import rbac_manager, get_current_user, Permission, Role
from ..core.celery_app import celery_app
from ..core.langgraph_orchestration import orchestrator, WorkflowTemplates
import joblib
import os
import pandas as pd
import requests
from urllib.parse import urlparse
from typing import Dict, List

router = APIRouter()

# Define tmp directory
TMP_DIR = "tmp"

# Authentication endpoints
@router.post("/auth/login")
async def login(username: str = Query(...), password: str = Query(...)):
    """Authenticate user and get access token"""
    try:
        user_data = rbac_manager.authenticate_user(username, password)
        if not user_data:
            raise HTTPException(status_code=401, detail="Invalid credentials")
        
        access_token = rbac_manager.create_access_token(user_data)
        return {
            "access_token": access_token,
            "token_type": "bearer",
            "user": user_data
        }
    except Exception as e:
        return {"error": str(e)}

@router.post("/auth/register")
async def register(
    username: str = Query(...),
    email: str = Query(...),
    password: str = Query(...),
    role: str = Query("data_scientist")
):
    """Register a new user"""
    try:
        result = rbac_manager.create_user(username, email, password, role)
        return result
    except Exception as e:
        return {"error": str(e)}

@router.get("/auth/me")
async def get_current_user_info(current_user: Dict = Depends(get_current_user)):
    """Get current user information"""
    return current_user

# File upload endpoints (with RBAC)
@router.post("/upload-csv/")
async def upload_csv(
    file: UploadFile = File(...),
    current_user: Dict = Depends(get_current_user)
):
    """Upload CSV file and trigger comprehensive analysis"""
    # Check permission
    if not rbac_manager.has_permission(current_user["role"], Permission.UPLOAD_DATA.value):
        raise HTTPException(status_code=403, detail="Permission denied: upload_data required")
    
    try:
        # Determine file type and read accordingly
        filename = file.filename.lower()
        if filename.endswith('.csv'):
            df = pd.read_csv(file.file)
        elif filename.endswith('.xlsx') or filename.endswith('.xls'):
            df = pd.read_excel(file.file)
        elif filename.endswith('.tsv') or filename.endswith('.txt'):
            df = pd.read_csv(file.file, sep='\t')
        elif filename.endswith('.json'):
            df = pd.read_json(file.file)
        else:
            df = pd.read_csv(file.file)
        df.to_csv(f"{TMP_DIR}/dataset.csv", index=False)
        gemini.reset()
        analysis_results = gemini.analyze_dataset(df)
        preprocessing_info = gemini.create_preprocessing_pipeline(df)
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

@router.post("/upload-url/")
async def upload_url(
    url: str = Query(..., description="URL to the dataset file"),
    current_user: Dict = Depends(get_current_user)
):
    """Upload dataset from URL and trigger comprehensive analysis"""
    # Check permission
    if not rbac_manager.has_permission(current_user["role"], Permission.UPLOAD_DATA.value):
        raise HTTPException(status_code=403, detail="Permission denied: upload_data required")
    
    try:
        parsed_url = urlparse(url)
        if not parsed_url.scheme or not parsed_url.netloc:
            return {"status": "error", "error": "Invalid URL provided"}
        response = requests.get(url, timeout=30)
        response.raise_for_status()
        filename = parsed_url.path.lower()
        import io
        file_content = io.BytesIO(response.content)
        if filename.endswith('.csv'):
            df = pd.read_csv(file_content)
        elif filename.endswith('.xlsx') or filename.endswith('.xls'):
            df = pd.read_excel(file_content)
        elif filename.endswith('.tsv') or filename.endswith('.txt'):
            df = pd.read_csv(file_content, sep='\t')
        elif filename.endswith('.json'):
            df = pd.read_json(file_content)
        else:
            df = pd.read_csv(file_content)
        df.to_csv(f"{TMP_DIR}/dataset.csv", index=False)
        gemini.reset()
        analysis_results = gemini.analyze_dataset(df)
        preprocessing_info = gemini.create_preprocessing_pipeline(df)
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

# Celery task endpoints
@router.post("/celery/train-model/")
async def celery_train_model(
    model_name: str = Query("RandomForest"),
    current_user: Dict = Depends(get_current_user)
):
    """Start distributed model training with Celery"""
    if not rbac_manager.has_permission(current_user["role"], Permission.TRAIN_MODEL.value):
        raise HTTPException(status_code=403, detail="Permission denied: train_model required")
    
    try:
        from .celery_tasks import train_model_task
        task = train_model_task.delay(model_name, "tmp/dataset.csv")
        return {
            "status": "success",
            "task_id": task.id,
            "message": f"Training task started with ID: {task.id}"
        }
    except Exception as e:
        return {"error": f"Failed to start training task: {str(e)}"}

@router.post("/celery/benchmark-models/")
async def celery_benchmark_models(current_user: Dict = Depends(get_current_user)):
    """Start distributed model benchmarking with Celery"""
    if not rbac_manager.has_permission(current_user["role"], Permission.RUN_BENCHMARK.value):
        raise HTTPException(status_code=403, detail="Permission denied: run_benchmark required")
    
    try:
        from .celery_tasks import benchmark_models_task
        task = benchmark_models_task.delay("tmp/dataset.csv")
        return {
            "status": "success",
            "task_id": task.id,
            "message": f"Benchmark task started with ID: {task.id}"
        }
    except Exception as e:
        return {"error": f"Failed to start benchmark task: {str(e)}"}

@router.get("/celery/task-status/{task_id}")
async def celery_task_status(task_id: str, current_user: Dict = Depends(get_current_user)):
    """Get Celery task status"""
    try:
        task = celery_app.AsyncResult(task_id)
        return {
            "task_id": task_id,
            "status": task.status,
            "result": task.result if task.ready() else None,
            "info": task.info if hasattr(task, 'info') else None
        }
    except Exception as e:
        return {"error": f"Failed to get task status: {str(e)}"}

# Workflow orchestration endpoints
@router.post("/workflow/create/")
async def create_workflow(
    workflow_id: str = Query(...),
    workflow_type: str = Query("basic_classification"),
    current_user: Dict = Depends(get_current_user)
):
    """Create a new workflow"""
    if not rbac_manager.has_permission(current_user["role"], Permission.TRAIN_MODEL.value):
        raise HTTPException(status_code=403, detail="Permission denied: train_model required")
    
    try:
        if workflow_type == "basic_classification":
            nodes = WorkflowTemplates.get_basic_classification_workflow()
        elif workflow_type == "advanced_mlops":
            nodes = WorkflowTemplates.get_advanced_mlops_workflow()
        else:
            return {"error": f"Unknown workflow type: {workflow_type}"}
        
        workflow_id = orchestrator.create_workflow(workflow_id, nodes)
        return {
            "status": "success",
            "workflow_id": workflow_id,
            "workflow_type": workflow_type,
            "nodes_count": len(nodes)
        }
    except Exception as e:
        return {"error": f"Failed to create workflow: {str(e)}"}

@router.post("/workflow/execute/{workflow_id}")
async def execute_workflow(
    workflow_id: str,
    initial_state: Dict = None,
    current_user: Dict = Depends(get_current_user)
):
    """Execute a workflow"""
    if not rbac_manager.has_permission(current_user["role"], Permission.TRAIN_MODEL.value):
        raise HTTPException(status_code=403, detail="Permission denied: train_model required")
    
    try:
        result = await orchestrator.execute_workflow(workflow_id, initial_state)
        return result
    except Exception as e:
        return {"error": f"Failed to execute workflow: {str(e)}"}

@router.get("/workflow/status/{workflow_id}")
async def get_workflow_status(workflow_id: str, current_user: Dict = Depends(get_current_user)):
    """Get workflow execution status"""
    try:
        if workflow_id in orchestrator.execution_history:
            history = orchestrator.execution_history[workflow_id]
            if history:
                latest_state = history[-1]
                return {
                    "workflow_id": workflow_id,
                    "status": latest_state.status,
                    "current_node": latest_state.current_node,
                    "completed_nodes": latest_state.completed_nodes,
                    "failed_nodes": latest_state.failed_nodes,
                    "results": latest_state.results
                }
        return {"error": f"Workflow {workflow_id} not found"}
    except Exception as e:
        return {"error": f"Failed to get workflow status: {str(e)}"}

# Existing endpoints with RBAC protection
@router.post("/analyze-data/")
async def analyze_data_endpoint(current_user: Dict = Depends(get_current_user)):
    """Analyze uploaded dataset"""
    if not rbac_manager.has_permission(current_user["role"], Permission.VIEW_DATA.value):
        raise HTTPException(status_code=403, detail="Permission denied: view_data required")
    return profile_data()

@router.post("/detect-target/")
async def detect_target_endpoint(current_user: Dict = Depends(get_current_user)):
    """Detect target column in dataset"""
    if not rbac_manager.has_permission(current_user["role"], Permission.VIEW_DATA.value):
        raise HTTPException(status_code=403, detail="Permission denied: view_data required")
    return find_target()

@router.post("/classify-task/")
async def classify_task_endpoint(current_user: Dict = Depends(get_current_user)):
    """Classify ML task type"""
    if not rbac_manager.has_permission(current_user["role"], Permission.VIEW_DATA.value):
        raise HTTPException(status_code=403, detail="Permission denied: view_data required")
    return classify_task()

@router.post("/suggest-model/")
async def suggest_model_endpoint(current_user: Dict = Depends(get_current_user)):
    """Suggest appropriate model for the task"""
    if not rbac_manager.has_permission(current_user["role"], Permission.TRAIN_MODEL.value):
        raise HTTPException(status_code=403, detail="Permission denied: train_model required")
    return select_model()

@router.post("/train-model/")
async def train_model_endpoint(current_user: Dict = Depends(get_current_user)):
    """Train the selected model"""
    if not rbac_manager.has_permission(current_user["role"], Permission.TRAIN_MODEL.value):
        raise HTTPException(status_code=403, detail="Permission denied: train_model required")
    return train_model()

@router.post("/evaluate-model/")
async def evaluate_model_endpoint(current_user: Dict = Depends(get_current_user)):
    """Evaluate the trained model"""
    if not rbac_manager.has_permission(current_user["role"], Permission.VIEW_MODEL.value):
        raise HTTPException(status_code=403, detail="Permission denied: view_model required")
    return evaluate()

@router.post("/explain-model/")
async def explain_model_endpoint(current_user: Dict = Depends(get_current_user)):
    """Explain model decisions and performance"""
    if not rbac_manager.has_permission(current_user["role"], Permission.VIEW_EXPLANATIONS.value):
        raise HTTPException(status_code=403, detail="Permission denied: view_explanations required")
    return explain()



@router.post("/chat/")
async def chat_endpoint(
    query: str = Query(...),
    current_user: Dict = Depends(get_current_user)
):
    """Chat with the AI assistant"""
    return gemini.chat(query)

@router.post("/benchmark-models/")
async def benchmark_models_endpoint(current_user: Dict = Depends(get_current_user)):
    """Benchmark multiple models on the dataset"""
    if not rbac_manager.has_permission(current_user["role"], Permission.RUN_BENCHMARK.value):
        raise HTTPException(status_code=403, detail="Permission denied: run_benchmark required")
    
    try:
        result = gemini.benchmark_models()
        return result
    except Exception as e:
        return {"error": f"Benchmark failed: {str(e)}"}

@router.get("/benchmark-summary/")
async def benchmark_summary_endpoint(current_user: Dict = Depends(get_current_user)):
    """Get a summary of benchmark results"""
    if not rbac_manager.has_permission(current_user["role"], Permission.VIEW_BENCHMARK.value):
        raise HTTPException(status_code=403, detail="Permission denied: view_benchmark required")
    
    try:
        result = gemini.get_benchmark_summary()
        return result
    except Exception as e:
        return {"error": f"Failed to get benchmark summary: {str(e)}"}

@router.get("/compare-models/")
async def compare_models_endpoint(current_user: Dict = Depends(get_current_user)):
    """Compare current model with benchmark results"""
    if not rbac_manager.has_permission(current_user["role"], Permission.VIEW_BENCHMARK.value):
        raise HTTPException(status_code=403, detail="Permission denied: view_benchmark required")
    
    try:
        result = gemini.compare_with_current_model()
        return result
    except Exception as e:
        return {"error": f"Failed to compare models: {str(e)}"}

@router.post("/save-best-benchmark-model/")
async def save_best_benchmark_model_endpoint(current_user: Dict = Depends(get_current_user)):
    """Save the best performing model from benchmark results"""
    if not rbac_manager.has_permission(current_user["role"], Permission.DEPLOY_MODEL.value):
        raise HTTPException(status_code=403, detail="Permission denied: deploy_model required")
    
    try:
        result = gemini.save_best_benchmark_model()
        return result
    except Exception as e:
        return {"error": f"Failed to save best benchmark model: {str(e)}"}

@router.post("/auto-train-best-model/")
async def auto_train_best_model(current_user: Dict = Depends(get_current_user)):
    """Automatically run benchmarking, save the best model, and load it into Gemini Brain"""
    if not rbac_manager.has_permission(current_user["role"], Permission.TRAIN_MODEL.value):
        raise HTTPException(status_code=403, detail="Permission denied: train_model required")

    try:
        # Run benchmark
        benchmark_result = gemini.benchmark_models()
        if "error" in benchmark_result:
            return {"error": benchmark_result["error"]}

        # Save best model
        success = gemini.save_best_benchmark_model()
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
async def download_model(current_user: Dict = Depends(get_current_user)):
    """Download the trained model"""
    if not rbac_manager.has_permission(current_user["role"], Permission.VIEW_MODEL.value):
        raise HTTPException(status_code=403, detail="Permission denied: view_model required")

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
async def download_best_benchmark_model(current_user: Dict = Depends(get_current_user)):
    """Download the best model from benchmark results"""
    if not rbac_manager.has_permission(current_user["role"], Permission.VIEW_MODEL.value):
        raise HTTPException(status_code=403, detail="Permission denied: view_model required")

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
    cv_folds: int = Query(5, description="Number of cross-validation folds"),
    current_user: Dict = Depends(get_current_user)
):
    """Perform hyperparameter tuning on the selected model"""
    if not rbac_manager.has_permission(current_user["role"], Permission.TRAIN_MODEL.value):
        raise HTTPException(status_code=403, detail="Permission denied: train_model required")
    return gemini.tune_hyperparameters(search_type, cv_folds)

@router.get("/shap-explanations/")
async def shap_explanations_endpoint(
    sample_index: int = Query(None, description="Sample index for individual explanation (optional)"),
    num_samples: int = Query(100, description="Number of samples for global explanation"),
    current_user: Dict = Depends(get_current_user)
):
    """Generate SHAP explanations for model predictions"""
    if not rbac_manager.has_permission(current_user["role"], Permission.VIEW_EXPLANATIONS.value):
        raise HTTPException(status_code=403, detail="Permission denied: view_explanations required")
    return gemini.generate_shap_explanations(sample_index, num_samples)

@router.get("/lime-explanations/")
async def lime_explanations_endpoint(
    sample_index: int = Query(0, description="Sample index for explanation"),
    num_features: int = Query(10, description="Number of features to include in explanation"),
    current_user: Dict = Depends(get_current_user)
):
    """Generate LIME explanations for model predictions"""
    if not rbac_manager.has_permission(current_user["role"], Permission.VIEW_EXPLANATIONS.value):
        raise HTTPException(status_code=403, detail="Permission denied: view_explanations required")
    return gemini.generate_lime_explanations(sample_index, num_features)

@router.get("/feature-importance/")
async def feature_importance_endpoint(current_user: Dict = Depends(get_current_user)):
    """Generate comprehensive feature importance analysis"""
    if not rbac_manager.has_permission(current_user["role"], Permission.VIEW_MODEL.value):
        raise HTTPException(status_code=403, detail="Permission denied: view_model required")
    return gemini.generate_feature_importance_analysis()

@router.get("/chat-history/")
async def chat_history_endpoint(num_messages: int = Query(10, description="Number of recent messages to retrieve"), current_user: Dict = Depends(get_current_user)):
    """Get recent chat conversation history"""
    if not rbac_manager.has_permission(current_user["role"], Permission.VIEW_CHAT.value):
        raise HTTPException(status_code=403, detail="Permission denied: view_chat required")
    try:
        history = gemini.get_conversation_history(num_messages)
        return {
            "status": "success",
            "history": history,
            "total_messages": len(history)
        }
    except Exception as e:
        return {"error": f"Failed to retrieve chat history: {str(e)}"}

@router.post("/clear-chat-memory/")
async def clear_chat_memory_endpoint(current_user: Dict = Depends(get_current_user)):
    """Clear chat conversation memory"""
    if not rbac_manager.has_permission(current_user["role"], Permission.VIEW_CHAT.value):
        raise HTTPException(status_code=403, detail="Permission denied: view_chat required")
    try:
        gemini.clear_memory()
        return {
            "status": "success",
            "message": "Chat memory has been cleared"
        }
    except Exception as e:
        return {"error": f"Failed to clear chat memory: {str(e)}"}

@router.post("/auto-train")
async def auto_train_model(
    model_name: str = Query("RandomForest"),
    current_user: Dict = Depends(get_current_user)
):
    """Auto-train a model with the specified name"""
    if not rbac_manager.has_permission(current_user["role"], Permission.TRAIN_MODEL.value):
        raise HTTPException(status_code=403, detail="Permission denied: train_model required")

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
async def suggest_models(current_user: Dict = Depends(get_current_user)):
    """Get model suggestions based on task type"""
    if not rbac_manager.has_permission(current_user["role"], Permission.TRAIN_MODEL.value):
        raise HTTPException(status_code=403, detail="Permission denied: train_model required")

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
async def get_evaluation_report(current_user: Dict = Depends(get_current_user)):
    """Get the current evaluation report"""
    if not rbac_manager.has_permission(current_user["role"], Permission.VIEW_MODEL.value):
        raise HTTPException(status_code=403, detail="Permission denied: view_model required")
    try:
        result = gemini.get_evaluation_report()
        return result
        
    except Exception as e:
        return {"error": f"Failed to get evaluation report: {str(e)}"}

@router.post("/generate-training-code")
async def generate_training_code(current_user: Dict = Depends(get_current_user)):
    """Generate training code using Gemini LLM with existing API key configuration"""
    if not rbac_manager.has_permission(current_user["role"], Permission.TRAIN_MODEL.value):
        raise HTTPException(status_code=403, detail="Permission denied: train_model required")

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
async def train_with_suggestions(current_user: Dict = Depends(get_current_user)):
    """Train models with all suggestions and return best one"""
    if not rbac_manager.has_permission(current_user["role"], Permission.TRAIN_MODEL.value):
        raise HTTPException(status_code=403, detail="Permission denied: train_model required")

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
async def predict_endpoint(
    data: Dict,
    current_user: Dict = Depends(get_current_user)
):
    """Make real-time predictions on input data"""
    if not rbac_manager.has_permission(current_user["role"], Permission.PREDICT_DATA.value):
        raise HTTPException(status_code=403, detail="Permission denied: predict_data required")
    try:
        result = predict_single(data)
        return result
    except Exception as e:
        return {"error": f"Prediction failed: {str(e)}"}

@router.post("/predict-batch/")
async def predict_batch_endpoint(
    data: List[Dict],
    current_user: Dict = Depends(get_current_user)
):
    """Make predictions on batch of data"""
    if not rbac_manager.has_permission(current_user["role"], Permission.PREDICT_DATA.value):
        raise HTTPException(status_code=403, detail="Permission denied: predict_data required")
    try:
        result = predict_batch(data)
        return result
    except Exception as e:
        return {"error": f"Batch prediction failed: {str(e)}"}

@router.get("/model-info/")
async def model_info_endpoint(current_user: Dict = Depends(get_current_user)):
    """Get information about the current model"""
    if not rbac_manager.has_permission(current_user["role"], Permission.VIEW_MODEL.value):
        raise HTTPException(status_code=403, detail="Permission denied: view_model required")
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
    tags: str = Query("", description="Comma-separated tags"),
    current_user: Dict = Depends(get_current_user)
):
    """Save current model as a new version"""
    if not rbac_manager.has_permission(current_user["role"], Permission.DEPLOY_MODEL.value):
        raise HTTPException(status_code=403, detail="Permission denied: deploy_model required")
    try:
        tag_list = [tag.strip() for tag in tags.split(",")] if tags else []
        result = save_model_version(model_name, version, description, tag_list)
        return result
    except Exception as e:
        return {"error": f"Failed to save model version: {str(e)}"}

@router.get("/list-models/")
async def list_models_endpoint(current_user: Dict = Depends(get_current_user)):
    """List all available model versions"""
    if not rbac_manager.has_permission(current_user["role"], Permission.VIEW_MODEL.value):
        raise HTTPException(status_code=403, detail="Permission denied: view_model required")
    try:
        result = list_models()
        return result
    except Exception as e:
        return {"error": f"Failed to list models: {str(e)}"}

@router.post("/load-model/")
async def load_model_endpoint(
    model_id: int = Query(..., description="Model ID to load"),
    current_user: Dict = Depends(get_current_user)
):
    """Load a specific model version"""
    if not rbac_manager.has_permission(current_user["role"], Permission.DEPLOY_MODEL.value):
        raise HTTPException(status_code=403, detail="Permission denied: deploy_model required")
    try:
        result = load_model(model_id)
        return result
    except Exception as e:
        return {"error": f"Failed to load model: {str(e)}"}

@router.delete("/delete-model/")
async def delete_model_endpoint(
    model_id: int = Query(..., description="Model ID to delete"),
    current_user: Dict = Depends(get_current_user)
):
    """Delete a specific model version"""
    if not rbac_manager.has_permission(current_user["role"], Permission.DEPLOY_MODEL.value):
        raise HTTPException(status_code=403, detail="Permission denied: deploy_model required")
    try:
        result = delete_model(model_id)
        return result
    except Exception as e:
        return {"error": f"Failed to delete model: {str(e)}"}

@router.get("/model-details/")
async def model_details_endpoint(
    model_id: int = Query(..., description="Model ID to get details for"),
    current_user: Dict = Depends(get_current_user)
):
    """Get detailed information about a specific model"""
    if not rbac_manager.has_permission(current_user["role"], Permission.VIEW_MODEL.value):
        raise HTTPException(status_code=403, detail="Permission denied: view_model required")
    try:
        result = get_model_details(model_id)
        return result
    except Exception as e:
        return {"error": f"Failed to get model details: {str(e)}"}

@router.get("/healthcheck/")
async def healthcheck_endpoint():
    """Get system health status"""
    try:
        result = gemini.get_health_status()
        return result
    except Exception as e:
        return {"error": f"Health check failed: {str(e)}"}

@router.get("/status/")
async def status_endpoint():
    """Get detailed system status"""
    try:
        result = gemini.get_health_status()
        return result
    except Exception as e:
        return {"error": f"Status check failed: {str(e)}"}

# Admin endpoints (admin only)
@router.get("/admin/users/")
async def list_users(current_user: Dict = Depends(get_current_user)):
    """List all users (admin only)"""
    if current_user["role"] != Role.ADMIN.value:
        raise HTTPException(status_code=403, detail="Admin access required")
    
    # Implementation would go here
    return {"message": "User list endpoint"}

@router.post("/admin/users/")
async def create_user_admin(
    username: str = Query(...),
    email: str = Query(...),
    password: str = Query(...),
    role: str = Query(...),
    current_user: Dict = Depends(get_current_user)
):
    """Create a new user (admin only)"""
    if current_user["role"] != Role.ADMIN.value:
        raise HTTPException(status_code=403, detail="Admin access required")
    
    try:
        result = rbac_manager.create_user(username, email, password, role)
        return result
    except Exception as e:
        return {"error": str(e)}
