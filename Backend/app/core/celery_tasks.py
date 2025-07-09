"""
Celery Tasks for Canis AI Backend
Distributed task processing for ML operations
"""

import os
import pandas as pd
import joblib
from celery import current_task
from .celery_app import celery_app
from .gemini_brain import gemini
from .benchmark_manager import BenchmarkManager
import logging

logger = logging.getLogger(__name__)

@celery_app.task(bind=True)
def train_model_task(self, model_name: str = "RandomForest", dataset_path: str = "tmp/dataset.csv"):
    """Distributed model training task"""
    try:
        # Update task state
        self.update_state(state="PROGRESS", meta={"progress": 10, "message": "Loading dataset"})
        
        # Load dataset
        df = pd.read_csv(dataset_path)
        
        self.update_state(state="PROGRESS", meta={"progress": 30, "message": "Analyzing dataset"})
        
        # Analyze dataset
        gemini.reset()
        analysis_results = gemini.analyze_dataset(df)
        
        self.update_state(state="PROGRESS", meta={"progress": 50, "message": "Training model"})
        
        # Train model
        training_result = gemini.auto_train_model(df, model_name)
        
        if "error" in training_result:
            self.update_state(state="FAILURE", meta={"error": training_result["error"]})
            return training_result
        
        self.update_state(state="PROGRESS", meta={"progress": 90, "message": "Saving model"})
        
        # Save model
        model_path = f"tmp/{model_name}_trained.joblib"
        joblib.dump(gemini.model, model_path)
        
        self.update_state(state="SUCCESS", meta={"progress": 100, "message": "Training completed"})
        
        return {
            "status": "success",
            "model_name": model_name,
            "model_path": model_path,
            "training_results": training_result,
            "task_id": self.request.id
        }
        
    except Exception as e:
        logger.error(f"Training task failed: {str(e)}")
        self.update_state(state="FAILURE", meta={"error": str(e)})
        return {"error": str(e)}

@celery_app.task(bind=True)
def benchmark_models_task(self, dataset_path: str = "tmp/dataset.csv"):
    """Distributed model benchmarking task"""
    try:
        self.update_state(state="PROGRESS", meta={"progress": 10, "message": "Loading dataset"})
        
        # Load dataset
        df = pd.read_csv(dataset_path)
        
        self.update_state(state="PROGRESS", meta={"progress": 30, "message": "Preparing data"})
        
        # Prepare data for benchmarking
        gemini.reset()
        analysis_results = gemini.analyze_dataset(df)
        
        # Create benchmark data
        from .trainer import train_model
        train_result = train_model()
        
        if "error" in train_result:
            self.update_state(state="FAILURE", meta={"error": train_result["error"]})
            return train_result
        
        self.update_state(state="PROGRESS", meta={"progress": 60, "message": "Running benchmarks"})
        
        # Run benchmarks
        benchmark_manager = BenchmarkManager()
        benchmark_results = benchmark_manager.run_benchmark()
        
        if "error" in benchmark_results:
            self.update_state(state="FAILURE", meta={"error": benchmark_results["error"]})
            return benchmark_results
        
        self.update_state(state="PROGRESS", meta={"progress": 90, "message": "Saving results"})
        
        # Save best model
        best_model = benchmark_results.get("best_model")
        if best_model:
            best_model_path = "tmp/best_benchmark_model.joblib"
            joblib.dump(best_model, best_model_path)
            benchmark_results["best_model_path"] = best_model_path
        
        self.update_state(state="SUCCESS", meta={"progress": 100, "message": "Benchmarking completed"})
        
        return {
            "status": "success",
            "benchmark_results": benchmark_results,
            "task_id": self.request.id
        }
        
    except Exception as e:
        logger.error(f"Benchmarking task failed: {str(e)}")
        self.update_state(state="FAILURE", meta={"error": str(e)})
        return {"error": str(e)}

@celery_app.task(bind=True)
def hyperparameter_tuning_task(self, model_name: str, dataset_path: str = "tmp/dataset.csv"):
    """Distributed hyperparameter tuning task"""
    try:
        self.update_state(state="PROGRESS", meta={"progress": 10, "message": "Loading dataset"})
        
        # Load dataset
        df = pd.read_csv(dataset_path)
        
        self.update_state(state="PROGRESS", meta={"progress": 30, "message": "Preparing data"})
        
        # Prepare data
        gemini.reset()
        analysis_results = gemini.analyze_dataset(df)
        
        self.update_state(state="PROGRESS", meta={"progress": 60, "message": "Tuning hyperparameters"})
        
        # Run hyperparameter tuning
        from .hyperparameter_tuner import tune_hyperparameters
        tuning_results = tune_hyperparameters(search_type="grid", cv_folds=5)
        
        if "error" in tuning_results:
            self.update_state(state="FAILURE", meta={"error": tuning_results["error"]})
            return tuning_results
        
        self.update_state(state="PROGRESS", meta={"progress": 90, "message": "Saving tuned model"})
        
        # Save tuned model
        if gemini.model:
            tuned_model_path = f"tmp/{model_name}_tuned.joblib"
            joblib.dump(gemini.model, tuned_model_path)
            tuning_results["tuned_model_path"] = tuned_model_path
        
        self.update_state(state="SUCCESS", meta={"progress": 100, "message": "Hyperparameter tuning completed"})
        
        return {
            "status": "success",
            "model_name": model_name,
            "tuning_results": tuning_results,
            "task_id": self.request.id
        }
        
    except Exception as e:
        logger.error(f"Hyperparameter tuning task failed: {str(e)}")
        self.update_state(state="FAILURE", meta={"error": str(e)})
        return {"error": str(e)}

@celery_app.task(bind=True)
def data_processing_task(self, dataset_path: str, processing_config: dict):
    """Distributed data processing task"""
    try:
        self.update_state(state="PROGRESS", meta={"progress": 10, "message": "Loading data"})
        
        # Load dataset
        df = pd.read_csv(dataset_path)
        
        self.update_state(state="PROGRESS", meta={"progress": 30, "message": "Processing data"})
        
        # Apply processing configuration
        if processing_config.get("remove_duplicates"):
            df = df.drop_duplicates()
        
        if processing_config.get("handle_missing"):
            # Handle missing values
            numeric_cols = df.select_dtypes(include=['int64', 'float64']).columns
            df[numeric_cols] = df[numeric_cols].fillna(df[numeric_cols].median())
            
            categorical_cols = df.select_dtypes(include=['object']).columns
            df[categorical_cols] = df[categorical_cols].fillna(df[categorical_cols].mode().iloc[0])
        
        if processing_config.get("outlier_removal"):
            # Remove outliers using IQR method
            numeric_cols = df.select_dtypes(include=['int64', 'float64']).columns
            for col in numeric_cols:
                Q1 = df[col].quantile(0.25)
                Q3 = df[col].quantile(0.75)
                IQR = Q3 - Q1
                lower_bound = Q1 - 1.5 * IQR
                upper_bound = Q3 + 1.5 * IQR
                df = df[(df[col] >= lower_bound) & (df[col] <= upper_bound)]
        
        self.update_state(state="PROGRESS", meta={"progress": 80, "message": "Saving processed data"})
        
        # Save processed data
        processed_path = f"tmp/processed_dataset.csv"
        df.to_csv(processed_path, index=False)
        
        self.update_state(state="SUCCESS", meta={"progress": 100, "message": "Data processing completed"})
        
        return {
            "status": "success",
            "original_rows": len(pd.read_csv(dataset_path)),
            "processed_rows": len(df),
            "processed_path": processed_path,
            "processing_config": processing_config,
            "task_id": self.request.id
        }
        
    except Exception as e:
        logger.error(f"Data processing task failed: {str(e)}")
        self.update_state(state="FAILURE", meta={"error": str(e)})
        return {"error": str(e)}

@celery_app.task(bind=True)
def model_deployment_task(self, model_path: str, deployment_config: dict):
    """Distributed model deployment task"""
    try:
        self.update_state(state="PROGRESS", meta={"progress": 10, "message": "Loading model"})
        
        # Load model
        model = joblib.load(model_path)
        
        self.update_state(state="PROGRESS", meta={"progress": 30, "message": "Validating model"})
        
        # Model validation
        if not hasattr(model, 'predict'):
            raise ValueError("Model must have predict method")
        
        self.update_state(state="PROGRESS", meta={"progress": 60, "message": "Preparing deployment"})
        
        # Prepare deployment
        deployment_path = f"models/deployed/{os.path.basename(model_path)}"
        os.makedirs(os.path.dirname(deployment_path), exist_ok=True)
        
        # Copy model to deployment location
        import shutil
        shutil.copy2(model_path, deployment_path)
        
        self.update_state(state="PROGRESS", meta={"progress": 90, "message": "Updating deployment status"})
        
        # Update deployment status in model registry
        from .model_versioning import model_versioning
        model_versioning.update_deployment_status(model_path, "deployed")
        
        self.update_state(state="SUCCESS", meta={"progress": 100, "message": "Model deployed successfully"})
        
        return {
            "status": "success",
            "model_path": model_path,
            "deployment_path": deployment_path,
            "deployment_config": deployment_config,
            "task_id": self.request.id
        }
        
    except Exception as e:
        logger.error(f"Model deployment task failed: {str(e)}")
        self.update_state(state="FAILURE", meta={"error": str(e)})
        return {"error": str(e)}

@celery_app.task(bind=True)
def explainability_task(self, model_path: str, sample_data: dict):
    """Distributed explainability task"""
    try:
        self.update_state(state="PROGRESS", meta={"progress": 10, "message": "Loading model"})
        
        # Load model
        model = joblib.load(model_path)
        
        self.update_state(state="PROGRESS", meta={"progress": 30, "message": "Preparing data"})
        
        # Prepare sample data
        sample_df = pd.DataFrame([sample_data])
        
        self.update_state(state="PROGRESS", meta={"progress": 60, "message": "Generating explanations"})
        
        # Generate explanations
        from .explainability import explain
        explanations = explain()
        
        if "error" in explanations:
            self.update_state(state="FAILURE", meta={"error": explanations["error"]})
            return explanations
        
        self.update_state(state="SUCCESS", meta={"progress": 100, "message": "Explanations generated"})
        
        return {
            "status": "success",
            "model_path": model_path,
            "sample_data": sample_data,
            "explanations": explanations,
            "task_id": self.request.id
        }
        
    except Exception as e:
        logger.error(f"Explainability task failed: {str(e)}")
        self.update_state(state="FAILURE", meta={"error": str(e)})
        return {"error": str(e)}

@celery_app.task(bind=True)
def model_evaluation_task(self, model_path: str, test_data_path: str):
    """Distributed model evaluation task"""
    try:
        self.update_state(state="PROGRESS", meta={"progress": 10, "message": "Loading model and data"})
        
        # Load model and test data
        model = joblib.load(model_path)
        test_data = pd.read_csv(test_data_path)
        
        self.update_state(state="PROGRESS", meta={"progress": 40, "message": "Running evaluation"})
        
        # Run evaluation
        from .evaluator import evaluate
        evaluation_results = evaluate()
        
        if "error" in evaluation_results:
            self.update_state(state="FAILURE", meta={"error": evaluation_results["error"]})
            return evaluation_results
        
        self.update_state(state="SUCCESS", meta={"progress": 100, "message": "Evaluation completed"})
        
        return {
            "status": "success",
            "model_path": model_path,
            "test_data_path": test_data_path,
            "evaluation_results": evaluation_results,
            "task_id": self.request.id
        }
        
    except Exception as e:
        logger.error(f"Model evaluation task failed: {str(e)}")
        self.update_state(state="FAILURE", meta={"error": str(e)})
        return {"error": str(e)} 