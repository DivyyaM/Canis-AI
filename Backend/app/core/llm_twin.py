import os
import httpx
import json
from dotenv import load_dotenv
import pandas as pd
import numpy as np
from collections import deque
from .gemini_brain import gemini

load_dotenv()

GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
GEMINI_URL = f"https://generativelanguage.googleapis.com/v1beta/models/gemini-2.0-flash:generateContent?key={GEMINI_API_KEY}"

# Chat memory storage (in-memory for now, can be extended to persistent storage)
chat_memory = deque(maxlen=50)  # Keep last 50 messages
session_context = {}

def add_to_memory(user_query: str, ai_response: str, context: dict = None):
    """Add conversation to memory"""
    memory_entry = {
        "user_query": user_query,
        "ai_response": ai_response,
        "context": context or {},
        "timestamp": pd.Timestamp.now().isoformat()
    }
    chat_memory.append(memory_entry)

def get_conversation_history(num_messages: int = 5):
    """Get recent conversation history"""
    return list(chat_memory)[-num_messages:] if chat_memory else []

def clear_memory():
    """Clear chat memory"""
    chat_memory.clear()
    session_context.clear()

def answer_query(query: str):
    """Answer user queries using Gemini Brain context"""
    try:
        # Get context from Gemini Brain
        context = gemini.get_context_for_chat()
        
        # Convert query to lowercase for easier matching
        query_lower = query.lower()
        
        # Target column questions
        if any(word in query_lower for word in ["target", "label", "y", "output", "prediction"]):
            if context["target_column"]:
                return {
                    "response": f"The target column is '{context['target_column']}' with {context['n_classes']} unique values. "
                               f"This was detected with {context['confidence_score']:.1f}% confidence."
                }
            else:
                return {"response": "No target column has been detected yet. Please upload a dataset first."}
        
        # Task type questions
        elif any(word in query_lower for word in ["task", "problem", "type", "classification", "regression"]):
            if context["task_type"]:
                return {
                    "response": f"This is a {context['task_type']} task. {context['task_reason']}. "
                               f"The target column '{context['target_column']}' has {context['n_classes']} classes."
                }
            else:
                return {"response": "Task type hasn't been determined yet. Please upload a dataset first."}
        
        # Model questions
        elif any(word in query_lower for word in ["model", "algorithm", "which model", "what model"]):
            if context["model"]:
                return {
                    "response": f"The current model is {context['model']}. "
                               f"Training score: {context['training_score']:.4f} if available."
                }
            else:
                return {"response": "No model has been trained yet. Please run the training step first."}
        
        # Dataset questions
        elif any(word in query_lower for word in ["dataset", "data", "rows", "columns", "features"]):
            return {
                "response": f"Dataset has {context['n_samples']} samples and {context['n_features']} features. "
                           f"Features: {', '.join(context['feature_columns'])}. "
                           f"Numeric features: {', '.join(context['numeric_features'])}. "
                           f"Categorical features: {', '.join(context['categorical_features'])}."
            }
        
        # Performance questions
        elif any(word in query_lower for word in ["accuracy", "score", "performance", "how well", "results"]):
            if context["training_score"]:
                return {
                    "response": f"Model performance: {context['training_score']:.4f} test score. "
                               f"Evaluation metrics: {context['evaluation_metrics']}."
                }
            else:
                return {"response": "No performance metrics available yet. Please train and evaluate the model first."}
        
        # General help
        elif any(word in query_lower for word in ["help", "what can", "how to", "guide"]):
            return {
                "response": "I can help you with:\n"
                           "• Target column detection\n"
                           "• Task classification (regression/classification)\n"
                           "• Model selection and training\n"
                           "• Dataset analysis\n"
                           "• Performance evaluation\n"
                           "Just ask me about any of these topics!"
            }
        
        # Default response
        else:
            return {
                "response": f"I'm your AI assistant. You asked: {query}\n\n"
                           f"Current context: {context['task_type']} task with {context['n_samples']} samples. "
                           f"Ask me about the target column, model, or dataset!"
            }
        
    except Exception as e:
        return {"response": f"Sorry, I encountered an error: {str(e)}"}

def chat(query: str):
    """Intelligent chat using Gemini Brain context"""
    try:
        # Get current context from Gemini Brain
        metadata = gemini.get_metadata()
        
        # Check if we have any context
        if not metadata:
            return {
                "response": f"I'm your AI assistant. You asked: {query}\n\nNo dataset loaded yet. Please upload a CSV file to get started!"
            }
        
        # Extract key information
        target_col = metadata.get("target_column", "unknown")
        task_type = metadata.get("task_type", "unknown")
        n_samples = metadata.get("n_samples", 0)
        n_features = metadata.get("n_features", 0)
        model_name = getattr(gemini.model, '__class__', None)
        model_name = model_name.__name__ if model_name else "None"
        
        # Create context-aware response
        context_info = f"Current context: {task_type} task with {n_samples} samples, {n_features} features. Target: {target_col}"
        
        # Handle specific query types
        query_lower = query.lower()
        
        if "target" in query_lower or "label" in query_lower:
            response = f"The target column is '{target_col}' with {metadata.get('n_classes', 'unknown')} classes."
            
        elif "task" in query_lower or "type" in query_lower:
            response = f"This is a {task_type} task. {metadata.get('task_reason', '')}"
            
        elif "model" in query_lower:
            if model_name != "None":
                response = f"Current model: {model_name}. {get_model_info(gemini.model_params, metadata)}"
            else:
                response = "No model trained yet. Use /train-model/ to train a model."
                
        elif "features" in query_lower or "columns" in query_lower:
            numeric_features = metadata.get("numeric_features", [])
            categorical_features = metadata.get("categorical_features", [])
            response = f"Features: {len(numeric_features)} numeric ({', '.join(numeric_features)}) and {len(categorical_features)} categorical ({', '.join(categorical_features)})"
            
        elif "performance" in query_lower or "accuracy" in query_lower or "score" in query_lower:
            if hasattr(gemini, 'training_results') and gemini.training_results:
                train_score = gemini.training_results.get("train_score", 0)
                test_score = gemini.training_results.get("test_score", 0)
                response = f"Model performance - Training: {train_score:.3f}, Test: {test_score:.3f}"
            elif hasattr(gemini, 'evaluation_results') and gemini.evaluation_results:
                eval_results = gemini.evaluation_results
                if "accuracy" in eval_results:
                    response = f"Model accuracy: {eval_results['accuracy']:.3f}"
                elif "r2_score" in eval_results:
                    response = f"Model R² score: {eval_results['r2_score']:.3f}"
                else:
                    response = "Performance metrics available but not displayed."
            else:
                response = "No performance metrics available. Train or evaluate the model first."
                
        elif "help" in query_lower or "what can you do" in query_lower:
            response = """I can help you with:
- Dataset analysis and target detection
- Task classification (classification, regression, clustering)
- Model selection and training
- Performance evaluation and explanation
- Code generation for your ML pipeline
- Feature importance and model interpretation

Just ask me about your data, model, or any ML-related questions!"""
            
        elif "hello" in query_lower or "hi" in query_lower:
            response = f"Hello! I'm your AI assistant for the {task_type} task. How can I help you today?"
            
        else:
            # Generic response with context
            response = f"You asked: {query}\n\n{context_info}\n\nAsk me about the target column, model, features, or performance!"
        
        return {
            "response": f"{response}\n\n{context_info}",
            "context": {
                "task_type": task_type,
                "target_column": target_col,
                "n_samples": n_samples,
                "n_features": n_features,
                "model": model_name
            }
        }
        
    except Exception as e:
        return {
            "response": f"I'm your AI assistant. You asked: {query}\n\nError: {str(e)}",
            "error": str(e)
        }

def get_model_info(model_params, metadata):
    """Get information about the current model"""
    if not model_params:
        return "No specific parameters set."
    
    param_info = []
    for param, value in model_params.items():
        param_info.append(f"{param}={value}")
    
    return f"Parameters: {', '.join(param_info)}"
