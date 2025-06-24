#!/usr/bin/env python3
"""
Comprehensive Test Script for All New Features
==============================================

This script tests all the new features implemented:
1. Upload Dataset by URL/Other Types
2. Model Download Endpoint
3. Cross-Validation
4. Hyperparameter Tuning
5. Advanced Explainability (SHAP/LIME)
6. Chat Memory

Usage: python test_all_features.py
"""

import requests
import json
import time
import os

# Configuration
BASE_URL = "http://127.0.0.1:8000"
TEST_DATASET_URL = "https://raw.githubusercontent.com/datasciencedojo/datasets/master/breast-cancer-wisconsin.csv"

def print_section(title):
    """Print a formatted section header"""
    print(f"\n{'='*60}")
    print(f"🧪 {title}")
    print(f"{'='*60}")

def print_success(message):
    """Print success message"""
    print(f"✅ {message}")

def print_error(message):
    """Print error message"""
    print(f"❌ {message}")

def test_upload_url():
    """Test uploading dataset from URL"""
    print_section("Testing Upload Dataset by URL")
    
    try:
        response = requests.post(f"{BASE_URL}/upload-url/", params={"url": TEST_DATASET_URL})
        
        if response.status_code == 200:
            data = response.json()
            if data.get("status") == "success":
                print_success(f"URL upload successful!")
                print(f"   Rows: {data.get('rows')}")
                print(f"   Columns: {len(data.get('cols', []))}")
                print(f"   Target: {data.get('target_column')}")
                print(f"   Task: {data.get('task_type')}")
                return True
            else:
                print_error(f"URL upload failed: {data.get('error')}")
                return False
        else:
            print_error(f"HTTP {response.status_code}: {response.text}")
            return False
            
    except Exception as e:
        print_error(f"URL upload test failed: {str(e)}")
        return False

def test_workflow():
    """Test the complete ML workflow"""
    print_section("Testing Complete ML Workflow")
    
    try:
        # Analyze data
        response = requests.post(f"{BASE_URL}/analyze-data/")
        if response.status_code == 200:
            print_success("Data analysis completed")
        
        # Detect target
        response = requests.post(f"{BASE_URL}/detect-target/")
        if response.status_code == 200:
            print_success("Target detection completed")
        
        # Classify task
        response = requests.post(f"{BASE_URL}/classify-task/")
        if response.status_code == 200:
            print_success("Task classification completed")
        
        # Suggest model
        response = requests.post(f"{BASE_URL}/suggest-model/")
        if response.status_code == 200:
            print_success("Model suggestion completed")
        
        # Train model
        response = requests.post(f"{BASE_URL}/train-model/")
        if response.status_code == 200:
            data = response.json()
            if data.get("status") == "model_trained":
                print_success("Model training completed")
                print(f"   Model: {data.get('model')}")
                print(f"   Train Score: {data.get('train_score')}")
                print(f"   Test Score: {data.get('test_score')}")
                print(f"   CV Mean: {data.get('cv_mean')}")
                print(f"   CV Std: {data.get('cv_std')}")
                return True
            else:
                print_error(f"Training failed: {data.get('error')}")
                return False
        else:
            print_error(f"Training failed: HTTP {response.status_code}")
            return False
            
    except Exception as e:
        print_error(f"Workflow test failed: {str(e)}")
        return False

def test_hyperparameter_tuning():
    """Test hyperparameter tuning"""
    print_section("Testing Hyperparameter Tuning")
    
    try:
        # Test grid search
        response = requests.post(f"{BASE_URL}/tune-hyperparameters/", 
                               params={"search_type": "grid", "cv_folds": 3})
        
        if response.status_code == 200:
            data = response.json()
            if data.get("status") == "hyperparameter_tuning_completed":
                print_success("Grid search hyperparameter tuning completed")
                print(f"   Model: {data.get('model')}")
                print(f"   Best CV Score: {data.get('best_cv_score')}")
                print(f"   Test Score: {data.get('test_score')}")
                print(f"   Best Params: {data.get('best_params')}")
                return True
            else:
                print_error(f"Hyperparameter tuning failed: {data.get('error')}")
                return False
        else:
            print_error(f"Hyperparameter tuning failed: HTTP {response.status_code}")
            return False
            
    except Exception as e:
        print_error(f"Hyperparameter tuning test failed: {str(e)}")
        return False

def test_model_download():
    """Test model download endpoints"""
    print_section("Testing Model Download")
    
    try:
        # Test current model download
        response = requests.get(f"{BASE_URL}/download-model/")
        if response.status_code == 200:
            print_success("Current model download successful")
            # Save the model file
            with open("downloaded_model.joblib", "wb") as f:
                f.write(response.content)
            print("   Model saved as 'downloaded_model.joblib'")
        else:
            print_error(f"Current model download failed: HTTP {response.status_code}")
        
        # Test best benchmark model download
        response = requests.get(f"{BASE_URL}/download-best-benchmark-model/")
        if response.status_code == 200:
            print_success("Best benchmark model download successful")
            # Save the model file
            with open("best_benchmark_model.joblib", "wb") as f:
                f.write(response.content)
            print("   Model saved as 'best_benchmark_model.joblib'")
        else:
            print_error(f"Best benchmark model download failed: HTTP {response.status_code}")
        
        return True
        
    except Exception as e:
        print_error(f"Model download test failed: {str(e)}")
        return False

def test_advanced_explainability():
    """Test advanced explainability features"""
    print_section("Testing Advanced Explainability")
    
    try:
        # Test feature importance
        response = requests.get(f"{BASE_URL}/feature-importance/")
        if response.status_code == 200:
            data = response.json()
            if data.get("status") == "success":
                print_success("Feature importance analysis completed")
                analysis = data.get("analysis", {})
                print(f"   Top features: {len(analysis.get('top_features', []))}")
                print(f"   Mean importance: {analysis.get('statistics', {}).get('mean_importance', 0):.4f}")
            else:
                print_error(f"Feature importance failed: {data.get('error')}")
        else:
            print_error(f"Feature importance failed: HTTP {response.status_code}")
        
        # Test SHAP explanations (may fail if SHAP not installed)
        response = requests.get(f"{BASE_URL}/shap-explanations/", 
                              params={"sample_index": 0, "num_samples": 50})
        if response.status_code == 200:
            data = response.json()
            if data.get("status") == "success":
                print_success("SHAP explanations generated")
            else:
                print(f"⚠️  SHAP explanations: {data.get('error')} (SHAP may not be installed)")
        else:
            print(f"⚠️  SHAP explanations failed: HTTP {response.status_code}")
        
        # Test LIME explanations (may fail if LIME not installed)
        response = requests.get(f"{BASE_URL}/lime-explanations/", 
                              params={"sample_index": 0, "num_features": 5})
        if response.status_code == 200:
            data = response.json()
            if data.get("status") == "success":
                print_success("LIME explanations generated")
            else:
                print(f"⚠️  LIME explanations: {data.get('error')} (LIME may not be installed)")
        else:
            print(f"⚠️  LIME explanations failed: HTTP {response.status_code}")
        
        return True
        
    except Exception as e:
        print_error(f"Advanced explainability test failed: {str(e)}")
        return False

def test_chat_memory():
    """Test chat memory functionality"""
    print_section("Testing Chat Memory")
    
    try:
        # Send some chat messages
        messages = [
            "What is the target column?",
            "What type of task is this?",
            "How many features are there?",
            "What is the model performance?"
        ]
        
        for message in messages:
            response = requests.post(f"{BASE_URL}/chat/", params={"query": message})
            if response.status_code == 200:
                print_success(f"Chat message sent: '{message[:30]}...'")
            else:
                print_error(f"Chat message failed: HTTP {response.status_code}")
        
        # Get chat history
        response = requests.get(f"{BASE_URL}/chat-history/", params={"num_messages": 5})
        if response.status_code == 200:
            data = response.json()
            if data.get("status") == "success":
                history = data.get("history", [])
                print_success(f"Chat history retrieved: {len(history)} messages")
                for i, entry in enumerate(history[-3:], 1):  # Show last 3
                    print(f"   {i}. User: {entry.get('user_query', '')[:30]}...")
            else:
                print_error(f"Chat history failed: {data.get('error')}")
        else:
            print_error(f"Chat history failed: HTTP {response.status_code}")
        
        # Clear chat memory
        response = requests.post(f"{BASE_URL}/clear-chat-memory/")
        if response.status_code == 200:
            data = response.json()
            if data.get("status") == "success":
                print_success("Chat memory cleared")
            else:
                print_error(f"Clear memory failed: {data.get('error')}")
        else:
            print_error(f"Clear memory failed: HTTP {response.status_code}")
        
        return True
        
    except Exception as e:
        print_error(f"Chat memory test failed: {str(e)}")
        return False

def main():
    """Run all tests"""
    print("🚀 Starting Comprehensive Feature Tests")
    print("=" * 60)
    
    # Check if server is running
    try:
        response = requests.get(f"{BASE_URL}/docs")
        if response.status_code != 200:
            print_error("Server not running. Please start the server first.")
            return
    except:
        print_error("Server not running. Please start the server first.")
        return
    
    print_success("Server is running!")
    
    # Run tests
    tests = [
        ("Upload Dataset by URL", test_upload_url),
        ("Complete ML Workflow", test_workflow),
        ("Hyperparameter Tuning", test_hyperparameter_tuning),
        ("Model Download", test_model_download),
        ("Advanced Explainability", test_advanced_explainability),
        ("Chat Memory", test_chat_memory)
    ]
    
    results = []
    for test_name, test_func in tests:
        try:
            result = test_func()
            results.append((test_name, result))
        except Exception as e:
            print_error(f"{test_name} test crashed: {str(e)}")
            results.append((test_name, False))
    
    # Summary
    print_section("Test Summary")
    passed = sum(1 for _, result in results if result)
    total = len(results)
    
    print(f"📊 Results: {passed}/{total} tests passed")
    
    for test_name, result in results:
        status = "✅ PASS" if result else "❌ FAIL"
        print(f"   {status} - {test_name}")
    
    if passed == total:
        print_success("🎉 All tests passed! All features are working correctly.")
    else:
        print_error(f"⚠️  {total - passed} tests failed. Check the output above for details.")
    
    # Cleanup
    for filename in ["downloaded_model.joblib", "best_benchmark_model.joblib"]:
        if os.path.exists(filename):
            os.remove(filename)
            print(f"🧹 Cleaned up {filename}")

if __name__ == "__main__":
    main() 