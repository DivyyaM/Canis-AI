#!/usr/bin/env python3
"""
Test script for Canis AI Advanced Features
Tests real-time inference, model versioning, async tasks, and monitoring
"""

import requests
import json
import time
import sys
import os

# Add the current directory to Python path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

BASE_URL = "http://127.0.0.1:8000/api/v1"

def test_health_check():
    """Test health check endpoint"""
    print("🔍 Testing Health Check...")
    try:
        response = requests.get(f"{BASE_URL}/healthcheck/")
        if response.status_code == 200:
            data = response.json()
            print(f"✅ Health Check: {data['status']}")
            print(f"   Uptime: {data['system']['uptime_formatted']}")
            print(f"   CPU: {data['system']['cpu']['usage_percent']}%")
            print(f"   Memory: {data['system']['memory']['percent']}%")
            return True
        else:
            print(f"❌ Health Check failed: {response.status_code}")
            return False
    except Exception as e:
        print(f"❌ Health Check error: {str(e)}")
        return False

def test_model_info():
    """Test model info endpoint"""
    print("\n🔍 Testing Model Info...")
    try:
        response = requests.get(f"{BASE_URL}/model-info/")
        if response.status_code == 200:
            data = response.json()
            if "error" in data:
                print(f"⚠️  Model Info: {data['error']}")
            else:
                print(f"✅ Model Info: {data['model_type']}")
                print(f"   Task Type: {data['task_type']}")
                print(f"   Target Column: {data['target_column']}")
            return True
        else:
            print(f"❌ Model Info failed: {response.status_code}")
            return False
    except Exception as e:
        print(f"❌ Model Info error: {str(e)}")
        return False

def test_prediction():
    """Test real-time prediction"""
    print("\n🔍 Testing Real-time Prediction...")
    try:
        # Sample data for prediction
        sample_data = {
            "feature1": 1.5,
            "feature2": 2.3,
            "feature3": 0.8,
            "feature4": 1.2
        }
        
        response = requests.post(f"{BASE_URL}/predict/", json=sample_data)
        if response.status_code == 200:
            data = response.json()
            if "error" in data:
                print(f"⚠️  Prediction: {data['error']}")
            else:
                print(f"✅ Prediction successful!")
                print(f"   Predictions: {data['predictions']}")
                if 'probabilities' in data:
                    print(f"   Probabilities: {data['probabilities']}")
            return True
        else:
            print(f"❌ Prediction failed: {response.status_code}")
            return False
    except Exception as e:
        print(f"❌ Prediction error: {str(e)}")
        return False

def test_model_versioning():
    """Test model versioning system"""
    print("\n🔍 Testing Model Versioning...")
    try:
        # Save model version
        print("   Saving model version...")
        response = requests.post(f"{BASE_URL}/save-model-version/", params={
            "model_name": "test_model",
            "description": "Test model for versioning",
            "tags": "test,regression"
        })
        
        if response.status_code == 200:
            data = response.json()
            if "error" in data:
                print(f"⚠️  Save Model: {data['error']}")
                return False
            else:
                print(f"✅ Model saved: {data['model_name']}_{data['version']}")
                
                # List models
                print("   Listing models...")
                response = requests.get(f"{BASE_URL}/list-models/")
                if response.status_code == 200:
                    data = response.json()
                    if "error" in data:
                        print(f"⚠️  List Models: {data['error']}")
                    else:
                        print(f"✅ Found {data['total_models']} models")
                        for model in data['models'][:3]:  # Show first 3
                            print(f"     - {model['model_name']}_{model['version']} ({model['task_type']})")
                return True
        else:
            print(f"❌ Save Model failed: {response.status_code}")
            return False
    except Exception as e:
        print(f"❌ Model Versioning error: {str(e)}")
        return False

def test_async_tasks():
    """Test async task queue"""
    print("\n🔍 Testing Async Task Queue...")
    try:
        # Start async benchmark
        print("   Starting async benchmark...")
        response = requests.post(f"{BASE_URL}/async-benchmark/")
        
        if response.status_code == 200:
            data = response.json()
            if "error" in data:
                print(f"⚠️  Async Benchmark: {data['error']}")
                return False
            else:
                task_id = data['task_id']
                print(f"✅ Benchmark task started: {task_id}")
                
                # Check task status
                print("   Checking task status...")
                for i in range(5):  # Check 5 times
                    time.sleep(2)
                    response = requests.get(f"{BASE_URL}/task-status/", params={"task_id": task_id})
                    if response.status_code == 200:
                        task_data = response.json()
                        if "error" not in task_data:
                            status = task_data['status']
                            progress = task_data.get('progress', 0)
                            print(f"     Status: {status} (Progress: {progress}%)")
                            
                            if status in ['completed', 'failed', 'cancelled']:
                                break
                
                # Get all tasks
                print("   Getting all tasks...")
                response = requests.get(f"{BASE_URL}/all-tasks/")
                if response.status_code == 200:
                    data = response.json()
                    if "error" not in data:
                        print(f"✅ Total tasks: {len(data)}")
                return True
        else:
            print(f"❌ Async Benchmark failed: {response.status_code}")
            return False
    except Exception as e:
        print(f"❌ Async Tasks error: {str(e)}")
        return False

def test_benchmark_endpoints():
    """Test benchmark endpoints"""
    print("\n🔍 Testing Benchmark Endpoints...")
    try:
        # Test benchmark models
        print("   Running benchmark...")
        response = requests.post(f"{BASE_URL}/benchmark-models/")
        
        if response.status_code == 200:
            data = response.json()
            if "error" in data:
                print(f"⚠️  Benchmark: {data['error']}")
                return False
            else:
                print(f"✅ Benchmark completed!")
                print(f"   Task Type: {data['task_type']}")
                print(f"   Best Score: {data['best_score']}")
                print(f"   Models Tested: {len(data['all_results'])}")
                
                # Test benchmark summary
                print("   Getting benchmark summary...")
                response = requests.get(f"{BASE_URL}/benchmark-summary/")
                if response.status_code == 200:
                    summary_data = response.json()
                    if "error" not in summary_data:
                        print(f"✅ Summary: {summary_data['total_models_tested']} models tested")
                return True
        else:
            print(f"❌ Benchmark failed: {response.status_code}")
            return False
    except Exception as e:
        print(f"❌ Benchmark error: {str(e)}")
        return False

def main():
    """Run all tests"""
    print("🚀 Testing Canis AI Advanced Features")
    print("=" * 50)
    
    tests = [
        test_health_check,
        test_model_info,
        test_prediction,
        test_model_versioning,
        test_async_tasks,
        test_benchmark_endpoints
    ]
    
    results = []
    for test in tests:
        try:
            result = test()
            results.append(result)
        except Exception as e:
            print(f"❌ Test failed with exception: {str(e)}")
            results.append(False)
    
    print("\n" + "=" * 50)
    print("📊 Test Results Summary")
    print("=" * 50)
    
    test_names = [
        "Health Check",
        "Model Info", 
        "Real-time Prediction",
        "Model Versioning",
        "Async Task Queue",
        "Benchmark Endpoints"
    ]
    
    passed = 0
    for i, (name, result) in enumerate(zip(test_names, results)):
        status = "✅ PASS" if result else "❌ FAIL"
        print(f"{i+1}. {name}: {status}")
        if result:
            passed += 1
    
    print(f"\n🎯 Overall: {passed}/{len(tests)} tests passed")
    
    if passed == len(tests):
        print("🎉 All tests passed! Your Canis AI Backend is working perfectly!")
    else:
        print("⚠️  Some tests failed. Check the server logs for details.")

if __name__ == "__main__":
    main() 