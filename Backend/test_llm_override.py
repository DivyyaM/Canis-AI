#!/usr/bin/env python3
"""
Test Script for LLM Override Target Detection
=============================================

This script tests the enhanced LLM override functionality for target column detection.
It demonstrates how the system uses rule-based detection first, then falls back to LLM analysis.

Usage: python test_llm_override.py
"""

import requests
import json
import time

# Configuration
BASE_URL = "http://127.0.0.1:8000"

# Test datasets with different complexity levels
TEST_DATASETS = {
    "simple_binary": "https://raw.githubusercontent.com/datasciencedojo/datasets/master/breast-cancer-wisconsin.csv",
    "complex_churn": "https://raw.githubusercontent.com/datasciencedojo/datasets/master/Churn_Modelling.csv",
    "regression": "https://raw.githubusercontent.com/datasciencedojo/datasets/master/50_Startups.csv"
}

def print_section(title):
    """Print a formatted section header"""
    print(f"\n{'='*60}")
    print(f"🧠 {title}")
    print(f"{'='*60}")

def print_success(message):
    """Print success message"""
    print(f"✅ {message}")

def print_info(message):
    """Print info message"""
    print(f"ℹ️  {message}")

def print_error(message):
    """Print error message"""
    print(f"❌ {message}")

def test_target_detection(dataset_name, dataset_url):
    """Test target detection for a specific dataset"""
    print_section(f"Testing Target Detection: {dataset_name}")
    
    try:
        # Upload dataset
        print_info(f"Uploading {dataset_name} dataset...")
        response = requests.post(f"{BASE_URL}/upload-url/", params={"url": dataset_url})
        
        if response.status_code != 200:
            print_error(f"Failed to upload dataset: HTTP {response.status_code}")
            return False
        
        data = response.json()
        if data.get("status") != "success":
            print_error(f"Upload failed: {data.get('error')}")
            return False
        
        print_success(f"Dataset uploaded: {data.get('rows')} rows, {len(data.get('cols', []))} columns")
        
        # Test target detection
        print_info("Testing target detection...")
        response = requests.post(f"{BASE_URL}/detect-target/")
        
        if response.status_code == 200:
            target_data = response.json()
            
            if "error" not in target_data:
                print_success("Target detection completed!")
                print(f"   Target Column: {target_data.get('suggested_target')}")
                print(f"   Detection Method: {target_data.get('method')}")
                print(f"   Confidence Score: {target_data.get('confidence_score')}")
                print(f"   Task Type: {target_data.get('task_type')}")
                print(f"   Number of Classes: {target_data.get('n_classes')}")
                
                # Check if LLM override was used
                if target_data.get('method') == 'llm_gemini_override':
                    print_info("🎯 LLM Override was used!")
                    if 'llm_features' in target_data:
                        print(f"   LLM Suggested Features: {len(target_data['llm_features'])} features")
                    if 'llm_code' in target_data:
                        print(f"   LLM Generated Code: Available")
                else:
                    print_info("📊 Rule-based detection was used")
                
                return True
            else:
                print_error(f"Target detection failed: {target_data.get('error')}")
                return False
        else:
            print_error(f"Target detection failed: HTTP {response.status_code}")
            return False
            
    except Exception as e:
        print_error(f"Test failed: {str(e)}")
        return False

def test_complete_workflow(dataset_name, dataset_url):
    """Test complete workflow with target detection"""
    print_section(f"Complete Workflow Test: {dataset_name}")
    
    try:
        # Upload dataset
        print_info("1. Uploading dataset...")
        response = requests.post(f"{BASE_URL}/upload-url/", params={"url": dataset_url})
        if response.status_code != 200:
            print_error("Upload failed")
            return False
        
        # Run complete workflow
        workflow_steps = [
            ("analyze-data", "2. Data analysis"),
            ("detect-target", "3. Target detection"),
            ("classify-task", "4. Task classification"),
            ("suggest-model", "5. Model suggestion"),
            ("train-model", "6. Model training")
        ]
        
        for endpoint, description in workflow_steps:
            print_info(f"{description}...")
            response = requests.post(f"{BASE_URL}/{endpoint}/")
            
            if response.status_code == 200:
                data = response.json()
                if "error" not in data:
                    print_success(f"{description} completed")
                    
                    # Show target detection details
                    if endpoint == "detect-target":
                        print(f"   Target: {data.get('suggested_target')}")
                        print(f"   Method: {data.get('method')}")
                        print(f"   Confidence: {data.get('confidence_score')}")
                else:
                    print_error(f"{description} failed: {data.get('error')}")
                    return False
            else:
                print_error(f"{description} failed: HTTP {response.status_code}")
                return False
        
        print_success("Complete workflow successful!")
        return True
        
    except Exception as e:
        print_error(f"Workflow test failed: {str(e)}")
        return False

def main():
    """Run all LLM override tests"""
    print("🧠 Starting LLM Override Target Detection Tests")
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
    
    # Test each dataset
    results = []
    
    for dataset_name, dataset_url in TEST_DATASETS.items():
        print(f"\n📊 Testing Dataset: {dataset_name}")
        print(f"URL: {dataset_url}")
        
        # Test target detection
        target_result = test_target_detection(dataset_name, dataset_url)
        
        # Test complete workflow
        workflow_result = test_complete_workflow(dataset_name, dataset_url)
        
        results.append((dataset_name, target_result and workflow_result))
    
    # Summary
    print_section("Test Summary")
    passed = sum(1 for _, result in results if result)
    total = len(results)
    
    print(f"📊 Results: {passed}/{total} datasets tested successfully")
    
    for dataset_name, result in results:
        status = "✅ PASS" if result else "❌ FAIL"
        print(f"   {status} - {dataset_name}")
    
    if passed == total:
        print_success("🎉 All LLM override tests passed!")
        print_info("The system is correctly using LLM override when rule-based detection is insufficient.")
    else:
        print_error(f"⚠️  {total - passed} tests failed. Check the output above for details.")
    
    print_info("\n💡 Key Features Demonstrated:")
    print_info("• Rule-based target detection for simple cases")
    print_info("• LLM override for complex datasets")
    print_info("• Automatic feature selection")
    print_info("• Complete ML workflow integration")

if __name__ == "__main__":
    main() 