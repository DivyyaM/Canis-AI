#!/usr/bin/env python3
"""
Test script for the new auto-train-best-model endpoint
Tests the complete workflow: benchmark -> save -> load into Gemini Brain
"""

import sys
import os
import pandas as pd
import numpy as np
import joblib
import asyncio
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestClassifier

# Add the app directory to the path
sys.path.append(os.path.join(os.path.dirname(__file__), 'app'))

def setup_test_data():
    """Create and save test data for the endpoint"""
    print("🔧 Setting up test data...")
    
    # Create classification dataset
    X, y = make_classification(
        n_samples=1000, n_features=10, n_informative=8, 
        n_redundant=2, n_classes=3, random_state=42
    )
    
    # Split data
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )
    
    # Create preprocessing pipeline
    preprocessor = StandardScaler()
    X_train_processed = preprocessor.fit_transform(X_train)
    X_test_processed = preprocessor.transform(X_test)
    
    # Create and fit a simple model
    model = RandomForestClassifier(n_estimators=50, random_state=42)
    model.fit(X_train_processed, y_train)
    
    # Create full pipeline
    full_pipeline = Pipeline([
        ('preprocessor', preprocessor),
        ('model', model)
    ])
    
    # Save everything
    os.makedirs('tmp', exist_ok=True)
    
    # Save the preprocessor
    joblib.dump(preprocessor, 'tmp/preprocessor.pkl')
    
    # Save the raw data for benchmarking
    joblib.dump((X_train, X_test, y_train, y_test), 'tmp/benchmark_data.pkl')
    
    print("✅ Test data setup complete")
    return full_pipeline

async def test_auto_train_endpoint():
    """Test the auto-train-best-model endpoint"""
    print("\n🚀 Testing Auto-Train-Best-Model Endpoint")
    print("=" * 50)
    
    try:
        # Setup test data
        pipeline = setup_test_data()
        
        # Import the endpoint function
        from app.api.routes import auto_train_best_model
        
        # Test the endpoint
        print("🔄 Testing auto_train_best_model endpoint...")
        result = await auto_train_best_model()
        
        if "error" in result:
            print(f"❌ Endpoint failed: {result['error']}")
            return False
        
        # Display results
        print("✅ Endpoint executed successfully!")
        print(f"   Status: {result.get('status')}")
        print(f"   Message: {result.get('message')}")
        print(f"   Best Score: {result.get('best_score')}")
        print(f"   Task Type: {result.get('task_type')}")
        print(f"   Best Model: {result.get('best_model_name')}")
        print(f"   Models Tested: {result.get('models_tested')}")
        
        # Check if model was loaded into Gemini Brain
        from app.core.gemini_brain import gemini
        if gemini.model:
            print(f"   Model loaded in Gemini Brain: {type(gemini.model).__name__}")
        else:
            print("   ❌ Model not loaded in Gemini Brain")
            return False
        
        # Check if training results were updated
        if gemini.training_results:
            print(f"   Training results updated: {len(gemini.training_results)} models")
        else:
            print("   ⚠️  Training results not updated")
        
        # Check if model file was created
        if os.path.exists('tmp/best_benchmark_model.joblib'):
            print("   ✅ Best model file created")
        else:
            print("   ❌ Best model file not found")
            return False
        
        return True
        
    except Exception as e:
        print(f"❌ Test failed: {str(e)}")
        import traceback
        traceback.print_exc()
        return False

def test_benchmark_integration():
    """Test that the benchmark system works correctly"""
    print("\n🔍 Testing Benchmark Integration")
    print("=" * 30)
    
    try:
        from app.core.benchmark_manager import BenchmarkManager
        
        # Create benchmark manager
        bm = BenchmarkManager('tmp')
        
        # Run benchmark
        print("🔄 Running benchmark...")
        results = bm.run_benchmark()
        
        if "error" in results:
            print(f"❌ Benchmark failed: {results['error']}")
            return False
        
        print("✅ Benchmark completed!")
        print(f"   Task Type: {results['task_type']}")
        print(f"   Best Score: {results['best_score']}")
        print(f"   Models Tested: {len(results['all_results'])}")
        
        # Check if any models succeeded
        successful_models = [name for name, result in results['all_results'].items() 
                           if "error" not in result]
        
        if successful_models:
            print(f"   Successful Models: {len(successful_models)}")
            for name in successful_models[:3]:  # Show top 3
                print(f"     - {name}")
        else:
            print("   ❌ No successful models")
            return False
        
        return True
        
    except Exception as e:
        print(f"❌ Benchmark test failed: {str(e)}")
        return False

def cleanup():
    """Clean up test files"""
    print("\n🧹 Cleaning up test files...")
    
    files_to_remove = [
        'tmp/preprocessor.pkl',
        'tmp/benchmark_data.pkl',
        'tmp/best_benchmark_model.joblib'
    ]
    
    for file_path in files_to_remove:
        if os.path.exists(file_path):
            os.remove(file_path)
            print(f"   Removed: {file_path}")
    
    # Reset Gemini Brain
    from app.core.gemini_brain import gemini
    gemini.reset()
    print("   Reset Gemini Brain")

async def main():
    """Main test function"""
    print("🧪 Auto-Train-Best-Model Endpoint Test Suite")
    print("=" * 50)
    
    success = True
    
    try:
        # Test benchmark integration first
        if not test_benchmark_integration():
            print("❌ Benchmark integration test failed")
            success = False
        
        # Test the auto-train endpoint
        if not await test_auto_train_endpoint():
            print("❌ Auto-train endpoint test failed")
            success = False
        
        if success:
            print("\n✅ All tests passed!")
        else:
            print("\n❌ Some tests failed")
        
    except Exception as e:
        print(f"\n❌ Test suite failed: {str(e)}")
        import traceback
        traceback.print_exc()
        success = False
    
    finally:
        # Cleanup
        cleanup()
        
    print(f"\n🎉 Test suite finished with {'success' if success else 'failure'}!")

if __name__ == "__main__":
    asyncio.run(main()) 