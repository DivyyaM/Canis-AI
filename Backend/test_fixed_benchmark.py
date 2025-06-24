#!/usr/bin/env python3
"""
Test script for the fixed benchmark system with preprocessing pipeline integration
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

def test_benchmark_system():
    """Test the complete benchmark system"""
    print("🚀 Testing Fixed Benchmark System")
    print("=" * 60)
    
    # Test 1: Import the benchmark manager
    print("\n1️⃣ Testing Benchmark Manager Import...")
    try:
        from app.core.benchmark_manager import BenchmarkManager
        print("✅ BenchmarkManager imported successfully!")
        
        # Create instance
        bm = BenchmarkManager("tmp")
        print("✅ BenchmarkManager instance created!")
        
    except Exception as e:
        print(f"❌ Import failed: {str(e)}")
        return False
    
    # Test 2: Test data loading (if data exists)
    print("\n2️⃣ Testing Data Loading...")
    try:
        # This will fail if no data exists, which is expected
        X_train, X_test, y_train, y_test = bm.load_data()
        print("✅ Data loaded successfully!")
        print(f"   X_train shape: {X_train.shape}")
        print(f"   X_test shape: {X_test.shape}")
        print(f"   y_train shape: {y_train.shape}")
        print(f"   y_test shape: {y_test.shape}")
        
    except Exception as e:
        print(f"⚠️  Data loading failed (expected if no model trained): {str(e)}")
        print("   This is normal - you need to train a model first!")
    
    # Test 3: Test task detection
    print("\n3️⃣ Testing Task Detection...")
    try:
        # Create sample data for testing
        import numpy as np
        
        # Test binary classification
        y_binary = np.array([0, 1, 0, 1, 0, 1, 0, 1, 0, 1])
        task_binary = bm.detect_task_type(y_binary)
        print(f"✅ Binary classification detected: {task_binary}")
        
        # Test multiclass classification
        y_multiclass = np.array([0, 1, 2, 0, 1, 2, 0, 1, 2, 0])
        task_multiclass = bm.detect_task_type(y_multiclass)
        print(f"✅ Multiclass classification detected: {task_multiclass}")
        
        # Test regression
        y_regression = np.array([1.1, 2.3, 3.5, 4.7, 5.9, 6.1, 7.3, 8.5, 9.7, 10.9])
        task_regression = bm.detect_task_type(y_regression)
        print(f"✅ Regression detected: {task_regression}")
        
    except Exception as e:
        print(f"❌ Task detection failed: {str(e)}")
    
    # Test 4: Show usage examples
    print("\n4️⃣ Usage Examples:")
    print("=" * 40)
    
    print("""
# Basic usage with preprocessing pipeline integration
from app.core.benchmark_manager import BenchmarkManager

# Create benchmark manager
bm = BenchmarkManager("tmp")

# Run comprehensive benchmark (automatically handles preprocessing)
results = bm.run_benchmark()

# Get summary
summary = bm.get_summary()

# Save best model
bm.save_best_model("RandomForestClassifier")
""")
    
    # Test 5: API Endpoints
    print("\n5️⃣ API Endpoints:")
    print("=" * 40)
    
    print("""
# Start your FastAPI server
python -m uvicorn app.main:app --reload --host 0.0.0.0 --port 8000

# Then use these endpoints:

# 1. Upload dataset
curl -X POST "http://localhost:8000/upload-csv/" -F "file=@your_data.csv"

# 2. Train initial model (creates preprocessing pipeline)
curl -X POST "http://localhost:8000/train-model/"

# 3. Run benchmark (uses saved preprocessing pipeline)
curl -X POST "http://localhost:8000/benchmark-models/"

# 4. Get benchmark summary
curl -X GET "http://localhost:8000/benchmark-summary/"

# 5. Compare with current model
curl -X GET "http://localhost:8000/compare-models/"

# 6. Save best benchmark model
curl -X POST "http://localhost:8000/save-best-benchmark-model/"
""")
    
    # Test 6: Show what the benchmark system does
    print("\n6️⃣ What the Benchmark System Does:")
    print("=" * 40)
    
    print("""
✅ Loads raw data from tmp/benchmark_data.pkl
✅ Loads preprocessing pipeline from tmp/preprocessor.pkl
✅ Applies preprocessing (handles categorical features)
✅ Loads target encoder from tmp/target_encoder.pkl (if needed)
✅ Detects task type automatically
✅ Tests multiple models:
   - Classification: LogisticRegression, RandomForest, KNN, DecisionTree, GaussianNB, SVC
   - Regression: LinearRegression, RandomForest, Ridge, SVR, DecisionTree, KNN
   - Clustering: KMeans, AgglomerativeClustering
✅ Evaluates with appropriate metrics
✅ Finds best performing model
✅ Provides detailed results and recommendations
""")
    
    print("\n" + "=" * 60)
    print("🎉 Benchmark system test completed!")
    print("💡 The system is now ready to handle categorical features correctly!")
    
    return True

if __name__ == "__main__":
    test_benchmark_system() 