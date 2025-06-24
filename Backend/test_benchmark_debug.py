#!/usr/bin/env python3
"""
Debug script to test benchmark functionality
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

import joblib
import pandas as pd
import numpy as np
from app.core.benchmark_manager import BenchmarkManager

def test_benchmark():
    """Test benchmark functionality step by step"""
    print("=== Testing Benchmark Functionality ===")
    
    # Check if required files exist
    tmp_dir = "tmp"
    required_files = [
        "benchmark_data.pkl",
        "preprocessor.pkl",
        "dataset.csv"
    ]
    
    print("\n1. Checking required files:")
    for file in required_files:
        path = f"{tmp_dir}/{file}"
        exists = os.path.exists(path)
        print(f"   {file}: {'✓' if exists else '✗'}")
        if exists:
            try:
                size = os.path.getsize(path)
                print(f"     Size: {size} bytes")
            except:
                print(f"     Size: unknown")
    
    # Test data loading
    print("\n2. Testing data loading:")
    try:
        benchmark_manager = BenchmarkManager(tmp_dir)
        
        # Try to load data
        print("   Attempting to load benchmark data...")
        X_train, X_test, y_train, y_test = benchmark_manager.load_data()
        print(f"   ✓ Data loaded successfully!")
        print(f"   X_train shape: {X_train.shape}")
        print(f"   X_test shape: {X_test.shape}")
        print(f"   y_train shape: {y_train.shape}")
        print(f"   y_test shape: {y_test.shape}")
        print(f"   y_train dtype: {y_train.dtype}")
        print(f"   y_train unique values: {np.unique(y_train)}")
        
        # Test task type detection
        print("\n3. Testing task type detection:")
        task_type = benchmark_manager.detect_task_type(y_train)
        print(f"   Detected task type: {task_type}")
        
        # Test benchmark run
        print("\n4. Testing benchmark run:")
        results = benchmark_manager.run_benchmark()
        if "error" in results:
            print(f"   ✗ Benchmark failed: {results['error']}")
        else:
            print(f"   ✓ Benchmark completed successfully!")
            print(f"   Task type: {results['task_type']}")
            print(f"   Best score: {results['best_score']}")
            print(f"   Models tested: {len(results['all_results'])}")
            
            # Show top models
            if 'all_results' in results:
                print("\n   Top models:")
                for name, result in results['all_results'].items():
                    if 'error' not in result:
                        if 'accuracy' in result:
                            print(f"     {name}: {result['accuracy']:.4f} (accuracy)")
                        elif 'r2_score' in result:
                            print(f"     {name}: {result['r2_score']:.4f} (r2_score)")
        
        # Test summary
        print("\n5. Testing summary:")
        summary = benchmark_manager.get_summary()
        if "error" in summary:
            print(f"   ✗ Summary failed: {summary['error']}")
        else:
            print(f"   ✓ Summary generated successfully!")
            print(f"   Task type: {summary['task_type']}")
            print(f"   Best score: {summary['best_score']}")
            print(f"   Models tested: {summary['total_models_tested']}")
            print(f"   Successful: {summary['successful_models']}")
            print(f"   Failed: {summary['failed_models']}")
        
    except Exception as e:
        print(f"   ✗ Error: {str(e)}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    test_benchmark() 