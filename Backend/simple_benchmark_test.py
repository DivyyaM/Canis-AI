#!/usr/bin/env python3
"""
Simple test script for the model benchmark module
Shows how to use the benchmark functionality
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

def test_benchmark_imports():
    """Test that we can import the benchmark module"""
    print("🧪 Testing Model Benchmark Module Imports...")
    print("=" * 50)
    
    try:
        from app.core.model_benchmark import (
            benchmark_models, 
            get_benchmark_summary, 
            compare_with_current_model,
            detect_task_type
        )
        print("✅ All benchmark functions imported successfully!")
        return True
    except ImportError as e:
        print(f"❌ Import error: {e}")
        return False
    except Exception as e:
        print(f"❌ Unexpected error: {e}")
        return False

def show_usage_examples():
    """Show how to use the benchmark module"""
    print("\n📚 Usage Examples:")
    print("=" * 50)
    
    print("\n1️⃣ Basic Usage:")
    print("""
from app.core.model_benchmark import benchmark_models

# Run benchmark (requires trained model first)
results = benchmark_models()
if "error" not in results:
    print(f"Best model: {results['best_model']}")
    print(f"Best score: {results['best_score']}")
""")
    
    print("\n2️⃣ Get Summary:")
    print("""
from app.core.model_benchmark import get_benchmark_summary

summary = get_benchmark_summary()
print(f"Top models: {summary.get('top_models', [])}")
""")
    
    print("\n3️⃣ Compare with Current Model:")
    print("""
from app.core.model_benchmark import compare_with_current_model

comparison = compare_with_current_model()
if comparison.get('recommendation'):
    print(f"Recommendation: {comparison['recommendation']}")
""")
    
    print("\n4️⃣ API Endpoints:")
    print("""
# Run benchmark
curl -X POST "http://localhost:8000/benchmark-models/"

# Get summary  
curl -X GET "http://localhost:8000/benchmark-summary/"

# Compare models
curl -X GET "http://localhost:8000/compare-models/"
""")

def show_prerequisites():
    """Show what's needed to run benchmarks"""
    print("\n🔧 Prerequisites:")
    print("=" * 50)
    
    print("""
To run model benchmarks, you need:

1. ✅ Upload a CSV file:
   POST /upload-csv/

2. ✅ Train a model (creates X_train.pkl, y_train.pkl, etc.):
   POST /train-model/

3. ✅ Then run benchmark:
   POST /benchmark-models/

The benchmark will test multiple models and find the best one!
""")

def main():
    """Main test function"""
    print("🚀 Model Benchmark Module Test")
    print("=" * 60)
    
    # Test imports
    if not test_benchmark_imports():
        return False
    
    # Show usage examples
    show_usage_examples()
    
    # Show prerequisites
    show_prerequisites()
    
    print("\n" + "=" * 60)
    print("🎉 Test completed! The benchmark module is ready to use.")
    print("💡 Start your FastAPI server and try the endpoints!")
    
    return True

if __name__ == "__main__":
    main() 