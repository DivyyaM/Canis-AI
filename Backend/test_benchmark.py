#!/usr/bin/env python3
"""
Test script for the model benchmark module
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from app.core.model_benchmark import benchmark_models, get_benchmark_summary, compare_with_current_model

def test_benchmark():
    """Test the benchmark functionality"""
    print("🧪 Testing Model Benchmark Module...")
    print("=" * 50)
    
    # Test 1: Run benchmark
    print("\n1️⃣ Running model benchmark...")
    try:
        results = benchmark_models()
        if "error" in results:
            print(f"❌ Benchmark failed: {results['error']}")
            return False
        else:
            print(f"✅ Benchmark completed successfully!")
            print(f"   Task: {results['task_type']}")
            print(f"   Best Model: {results['best_model']}")
            print(f"   Best Score: {results['best_score']}")
            print(f"   Models Tested: {len(results['all_results'])}")
    except Exception as e:
        print(f"❌ Benchmark error: {str(e)}")
        return False
    
    # Test 2: Get summary
    print("\n2️⃣ Getting benchmark summary...")
    try:
        summary = get_benchmark_summary()
        if "error" in summary:
            print(f"❌ Summary failed: {summary['error']}")
        else:
            print(f"✅ Summary generated!")
            top_models = summary.get('top_models', [])
            if top_models:
                model_names = []
                for m in top_models:
                    if isinstance(m, dict) and 'name' in m:
                        model_names.append(m['name'])
                print(f"   Top Models: {model_names}")
            else:
                print("   No top models found")
    except Exception as e:
        print(f"❌ Summary error: {str(e)}")
    
    # Test 3: Compare with current model
    print("\n3️⃣ Comparing with current model...")
    try:
        comparison = compare_with_current_model()
        if "error" in comparison:
            print(f"❌ Comparison failed: {comparison['error']}")
        else:
            print(f"✅ Comparison completed!")
            print(f"   Current: {comparison.get('current_model', 'None')}")
            print(f"   Best Benchmark: {comparison.get('benchmark_best_model', 'None')}")
            if comparison.get('recommendation'):
                print(f"   Recommendation: {comparison['recommendation']}")
    except Exception as e:
        print(f"❌ Comparison error: {str(e)}")
    
    print("\n" + "=" * 50)
    print("🎉 Benchmark testing completed!")
    return True

if __name__ == "__main__":
    test_benchmark() 