#!/usr/bin/env python3
"""
Test script for the improved benchmark system
Tests the new pipeline-based approach that uses fitted preprocessors
"""

import sys
import os
import pandas as pd
import numpy as np
import joblib
from sklearn.datasets import make_classification, make_regression
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor

# Add the app directory to the path
sys.path.append(os.path.join(os.path.dirname(__file__), 'app'))

def create_test_data():
    """Create test datasets for different task types"""
    print("🔧 Creating test datasets...")
    
    # Create classification dataset
    X_clf, y_clf = make_classification(
        n_samples=1000, n_features=10, n_informative=8, 
        n_redundant=2, n_classes=3, random_state=42
    )
    
    # Create regression dataset
    X_reg, y_reg = make_regression(
        n_samples=1000, n_features=10, n_informative=8, 
        noise=0.1, random_state=42
    )
    
    # Create dataset with categorical features
    X_cat = np.random.rand(1000, 8)
    categorical_col = np.random.choice(['A', 'B', 'C'], size=1000)
    X_cat = np.column_stack([X_cat, categorical_col])
    
    # Create target with categorical values
    y_cat = np.random.choice(['Low', 'Medium', 'High'], size=1000)
    
    return {
        'classification': (X_clf, y_clf),
        'regression': (X_reg, y_reg),
        'categorical': (X_cat, y_cat)
    }

def setup_pipeline_and_data(dataset_type, X, y):
    """Set up preprocessing pipeline and save data"""
    print(f"🔧 Setting up {dataset_type} pipeline...")
    
    # Split data
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )
    
    # Create preprocessing pipeline
    if dataset_type == 'categorical':
        # For categorical data, we need to handle string columns
        # Identify categorical columns (last column in our test data)
        categorical_features = [X.shape[1] - 1]  # Last column index
        numerical_features = list(range(X.shape[1] - 1))  # All other columns
        
        preprocessor = ColumnTransformer(
            transformers=[
                ('num', StandardScaler(), numerical_features),
                ('cat', OneHotEncoder(drop='first', sparse_output=False), categorical_features)
            ]
        )
        
        # Create target encoder
        target_encoder = LabelEncoder()
        y_train_encoded = target_encoder.fit_transform(y_train)
        y_test_encoded = target_encoder.transform(y_test)
        
        # Use classification model
        model = RandomForestClassifier(n_estimators=50, random_state=42)
        
    elif dataset_type == 'regression':
        # For regression data
        preprocessor = StandardScaler()
        y_train_encoded = y_train
        y_test_encoded = y_test
        
        # Use regression model
        model = RandomForestRegressor(n_estimators=50, random_state=42)
        
    else:
        # For classification data
        preprocessor = StandardScaler()
        y_train_encoded = y_train
        y_test_encoded = y_test
        
        # Use classification model
        model = RandomForestClassifier(n_estimators=50, random_state=42)
    
    # Fit the preprocessor
    if dataset_type == 'categorical':
        X_train_processed = preprocessor.fit_transform(X_train)
        X_test_processed = preprocessor.transform(X_test)
    else:
        X_train_processed = preprocessor.fit_transform(X_train)
        X_test_processed = preprocessor.transform(X_test)
    
    # Fit the model
    model.fit(X_train_processed, y_train_encoded)
    
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
    
    # Save target encoder if needed
    if dataset_type == 'categorical':
        joblib.dump(target_encoder, 'tmp/target_encoder.pkl')
    
    print(f"✅ {dataset_type} pipeline setup complete")
    return full_pipeline

def test_benchmark_system():
    """Test the improved benchmark system"""
    print("\n🚀 Testing Improved Benchmark System")
    print("=" * 50)
    
    # Test with different dataset types
    datasets = create_test_data()
    
    for dataset_type, (X, y) in datasets.items():
        print(f"\n📊 Testing {dataset_type.upper()} dataset")
        print("-" * 30)
        
        try:
            # Setup pipeline and data
            pipeline = setup_pipeline_and_data(dataset_type, X, y)
            
            # Import and test benchmark manager
            from app.core.benchmark_manager import BenchmarkManager
            
            # Create benchmark manager
            bm = BenchmarkManager('tmp')
            
            # Run benchmark
            print("🔄 Running benchmark...")
            results = bm.run_benchmark()
            
            if "error" in results:
                print(f"❌ Benchmark failed: {results['error']}")
                continue
            
            # Display results
            print(f"✅ Benchmark completed successfully!")
            print(f"   Task Type: {results['task_type']}")
            print(f"   Best Score: {results['best_score']}")
            print(f"   Best Metric: {results['best_metric']}")
            print(f"   Models Tested: {len(results['all_results'])}")
            
            # Show top 3 models
            print("\n🏆 Top 3 Models:")
            metric = results['best_metric']
            sorted_models = []
            
            for name, result in results['all_results'].items():
                if "error" not in result and metric in result:
                    sorted_models.append((name, result[metric]))
            
            sorted_models.sort(key=lambda x: x[1], reverse=True)
            
            for i, (name, score) in enumerate(sorted_models[:3], 1):
                print(f"   {i}. {name}: {score:.4f}")
            
            # Test saving best model
            print("\n💾 Testing save functionality...")
            success = bm.save_best_model("")
            if success:
                print("✅ Best model saved successfully")
                
                # Verify file exists
                if os.path.exists('tmp/best_benchmark_model.joblib'):
                    print("✅ Best model file created")
                else:
                    print("❌ Best model file not found")
            else:
                print("❌ Failed to save best model")
            
            # Test summary
            print("\n📋 Testing summary...")
            summary = bm.get_summary()
            if "error" not in summary:
                print("✅ Summary generated successfully")
                print(f"   Total models: {summary['total_models_tested']}")
                print(f"   Successful: {summary['successful_models']}")
                print(f"   Failed: {summary['failed_models']}")
            else:
                print(f"❌ Summary failed: {summary['error']}")
            
        except Exception as e:
            print(f"❌ Test failed for {dataset_type}: {str(e)}")
            import traceback
            traceback.print_exc()
        
        print("\n" + "="*50)

def test_api_integration():
    """Test the API integration"""
    print("\n🌐 Testing API Integration")
    print("=" * 30)
    
    try:
        from app.core.model_benchmark import (
            benchmark_models, 
            get_benchmark_summary, 
            compare_with_current_model, 
            save_best_benchmark_model
        )
        
        # Test benchmark_models
        print("🔄 Testing benchmark_models()...")
        results = benchmark_models()
        if "error" not in results:
            print("✅ benchmark_models() works")
        else:
            print(f"❌ benchmark_models() failed: {results['error']}")
        
        # Test get_benchmark_summary
        print("📋 Testing get_benchmark_summary()...")
        summary = get_benchmark_summary()
        if "error" not in summary:
            print("✅ get_benchmark_summary() works")
        else:
            print(f"❌ get_benchmark_summary() failed: {summary['error']}")
        
        # Test save_best_benchmark_model
        print("💾 Testing save_best_benchmark_model()...")
        save_result = save_best_benchmark_model()
        if "error" not in save_result:
            print("✅ save_best_benchmark_model() works")
        else:
            print(f"❌ save_best_benchmark_model() failed: {save_result['error']}")
        
    except Exception as e:
        print(f"❌ API integration test failed: {str(e)}")
        import traceback
        traceback.print_exc()

def cleanup():
    """Clean up test files"""
    print("\n🧹 Cleaning up test files...")
    
    files_to_remove = [
        'tmp/preprocessor.pkl',
        'tmp/benchmark_data.pkl',
        'tmp/target_encoder.pkl',
        'tmp/best_benchmark_model.joblib'
    ]
    
    for file_path in files_to_remove:
        if os.path.exists(file_path):
            os.remove(file_path)
            print(f"   Removed: {file_path}")
    
    # Remove tmp directory if empty
    if os.path.exists('tmp') and not os.listdir('tmp'):
        os.rmdir('tmp')
        print("   Removed: tmp/ directory")

if __name__ == "__main__":
    print("🧪 Improved Benchmark System Test Suite")
    print("=" * 50)
    
    try:
        # Test the benchmark system
        test_benchmark_system()
        
        # Test API integration
        test_api_integration()
        
        print("\n✅ All tests completed!")
        
    except Exception as e:
        print(f"\n❌ Test suite failed: {str(e)}")
        import traceback
        traceback.print_exc()
    
    finally:
        # Cleanup
        cleanup()
        
    print("\n🎉 Test suite finished!") 