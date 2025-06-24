import pandas as pd
import numpy as np

def generate_pipeline_code():
    """Generate complete, working ML pipeline code"""
    try:
        # Get all the analysis results
        from .target_identifier import find_target
        from .task_classifier import classify_task
        from .model_selector import select_model
        from .preprocessor import create_preprocessing_pipeline
        
        target_info = find_target()
        task_info = classify_task()
        model_info = select_model()
        preprocessor_info = create_preprocessing_pipeline()
        
        # Extract information with defaults
        target_col = target_info.get("suggested_target", "target")
        task = task_info.get("task", "classification")
        model_name = model_info.get("selected_model", "RandomForestClassifier")
        model_params = model_info.get("model_params", {}) or {}
        
        # Generate code based on task
        if task.startswith("classification"):
            return generate_classification_code(target_col, model_name, model_params, task)
        elif task == "regression":
            return generate_regression_code(target_col, model_name, model_params)
        elif task == "clustering":
            return generate_clustering_code(model_name, model_params)
        else:
            return generate_generic_code(target_col, model_name, model_params, task)
            
    except Exception as e:
        return {"error": str(e)}

def generate_classification_code(target_col, model_name, model_params, task):
    """Generate classification pipeline code"""
    
    # Import statements
    imports = """import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder, OneHotEncoder
from sklearn.impute import SimpleImputer
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
"""
    
    # Add model-specific imports
    if "RandomForest" in model_name:
        imports += "from sklearn.ensemble import RandomForestClassifier\n"
    elif "Logistic" in model_name:
        imports += "from sklearn.linear_model import LogisticRegression\n"
    elif "XGB" in model_name:
        imports += "import xgboost as xgb\n"
    elif "SVC" in model_name:
        imports += "from sklearn.svm import SVC\n"
    elif "KNeighbors" in model_name:
        imports += "from sklearn.neighbors import KNeighborsClassifier\n"
    
    # Data loading
    data_loading = f"""
# Load your data
df = pd.read_csv('your_data.csv')

# Separate features and target
X = df.drop(columns=['{target_col}'])
y = df['{target_col}']

# Split the data
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)
"""
    
    # Preprocessing pipeline
    preprocessing = """
# Create preprocessing pipeline
numeric_features = X.select_dtypes(include=['int64', 'float64']).columns.tolist()
categorical_features = X.select_dtypes(include=['object', 'category']).columns.tolist()

preprocessing_steps = []

# Numeric preprocessing
if numeric_features:
    numeric_transformer = Pipeline([
        ('imputer', SimpleImputer(strategy='median')),
        ('scaler', StandardScaler())
    ])
    preprocessing_steps.append(('numeric', numeric_transformer, numeric_features))

# Categorical preprocessing
if categorical_features:
    categorical_transformer = Pipeline([
        ('imputer', SimpleImputer(strategy='most_frequent')),
        ('encoder', OneHotEncoder(drop='first', sparse_output=False))
    ])
    preprocessing_steps.append(('categorical', categorical_transformer, categorical_features))

preprocessor = ColumnTransformer(transformers=preprocessing_steps, remainder='passthrough')
"""
    
    # Model creation
    model_params_str = ", ".join([f"{k}={v}" for k, v in model_params.items()])
    model_creation = f"""
# Create and train the model
model = {model_name}({model_params_str})

# Create full pipeline
pipeline = Pipeline([
    ('preprocessor', preprocessor),
    ('classifier', model)
])

# Train the model
pipeline.fit(X_train, y_train)
"""
    
    # Evaluation
    evaluation = f"""
# Make predictions
y_pred = pipeline.predict(X_test)

# Evaluate the model
accuracy = accuracy_score(y_test, y_pred)
print(f"Accuracy: {{accuracy:.4f}}")

# Classification report
print("\\nClassification Report:")
print(classification_report(y_test, y_pred))

# Confusion matrix
print("\\nConfusion Matrix:")
print(confusion_matrix(y_test, y_pred))

# Feature importance (if available)
if hasattr(model, 'feature_importances_'):
    feature_names = preprocessor.get_feature_names_out()
    importances = model.feature_importances_
    feature_importance_df = pd.DataFrame({{
        'feature': feature_names,
        'importance': importances
    }}).sort_values('importance', ascending=False)
    print("\\nTop 10 Most Important Features:")
    print(feature_importance_df.head(10))
"""
    
    # Prediction function
    prediction = """
# Function to make predictions on new data
def predict_new_data(new_data):
    '''
    Make predictions on new data
    
    Parameters:
    new_data: DataFrame with same columns as training data (excluding target)
    
    Returns:
    predictions: Array of predicted classes
    '''
    return pipeline.predict(new_data)

# Example usage:
# new_data = pd.read_csv('new_data.csv')
# predictions = predict_new_data(new_data)
# print("Predictions:", predictions)
"""
    
    # Complete code
    complete_code = imports + data_loading + preprocessing + model_creation + evaluation + prediction
    
    return {
        "code": complete_code,
        "task": task,
        "model": model_name,
        "target_column": target_col,
        "description": f"Complete {task} pipeline using {model_name}"
    }

def generate_regression_code(target_col, model_name, model_params):
    """Generate regression pipeline code"""
    
    imports = """import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.impute import SimpleImputer
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error
"""
    
    # Add model-specific imports
    if "RandomForest" in model_name:
        imports += "from sklearn.ensemble import RandomForestRegressor\n"
    elif "Linear" in model_name:
        imports += "from sklearn.linear_model import LinearRegression\n"
    elif "Ridge" in model_name:
        imports += "from sklearn.linear_model import Ridge\n"
    elif "XGB" in model_name:
        imports += "import xgboost as xgb\n"
    
    data_loading = f"""
# Load your data
df = pd.read_csv('your_data.csv')

# Separate features and target
X = df.drop(columns=['{target_col}'])
y = df['{target_col}']

# Split the data
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)
"""
    
    preprocessing = """
# Create preprocessing pipeline
numeric_features = X.select_dtypes(include=['int64', 'float64']).columns.tolist()
categorical_features = X.select_dtypes(include=['object', 'category']).columns.tolist()

preprocessing_steps = []

# Numeric preprocessing
if numeric_features:
    numeric_transformer = Pipeline([
        ('imputer', SimpleImputer(strategy='median')),
        ('scaler', StandardScaler())
    ])
    preprocessing_steps.append(('numeric', numeric_transformer, numeric_features))

# Categorical preprocessing
if categorical_features:
    categorical_transformer = Pipeline([
        ('imputer', SimpleImputer(strategy='most_frequent')),
        ('encoder', OneHotEncoder(drop='first', sparse_output=False))
    ])
    preprocessing_steps.append(('categorical', categorical_transformer, categorical_features))

preprocessor = ColumnTransformer(transformers=preprocessing_steps, remainder='passthrough')
"""
    
    model_params_str = ", ".join([f"{k}={v}" for k, v in model_params.items()])
    model_creation = f"""
# Create and train the model
model = {model_name}({model_params_str})

# Create full pipeline
pipeline = Pipeline([
    ('preprocessor', preprocessor),
    ('regressor', model)
])

# Train the model
pipeline.fit(X_train, y_train)
"""
    
    evaluation = f"""
# Make predictions
y_pred = pipeline.predict(X_test)

# Evaluate the model
r2 = r2_score(y_test, y_pred)
mse = mean_squared_error(y_test, y_pred)
rmse = np.sqrt(mse)
mae = mean_absolute_error(y_test, y_pred)

print(f"R² Score: {{r2:.4f}}")
print(f"Mean Squared Error: {{mse:.4f}}")
print(f"Root Mean Squared Error: {{rmse:.4f}}")
print(f"Mean Absolute Error: {{mae:.4f}}")

# Compare actual vs predicted
comparison_df = pd.DataFrame({{
    'Actual': y_test,
    'Predicted': y_pred,
    'Difference': y_test - y_pred
}})
print("\\nActual vs Predicted (first 10 samples):")
print(comparison_df.head(10))
"""
    
    prediction = """
# Function to make predictions on new data
def predict_new_data(new_data):
    '''
    Make predictions on new data
    
    Parameters:
    new_data: DataFrame with same columns as training data (excluding target)
    
    Returns:
    predictions: Array of predicted values
    '''
    return pipeline.predict(new_data)

# Example usage:
# new_data = pd.read_csv('new_data.csv')
# predictions = predict_new_data(new_data)
# print("Predictions:", predictions)
"""
    
    complete_code = imports + data_loading + preprocessing + model_creation + evaluation + prediction
    
    return {
        "code": complete_code,
        "task": "regression",
        "model": model_name,
        "target_column": target_col,
        "description": f"Complete regression pipeline using {model_name}"
    }

def generate_clustering_code(model_name, model_params):
    """Generate clustering pipeline code"""
    
    imports = """import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.metrics import silhouette_score
"""
    
    if "KMeans" in model_name:
        imports += "from sklearn.cluster import KMeans\n"
    elif "DBSCAN" in model_name:
        imports += "from sklearn.cluster import DBSCAN\n"
    
    data_loading = """
# Load your data
df = pd.read_csv('your_data.csv')

# Select features for clustering (exclude any target columns if present)
# Modify this based on your data
features_for_clustering = df.select_dtypes(include=['int64', 'float64']).columns.tolist()
X = df[features_for_clustering]
"""
    
    preprocessing = """
# Create preprocessing pipeline
preprocessor = Pipeline([
    ('imputer', SimpleImputer(strategy='median')),
    ('scaler', StandardScaler())
])

# Preprocess the data
X_scaled = preprocessor.fit_transform(X)
"""
    
    model_params_str = ", ".join([f"{k}={v}" for k, v in model_params.items()])
    model_creation = f"""
# Create and fit the clustering model
model = {model_name}({model_params_str})
clusters = model.fit_predict(X_scaled)

# Add cluster labels to original data
df['cluster'] = clusters
"""
    
    evaluation = """
# Evaluate clustering
silhouette = silhouette_score(X_scaled, clusters)
print(f"Silhouette Score: {silhouette:.4f}")

# Analyze clusters
cluster_analysis = df.groupby('cluster').agg({
    'cluster': 'count'
}).rename(columns={'cluster': 'count'})
print("\\nCluster Distribution:")
print(cluster_analysis)

# Feature analysis by cluster
print("\\nFeature Analysis by Cluster:")
for feature in X.columns:
    cluster_means = df.groupby('cluster')[feature].mean()
    print(f"\\n{feature} - Mean by Cluster:")
    print(cluster_means)
"""
    
    prediction = """
# Function to assign new data to clusters
def assign_clusters(new_data):
    '''
    Assign new data to clusters
    
    Parameters:
    new_data: DataFrame with same features as training data
    
    Returns:
    clusters: Array of cluster assignments
    '''
    X_new = new_data[features_for_clustering]
    X_new_scaled = preprocessor.transform(X_new)
    return model.predict(X_new_scaled)

# Example usage:
# new_data = pd.read_csv('new_data.csv')
# new_clusters = assign_clusters(new_data)
# print("New data cluster assignments:", new_clusters)
"""
    
    complete_code = imports + data_loading + preprocessing + model_creation + evaluation + prediction
    
    return {
        "code": complete_code,
        "task": "clustering",
        "model": model_name,
        "description": f"Complete clustering pipeline using {model_name}"
    }

def generate_generic_code(target_col, model_name, model_params, task):
    """Generate generic pipeline code"""
    return {
        "code": f"""# Generic {task} pipeline using {model_name}
# This is a template - customize based on your specific needs

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline

# Load data
df = pd.read_csv('your_data.csv')

# Separate features and target
X = df.drop(columns=['{target_col}'])
y = df['{target_col}']

# Split data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Create pipeline
pipeline = Pipeline([
    ('scaler', StandardScaler()),
    ('model', {model_name}({', '.join([f'{k}={v}' for k, v in model_params.items()])}))
])

# Train and evaluate
pipeline.fit(X_train, y_train)
predictions = pipeline.predict(X_test)

print("Model trained successfully!")
print(f"Task: {task}")
print(f"Model: {model_name}")
""",
        "task": task,
        "model": model_name,
        "description": f"Generic {task} pipeline"
    }
