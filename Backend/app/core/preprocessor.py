import pandas as pd
import numpy as np
from pandas.api.types import is_numeric_dtype, is_object_dtype
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, LabelEncoder, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
import joblib

def create_preprocessing_pipeline():
    """Create intelligent preprocessing pipeline based on data characteristics"""
    try:
        df = pd.read_csv("tmp/dataset.csv")
        
        # Get target column
        from .target_identifier import find_target
        target_info = find_target()
        target_col = target_info.get("suggested_target")
        
        if target_col is None:
            return {"error": "No target column detected"}
        
        # Separate features and target
        X = df.drop(columns=[target_col])
        y = df[target_col]
        
        # Analyze data characteristics
        numeric_features = X.select_dtypes(include=['int64', 'float64']).columns.tolist()
        categorical_features = X.select_dtypes(include=['object', 'category']).columns.tolist()
        
        # Check for missing values
        missing_numeric = X[numeric_features].isnull().sum().sum() > 0 if numeric_features else False
        missing_categorical = X[categorical_features].isnull().sum().sum() > 0 if categorical_features else False
        
        # Build preprocessing steps
        preprocessing_steps = []
        
        # Numeric preprocessing
        if numeric_features:
            numeric_transformer = []
            
            if missing_numeric:
                numeric_transformer.append(('imputer', SimpleImputer(strategy='median')))
            
            # Always scale numeric features for most algorithms
            numeric_transformer.append(('scaler', StandardScaler()))
            
            preprocessing_steps.append(('numeric', Pipeline(numeric_transformer), numeric_features))
        
        # Categorical preprocessing
        if categorical_features:
            categorical_transformer = []
            
            if missing_categorical:
                categorical_transformer.append(('imputer', SimpleImputer(strategy='most_frequent')))
            
            # Use OneHotEncoder for categorical features
            categorical_transformer.append(('encoder', OneHotEncoder(drop='first', sparse_output=False)))
            
            preprocessing_steps.append(('categorical', Pipeline(categorical_transformer), categorical_features))
        
        # Create the preprocessing pipeline
        if preprocessing_steps:
            preprocessor = ColumnTransformer(transformers=preprocessing_steps, remainder='passthrough')
            
            # Save preprocessing pipeline
            joblib.dump(preprocessor, "tmp/preprocessor.pkl")
            
            # Handle target encoding
            target_encoder = None
            if y.dtype == 'object':
                target_encoder = LabelEncoder()
                y_encoded = target_encoder.fit_transform(y)
                joblib.dump(target_encoder, "tmp/target_encoder.pkl")
            else:
                y_encoded = y.values
            
            return {
                "preprocessor_created": True,
                "target_column": target_col,
                "target_encoder": "LabelEncoder" if target_encoder else "None",
                "numeric_features": numeric_features,
                "categorical_features": categorical_features,
                "missing_numeric": missing_numeric,
                "missing_categorical": missing_categorical,
                "steps": [step[0] for step in preprocessing_steps],
                "target_unique_values": y.nunique(),
                "target_dtype": str(y.dtype)
            }
        else:
            return {"error": "No features to preprocess"}
            
    except Exception as e:
        return {"error": str(e)}

def get_preprocessed_data():
    """Get preprocessed features and target"""
    try:
        df = pd.read_csv("tmp/dataset.csv")
        
        # Get target column
        from .target_identifier import find_target
        target_info = find_target()
        target_col = target_info.get("suggested_target")
        
        if target_col is None:
            return {"error": "No target column detected"}
        
        # Separate features and target
        X = df.drop(columns=[target_col])
        y = df[target_col]
        
        # Load preprocessor
        preprocessor = joblib.load("tmp/preprocessor.pkl")
        
        # Transform features
        X_transformed = preprocessor.fit_transform(X)
        
        # Handle target encoding if needed
        if y.dtype == 'object':
            try:
                target_encoder = joblib.load("tmp/target_encoder.pkl")
                y_encoded = target_encoder.transform(y)
            except:
                # Create new encoder if not found
                target_encoder = LabelEncoder()
                y_encoded = target_encoder.fit_transform(y)
                joblib.dump(target_encoder, "tmp/target_encoder.pkl")
        else:
            y_encoded = y.values
        
        return {
            "X_shape": X_transformed.shape,
            "y_shape": y_encoded.shape,
            "feature_names": preprocessor.get_feature_names_out().tolist(),
            "target_column": target_col,
            "target_encoded": y.dtype == 'object'
        }
        
    except Exception as e:
        return {"error": str(e)}
