"""
Gemini Brain - Central Intelligence Layer for Canis AI Backend
Maintains context and metadata across all modules
"""

import pandas as pd
import numpy as np
import joblib
import os
import google.generativeai as genai
from typing import Dict, List, Any, Optional
from dataclasses import dataclass, asdict
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, f1_score, mean_squared_error, precision_score, recall_score, r2_score, mean_absolute_error
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor, GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression, LinearRegression, Ridge
from sklearn.svm import SVC, SVR
from sklearn.neighbors import KNeighborsClassifier
from xgboost import XGBClassifier, XGBRegressor
import traceback

@dataclass
class DatasetMetadata:
    """Structured metadata for dataset analysis"""
    target_column: Optional[str] = None
    feature_columns: List[str] = None
    task_type: Optional[str] = None
    task_reason: Optional[str] = None
    n_samples: int = 0
    n_features: int = 0
    n_classes: Optional[int] = None
    target_dtype: Optional[str] = None
    numeric_features: List[str] = None
    categorical_features: List[str] = None
    missing_values: Dict[str, int] = None
    confidence_score: float = 0.0

    def __post_init__(self):
        if self.feature_columns is None:
            self.feature_columns = []
        if self.numeric_features is None:
            self.numeric_features = []
        if self.categorical_features is None:
            self.categorical_features = []
        if self.missing_values is None:
            self.missing_values = {}

class GeminiBrain:
    """Central intelligence layer for the Canis AI system"""

    def __init__(self):
        self.metadata = DatasetMetadata()
        self.encoders = {}
        self.preprocessor = None
        self.model = None
        self.model_params = {}
        self.training_results = {}
        self.evaluation_results = {}

    def reset(self):
        self.metadata = DatasetMetadata()
        self.encoders = {}
        self.preprocessor = None
        self.model = None
        self.model_params = {}
        self.training_results = {}
        self.evaluation_results = {}

    def analyze_dataset(self, df: pd.DataFrame) -> Dict[str, Any]:
        try:
            self.metadata.n_samples, self.metadata.n_features = df.shape

            target_info = self._detect_target_column(df)
            self.metadata.target_column = target_info.get("suggested_target")
            self.metadata.confidence_score = target_info.get("confidence_score", 0.0)

            if self.metadata.target_column:
                self.metadata.feature_columns = [col for col in df.columns if col != self.metadata.target_column]

                y = df[self.metadata.target_column]
                self.metadata.target_dtype = str(y.dtype)
                self.metadata.n_classes = y.nunique()

                task_info = self._classify_task(df, self.metadata.target_column)
                self.metadata.task_type = task_info.get("task")
                self.metadata.task_reason = task_info.get("reason")

                X = df[self.metadata.feature_columns]
                self.metadata.numeric_features = X.select_dtypes(include=['int64', 'float64']).columns.tolist()
                self.metadata.categorical_features = X.select_dtypes(include=['object', 'category']).columns.tolist()
                self.metadata.missing_values = X.isnull().sum().to_dict()

            return asdict(self.metadata)

        except Exception as e:
            return {"error": str(e)}

    def _detect_target_column(self, df: pd.DataFrame) -> Dict[str, Any]:
        """Enhanced target column detection with intelligent LLM override"""
        scores = {}
        
        # Rule-based scoring
        for col in df.columns:
            nunique = df[col].nunique()
            dtype = df[col].dtype
            
            # Skip obvious feature columns
            feature_keywords = ['id', 'name', 'country', 'state', 'city', 'zip', 'address', 
                              'phone', 'email', 'date', 'time', 'index', 'row', 'sample']
            if any(keyword in col.lower() for keyword in feature_keywords):
                continue
                
            # Score based on uniqueness and data type
            if nunique == 2:
                scores[col] = 10 if dtype == 'object' else 8
            elif 3 <= nunique <= 10:
                scores[col] = 6 if dtype == 'object' else 4
            elif nunique > 10 and dtype in ['int64', 'float64']:
                scores[col] = 2

        # If we have good candidates, use rule-based selection
        if scores:
            best_target = max(scores, key=scores.get)
            best_score = scores[best_target]
            
            if best_score >= 8:
                method = "binary_classification_target"
            elif best_score >= 4:
                method = "multiclass_classification_target"
            elif best_score >= 2:
                method = "regression_target"
            else:
                method = "fallback_selection"

            return {
                "suggested_target": best_target,
                "method": method,
                "confidence_score": best_score,
                "all_scores": scores,
                "reason": f"Rule-based detection: {best_target} scored {best_score}"
            }

        # If no good candidates found, try LLM override
        from dotenv import load_dotenv
        load_dotenv()
        gemini_key = os.getenv("GEMINI_API_KEY")
        
        if gemini_key:
            try:
                llm_response = self.llm_suggest_target_and_features(df, gemini_key)
                if "target_column" in llm_response and llm_response["target_column"] in df.columns:
                    return {
                        "suggested_target": llm_response["target_column"],
                        "method": "llm_gemini_override",
                        "confidence_score": 10,
                        "llm_code": llm_response.get("code", ""),
                        "llm_features": llm_response.get("feature_columns", []),
                        "reason": "LLM analysis suggested better target column"
                    }
                else:
                    print(f"LLM response invalid: {llm_response}")
            except Exception as e:
                print(f"LLM override failed: {str(e)}")

        # Final fallback: last column convention
        fallback_target = df.columns[-1]
        return {
            "suggested_target": fallback_target,
            "method": "fallback_last_column",
            "confidence_score": 1,
            "reason": "No clear target identified, using last column as fallback"
        }

    def llm_suggest_target_and_features(self, df: pd.DataFrame, api_key: str) -> Dict[str, Any]:
        """Enhanced LLM-based target and feature suggestion"""
        try:
            genai.configure(api_key=api_key)
            model = genai.GenerativeModel("gemini-1.5-pro")

            # Create a more detailed prompt
            prompt = f"""
You are an expert machine learning engineer. Analyze this dataset and identify:

1. **Target Variable**: Which column should be the target (output) variable?
2. **Feature Variables**: Which columns should be used as features (inputs)?
3. **Python Code**: Generate working code to split X and y.

Dataset Information:
- Shape: {df.shape}
- Columns: {list(df.columns)}
- Data Types: {df.dtypes.to_dict()}
- Missing Values: {df.isnull().sum().to_dict()}

Sample Data (first 5 rows):
{df.head(5).to_markdown()}

Important Guidelines:
- Target should be the variable you want to predict
- Features should be variables that help predict the target
- Avoid using ID columns, names, or obvious identifiers as features
- For classification: target should have limited unique values
- For regression: target should be continuous numeric

Respond ONLY with valid JSON in this exact format:
{{
  "target_column": "column_name",
  "feature_columns": ["col1", "col2", "col3"],
  "code": "X = df[['col1', 'col2', 'col3']]\\ny = df['column_name']",
  "reasoning": "Brief explanation of why this target was chosen"
}}
            """

            response = model.generate_content(prompt)
            response_text = response.text.strip()
            
            # Improved JSON extraction
            import json
            import re
            
            # Try to find JSON in the response
            json_match = re.search(r'\{[\s\S]*\}', response_text)
            if json_match:
                json_str = json_match.group(0)
                result = json.loads(json_str)
                
                # Validate the response
                if "target_column" in result and "feature_columns" in result:
                    # Ensure target column exists in dataset
                    if result["target_column"] in df.columns:
                        # Update metadata with LLM suggestions
                        self.metadata.target_column = result["target_column"]
                        self.metadata.feature_columns = result.get("feature_columns", [])
                        return result
                    else:
                        return {"error": f"LLM suggested target column '{result['target_column']}' not found in dataset"}
                else:
                    return {"error": "LLM response missing required fields"}
            else:
                return {"error": "Could not extract JSON from LLM response"}

        except json.JSONDecodeError as e:
            return {"error": f"Invalid JSON from LLM: {str(e)}"}
        except Exception as e:
            return {"error": f"LLM analysis failed: {str(e)}"}

    def _classify_task(self, df: pd.DataFrame, target_col: str) -> Dict[str, Any]:
        y = df[target_col]
        text_columns = []
        for col in df.columns:
            if col != target_col and df[col].dtype == 'object':
                sample_values = df[col].dropna().head(10)
                if len(sample_values) > 0:
                    avg_length = sample_values.astype(str).str.len().mean()
                    if avg_length > 20:
                        text_columns.append(col)
        if text_columns:
            return {
                "task": "nlp",
                "reason": f"text_columns_detected: {text_columns}",
                "text_columns": text_columns
            }
        if y.dtype == 'object':
            unique_count = y.nunique()
            return {
                "task": "binary_classification" if unique_count == 2 else "multiclass_classification",
                "reason": f"categorical_target_with_{unique_count}_classes",
                "classes": sorted(y.unique().tolist())
            }
        else:
            unique_count = y.nunique()
            if unique_count <= 20:
                return {
                    "task": "binary_classification" if unique_count == 2 else "multiclass_classification",
                    "reason": f"categorical_target_with_{unique_count}_classes",
                    "classes": sorted(y.unique().tolist())
                }
            else:
                return {
                    "task": "regression",
                    "reason": f"continuous_target_with_{unique_count}_unique_values"
                }

    def create_preprocessing_pipeline(self, df: pd.DataFrame) -> Dict[str, Any]:
        try:
            if not self.metadata.target_column:
                return {"error": "No target column detected"}

            X = df[self.metadata.feature_columns]
            missing_numeric = X[self.metadata.numeric_features].isnull().sum().sum() > 0 if self.metadata.numeric_features else False
            missing_categorical = X[self.metadata.categorical_features].isnull().sum().sum() > 0 if self.metadata.categorical_features else False
            preprocessing_steps = []
            if self.metadata.numeric_features:
                numeric_transformer = []
                if missing_numeric:
                    numeric_transformer.append(('imputer', SimpleImputer(strategy='median')))
                numeric_transformer.append(('scaler', StandardScaler()))
                preprocessing_steps.append(('numeric', numeric_transformer, self.metadata.numeric_features))
            if self.metadata.categorical_features:
                categorical_transformer = []
                if missing_categorical:
                    categorical_transformer.append(('imputer', SimpleImputer(strategy='most_frequent')))
                categorical_transformer.append(('encoder', 'onehot'))
                preprocessing_steps.append(('categorical', categorical_transformer, self.metadata.categorical_features))
            y = df[self.metadata.target_column]
            if y.dtype == 'object':
                target_encoder = LabelEncoder()
                target_encoder.fit(y)
                self.encoders['target'] = target_encoder
            return {
                "preprocessing_steps": preprocessing_steps,
                "target_encoder": "LabelEncoder" if y.dtype == 'object' else "None",
                "missing_numeric": missing_numeric,
                "missing_categorical": missing_categorical
            }
        except Exception as e:
            return {"error": str(e)}

    def get_metadata(self) -> Dict[str, Any]:
        metadata_dict = asdict(self.metadata)
        def make_json_serializable(obj):
            if isinstance(obj, dict):
                return {k: make_json_serializable(v) for k, v in obj.items()}
            elif isinstance(obj, list):
                return [make_json_serializable(item) for item in obj]
            elif hasattr(obj, 'item'):
                return obj.item()
            elif hasattr(obj, 'tolist'):
                return obj.tolist()
            else:
                return obj
        return make_json_serializable(metadata_dict)

    def get_context_for_chat(self) -> Dict[str, Any]:
        context = {
            "target_column": self.metadata.target_column,
            "task_type": self.metadata.task_type,
            "task_reason": self.metadata.task_reason,
            "n_samples": int(self.metadata.n_samples),
            "n_features": int(self.metadata.n_features),
            "n_classes": int(self.metadata.n_classes) if self.metadata.n_classes else None,
            "feature_columns": list(self.metadata.feature_columns),
            "numeric_features": list(self.metadata.numeric_features),
            "categorical_features": list(self.metadata.categorical_features),
            "confidence_score": float(self.metadata.confidence_score),
            "model": getattr(self.model, '__class__.__name__', None) if self.model else None,
            "training_score": float(self.training_results.get("test_score", 0.0)) if self.training_results.get("test_score") else None,
            "evaluation_metrics": self.evaluation_results
        }
        def make_json_serializable(obj):
            if isinstance(obj, dict):
                return {k: make_json_serializable(v) for k, v in obj.items()}
            elif isinstance(obj, list):
                return [make_json_serializable(item) for item in obj]
            elif hasattr(obj, 'item'):
                return obj.item()
            elif hasattr(obj, 'tolist'):
                return obj.tolist()
            else:
                return obj
        return make_json_serializable(context)

    def load_model(self, model_path: str) -> Dict[str, Any]:
        try:
            if not os.path.exists(model_path):
                return {"error": f"Model file not found: {model_path}"}
            loaded_model = joblib.load(model_path)
            if hasattr(loaded_model, 'steps') and isinstance(loaded_model.steps, list):
                self.model = loaded_model
                if loaded_model.steps:
                    last_step = loaded_model.steps[-1]
                    if isinstance(last_step, tuple) and len(last_step) >= 2:
                        actual_model = last_step[1]
                        self.model_params = actual_model.get_params() if hasattr(actual_model, 'get_params') else {}
            else:
                self.model = loaded_model
                self.model_params = loaded_model.get_params() if hasattr(loaded_model, 'get_params') else {}
            return {
                "status": "success",
                "message": f"Model loaded successfully from {model_path}",
                "model_type": getattr(self.model, '__class__.__name__', "Unknown")
            }
        except Exception as e:
            return {"error": f"Failed to load model: {str(e)}"}

    def suggest_models(self) -> List[str]:
        """Suggest models based on the task type"""
        if self.metadata.task_type == "binary_classification":
            return ["LogisticRegression", "RandomForestClassifier", "XGBClassifier", "SVC", "KNeighborsClassifier"]
        elif self.metadata.task_type == "multiclass_classification":
            return ["RandomForestClassifier", "XGBClassifier", "GradientBoostingClassifier", "SVC", "KNeighborsClassifier"]
        elif self.metadata.task_type == "regression":
            return ["LinearRegression", "RandomForestRegressor", "XGBRegressor", "SVR", "Ridge"]
        else:
            return ["RandomForestClassifier"]  # Default fallback

    def auto_train_model(self, df: pd.DataFrame, model_name: str = "RandomForest") -> Dict[str, Any]:
        """Train model based on metadata and user-selected model"""
        try:
            if not self.metadata.target_column:
                return {"error": "Target column not set"}

            X = df[self.metadata.feature_columns]
            y = df[self.metadata.target_column]

            # Handle target encoding if needed
            if y.dtype == 'object':
                from sklearn.preprocessing import LabelEncoder
                label_encoder = LabelEncoder()
                y = label_encoder.fit_transform(y)
                self.encoders['target'] = label_encoder

            X_train, X_test, y_train, y_test = train_test_split(
                X, y, test_size=0.2, random_state=42, 
                stratify=y if self.metadata.task_type.startswith('classification') else None
            )

            # Create model based on task type
            model = self._create_model(model_name)
            if not model:
                return {"error": f"Model '{model_name}' not supported for task type '{self.metadata.task_type}'"}

            # Train the model
            model.fit(X_train, y_train)
            self.model = model
            
            # Calculate scores
            train_score = model.score(X_train, y_train)
            test_score = model.score(X_test, y_test)
            
            self.training_results = {
                "model_name": model_name,
                "train_score": train_score,
                "test_score": test_score,
                "train_samples": len(X_train),
                "test_samples": len(X_test)
            }

            # Store predictions for evaluation
            y_pred = model.predict(X_test)
            self.evaluation_results = self._generate_evaluation_report(y_test, y_pred)

            # Save model and test data for later use
            import joblib
            joblib.dump(model, "tmp/model.pkl")
            joblib.dump(X_test, "tmp/X_test.pkl")
            joblib.dump(y_test, "tmp/y_test.pkl")
            joblib.dump((X_train, X_test, y_train, y_test), "tmp/benchmark_data.pkl")

            return {
                "status": "success",
                "message": f"Model '{model_name}' trained successfully.",
                "model_name": model_name,
                "train_score": round(train_score, 4),
                "test_score": round(test_score, 4),
                "train_samples": len(X_train),
                "test_samples": len(X_test),
                "evaluation": self.evaluation_results
            }

        except Exception as e:
            return {
                "error": str(e),
                "trace": traceback.format_exc()
            }

    def _create_model(self, model_name: str):
        """Create model instance based on name and task type"""
        try:
            # Normalize model name
            model_name = model_name.strip()
            
            if self.metadata.task_type == "binary_classification":
                model_map = {
                    "LogisticRegression": lambda: LogisticRegression(random_state=42, max_iter=1000),
                    "RandomForestClassifier": lambda: RandomForestClassifier(random_state=42),
                    "RandomForest": lambda: RandomForestClassifier(random_state=42),  # Alias
                    "XGBClassifier": lambda: XGBClassifier(random_state=42),
                    "XGB": lambda: XGBClassifier(random_state=42),  # Alias
                    "SVC": lambda: SVC(random_state=42),
                    "KNeighborsClassifier": lambda: KNeighborsClassifier(),
                    "KNN": lambda: KNeighborsClassifier()  # Alias
                }
            elif self.metadata.task_type == "multiclass_classification":
                model_map = {
                    "RandomForestClassifier": lambda: RandomForestClassifier(random_state=42),
                    "RandomForest": lambda: RandomForestClassifier(random_state=42),  # Alias
                    "XGBClassifier": lambda: XGBClassifier(random_state=42),
                    "XGB": lambda: XGBClassifier(random_state=42),  # Alias
                    "GradientBoostingClassifier": lambda: GradientBoostingClassifier(random_state=42),
                    "SVC": lambda: SVC(random_state=42),
                    "KNeighborsClassifier": lambda: KNeighborsClassifier(),
                    "KNN": lambda: KNeighborsClassifier()  # Alias
                }
            elif self.metadata.task_type == "regression":
                model_map = {
                    "LinearRegression": lambda: LinearRegression(),
                    "RandomForestRegressor": lambda: RandomForestRegressor(random_state=42),
                    "RandomForest": lambda: RandomForestRegressor(random_state=42),  # Alias
                    "XGBRegressor": lambda: XGBRegressor(random_state=42),
                    "XGB": lambda: XGBRegressor(random_state=42),  # Alias
                    "SVR": lambda: SVR(),
                    "Ridge": lambda: Ridge()
                }
            else:
                return None

            # Try to get the model from the map
            if model_name in model_map:
                return model_map[model_name]()
            else:
                # Fallback to a default model for the task type
                if self.metadata.task_type == "binary_classification":
                    return LogisticRegression(random_state=42, max_iter=1000)
                elif self.metadata.task_type == "multiclass_classification":
                    return RandomForestClassifier(random_state=42)
                elif self.metadata.task_type == "regression":
                    return RandomForestRegressor(random_state=42)
                else:
                    return None

        except Exception as e:
            print(f"Error creating model {model_name}: {str(e)}")
            return None

    def _generate_evaluation_report(self, y_true, y_pred) -> Dict[str, Any]:
        """Generate comprehensive evaluation report"""
        try:
            if self.metadata.task_type == "binary_classification":
                # Handle binary classification with dynamic pos_label
                unique_labels = sorted(np.unique(y_true))
                pos_label = max(unique_labels) if len(unique_labels) == 2 else 1
                
                return {
                    "accuracy": round(accuracy_score(y_true, y_pred), 4),
                    "precision": round(precision_score(y_true, y_pred, pos_label=pos_label, average='binary', zero_division=0), 4),
                    "recall": round(recall_score(y_true, y_pred, pos_label=pos_label, average='binary', zero_division=0), 4),
                    "f1_score": round(f1_score(y_true, y_pred, pos_label=pos_label, average='binary', zero_division=0), 4),
                    "pos_label": int(pos_label)
                }
            elif self.metadata.task_type == "multiclass_classification":
                return {
                    "accuracy": round(accuracy_score(y_true, y_pred), 4),
                    "precision": round(precision_score(y_true, y_pred, average='weighted', zero_division=0), 4),
                    "recall": round(recall_score(y_true, y_pred, average='weighted', zero_division=0), 4),
                    "f1_score": round(f1_score(y_true, y_pred, average='weighted', zero_division=0), 4)
                }
            elif self.metadata.task_type == "regression":
                return {
                    "r2_score": round(r2_score(y_true, y_pred), 4),
                    "rmse": round(mean_squared_error(y_true, y_pred, squared=False), 4),
                    "mse": round(mean_squared_error(y_true, y_pred), 4),
                    "mae": round(mean_absolute_error(y_true, y_pred), 4)
                }
            else:
                return {"error": f"Unknown task type: {self.metadata.task_type}"}
                
        except Exception as e:
            return {"error": f"Evaluation failed: {str(e)}"}

    def generate_training_code_llm(self, df: pd.DataFrame) -> str:
        """Use Gemini to generate complete training code block using existing API key configuration"""
        try:
            # Load API key from environment (same as other LLM methods)
            from dotenv import load_dotenv
            load_dotenv()
            api_key = os.getenv("GEMINI_API_KEY")
            
            if not api_key:
                return "# Error: GEMINI_API_KEY not found in environment variables.\n# Please set your Gemini API key in .env file or environment variables."
            
            genai.configure(api_key=api_key)
            model = genai.GenerativeModel("gemini-1.5-pro")

            prompt = f"""
You're an expert ML engineer. Generate a complete Python script for training a machine learning model.

Dataset Information:
- Shape: {df.shape}
- Target Column: {self.metadata.target_column}
- Feature Columns: {self.metadata.feature_columns}
- Task Type: {self.metadata.task_type}
- Data Types: {df.dtypes.to_dict()}

Requirements:
1. Import necessary libraries (pandas, numpy, sklearn, etc.)
2. Load and preprocess the dataset
3. Handle missing values and categorical features
4. Split data into X and y
5. Split into train/test sets
6. Train a {self.metadata.task_type} model
7. Evaluate the model with appropriate metrics
8. Print results and save the model

Generate ONLY the Python code, no explanations. Make it production-ready with proper error handling.
            """

            response = model.generate_content(prompt)
            return response.text

        except Exception as e:
            return f"# Error generating training code: {str(e)}\n\n# Please check your GEMINI_API_KEY environment variable and try again."

    def get_evaluation_report(self) -> Dict[str, Any]:
        """Get the current evaluation report"""
        if not self.evaluation_results:
            return {"error": "No evaluation results available. Please train a model first."}
        
        return {
            "status": "success",
            "task_type": self.metadata.task_type,
            "target_column": self.metadata.target_column,
            "model_name": self.training_results.get("model_name", "Unknown"),
            "evaluation": self.evaluation_results,
            "training_info": self.training_results
        }

# Global singleton instance
gemini = GeminiBrain()