"""
LangGraph-style Modular Orchestration for Canis AI Backend
Provides team-scale MLOps workflow orchestration
"""

import asyncio
from typing import Dict, List, Any, Optional, Callable
from enum import Enum
from dataclasses import dataclass, field
import logging
from .gemini_brain import gemini
import pandas as pd

logger = logging.getLogger(__name__)

class NodeType(Enum):
    """Types of workflow nodes"""
    DATA_LOADING = "data_loading"
    DATA_PREPROCESSING = "data_preprocessing"
    FEATURE_ENGINEERING = "feature_engineering"
    MODEL_TRAINING = "model_training"
    MODEL_EVALUATION = "model_evaluation"
    MODEL_DEPLOYMENT = "model_deployment"
    BENCHMARKING = "benchmarking"
    EXPLAINABILITY = "explainability"
    CONDITIONAL = "conditional"
    PARALLEL = "parallel"
    SEQUENTIAL = "sequential"

@dataclass
class WorkflowNode:
    """Represents a node in the workflow"""
    id: str
    name: str
    node_type: NodeType
    function: Callable
    inputs: List[str] = field(default_factory=list)
    outputs: List[str] = field(default_factory=list)
    config: Dict[str, Any] = field(default_factory=dict)
    dependencies: List[str] = field(default_factory=list)
    timeout: int = 300  # 5 minutes default
    retries: int = 3

@dataclass
class WorkflowState:
    """Represents the state of a workflow execution"""
    workflow_id: str
    current_node: str
    completed_nodes: List[str] = field(default_factory=list)
    failed_nodes: List[str] = field(default_factory=list)
    results: Dict[str, Any] = field(default_factory=dict)
    metadata: Dict[str, Any] = field(default_factory=dict)
    status: str = "running"  # running, completed, failed, paused

class WorkflowOrchestrator:
    """Orchestrates complex ML workflows"""
    
    def __init__(self):
        self.workflows: Dict[str, Dict[str, WorkflowNode]] = {}
        self.execution_history: Dict[str, List[WorkflowState]] = {}
        self.node_registry: Dict[str, Callable] = {}
        self._register_default_nodes()
    
    def _register_default_nodes(self):
        """Register default workflow nodes"""
        self.register_node("load_csv", self._load_csv_node, NodeType.DATA_LOADING)
        self.register_node("preprocess_data", self._preprocess_data_node, NodeType.DATA_PREPROCESSING)
        self.register_node("train_model", self._train_model_node, NodeType.MODEL_TRAINING)
        self.register_node("evaluate_model", self._evaluate_model_node, NodeType.MODEL_EVALUATION)
        self.register_node("benchmark_models", self._benchmark_models_node, NodeType.BENCHMARKING)
        self.register_node("generate_explanations", self._generate_explanations_node, NodeType.EXPLAINABILITY)
        self.register_node("deploy_model", self._deploy_model_node, NodeType.MODEL_DEPLOYMENT)
    
    def register_node(self, name: str, function: Callable, node_type: NodeType):
        """Register a new workflow node"""
        self.node_registry[name] = function
        logger.info(f"Registered node: {name} ({node_type.value})")
    
    def create_workflow(self, workflow_id: str, nodes: List[WorkflowNode]) -> str:
        """Create a new workflow"""
        workflow = {}
        for node in nodes:
            workflow[node.id] = node
        
        self.workflows[workflow_id] = workflow
        self.execution_history[workflow_id] = []
        
        logger.info(f"Created workflow: {workflow_id} with {len(nodes)} nodes")
        return workflow_id
    
    async def execute_workflow(self, workflow_id: str, initial_state: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """Execute a workflow"""
        if workflow_id not in self.workflows:
            raise ValueError(f"Workflow {workflow_id} not found")
        
        workflow = self.workflows[workflow_id]
        state = WorkflowState(
            workflow_id=workflow_id,
            current_node=list(workflow.keys())[0],
            metadata=initial_state or {}
        )
        
        self.execution_history[workflow_id].append(state)
        
        try:
            # Execute nodes in dependency order
            execution_order = self._get_execution_order(workflow)
            
            for node_id in execution_order:
                node = workflow[node_id]
                state.current_node = node_id
                
                logger.info(f"Executing node: {node.name} ({node.node_type.value})")
                
                # Execute node
                result = await self._execute_node(node, state)
                
                if result.get("status") == "success":
                    state.completed_nodes.append(node_id)
                    state.results[node_id] = result.get("data", {})
                else:
                    state.failed_nodes.append(node_id)
                    state.status = "failed"
                    logger.error(f"Node {node.name} failed: {result.get('error')}")
                    break
            
            if not state.failed_nodes:
                state.status = "completed"
            
            return {
                "workflow_id": workflow_id,
                "status": state.status,
                "completed_nodes": state.completed_nodes,
                "failed_nodes": state.failed_nodes,
                "results": state.results,
                "metadata": state.metadata
            }
            
        except Exception as e:
            logger.error(f"Workflow execution failed: {str(e)}")
            state.status = "failed"
            return {
                "workflow_id": workflow_id,
                "status": "failed",
                "error": str(e)
            }
    
    def _get_execution_order(self, workflow: Dict[str, WorkflowNode]) -> List[str]:
        """Get execution order based on dependencies"""
        # Simple topological sort
        visited = set()
        order = []
        
        def visit(node_id):
            if node_id in visited:
                return
            visited.add(node_id)
            
            node = workflow[node_id]
            for dep in node.dependencies:
                if dep in workflow:
                    visit(dep)
            
            order.append(node_id)
        
        for node_id in workflow:
            visit(node_id)
        
        return order
    
    async def _execute_node(self, node: WorkflowNode, state: WorkflowState) -> Dict[str, Any]:
        """Execute a single workflow node"""
        try:
            # Prepare inputs
            inputs = {}
            for input_name in node.inputs:
                if input_name in state.results:
                    inputs[input_name] = state.results[input_name]
                elif input_name in state.metadata:
                    inputs[input_name] = state.metadata[input_name]
            
            # Add config to inputs
            inputs.update(node.config)
            
            # Execute node function
            if asyncio.iscoroutinefunction(node.function):
                result = await node.function(**inputs)
            else:
                result = node.function(**inputs)
            
            return {"status": "success", "data": result}
            
        except Exception as e:
            logger.error(f"Node execution failed: {str(e)}")
            return {"status": "error", "error": str(e)}
    
    # Default node implementations
    def _load_csv_node(self, file_path: str, **kwargs) -> Dict[str, Any]:
        """Load CSV data node"""
        df = pd.read_csv(file_path)
        return {
            "data": df,
            "shape": df.shape,
            "columns": list(df.columns)
        }
    
    def _preprocess_data_node(self, data: pd.DataFrame, target_column: Optional[str] = None, **kwargs) -> Dict[str, Any]:
        """Preprocess data node"""
        from .preprocessor import create_preprocessing_pipeline
        
        # Save data temporarily
        data.to_csv("tmp/workflow_dataset.csv", index=False)
        
        # Create preprocessing pipeline
        preprocessor_info = create_preprocessing_pipeline()
        
        return {
            "preprocessor_info": preprocessor_info,
            "data_shape": data.shape
        }
    
    def _train_model_node(self, model_name: str = "RandomForest", **kwargs) -> Dict[str, Any]:
        """Train model node"""
        from .trainer import train_model
        
        result = train_model()
        return result
    
    def _evaluate_model_node(self, **kwargs) -> Dict[str, Any]:
        """Evaluate model node"""
        from .evaluator import evaluate
        
        result = evaluate()
        return result
    
    def _benchmark_models_node(self, **kwargs) -> Dict[str, Any]:
        """Benchmark models node"""
        from .benchmark_manager import BenchmarkManager
        
        benchmark_manager = BenchmarkManager()
        result = benchmark_manager.run_benchmark()
        return result
    
    def _generate_explanations_node(self, **kwargs) -> Dict[str, Any]:
        """Generate explanations node"""
        from .explainability import explain
        
        result = explain()
        return result
    
    def _deploy_model_node(self, model_path: str, **kwargs) -> Dict[str, Any]:
        """Deploy model node"""
        import shutil
        import os
        
        deployment_path = f"models/deployed/{os.path.basename(model_path)}"
        os.makedirs(os.path.dirname(deployment_path), exist_ok=True)
        shutil.copy2(model_path, deployment_path)
        
        return {
            "deployment_path": deployment_path,
            "status": "deployed"
        }

# Predefined workflow templates
class WorkflowTemplates:
    """Predefined workflow templates for common ML tasks"""
    
    @staticmethod
    def get_basic_classification_workflow() -> List[WorkflowNode]:
        """Basic classification workflow"""
        return [
            WorkflowNode(
                id="load_data",
                name="Load Data",
                node_type=NodeType.DATA_LOADING,
                function=lambda **kwargs: WorkflowOrchestrator()._load_csv_node(**kwargs),
                outputs=["data"]
            ),
            WorkflowNode(
                id="preprocess",
                name="Preprocess Data",
                node_type=NodeType.DATA_PREPROCESSING,
                function=lambda **kwargs: WorkflowOrchestrator()._preprocess_data_node(**kwargs),
                inputs=["data"],
                outputs=["preprocessor_info"],
                dependencies=["load_data"]
            ),
            WorkflowNode(
                id="train",
                name="Train Model",
                node_type=NodeType.MODEL_TRAINING,
                function=lambda **kwargs: WorkflowOrchestrator()._train_model_node(**kwargs),
                inputs=["preprocessor_info"],
                outputs=["model"],
                dependencies=["preprocess"]
            ),
            WorkflowNode(
                id="evaluate",
                name="Evaluate Model",
                node_type=NodeType.MODEL_EVALUATION,
                function=lambda **kwargs: WorkflowOrchestrator()._evaluate_model_node(**kwargs),
                inputs=["model"],
                outputs=["evaluation_results"],
                dependencies=["train"]
            )
        ]
    
    @staticmethod
    def get_advanced_mlops_workflow() -> List[WorkflowNode]:
        """Advanced MLOps workflow with benchmarking and deployment"""
        return [
            WorkflowNode(
                id="load_data",
                name="Load Data",
                node_type=NodeType.DATA_LOADING,
                function=lambda **kwargs: WorkflowOrchestrator()._load_csv_node(**kwargs),
                outputs=["data"]
            ),
            WorkflowNode(
                id="preprocess",
                name="Preprocess Data",
                node_type=NodeType.DATA_PREPROCESSING,
                function=lambda **kwargs: WorkflowOrchestrator()._preprocess_data_node(**kwargs),
                inputs=["data"],
                outputs=["preprocessor_info"],
                dependencies=["load_data"]
            ),
            WorkflowNode(
                id="benchmark",
                name="Benchmark Models",
                node_type=NodeType.BENCHMARKING,
                function=lambda **kwargs: WorkflowOrchestrator()._benchmark_models_node(**kwargs),
                inputs=["preprocessor_info"],
                outputs=["benchmark_results"],
                dependencies=["preprocess"]
            ),
            WorkflowNode(
                id="train_best",
                name="Train Best Model",
                node_type=NodeType.MODEL_TRAINING,
                function=lambda **kwargs: WorkflowOrchestrator()._train_model_node(**kwargs),
                inputs=["benchmark_results"],
                outputs=["model"],
                dependencies=["benchmark"]
            ),
            WorkflowNode(
                id="evaluate",
                name="Evaluate Model",
                node_type=NodeType.MODEL_EVALUATION,
                function=lambda **kwargs: WorkflowOrchestrator()._evaluate_model_node(**kwargs),
                inputs=["model"],
                outputs=["evaluation_results"],
                dependencies=["train_best"]
            ),
            WorkflowNode(
                id="explain",
                name="Generate Explanations",
                node_type=NodeType.EXPLAINABILITY,
                function=lambda **kwargs: WorkflowOrchestrator()._generate_explanations_node(**kwargs),
                inputs=["model"],
                outputs=["explanations"],
                dependencies=["evaluate"]
            ),
            WorkflowNode(
                id="deploy",
                name="Deploy Model",
                node_type=NodeType.MODEL_DEPLOYMENT,
                function=lambda **kwargs: WorkflowOrchestrator()._deploy_model_node(**kwargs),
                inputs=["model"],
                outputs=["deployment_info"],
                dependencies=["explain"]
            )
        ]

# Global orchestrator instance
orchestrator = WorkflowOrchestrator() 