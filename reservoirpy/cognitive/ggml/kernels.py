"""
Symbolic Kernels for Neural-Symbolic Computation
==============================================

Custom kernels implementing symbolic tensor operations for cognitive computing.
"""

from typing import Any, Callable, Dict, List, Optional, Tuple, Union
import numpy as np
from abc import ABC, abstractmethod

from .core import GGMLTensor, GGMLContext, TensorType, SymbolicAnnotation
from ..hypergraph import HypergraphNode, HypergraphLink, AtomSpace
from ..attention.ecan import AttentionValue


class SymbolicKernel(ABC):
    """
    Abstract base class for symbolic tensor operation kernels.
    
    Symbolic kernels implement neural-symbolic operations that bridge
    numerical computation with symbolic reasoning.
    """
    
    def __init__(self, name: str):
        self.name = name
        self.operation_count = 0
        self.performance_metrics = {}
        
    @abstractmethod
    def forward(self, *inputs: GGMLTensor, **kwargs) -> GGMLTensor:
        """Execute the forward pass of the kernel."""
        pass
        
    def __call__(self, *inputs: GGMLTensor, **kwargs) -> GGMLTensor:
        """Execute the kernel."""
        self.operation_count += 1
        return self.forward(*inputs, **kwargs)


class HypergraphConvKernel(SymbolicKernel):
    """
    Hypergraph convolution kernel for processing symbolic structures.
    
    Implements convolution operations over hypergraph structures, enabling
    message passing and feature aggregation in symbolic representations.
    
    Parameters
    ----------
    hidden_dim : int
        Hidden dimension for convolution
    num_heads : int
        Number of attention heads for multi-head convolution
    """
    
    def __init__(self, hidden_dim: int = 64, num_heads: int = 4):
        super().__init__("hypergraph_conv")
        self.hidden_dim = hidden_dim
        self.num_heads = num_heads
        
        # Initialize learnable parameters - will be resized dynamically
        self.W_node = None
        self.W_edge = None
        self.attention_weights = None
        
    def _initialize_weights(self, feature_dim: int):
        """Initialize weights based on actual feature dimension."""
        if self.W_node is None or self.W_node.shape[0] != feature_dim:
            self.W_node = np.random.randn(feature_dim, self.hidden_dim) * 0.1
            self.W_edge = np.random.randn(feature_dim, self.hidden_dim) * 0.1
            self.attention_weights = np.random.randn(self.num_heads, self.hidden_dim) * 0.1
        
    def forward(self, node_features: GGMLTensor, edge_indices: GGMLTensor, 
                edge_features: Optional[GGMLTensor] = None, **kwargs) -> GGMLTensor:
        """
        Execute hypergraph convolution.
        
        Parameters
        ----------
        node_features : GGMLTensor
            Node feature tensor [num_nodes, feature_dim]
        edge_indices : GGMLTensor
            Hyperedge indices [num_edges, max_edge_size]
        edge_features : GGMLTensor, optional
            Edge feature tensor [num_edges, feature_dim]
            
        Returns
        -------
        GGMLTensor
            Updated node features after convolution
        """
        # Basic hypergraph convolution implementation
        num_nodes = node_features.shape[0]
        feature_dim = node_features.shape[1]
        
        # Initialize weights if needed
        self._initialize_weights(feature_dim)
        
        # Node transformation
        transformed_nodes = node_features.data @ self.W_node
        
        # Edge aggregation (simplified)
        if edge_features is not None:
            transformed_edges = edge_features.data @ self.W_edge
            # Aggregate edge information to nodes (placeholder implementation)
            aggregated = transformed_nodes + np.mean(transformed_edges, axis=0, keepdims=True)
        else:
            aggregated = transformed_nodes
            
        # Apply attention mechanism
        attention_scores = np.tanh(aggregated @ self.attention_weights.T)
        attended_features = aggregated * np.mean(attention_scores, axis=1, keepdims=True)
        
        # Create result tensor with symbolic annotation
        result = GGMLTensor(
            data=attended_features,
            tensor_type=TensorType.SYMBOLIC,
            name=f"hypergraph_conv_{node_features.name}"
        )
        
        # Update symbolic annotation
        result.symbolic_annotation = SymbolicAnnotation(
            semantic_context={
                "operation": "hypergraph_convolution",
                "input_nodes": num_nodes,
                "hidden_dim": self.hidden_dim,
                "num_heads": self.num_heads
            }
        )
        
        return result


class SymbolicActivationKernel(SymbolicKernel):
    """
    Symbolic activation functions for cognitive processing.
    
    Implements activation functions that maintain symbolic meaning
    while performing numerical computation.
    
    Parameters
    ----------
    activation_type : str
        Type of activation ('cognitive_tanh', 'symbolic_relu', 'attention_softmax')
    cognitive_params : dict
        Parameters for cognitive processing
    """
    
    def __init__(self, activation_type: str = "cognitive_tanh", cognitive_params: Optional[Dict] = None):
        super().__init__(f"symbolic_activation_{activation_type}")
        self.activation_type = activation_type
        self.cognitive_params = cognitive_params or {}
        
    def forward(self, input_tensor: GGMLTensor, **kwargs) -> GGMLTensor:
        """
        Execute symbolic activation.
        
        Parameters
        ----------
        input_tensor : GGMLTensor
            Input tensor to activate
            
        Returns
        -------
        GGMLTensor
            Activated tensor with symbolic properties
        """
        if self.activation_type == "cognitive_tanh":
            # Cognitive tanh with symbolic interpretation
            activated_data = np.tanh(input_tensor.data)
            
            # Add cognitive interpretation: values close to ±1 are "certain",
            # values close to 0 are "uncertain"
            certainty_mask = np.abs(activated_data) > 0.7
            
        elif self.activation_type == "symbolic_relu":
            # ReLU with symbolic sparsity interpretation
            activated_data = np.maximum(0, input_tensor.data)
            
            # Symbolic interpretation: zero values represent "inactive concepts"
            certainty_mask = activated_data > 0
            
        elif self.activation_type == "attention_softmax":
            # Softmax with attention interpretation
            exp_data = np.exp(input_tensor.data - np.max(input_tensor.data, axis=-1, keepdims=True))
            activated_data = exp_data / np.sum(exp_data, axis=-1, keepdims=True)
            
            # Symbolic interpretation: high values are "attended", low values are "ignored"
            certainty_mask = activated_data > (1.0 / activated_data.shape[-1])  # Above uniform
            
        else:
            raise ValueError(f"Unknown activation type: {self.activation_type}")
            
        # Create result tensor
        result = GGMLTensor(
            data=activated_data,
            tensor_type=TensorType.SYMBOLIC,
            name=f"{self.activation_type}_{input_tensor.name}"
        )
        
        # Update symbolic annotation with cognitive interpretation
        result.symbolic_annotation = SymbolicAnnotation(
            semantic_context={
                "activation_type": self.activation_type,
                "certainty_ratio": np.mean(certainty_mask),
                "symbolic_sparsity": 1.0 - np.mean(activated_data != 0),
                "cognitive_interpretation": self._get_cognitive_interpretation(activated_data)
            }
        )
        
        return result
        
    def _get_cognitive_interpretation(self, data: np.ndarray) -> Dict[str, Any]:
        """Generate cognitive interpretation of activation pattern."""
        return {
            "mean_activation": float(np.mean(data)),
            "activation_variance": float(np.var(data)),
            "positive_ratio": float(np.mean(data > 0)),
            "high_confidence_ratio": float(np.mean(np.abs(data) > 0.8))
        }


class AttentionWeightedOpKernel(SymbolicKernel):
    """
    Attention-weighted tensor operations for cognitive focus.
    
    Implements tensor operations modulated by attention weights from
    the ECAN attention system.
    
    Parameters
    ----------
    operation_type : str
        Type of operation ('weighted_sum', 'attention_gating', 'focus_selection')
    """
    
    def __init__(self, operation_type: str = "weighted_sum"):
        super().__init__(f"attention_weighted_{operation_type}")
        self.operation_type = operation_type
        
    def forward(self, input_tensor: GGMLTensor, attention_weights: GGMLTensor, **kwargs) -> GGMLTensor:
        """
        Execute attention-weighted operation.
        
        Parameters
        ----------
        input_tensor : GGMLTensor
            Input tensor to process
        attention_weights : GGMLTensor
            Attention weight tensor
            
        Returns
        -------
        GGMLTensor
            Attention-modulated tensor
        """
        if self.operation_type == "weighted_sum":
            # Weighted sum along specified dimension
            axis = kwargs.get("axis", -1)
            weighted_data = input_tensor.data * attention_weights.data
            result_data = np.sum(weighted_data, axis=axis, keepdims=True)
            
        elif self.operation_type == "attention_gating":
            # Element-wise gating with attention
            result_data = input_tensor.data * attention_weights.data
            
        elif self.operation_type == "focus_selection":
            # Select top-k based on attention weights
            k = kwargs.get("k", 5)
            if attention_weights.data.ndim == 1:
                top_k_indices = np.argsort(attention_weights.data)[-k:]
                result_data = input_tensor.data[top_k_indices]
            else:
                # For multi-dimensional attention, select along last axis
                top_k_indices = np.argsort(attention_weights.data, axis=-1)[..., -k:]
                result_data = np.take_along_axis(input_tensor.data, top_k_indices, axis=-1)
                
        else:
            raise ValueError(f"Unknown attention operation: {self.operation_type}")
            
        # Create result tensor
        result = GGMLTensor(
            data=result_data,
            tensor_type=TensorType.HYBRID,
            name=f"attention_{self.operation_type}_{input_tensor.name}"
        )
        
        # Update symbolic annotation
        attention_stats = {
            "mean_attention": float(np.mean(attention_weights.data)),
            "attention_entropy": float(-np.sum(attention_weights.data * np.log(attention_weights.data + 1e-8))),
            "focus_ratio": float(np.mean(attention_weights.data > np.mean(attention_weights.data)))
        }
        
        result.symbolic_annotation = SymbolicAnnotation(
            semantic_context={
                "operation": f"attention_{self.operation_type}",
                "attention_statistics": attention_stats,
                "cognitive_focus": self._interpret_attention_pattern(attention_weights.data)
            }
        )
        
        return result
        
    def _interpret_attention_pattern(self, attention: np.ndarray) -> Dict[str, Any]:
        """Interpret attention pattern for cognitive insights."""
        # Compute attention distribution statistics
        attention_flat = attention.flatten()
        sorted_attention = np.sort(attention_flat)[::-1]
        
        return {
            "focus_concentration": float(np.sum(sorted_attention[:len(sorted_attention)//10])),  # Top 10%
            "attention_diversity": float(1.0 / (1.0 + np.var(attention_flat))),
            "peak_attention": float(np.max(attention_flat)),
            "attention_sparsity": float(np.mean(attention_flat < 0.1))
        }


class PatternMatchingKernel(SymbolicKernel):
    """
    Symbolic pattern matching kernel for cognitive recognition.
    
    Implements pattern matching operations for recognizing symbolic
    structures in neural representations.
    
    Parameters
    ----------
    pattern_library : dict
        Library of patterns to match against
    matching_threshold : float
        Threshold for pattern matching
    """
    
    def __init__(self, pattern_library: Optional[Dict] = None, matching_threshold: float = 0.8):
        super().__init__("pattern_matching")
        self.pattern_library = pattern_library or {}
        self.matching_threshold = matching_threshold
        
    def forward(self, input_tensor: GGMLTensor, **kwargs) -> GGMLTensor:
        """
        Execute pattern matching.
        
        Parameters
        ----------
        input_tensor : GGMLTensor
            Input tensor to match patterns against
            
        Returns
        -------
        GGMLTensor
            Pattern matching results
        """
        pattern_name = kwargs.get("pattern_name", None)
        
        if pattern_name and pattern_name in self.pattern_library:
            # Match against specific pattern
            pattern = self.pattern_library[pattern_name]
            similarity = self._compute_similarity(input_tensor.data, pattern)
            matches = similarity > self.matching_threshold
            result_data = matches.astype(np.float32)
            
        else:
            # Generic pattern detection (placeholder)
            # Detect repeating patterns, symmetries, etc.
            result_data = self._detect_generic_patterns(input_tensor.data)
            
        # Create result tensor
        result = GGMLTensor(
            data=result_data,
            tensor_type=TensorType.SYMBOLIC,
            name=f"pattern_match_{input_tensor.name}"
        )
        
        # Update symbolic annotation
        result.symbolic_annotation = SymbolicAnnotation(
            semantic_context={
                "operation": "pattern_matching",
                "patterns_detected": int(np.sum(result_data > 0)),
                "match_confidence": float(np.mean(result_data)),
                "pattern_name": pattern_name
            }
        )
        
        return result
        
    def _compute_similarity(self, data: np.ndarray, pattern: np.ndarray) -> np.ndarray:
        """Compute similarity between data and pattern."""
        # Normalized cross-correlation
        data_norm = data / (np.linalg.norm(data) + 1e-8)
        pattern_norm = pattern / (np.linalg.norm(pattern) + 1e-8)
        return np.dot(data_norm.flatten(), pattern_norm.flatten())
        
    def _detect_generic_patterns(self, data: np.ndarray) -> np.ndarray:
        """Detect generic patterns in data."""
        # Simple pattern detection: identify local maxima
        if data.ndim == 1:
            patterns = np.zeros_like(data)
            for i in range(1, len(data) - 1):
                if data[i] > data[i-1] and data[i] > data[i+1]:
                    patterns[i] = 1.0
        else:
            # For multi-dimensional data, detect patterns along each dimension
            patterns = np.zeros_like(data)
            for dim in range(data.ndim):
                axis_data = np.mean(data, axis=tuple(i for i in range(data.ndim) if i != dim))
                axis_patterns = self._detect_generic_patterns(axis_data)
                # Broadcast back to original shape
                expand_dims = [np.newaxis] * data.ndim
                expand_dims[dim] = slice(None)
                patterns += axis_patterns[tuple(expand_dims)]
                
        return patterns


# Convenience functions for kernel operations
def hypergraph_conv(node_features: GGMLTensor, edge_indices: GGMLTensor, 
                   edge_features: Optional[GGMLTensor] = None, 
                   hidden_dim: int = 64, num_heads: int = 4) -> GGMLTensor:
    """Hypergraph convolution operation."""
    kernel = HypergraphConvKernel(hidden_dim=hidden_dim, num_heads=num_heads)
    return kernel(node_features, edge_indices, edge_features)


def symbolic_activation(input_tensor: GGMLTensor, activation_type: str = "cognitive_tanh") -> GGMLTensor:
    """Symbolic activation function."""
    kernel = SymbolicActivationKernel(activation_type=activation_type)
    return kernel(input_tensor)


def attention_weighted_op(input_tensor: GGMLTensor, attention_weights: GGMLTensor, 
                         operation_type: str = "weighted_sum", **kwargs) -> GGMLTensor:
    """Attention-weighted tensor operation."""
    kernel = AttentionWeightedOpKernel(operation_type=operation_type)
    return kernel(input_tensor, attention_weights, **kwargs)


def pattern_matching(input_tensor: GGMLTensor, pattern_name: Optional[str] = None, 
                    matching_threshold: float = 0.8) -> GGMLTensor:
    """Pattern matching operation."""
    kernel = PatternMatchingKernel(matching_threshold=matching_threshold)
    return kernel(input_tensor, pattern_name=pattern_name)