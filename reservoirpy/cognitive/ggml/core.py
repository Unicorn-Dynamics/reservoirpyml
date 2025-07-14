"""
Core GGML Interface Components
=============================

Core tensor and context classes for neural-symbolic computation.
"""

from typing import Any, Dict, List, Optional, Tuple, Union
import numpy as np
from dataclasses import dataclass, field
import uuid
from enum import Enum

from ..hypergraph import HypergraphNode, HypergraphLink, AtomSpace
from ..tensor_fragment import TensorSignature


class TensorType(Enum):
    """Tensor type enumeration for neural-symbolic operations."""
    NEURAL = "neural"
    SYMBOLIC = "symbolic" 
    HYBRID = "hybrid"


@dataclass
class SymbolicAnnotation:
    """
    Symbolic annotation for tensor data linking to hypergraph representations.
    
    Attributes
    ----------
    atom_space_ref : str
        Reference to AtomSpace representation
    hypergraph_nodes : List[str]
        Associated hypergraph node IDs
    semantic_context : Dict[str, Any]
        Semantic context information
    attention_weight : float
        Attention weight for this annotation
    """
    atom_space_ref: Optional[str] = None
    hypergraph_nodes: List[str] = field(default_factory=list)
    semantic_context: Dict[str, Any] = field(default_factory=dict)
    attention_weight: float = 1.0


class GGMLTensor:
    """
    GGML-compatible tensor with symbolic annotations.
    
    A tensor that combines numerical computation with symbolic representations,
    enabling neural-symbolic reasoning in reservoir computing.
    
    Parameters
    ----------
    data : np.ndarray
        Numerical tensor data
    tensor_type : TensorType
        Type of tensor (neural, symbolic, or hybrid)
    symbolic_annotation : SymbolicAnnotation, optional
        Symbolic annotation linking to hypergraph
    tensor_signature : TensorSignature, optional
        Cognitive tensor signature
    name : str, optional
        Human-readable name for the tensor
    """
    
    def __init__(
        self,
        data: np.ndarray,
        tensor_type: TensorType = TensorType.NEURAL,
        symbolic_annotation: Optional[SymbolicAnnotation] = None,
        tensor_signature: Optional[TensorSignature] = None,
        name: Optional[str] = None
    ):
        self.data = np.asarray(data)
        self.tensor_type = tensor_type
        self.symbolic_annotation = symbolic_annotation or SymbolicAnnotation()
        self.tensor_signature = tensor_signature
        self.name = name or f"tensor_{str(uuid.uuid4())[:8]}"
        self.id = str(uuid.uuid4())
        
        # Metadata for neural-symbolic operations
        self.grad = None  # Gradient tensor
        self.requires_grad = False
        self.computation_graph = []  # Track symbolic operations
        
    @property
    def shape(self) -> Tuple[int, ...]:
        """Get tensor shape."""
        return self.data.shape
        
    @property
    def dtype(self) -> np.dtype:
        """Get tensor data type."""
        return self.data.dtype
        
    @property
    def size(self) -> int:
        """Get total number of elements."""
        return self.data.size
        
    def to_symbolic(self, atom_space: AtomSpace) -> "GGMLTensor":
        """
        Convert tensor to symbolic representation.
        
        Parameters
        ----------
        atom_space : AtomSpace
            AtomSpace for symbolic representation
            
        Returns
        -------
        GGMLTensor
            Tensor with symbolic type and annotation
        """
        # Create hypergraph representation of tensor structure
        tensor_node = HypergraphNode(
            name=f"tensor_{self.name}",
            node_type="tensor",
            properties={
                "shape": self.shape,
                "dtype": str(self.dtype),
                "tensor_id": self.id
            },
            tensor_data=self.data
        )
        
        # Add to AtomSpace
        atom_space.add_node(tensor_node)
        
        # Create symbolic annotation
        annotation = SymbolicAnnotation(
            atom_space_ref=str(id(atom_space)),
            hypergraph_nodes=[tensor_node.id],
            semantic_context={
                "tensor_name": self.name,
                "creation_context": "to_symbolic_conversion"
            }
        )
        
        return GGMLTensor(
            data=self.data.copy(),
            tensor_type=TensorType.SYMBOLIC,
            symbolic_annotation=annotation,
            tensor_signature=self.tensor_signature,
            name=f"symbolic_{self.name}"
        )
        
    def __add__(self, other: "GGMLTensor") -> "GGMLTensor":
        """Element-wise addition with symbolic tracking."""
        result_data = self.data + other.data
        
        # Track symbolic operation
        op_record = {
            "operation": "add",
            "inputs": [self.id, other.id],
            "output_shape": result_data.shape
        }
        
        result = GGMLTensor(
            data=result_data,
            tensor_type=TensorType.HYBRID if self.tensor_type != other.tensor_type else self.tensor_type,
            name=f"add_{self.name}_{other.name}"
        )
        result.computation_graph.append(op_record)
        
        return result
        
    def __mul__(self, other: Union["GGMLTensor", float]) -> "GGMLTensor":
        """Element-wise multiplication with symbolic tracking."""
        if isinstance(other, GGMLTensor):
            result_data = self.data * other.data
            other_id = other.id
        else:
            result_data = self.data * other
            other_id = f"scalar_{other}"
            
        # Track symbolic operation
        op_record = {
            "operation": "mul",
            "inputs": [self.id, other_id],
            "output_shape": result_data.shape
        }
        
        result = GGMLTensor(
            data=result_data,
            tensor_type=self.tensor_type,
            name=f"mul_{self.name}"
        )
        result.computation_graph.append(op_record)
        
        return result
        
    def matmul(self, other: "GGMLTensor") -> "GGMLTensor":
        """Matrix multiplication with symbolic tracking."""
        result_data = np.matmul(self.data, other.data)
        
        # Track symbolic operation
        op_record = {
            "operation": "matmul",
            "inputs": [self.id, other.id],
            "output_shape": result_data.shape
        }
        
        result = GGMLTensor(
            data=result_data,
            tensor_type=TensorType.HYBRID if self.tensor_type != other.tensor_type else self.tensor_type,
            name=f"matmul_{self.name}_{other.name}"
        )
        result.computation_graph.append(op_record)
        
        return result
        
    def __repr__(self) -> str:
        return f"GGMLTensor(shape={self.shape}, type={self.tensor_type.value}, name='{self.name}')"


class GGMLContext:
    """
    Computation context for neural-symbolic operations.
    
    Manages the computational graph, memory allocation, and symbolic reasoning
    context for GGML tensor operations.
    
    Parameters
    ----------
    mem_size : int
        Memory size for tensor operations (bytes)
    atom_space : AtomSpace, optional
        AtomSpace for symbolic reasoning
    enable_gradients : bool
        Whether to compute gradients
    """
    
    def __init__(
        self,
        mem_size: int = 16 * 1024 * 1024,  # 16MB default
        atom_space: Optional[AtomSpace] = None,
        enable_gradients: bool = False
    ):
        self.mem_size = mem_size
        self.atom_space = atom_space or AtomSpace()
        self.enable_gradients = enable_gradients
        
        # Computation tracking
        self.computation_graph = []
        self.tensors = {}  # tensor_id -> GGMLTensor
        self.memory_usage = 0
        
        # Performance metrics
        self.operation_count = 0
        self.symbolic_operations = 0
        self.neural_operations = 0
        
    def create_tensor(
        self,
        data: np.ndarray,
        tensor_type: TensorType = TensorType.NEURAL,
        name: Optional[str] = None
    ) -> GGMLTensor:
        """
        Create a new tensor in this context.
        
        Parameters
        ----------
        data : np.ndarray
            Tensor data
        tensor_type : TensorType
            Type of tensor to create
        name : str, optional
            Name for the tensor
            
        Returns
        -------
        GGMLTensor
            Created tensor
        """
        tensor = GGMLTensor(data=data, tensor_type=tensor_type, name=name)
        self.tensors[tensor.id] = tensor
        self.memory_usage += tensor.data.nbytes
        
        return tensor
        
    def symbolic_op(self, op_name: str, *tensors: GGMLTensor, **kwargs) -> GGMLTensor:
        """
        Execute a symbolic operation on tensors.
        
        Parameters
        ----------
        op_name : str
            Name of the symbolic operation
        tensors : GGMLTensor
            Input tensors
        kwargs : dict
            Additional operation parameters
            
        Returns
        -------
        GGMLTensor
            Result tensor
        """
        self.operation_count += 1
        self.symbolic_operations += 1
        
        # Record operation in computation graph
        op_record = {
            "operation": op_name,
            "inputs": [t.id for t in tensors],
            "parameters": kwargs,
            "timestamp": self.operation_count
        }
        self.computation_graph.append(op_record)
        
        # This is a placeholder - specific operations implemented in kernels
        if op_name == "identity":
            return tensors[0]
        else:
            raise NotImplementedError(f"Symbolic operation '{op_name}' not implemented")
            
    def get_memory_usage(self) -> Dict[str, int]:
        """Get current memory usage statistics."""
        return {
            "total_bytes": self.memory_usage,
            "total_mb": self.memory_usage / (1024 * 1024),
            "tensor_count": len(self.tensors),
            "operation_count": self.operation_count
        }
        
    def get_computation_stats(self) -> Dict[str, Any]:
        """Get computation statistics."""
        return {
            "total_operations": self.operation_count,
            "symbolic_operations": self.symbolic_operations,
            "neural_operations": self.neural_operations,
            "symbolic_ratio": self.symbolic_operations / max(1, self.operation_count),
            "computation_graph_size": len(self.computation_graph)
        }
        
    def __repr__(self) -> str:
        return f"GGMLContext(tensors={len(self.tensors)}, operations={self.operation_count}, mem={self.memory_usage/(1024*1024):.1f}MB)"