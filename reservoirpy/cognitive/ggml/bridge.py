"""
Neural-Symbolic Bridge
=====================

Bridge between ReservoirPy neural computation and symbolic reasoning via GGML.
"""

from typing import Any, Dict, List, Optional, Tuple, Union
import numpy as np

from .core import GGMLTensor, GGMLContext, TensorType, SymbolicAnnotation
from .kernels import (
    hypergraph_conv, symbolic_activation, attention_weighted_op, 
    HypergraphConvKernel, SymbolicActivationKernel, AttentionWeightedOpKernel
)
from ..hypergraph import HypergraphNode, HypergraphLink, AtomSpace
from ..tensor_fragment import TensorSignature, TensorFragment
from ..attention.ecan import ECANAttentionSystem, AttentionValue
from ..attention.attention_reservoir import AttentionReservoir

import reservoirpy as rpy
from reservoirpy.nodes import Reservoir, Ridge


class NeuralSymbolicBridge:
    """
    Bridge between neural computation (ReservoirPy) and symbolic reasoning (AtomSpace).
    
    This class provides seamless integration between reservoir computing and
    symbolic AI, enabling hybrid neural-symbolic computation.
    
    Parameters
    ----------
    ggml_context : GGMLContext
        GGML computation context
    atom_space : AtomSpace
        AtomSpace for symbolic reasoning
    attention_system : ECANAttentionSystem, optional
        ECAN attention system for resource allocation
    """
    
    def __init__(
        self,
        ggml_context: GGMLContext,
        atom_space: AtomSpace,
        attention_system: Optional[ECANAttentionSystem] = None
    ):
        self.ggml_context = ggml_context
        self.atom_space = atom_space
        self.attention_system = attention_system
        
        # Neural-symbolic mapping
        self.neural_to_symbolic = {}  # ReservoirPy node ID -> symbolic representation
        self.symbolic_to_neural = {}  # Symbolic ID -> ReservoirPy node
        
        # Operation statistics
        self.bridge_operations = 0
        self.neural_operations = 0
        self.symbolic_operations = 0
        
    def reservoir_to_symbolic(self, reservoir: Reservoir, input_data: np.ndarray) -> GGMLTensor:
        """
        Convert ReservoirPy reservoir to symbolic representation.
        
        Parameters
        ----------
        reservoir : Reservoir
            ReservoirPy reservoir node
        input_data : np.ndarray
            Input data to process through reservoir
            
        Returns
        -------
        GGMLTensor
            Symbolic tensor representation of reservoir state
        """
        self.bridge_operations += 1
        
        # Run reservoir to get internal state
        reservoir_state = reservoir(input_data)
        
        # Create GGML tensor from reservoir state
        ggml_tensor = self.ggml_context.create_tensor(
            data=reservoir_state,
            tensor_type=TensorType.NEURAL,
            name=f"reservoir_{reservoir.name if hasattr(reservoir, 'name') else 'unnamed'}"
        )
        
        # Convert to symbolic representation
        symbolic_tensor = ggml_tensor.to_symbolic(self.atom_space)
        
        # Create hypergraph representation of reservoir
        reservoir_node = HypergraphNode(
            name=f"reservoir_{id(reservoir)}",
            node_type="reservoir",
            properties={
                "units": getattr(reservoir, "units", "unknown"),
                "spectral_radius": getattr(reservoir, "sr", "unknown"),
                "leak_rate": getattr(reservoir, "lr", "unknown"),
                "input_scaling": getattr(reservoir, "input_scaling", "unknown"),
                "state_shape": reservoir_state.shape
            },
            tensor_data=reservoir_state
        )
        
        # Add to AtomSpace
        self.atom_space.add_node(reservoir_node)
        
        # Update mapping
        self.neural_to_symbolic[id(reservoir)] = reservoir_node.id
        self.symbolic_to_neural[reservoir_node.id] = reservoir
        
        # Update symbolic annotation
        symbolic_tensor.symbolic_annotation.hypergraph_nodes.append(reservoir_node.id)
        symbolic_tensor.symbolic_annotation.semantic_context.update({
            "reservoir_id": id(reservoir),
            "conversion_type": "reservoir_to_symbolic",
            "reservoir_properties": reservoir_node.properties
        })
        
        return symbolic_tensor
        
    def symbolic_to_reservoir_state(self, symbolic_tensor: GGMLTensor) -> np.ndarray:
        """
        Extract reservoir state from symbolic tensor.
        
        Parameters
        ----------
        symbolic_tensor : GGMLTensor
            Symbolic tensor containing reservoir state
            
        Returns
        -------
        np.ndarray
            Reservoir state data
        """
        self.bridge_operations += 1
        
        # Extract numerical data
        return symbolic_tensor.data
        
    def apply_symbolic_reasoning(
        self,
        input_tensor: GGMLTensor,
        reasoning_type: str = "hypergraph_conv",
        **kwargs
    ) -> GGMLTensor:
        """
        Apply symbolic reasoning to tensor data.
        
        Parameters
        ----------
        input_tensor : GGMLTensor
            Input tensor for reasoning
        reasoning_type : str
            Type of symbolic reasoning to apply
        kwargs : dict
            Additional parameters for reasoning
            
        Returns
        -------
        GGMLTensor
            Result of symbolic reasoning
        """
        self.symbolic_operations += 1
        
        if reasoning_type == "hypergraph_conv":
            # Create dummy edge indices for demonstration
            num_nodes = input_tensor.shape[0] if input_tensor.data.ndim > 1 else 1
            edge_indices = self.ggml_context.create_tensor(
                data=np.array([[i, (i+1) % num_nodes] for i in range(num_nodes)]),
                tensor_type=TensorType.SYMBOLIC,
                name="edge_indices"
            )
            
            result = hypergraph_conv(
                input_tensor, edge_indices, 
                hidden_dim=kwargs.get("hidden_dim", 64),
                num_heads=kwargs.get("num_heads", 4)
            )
            
        elif reasoning_type == "symbolic_activation":
            result = symbolic_activation(
                input_tensor,
                activation_type=kwargs.get("activation_type", "cognitive_tanh")
            )
            
        elif reasoning_type == "attention_reasoning":
            # Use attention system if available
            if self.attention_system:
                # Create attention weights from ECAN system
                attention_concepts = list(self.attention_system.attention_values.keys())[:input_tensor.shape[0]]
                attention_weights = np.array([
                    self.attention_system.attention_values.get(concept, AttentionValue()).sti
                    for concept in attention_concepts
                ])
                # Normalize attention weights
                attention_weights = attention_weights / (np.sum(attention_weights) + 1e-8)
                
                attention_tensor = self.ggml_context.create_tensor(
                    data=attention_weights.reshape(-1, 1),
                    tensor_type=TensorType.SYMBOLIC,
                    name="ecan_attention"
                )
                
                result = attention_weighted_op(
                    input_tensor, attention_tensor,
                    operation_type=kwargs.get("operation_type", "attention_gating")
                )
            else:
                # Fallback to uniform attention
                uniform_attention = self.ggml_context.create_tensor(
                    data=np.ones((input_tensor.shape[0], 1)) / input_tensor.shape[0],
                    tensor_type=TensorType.SYMBOLIC,
                    name="uniform_attention"
                )
                result = attention_weighted_op(input_tensor, uniform_attention)
                
        else:
            raise ValueError(f"Unknown reasoning type: {reasoning_type}")
            
        return result
        
    def create_hybrid_computation(
        self,
        reservoir: Reservoir,
        symbolic_reasoning_steps: List[Dict[str, Any]],
        input_data: np.ndarray
    ) -> Tuple[GGMLTensor, Dict[str, Any]]:
        """
        Create hybrid neural-symbolic computation pipeline.
        
        Parameters
        ----------
        reservoir : Reservoir
            ReservoirPy reservoir for neural computation
        symbolic_reasoning_steps : List[Dict]
            List of symbolic reasoning steps to apply
        input_data : np.ndarray
            Input data for the computation
            
        Returns
        -------
        Tuple[GGMLTensor, Dict]
            Final result tensor and computation statistics
        """
        self.bridge_operations += 1
        
        # Step 1: Neural computation
        symbolic_tensor = self.reservoir_to_symbolic(reservoir, input_data)
        self.neural_operations += 1
        
        # Step 2: Apply symbolic reasoning steps
        current_tensor = symbolic_tensor
        reasoning_history = []
        
        for step in symbolic_reasoning_steps:
            reasoning_type = step.get("type", "hypergraph_conv")
            step_params = step.get("params", {})
            
            previous_tensor = current_tensor
            current_tensor = self.apply_symbolic_reasoning(
                current_tensor, reasoning_type, **step_params
            )
            
            # Record reasoning step
            reasoning_history.append({
                "step": len(reasoning_history),
                "reasoning_type": reasoning_type,
                "input_shape": previous_tensor.shape,
                "output_shape": current_tensor.shape,
                "semantic_context": current_tensor.symbolic_annotation.semantic_context
            })
            
        # Step 3: Create computation statistics
        stats = {
            "neural_operations": self.neural_operations,
            "symbolic_operations": self.symbolic_operations,
            "bridge_operations": self.bridge_operations,
            "reasoning_steps": len(reasoning_history),
            "reasoning_history": reasoning_history,
            "final_tensor_type": current_tensor.tensor_type.value,
            "memory_usage": self.ggml_context.get_memory_usage(),
            "computation_stats": self.ggml_context.get_computation_stats()
        }
        
        return current_tensor, stats
        
    def integrate_with_attention_reservoir(
        self,
        attention_reservoir: AttentionReservoir,
        input_data: np.ndarray,
        symbolic_enhancement: bool = True
    ) -> Tuple[np.ndarray, Dict[str, Any]]:
        """
        Integrate GGML symbolic reasoning with AttentionReservoir.
        
        Parameters
        ----------
        attention_reservoir : AttentionReservoir
            Attention-aware reservoir from Phase 2
        input_data : np.ndarray
            Input data
        symbolic_enhancement : bool
            Whether to apply symbolic enhancement
            
        Returns
        -------
        Tuple[np.ndarray, Dict]
            Enhanced reservoir output and statistics
        """
        self.bridge_operations += 1
        
        # Get standard reservoir output
        reservoir_output = attention_reservoir(input_data)
        
        if not symbolic_enhancement:
            return reservoir_output, {"symbolic_enhancement": False}
            
        # Convert to symbolic representation
        symbolic_tensor = self.ggml_context.create_tensor(
            data=reservoir_output,
            tensor_type=TensorType.NEURAL,
            name="attention_reservoir_output"
        ).to_symbolic(self.atom_space)
        
        # Apply symbolic reasoning based on attention state
        if hasattr(attention_reservoir, 'attention_system') and attention_reservoir.attention_system:
            # Use actual attention values
            enhanced_tensor = self.apply_symbolic_reasoning(
                symbolic_tensor,
                reasoning_type="attention_reasoning",
                operation_type="attention_gating"
            )
        else:
            # Apply general symbolic enhancement
            enhanced_tensor = self.apply_symbolic_reasoning(
                symbolic_tensor,
                reasoning_type="symbolic_activation",
                activation_type="cognitive_tanh"
            )
            
        # Apply hypergraph convolution for additional symbolic processing
        final_tensor = self.apply_symbolic_reasoning(
            enhanced_tensor,
            reasoning_type="hypergraph_conv",
            hidden_dim=min(64, enhanced_tensor.shape[-1])
        )
        
        # Extract enhanced output
        enhanced_output = final_tensor.data
        
        # Compute enhancement statistics
        original_norm = np.linalg.norm(reservoir_output)
        enhanced_norm = np.linalg.norm(enhanced_output)
        
        stats = {
            "symbolic_enhancement": True,
            "original_norm": float(original_norm),
            "enhanced_norm": float(enhanced_norm),
            "enhancement_ratio": float(enhanced_norm / (original_norm + 1e-8)),
            "symbolic_context": final_tensor.symbolic_annotation.semantic_context,
            "reasoning_steps": 2,  # attention + hypergraph conv
            "tensor_type": final_tensor.tensor_type.value
        }
        
        return enhanced_output, stats
        
    def create_tensor_signature_from_reservoir(self, reservoir: Reservoir, input_data: np.ndarray) -> TensorSignature:
        """
        Create tensor signature from reservoir computation.
        
        Parameters
        ----------
        reservoir : Reservoir
            ReservoirPy reservoir
        input_data : np.ndarray
            Input data
            
        Returns
        -------
        TensorSignature
            Cognitive tensor signature
        """
        # Process data through reservoir
        reservoir_state = reservoir(input_data)
        
        # Analyze reservoir properties for signature creation
        modality = 7  # Multimodal default
        depth = int(np.log2(getattr(reservoir, "units", 100))) + 1  # Based on reservoir size
        context = len(input_data) if input_data.ndim > 1 else 1  # Context from input length
        salience = int(np.mean(np.abs(reservoir_state)) * 10) + 1  # Based on activation level
        autonomy_index = 2  # Default autonomy level
        
        return TensorSignature(
            modality=modality,
            depth=depth,
            context=context,
            salience=salience,
            autonomy_index=autonomy_index
        )
        
    def get_bridge_statistics(self) -> Dict[str, Any]:
        """Get comprehensive statistics about bridge operations."""
        total_ops = self.neural_operations + self.symbolic_operations + self.bridge_operations
        
        return {
            "total_operations": total_ops,
            "neural_operations": self.neural_operations,
            "symbolic_operations": self.symbolic_operations,
            "bridge_operations": self.bridge_operations,
            "neural_ratio": self.neural_operations / max(1, total_ops),
            "symbolic_ratio": self.symbolic_operations / max(1, total_ops),
            "bridge_ratio": self.bridge_operations / max(1, total_ops),
            "mappings": {
                "neural_to_symbolic": len(self.neural_to_symbolic),
                "symbolic_to_neural": len(self.symbolic_to_neural)
            },
            "ggml_context_stats": self.ggml_context.get_computation_stats(),
            "atom_space_stats": {
                "nodes": len(self.atom_space.nodes),
                "links": len(self.atom_space.links)
            }
        }