"""
Tensor Fragment Architecture for Cognitive States
================================================

Implements the tensor fragment encoding scheme for representing agent/state
as hypergraph nodes/links with tensor shapes: [modality, depth, context, salience, autonomy_index].
"""

from typing import Any, Dict, List, Optional, Tuple, Union
import numpy as np
from dataclasses import dataclass
import math

from .hypergraph import HypergraphNode, HypergraphLink


@dataclass
class TensorSignature:
    """
    Mathematical signature for tensor fragments with prime factorization mapping.
    
    The tensor signature encodes cognitive state dimensions using prime factorization
    for efficient compression and mathematical soundness.
    
    Attributes
    ----------
    modality : int
        Input modality dimension (visual=2, audio=3, text=5, multimodal=7)
    depth : int
        Processing depth in the reservoir hierarchy (prime factorized)
    context : int
        Contextual memory span (Fibonacci-based for temporal dynamics)
    salience : int
        Attention/salience weighting (log-scaled prime)
    autonomy_index : int
        Agent autonomy level (prime factorized behavioral complexity)
    """
    modality: int
    depth: int
    context: int
    salience: int
    autonomy_index: int
    
    @classmethod
    def create_signature(
        cls,
        modality_type: str = "multimodal",
        processing_depth: int = 1,
        context_span: int = 10,
        salience_weight: float = 1.0,
        autonomy_level: int = 1
    ) -> "TensorSignature":
        """
        Create a tensor signature from semantic parameters.
        
        Parameters
        ----------
        modality_type : str
            Type of input modality ('visual', 'audio', 'text', 'multimodal')
        processing_depth : int
            Depth of processing in reservoir hierarchy
        context_span : int
            Length of contextual memory
        salience_weight : float
            Attention/salience weight (0.0 to 1.0)
        autonomy_level : int
            Level of agent autonomy
            
        Returns
        -------
        TensorSignature
            Mathematically encoded tensor signature
        """
        # Modality encoding using primes
        modality_map = {
            "visual": 2,
            "audio": 3, 
            "text": 5,
            "multimodal": 7
        }
        modality = modality_map.get(modality_type, 7)
        
        # Depth encoding using prime factorization
        depth = cls._prime_factorize_depth(processing_depth)
        
        # Context encoding using Fibonacci-like sequence for temporal dynamics
        context = cls._fibonacci_context(context_span)
        
        # Salience encoding using log-scaled primes
        salience = cls._encode_salience(salience_weight)
        
        # Autonomy encoding using prime factorization of complexity
        autonomy_index = cls._encode_autonomy(autonomy_level)
        
        return cls(modality, depth, context, salience, autonomy_index)
    
    @staticmethod
    def _prime_factorize_depth(depth: int) -> int:
        """Encode processing depth using prime factorization."""
        primes = [2, 3, 5, 7, 11, 13, 17, 19, 23, 29]
        if depth <= 0:
            return 1
        if depth >= len(primes):
            return primes[-1] * (depth - len(primes) + 1)
        return primes[depth - 1]
    
    @staticmethod
    def _fibonacci_context(span: int) -> int:
        """Encode context span using Fibonacci sequence for temporal dynamics."""
        if span <= 0:
            return 1
        if span == 1:
            return 1
        
        # Generate Fibonacci number for context encoding
        a, b = 1, 1
        for _ in range(span - 1):
            a, b = b, a + b
        return min(b, 1000)  # Cap at reasonable size
    
    @staticmethod
    def _encode_salience(weight: float) -> int:
        """Encode salience weight using log-scaled primes."""
        if weight <= 0:
            return 1
        
        # Log scale the weight and map to primes
        log_weight = max(1, int(np.log10(weight * 10) * 3))
        primes = [2, 3, 5, 7, 11, 13, 17, 19, 23, 29]
        return primes[min(log_weight - 1, len(primes) - 1)]
    
    @staticmethod
    def _encode_autonomy(level: int) -> int:
        """Encode autonomy level using prime factorization of behavioral complexity."""
        if level <= 0:
            return 1
        
        # Map autonomy level to composite numbers for complexity encoding
        autonomy_map = {
            1: 2,      # Basic reactive
            2: 6,      # 2*3 - Simple planning
            3: 30,     # 2*3*5 - Multi-goal
            4: 210,    # 2*3*5*7 - Strategic
            5: 2310,   # 2*3*5*7*11 - Autonomous
        }
        
        if level <= 5:
            return autonomy_map[level]
        else:
            # For higher levels, continue the prime product pattern
            return autonomy_map[5] * (level - 4)
    
    def get_tensor_shape(self) -> Tuple[int, int, int, int, int]:
        """Get the tensor shape tuple."""
        return (self.modality, self.depth, self.context, self.salience, self.autonomy_index)
    
    def get_total_dimensions(self) -> int:
        """Get total number of tensor dimensions."""
        return self.modality * self.depth * self.context * self.salience * self.autonomy_index
    
    def get_prime_factorization(self) -> Dict[str, List[int]]:
        """Get prime factorization for each dimension."""
        return {
            "modality": self._factorize(self.modality),
            "depth": self._factorize(self.depth),
            "context": self._factorize(self.context),
            "salience": self._factorize(self.salience),
            "autonomy_index": self._factorize(self.autonomy_index)
        }
    
    @staticmethod
    def _factorize(n: int) -> List[int]:
        """Get prime factorization of a number."""
        factors = []
        d = 2
        while d * d <= n:
            while n % d == 0:
                factors.append(d)
                n //= d
            d += 1
        if n > 1:
            factors.append(n)
        return factors
    

class TensorFragment:
    """
    Tensor fragment for encoding cognitive states in hypergraph representation.
    
    A TensorFragment encodes agent/state information using the five-dimensional
    tensor signature: [modality, depth, context, salience, autonomy_index].
    
    Parameters
    ----------
    signature : TensorSignature
        Mathematical signature defining tensor dimensions
    data : np.ndarray, optional
        Actual tensor data matching the signature shape
    metadata : dict, optional
        Additional metadata about the cognitive state
    """
    
    def __init__(
        self,
        signature: TensorSignature,
        data: Optional[np.ndarray] = None,
        metadata: Optional[Dict[str, Any]] = None
    ):
        self.signature = signature
        self.metadata = metadata or {}
        
        # Initialize tensor data if not provided
        if data is None:
            shape = signature.get_tensor_shape()
            self.data = np.zeros(shape, dtype=np.float32)
        else:
            expected_shape = signature.get_tensor_shape()
            if data.shape != expected_shape:
                raise ValueError(f"Data shape {data.shape} doesn't match signature shape {expected_shape}")
            self.data = data.astype(np.float32)
    
    def encode_reservoir_state(self, state: np.ndarray, reservoir_info: Dict[str, Any]) -> None:
        """
        Encode a ReservoirPy reservoir state into the tensor fragment.
        
        Parameters
        ----------
        state : np.ndarray
            Reservoir state vector
        reservoir_info : dict
            Information about the reservoir (type, parameters, etc.)
        """
        if state is None or len(state) == 0:
            return
            
        # Reshape state to fit tensor signature dimensions
        target_shape = self.signature.get_tensor_shape()
        total_elements = np.prod(target_shape)
        
        # Pad or truncate state to fit
        if len(state.flatten()) < total_elements:
            # Pad with zeros
            padded_state = np.zeros(total_elements)
            padded_state[:len(state.flatten())] = state.flatten()
            reshaped_state = padded_state.reshape(target_shape)
        else:
            # Truncate or subsample
            truncated_state = state.flatten()[:total_elements]
            reshaped_state = truncated_state.reshape(target_shape)
        
        self.data = reshaped_state.astype(np.float32)
        
        # Store encoding metadata
        self.metadata.update({
            "original_state_shape": state.shape,
            "reservoir_info": reservoir_info,
            "encoding_method": "reshape_pad_truncate"
        })
    
    def decode_to_reservoir_state(self, target_shape: Optional[Tuple[int, ...]] = None) -> np.ndarray:
        """
        Decode the tensor fragment back to a reservoir state.
        
        Parameters
        ----------
        target_shape : tuple, optional
            Target shape for the decoded state. If None, uses original shape from metadata.
            
        Returns
        -------
        np.ndarray
            Decoded reservoir state
        """
        # Get target shape from metadata or parameter
        if target_shape is None:
            target_shape = self.metadata.get("original_state_shape")
        
        if target_shape is None:
            # Default to flattened tensor
            return self.data.flatten()
        
        # Flatten tensor data
        flat_data = self.data.flatten()
        target_elements = np.prod(target_shape)
        
        if len(flat_data) < target_elements:
            # Pad if needed
            padded_data = np.zeros(target_elements)
            padded_data[:len(flat_data)] = flat_data
            return padded_data.reshape(target_shape)
        else:
            # Truncate if needed
            return flat_data[:target_elements].reshape(target_shape)
    
    def to_hypergraph_node(self, name: str, node_type: str = "tensor_fragment") -> HypergraphNode:
        """
        Convert the tensor fragment to a hypergraph node.
        
        Parameters
        ----------
        name : str
            Name for the hypergraph node
        node_type : str
            Type classification for the node
            
        Returns
        -------
        HypergraphNode
            Hypergraph node representation
        """
        properties = {
            "signature": {
                "modality": self.signature.modality,
                "depth": self.signature.depth,
                "context": self.signature.context,
                "salience": self.signature.salience,
                "autonomy_index": self.signature.autonomy_index
            },
            "tensor_shape": self.signature.get_tensor_shape(),
            "total_dimensions": self.signature.get_total_dimensions(),
            "prime_factorization": self.signature.get_prime_factorization(),
            "metadata": self.metadata.copy()
        }
        
        return HypergraphNode(
            name=name,
            node_type=node_type,
            properties=properties,
            tensor_data=self.data.copy()
        )
    
    @classmethod
    def from_hypergraph_node(cls, node: HypergraphNode) -> "TensorFragment":
        """
        Create a tensor fragment from a hypergraph node.
        
        Parameters
        ----------
        node : HypergraphNode
            Source hypergraph node
            
        Returns
        -------
        TensorFragment
            Reconstructed tensor fragment
        """
        if "signature" not in node.properties:
            raise ValueError("Node does not contain tensor signature information")
        
        sig_data = node.properties["signature"]
        signature = TensorSignature(
            modality=sig_data["modality"],
            depth=sig_data["depth"],
            context=sig_data["context"],
            salience=sig_data["salience"],
            autonomy_index=sig_data["autonomy_index"]
        )
        
        metadata = node.properties.get("metadata", {})
        
        return cls(
            signature=signature,
            data=node.tensor_data,
            metadata=metadata
        )
    
    def compute_similarity(self, other: "TensorFragment") -> float:
        """
        Compute similarity between tensor fragments.
        
        Parameters
        ----------
        other : TensorFragment
            Other tensor fragment to compare
            
        Returns
        -------
        float
            Similarity score between 0 and 1
        """
        # Signature similarity
        sig_similarity = self._signature_similarity(other.signature)
        
        # Data similarity (normalized correlation)
        data_similarity = self._data_similarity(other)
        
        # Weighted combination
        return 0.3 * sig_similarity + 0.7 * data_similarity
    
    def _signature_similarity(self, other_sig: TensorSignature) -> float:
        """Compute similarity between tensor signatures."""
        # Compare dimensions using relative differences
        dims1 = np.array(self.signature.get_tensor_shape())
        dims2 = np.array(other_sig.get_tensor_shape())
        
        # Avoid division by zero
        max_dims = np.maximum(dims1, dims2)
        max_dims[max_dims == 0] = 1
        
        relative_diff = np.abs(dims1 - dims2) / max_dims
        similarity = 1 - np.mean(relative_diff)
        
        return max(0, similarity)
    
    def _data_similarity(self, other: "TensorFragment") -> float:
        """Compute similarity between tensor data."""
        if self.data.shape != other.data.shape:
            # Reshape to common shape for comparison
            flat1 = self.data.flatten()
            flat2 = other.data.flatten()
            
            min_len = min(len(flat1), len(flat2))
            flat1 = flat1[:min_len]
            flat2 = flat2[:min_len]
        else:
            flat1 = self.data.flatten()
            flat2 = other.data.flatten()
        
        if len(flat1) == 0:
            return 1.0
        
        # Normalized correlation
        if np.std(flat1) == 0 and np.std(flat2) == 0:
            return 1.0 if np.allclose(flat1, flat2) else 0.0
        elif np.std(flat1) == 0 or np.std(flat2) == 0:
            return 0.0
        
        correlation = np.corrcoef(flat1, flat2)[0, 1]
        return max(0, correlation) if not np.isnan(correlation) else 0.0
    
    def __repr__(self) -> str:
        shape = self.signature.get_tensor_shape()
        return f"TensorFragment(shape={shape}, total_dims={self.signature.get_total_dimensions()})"