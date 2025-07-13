"""
Main Hypergraph Encoder for ReservoirPy Cognitive Primitives
==========================================================

Primary interface for encoding ReservoirPy primitives as hypergraph patterns
with bidirectional translation capabilities.
"""

from typing import Any, Dict, List, Optional, Union, Tuple
import numpy as np

from ..node import Node
from ..model import Model
from .hypergraph import AtomSpace, HypergraphNode, HypergraphLink
from .tensor_fragment import TensorFragment, TensorSignature
from .translator import NodeTranslator, StateTranslator, ModelTranslator
from .scheme_adapter import SchemeAdapter, SchemeExpression


class HypergraphEncoder:
    """
    Main encoder for ReservoirPy → Hypergraph translation.
    
    Provides a unified interface for converting ReservoirPy primitives
    to hypergraph representations and back, maintaining semantic integrity.
    
    Parameters
    ----------
    use_tensor_fragments : bool, default=True
        Whether to use tensor fragment encoding for states
    scheme_mode : bool, default=False
        Whether to enable Scheme S-expression output
    """
    
    def __init__(self, use_tensor_fragments: bool = True, scheme_mode: bool = False):
        self.use_tensor_fragments = use_tensor_fragments
        self.scheme_mode = scheme_mode
        
        # Initialize translators
        self.node_translator = NodeTranslator()
        self.state_translator = StateTranslator()
        self.model_translator = ModelTranslator()
        self.scheme_adapter = SchemeAdapter()
        
        # Encoding statistics
        self.encoding_stats = {
            "nodes_encoded": 0,
            "models_encoded": 0,
            "states_encoded": 0,
            "round_trips_validated": 0,
            "errors": []
        }
    
    def encode_node(
        self, 
        node: Node, 
        include_state: bool = True,
        name_override: Optional[str] = None
    ) -> Union[HypergraphNode, SchemeExpression]:
        """
        Encode a ReservoirPy Node as a hypergraph pattern.
        
        Parameters
        ----------
        node : Node
            ReservoirPy node to encode
        include_state : bool, default=True
            Whether to include current state in encoding
        name_override : str, optional
            Override name for the encoded node
            
        Returns
        -------
        HypergraphNode or SchemeExpression
            Encoded hypergraph representation
        """
        try:
            # Convert to hypergraph node
            hypergraph_node = self.node_translator.node_to_hypergraph(
                node, include_state=include_state
            )
            
            # Override name if requested
            if name_override:
                hypergraph_node.name = name_override
            
            self.encoding_stats["nodes_encoded"] += 1
            
            # Return Scheme representation if requested
            if self.scheme_mode:
                # Create simple AtomSpace with just this node
                temp_atomspace = AtomSpace("temp")
                temp_atomspace.add_node(hypergraph_node)
                return self.scheme_adapter.atomspace_to_scheme(temp_atomspace)
            
            return hypergraph_node
            
        except Exception as e:
            error_msg = f"Failed to encode node {getattr(node, 'name', 'unknown')}: {e}"
            self.encoding_stats["errors"].append(error_msg)
            raise ValueError(error_msg)
    
    def decode_node(
        self, 
        encoded_node: Union[HypergraphNode, SchemeExpression]
    ) -> Node:
        """
        Decode a hypergraph pattern back to a ReservoirPy Node.
        
        Parameters
        ----------
        encoded_node : HypergraphNode or SchemeExpression
            Encoded hypergraph representation
            
        Returns
        -------
        Node
            Reconstructed ReservoirPy node
        """
        try:
            # Handle Scheme expression input
            if isinstance(encoded_node, SchemeExpression):
                atomspace = self.scheme_adapter.scheme_to_atomspace(encoded_node)
                if atomspace.nodes:
                    hypergraph_node = next(iter(atomspace.nodes))
                else:
                    raise ValueError("No nodes found in Scheme expression")
            else:
                hypergraph_node = encoded_node
            
            return self.node_translator.hypergraph_to_node(hypergraph_node)
            
        except Exception as e:
            error_msg = f"Failed to decode node: {e}"
            self.encoding_stats["errors"].append(error_msg)
            raise ValueError(error_msg)
    
    def encode_model(
        self, 
        model: Model, 
        include_states: bool = True,
        model_name: Optional[str] = None
    ) -> Union[AtomSpace, SchemeExpression]:
        """
        Encode a complete ReservoirPy Model as hypergraph patterns.
        
        Parameters
        ----------
        model : Model
            ReservoirPy model to encode
        include_states : bool, default=True
            Whether to include current states in encoding
        model_name : str, optional
            Override name for the encoded model
            
        Returns
        -------
        AtomSpace or SchemeExpression
            Complete hypergraph representation
        """
        try:
            # Convert to AtomSpace
            atomspace = self.model_translator.model_to_atomspace(
                model, include_states=include_states
            )
            
            # Override name if requested
            if model_name:
                atomspace.name = model_name
            
            self.encoding_stats["models_encoded"] += 1
            
            # Return Scheme representation if requested
            if self.scheme_mode:
                return self.scheme_adapter.atomspace_to_scheme(atomspace)
            
            return atomspace
            
        except Exception as e:
            error_msg = f"Failed to encode model: {e}"
            self.encoding_stats["errors"].append(error_msg)
            raise ValueError(error_msg)
    
    def decode_model(
        self, 
        encoded_model: Union[AtomSpace, SchemeExpression]
    ) -> Model:
        """
        Decode a hypergraph representation back to a ReservoirPy Model.
        
        Parameters
        ----------
        encoded_model : AtomSpace or SchemeExpression
            Encoded hypergraph representation
            
        Returns
        -------
        Model
            Reconstructed ReservoirPy model
        """
        try:
            # Handle Scheme expression input
            if isinstance(encoded_model, SchemeExpression):
                atomspace = self.scheme_adapter.scheme_to_atomspace(encoded_model)
            else:
                atomspace = encoded_model
            
            return self.model_translator.atomspace_to_model(atomspace)
            
        except Exception as e:
            error_msg = f"Failed to decode model: {e}"
            self.encoding_stats["errors"].append(error_msg)
            raise ValueError(error_msg)
    
    def encode_state(
        self, 
        state: np.ndarray, 
        context: Optional[Dict[str, Any]] = None
    ) -> TensorFragment:
        """
        Encode a reservoir state as a tensor fragment.
        
        Parameters
        ----------
        state : np.ndarray
            Reservoir state vector
        context : dict, optional
            Context information for tensor signature creation
            
        Returns
        -------
        TensorFragment
            Encoded tensor fragment
        """
        if not self.use_tensor_fragments:
            raise ValueError("Tensor fragment encoding is disabled")
        
        try:
            context = context or {}
            fragment = self.state_translator.state_to_fragment(state, context)
            self.encoding_stats["states_encoded"] += 1
            return fragment
            
        except Exception as e:
            error_msg = f"Failed to encode state: {e}"
            self.encoding_stats["errors"].append(error_msg)
            raise ValueError(error_msg)
    
    def decode_state(
        self, 
        fragment: TensorFragment, 
        target_shape: Optional[tuple] = None
    ) -> np.ndarray:
        """
        Decode a tensor fragment back to a reservoir state.
        
        Parameters
        ----------
        fragment : TensorFragment
            Source tensor fragment
        target_shape : tuple, optional
            Target shape for decoded state
            
        Returns
        -------
        np.ndarray
            Decoded reservoir state
        """
        try:
            return self.state_translator.fragment_to_state(fragment, target_shape)
            
        except Exception as e:
            error_msg = f"Failed to decode state: {e}"
            self.encoding_stats["errors"].append(error_msg)
            raise ValueError(error_msg)
    
    def validate_round_trip(
        self, 
        original: Union[Node, Model], 
        tolerance: float = 1e-6
    ) -> Dict[str, Any]:
        """
        Validate round-trip translation fidelity.
        
        Parameters
        ----------
        original : Node or Model
            Original ReservoirPy object
        tolerance : float, default=1e-6
            Numerical tolerance for comparison
            
        Returns
        -------
        dict
            Validation results and fidelity metrics
        """
        try:
            if isinstance(original, Node):
                # Node round-trip validation
                encoded = self.encode_node(original, include_state=True)
                decoded = self.decode_node(encoded)
                
                results = {
                    "object_type": "Node",
                    "round_trip_successful": True,
                    "class_match": decoded.__class__.__name__ == original.__class__.__name__,
                    "state_fidelity": None,
                    "errors": []
                }
                
                # Compare states if available
                if hasattr(original, 'state') and hasattr(decoded, 'state'):
                    if original.state is not None and decoded.state is not None:
                        orig_flat = original.state.flatten()
                        dec_flat = decoded.state.flatten()
                        
                        min_len = min(len(orig_flat), len(dec_flat))
                        if min_len > 0:
                            mse = np.mean((orig_flat[:min_len] - dec_flat[:min_len]) ** 2)
                            correlation = np.corrcoef(orig_flat[:min_len], dec_flat[:min_len])[0, 1]
                            
                            results["state_fidelity"] = {
                                "mse": float(mse),
                                "correlation": float(correlation) if not np.isnan(correlation) else 0.0,
                                "within_tolerance": mse < tolerance
                            }
                
            elif isinstance(original, Model):
                # Model round-trip validation
                results = self.model_translator.validate_round_trip(original, tolerance)
                results["object_type"] = "Model"
            else:
                raise ValueError(f"Unsupported object type: {type(original)}")
            
            self.encoding_stats["round_trips_validated"] += 1
            return results
            
        except Exception as e:
            error_msg = f"Round-trip validation failed: {e}"
            self.encoding_stats["errors"].append(error_msg)
            return {
                "object_type": type(original).__name__,
                "round_trip_successful": False,
                "errors": [error_msg]
            }
    
    def create_cognitive_pattern(
        self, 
        pattern_name: str,
        nodes: List[HypergraphNode],
        relationships: List[Tuple[str, str, str]],
        description: Optional[str] = None
    ) -> AtomSpace:
        """
        Create a reusable cognitive pattern from nodes and relationships.
        
        Parameters
        ----------
        pattern_name : str
            Name for the cognitive pattern
        nodes : List[HypergraphNode]
            List of hypergraph nodes in the pattern
        relationships : List[Tuple[str, str, str]]
            List of (source_name, target_name, relation_type) tuples
        description : str, optional
            Description of the cognitive pattern
            
        Returns
        -------
        AtomSpace
            AtomSpace containing the cognitive pattern
        """
        atomspace = AtomSpace(pattern_name)
        
        # Add all nodes
        node_map = {}
        for node in nodes:
            atomspace.add_node(node)
            node_map[node.name] = node
        
        # Create relationship links
        for source_name, target_name, relation_type in relationships:
            if source_name in node_map and target_name in node_map:
                link = HypergraphLink(
                    nodes=[node_map[source_name], node_map[target_name]],
                    link_type=relation_type,
                    properties={"pattern": pattern_name}
                )
                atomspace.add_link(link)
        
        # Add pattern metadata
        if description:
            pattern_node = HypergraphNode(
                name=f"{pattern_name}_metadata",
                node_type="pattern_metadata",
                properties={
                    "description": description,
                    "node_count": len(nodes),
                    "relationship_count": len(relationships)
                }
            )
            atomspace.add_node(pattern_node)
        
        return atomspace
    
    def get_encoding_statistics(self) -> Dict[str, Any]:
        """Get encoding statistics and performance metrics."""
        return self.encoding_stats.copy()
    
    def reset_statistics(self) -> None:
        """Reset encoding statistics."""
        self.encoding_stats = {
            "nodes_encoded": 0,
            "models_encoded": 0,
            "states_encoded": 0,
            "round_trips_validated": 0,
            "errors": []
        }
    
    def set_scheme_mode(self, enabled: bool) -> None:
        """Enable or disable Scheme S-expression output mode."""
        self.scheme_mode = enabled
    
    def set_tensor_fragments(self, enabled: bool) -> None:
        """Enable or disable tensor fragment encoding for states."""
        self.use_tensor_fragments = enabled