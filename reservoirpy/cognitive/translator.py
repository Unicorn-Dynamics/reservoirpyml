"""
Translation Components for ReservoirPy ↔ Hypergraph Conversion
=============================================================

Bidirectional translators for converting between ReservoirPy primitives 
and hypergraph representations while maintaining semantic integrity.
"""

from typing import Any, Dict, List, Optional, Tuple, Union
import numpy as np

from ..node import Node
from ..model import Model
from .hypergraph import AtomSpace, HypergraphNode, HypergraphLink
from .tensor_fragment import TensorFragment, TensorSignature


class NodeTranslator:
    """
    Translator for ReservoirPy Node objects to hypergraph representation.
    
    Handles the conversion of all ReservoirPy node types including reservoirs,
    readouts, activation functions, and input/output nodes.
    """
    
    def __init__(self):
        self._node_type_map = {
            "Reservoir": "reservoir",
            "ESN": "esn", 
            "Ridge": "readout",
            "FORCE": "readout",
            "LMS": "readout",
            "RLS": "readout",
            "Input": "input",
            "Output": "output",
            "Tanh": "activation",
            "Sigmoid": "activation", 
            "ReLU": "activation",
            "Softmax": "activation",
            "Identity": "activation",
            "Concat": "operator",
            "Delay": "operator"
        }
    
    def node_to_hypergraph(self, node: Node, include_state: bool = True) -> HypergraphNode:
        """
        Convert a ReservoirPy Node to a HypergraphNode.
        
        Parameters
        ----------
        node : Node
            ReservoirPy node to convert
        include_state : bool
            Whether to include current state in the conversion
            
        Returns
        -------
        HypergraphNode
            Hypergraph representation of the node
        """
        # Determine node type
        node_class_name = node.__class__.__name__
        hypergraph_type = self._node_type_map.get(node_class_name, "unknown")
        
        # Extract node properties
        properties = {
            "class_name": node_class_name,
            "input_dim": getattr(node, 'input_dim', None),
            "output_dim": getattr(node, 'output_dim', None), 
            "feedback_dim": getattr(node, 'feedback_dim', None),
            "is_trainable": getattr(node, 'is_trainable', False),
            "is_initialized": getattr(node, 'is_initialized', False),
            "fitted": getattr(node, 'fitted', False),
            "hypers": getattr(node, 'hypers', {}).copy() if hasattr(node, 'hypers') else {}
        }
        
        # Include parameters if available
        if hasattr(node, 'params'):
            properties["params"] = {}
            for key, value in node.params.items():
                if isinstance(value, np.ndarray):
                    properties["params"][key] = {
                        "shape": value.shape,
                        "dtype": str(value.dtype),
                        "mean": float(np.mean(value)),
                        "std": float(np.std(value))
                    }
                else:
                    properties["params"][key] = value
        
        # Include state if requested and available
        tensor_data = None
        if include_state and hasattr(node, 'state') and node.state is not None:
            # Handle different state representations
            state_value = node.state
            if callable(state_value):
                # If state is a callable, try to get the actual state
                try:
                    state_value = state_value()
                except:
                    state_value = None
            
            if state_value is not None and hasattr(state_value, 'shape'):
                # Create tensor fragment for state
                try:
                    signature = self._create_signature_for_node(node, hypergraph_type)
                    fragment = TensorFragment(signature)
                    fragment.encode_reservoir_state(
                        state_value, 
                        {"node_type": hypergraph_type, "class_name": node_class_name}
                    )
                    tensor_data = fragment.data
                    properties["tensor_signature"] = signature.get_tensor_shape()
                except Exception as e:
                    # If tensor fragment creation fails, store raw state info
                    properties["state_shape"] = state_value.shape
                    properties["state_encoding_error"] = str(e)
            else:
                # No valid state data
                properties["state_available"] = False
        
        return HypergraphNode(
            name=getattr(node, 'name', f"{node_class_name}_{id(node)}"),
            node_type=hypergraph_type,
            properties=properties,
            tensor_data=tensor_data
        )
    
    def hypergraph_to_node(self, hypergraph_node: HypergraphNode) -> Node:
        """
        Convert a HypergraphNode back to a ReservoirPy Node.
        
        Parameters
        ----------
        hypergraph_node : HypergraphNode
            Hypergraph node to convert
            
        Returns
        -------
        Node
            Reconstructed ReservoirPy node
        """
        properties = hypergraph_node.properties
        class_name = properties.get("class_name", "Node")
        
        # Import the appropriate node class
        try:
            if class_name in ["Reservoir", "ESN"]:
                from ..nodes.reservoirs import Reservoir
                node_class = Reservoir
            elif class_name == "Ridge":
                from ..nodes.readouts import Ridge
                node_class = Ridge
            elif class_name in ["Input", "Output"]:
                from ..nodes.io import Input, Output
                node_class = Input if class_name == "Input" else Output
            elif class_name in ["Tanh", "Sigmoid", "ReLU", "Softmax", "Identity"]:
                from ..nodes.activations import Tanh, Sigmoid, ReLU, Softmax, Identity
                activation_map = {
                    "Tanh": Tanh, "Sigmoid": Sigmoid, "ReLU": ReLU,
                    "Softmax": Softmax, "Identity": Identity
                }
                node_class = activation_map[class_name]
            else:
                # Fallback to base Node class
                from ..node import Node
                node_class = Node
        except ImportError:
            from ..node import Node
            node_class = Node
        
        # Extract hyperparameters
        hypers = properties.get("hypers", {})
        
        # Create node instance
        try:
            if class_name == "Reservoir":
                # Special handling for Reservoir nodes
                units = hypers.get("units", 100)
                lr = hypers.get("lr", 1.0)
                sr = hypers.get("sr", 0.9)
                node = node_class(units=units, lr=lr, sr=sr, **hypers)
            elif hasattr(node_class, '__init__'):
                # Try to create with hyperparameters
                node = node_class(**hypers)
            else:
                node = node_class()
        except Exception:
            # Fallback to default construction
            from ..node import Node
            node = Node()
        
        # Set dimensions if available
        if properties.get("input_dim"):
            node.set_input_dim(properties["input_dim"])
        if properties.get("output_dim"):
            node.set_output_dim(properties["output_dim"])
        
        # Restore state if available
        if hypergraph_node.tensor_data is not None:
            try:
                # Decode tensor fragment back to state
                if "tensor_signature" in properties:
                    signature_shape = properties["tensor_signature"]
                    signature = TensorSignature(*signature_shape)
                    fragment = TensorFragment(signature, hypergraph_node.tensor_data)
                    original_shape = properties.get("state_shape")
                    restored_state = fragment.decode_to_reservoir_state(original_shape)
                    node.set_state_proxy(restored_state)
                else:
                    # Direct tensor data
                    node.set_state_proxy(hypergraph_node.tensor_data)
            except Exception:
                # Fallback: try to set as-is
                try:
                    node.set_state_proxy(hypergraph_node.tensor_data.flatten())
                except Exception:
                    pass
        
        return node
    
    def _create_signature_for_node(self, node: Node, hypergraph_type: str) -> TensorSignature:
        """Create appropriate tensor signature for a node type."""
        # Determine modality based on node type and properties
        modality_type = "multimodal"  # Default
        
        if hypergraph_type == "input":
            modality_type = "multimodal"  # Input can handle any modality
        elif hypergraph_type == "activation":
            modality_type = "text"  # Activations are more like text processing
        elif hypergraph_type == "reservoir":
            modality_type = "multimodal"  # Reservoirs handle complex patterns
        
        # Determine processing depth
        depth = 1
        if hasattr(node, 'hypers'):
            # Use reservoir size or complexity as depth indicator
            units = node.hypers.get('units', 100)
            if units > 500:
                depth = 3
            elif units > 100:
                depth = 2
        
        # Context span based on state size
        context_span = 10  # Default
        if hasattr(node, 'state') and node.state is not None:
            state_value = node.state
            if callable(state_value):
                try:
                    state_value = state_value()
                except:
                    state_value = None
            
            if state_value is not None and hasattr(state_value, 'flatten'):
                context_span = min(20, max(5, len(state_value.flatten()) // 10))
        
        # Salience based on spectral radius or learning rate
        salience_weight = 1.0
        if hasattr(node, 'hypers'):
            sr = node.hypers.get('sr', 1.0)
            lr = node.hypers.get('lr', 1.0)
            salience_weight = min(1.0, max(0.1, (sr + lr) / 2))
        
        # Autonomy based on trainability and complexity
        autonomy_level = 1
        if getattr(node, 'is_trainable', False):
            autonomy_level = 2
        if hasattr(node, 'hypers') and len(node.hypers) > 3:
            autonomy_level = 3
        
        return TensorSignature.create_signature(
            modality_type=modality_type,
            processing_depth=depth,
            context_span=context_span,
            salience_weight=salience_weight,
            autonomy_level=autonomy_level
        )


class StateTranslator:
    """
    Translator for reservoir states and tensor data.
    
    Handles conversion of raw state vectors and tensor data between
    ReservoirPy format and hypergraph tensor fragments.
    """
    
    def state_to_fragment(
        self, 
        state: np.ndarray, 
        context: Dict[str, Any]
    ) -> TensorFragment:
        """
        Convert a reservoir state to a tensor fragment.
        
        Parameters
        ----------
        state : np.ndarray
            Reservoir state vector
        context : dict
            Context information for creating appropriate signature
            
        Returns
        -------
        TensorFragment
            Encoded tensor fragment
        """
        # Create signature based on context
        signature = self._create_signature_from_context(state, context)
        
        # Create fragment and encode state
        fragment = TensorFragment(signature)
        fragment.encode_reservoir_state(state, context)
        
        return fragment
    
    def fragment_to_state(
        self, 
        fragment: TensorFragment, 
        target_shape: Optional[Tuple[int, ...]] = None
    ) -> np.ndarray:
        """
        Convert a tensor fragment back to a reservoir state.
        
        Parameters
        ----------
        fragment : TensorFragment
            Source tensor fragment
        target_shape : tuple, optional
            Target shape for the decoded state
            
        Returns
        -------
        np.ndarray
            Decoded reservoir state
        """
        return fragment.decode_to_reservoir_state(target_shape)
    
    def _create_signature_from_context(
        self, 
        state: np.ndarray, 
        context: Dict[str, Any]
    ) -> TensorSignature:
        """Create tensor signature from state and context."""
        # Analyze state characteristics
        state_size = len(state.flatten())
        state_std = np.std(state)
        state_mean = np.abs(np.mean(state))
        
        # Modality from context or state characteristics
        modality_type = context.get('modality', 'multimodal')
        
        # Depth based on state complexity
        if state_size > 1000:
            depth = 4
        elif state_size > 500:
            depth = 3
        elif state_size > 100:
            depth = 2
        else:
            depth = 1
        
        # Context span based on state size
        context_span = max(5, min(25, state_size // 20))
        
        # Salience based on state variance
        salience_weight = min(1.0, max(0.1, state_std))
        
        # Autonomy based on state characteristics and context
        autonomy_level = context.get('autonomy_level', 1)
        if state_mean > 0.5:  # High activation suggests more autonomy
            autonomy_level += 1
        
        return TensorSignature.create_signature(
            modality_type=modality_type,
            processing_depth=depth,
            context_span=context_span,
            salience_weight=salience_weight,
            autonomy_level=min(5, autonomy_level)
        )


class ModelTranslator:
    """
    Translator for ReservoirPy Model objects to hypergraph representation.
    
    Handles conversion of complete models including node connections and
    data flow patterns.
    """
    
    def __init__(self):
        self.node_translator = NodeTranslator()
        self.state_translator = StateTranslator()
    
    def model_to_atomspace(self, model: Model, include_states: bool = True) -> AtomSpace:
        """
        Convert a ReservoirPy Model to an AtomSpace hypergraph.
        
        Parameters
        ----------
        model : Model
            ReservoirPy model to convert
        include_states : bool
            Whether to include current states in the conversion
            
        Returns
        -------
        AtomSpace
            Hypergraph representation of the model
        """
        atomspace = AtomSpace(name=getattr(model, 'name', 'model'))
        
        # Convert all nodes
        node_map = {}
        if hasattr(model, 'nodes'):
            for node in model.nodes:
                hypergraph_node = self.node_translator.node_to_hypergraph(
                    node, include_state=include_states
                )
                atomspace.add_node(hypergraph_node)
                node_map[node] = hypergraph_node
        
        # Convert connections
        if hasattr(model, 'nodes'):
            for i, source_node in enumerate(model.nodes):
                if hasattr(source_node, 'successors'):
                    for target_node in source_node.successors:
                        if target_node in node_map:
                            # Create connection link
                            connection_link = HypergraphLink(
                                nodes=[node_map[source_node], node_map[target_node]],
                                link_type="connection",
                                properties={
                                    "flow_direction": "forward",
                                    "connection_index": i
                                }
                            )
                            atomspace.add_link(connection_link)
                
                # Handle feedback connections
                if hasattr(source_node, 'feedback') and source_node.feedback:
                    for fb_node in source_node.feedback:
                        if fb_node in node_map:
                            feedback_link = HypergraphLink(
                                nodes=[node_map[fb_node], node_map[source_node]],
                                link_type="feedback",
                                properties={
                                    "flow_direction": "backward",
                                    "feedback_connection": True
                                }
                            )
                            atomspace.add_link(feedback_link)
        
        return atomspace
    
    def atomspace_to_model(self, atomspace: AtomSpace) -> Model:
        """
        Convert an AtomSpace hypergraph back to a ReservoirPy Model.
        
        Parameters
        ----------
        atomspace : AtomSpace
            Source hypergraph representation
            
        Returns
        -------
        Model
            Reconstructed ReservoirPy model
        """
        # Convert nodes back to ReservoirPy nodes
        nodes = []
        hypergraph_to_node_map = {}
        
        for hypergraph_node in atomspace.nodes:
            try:
                reservoirpy_node = self.node_translator.hypergraph_to_node(hypergraph_node)
                nodes.append(reservoirpy_node)
                hypergraph_to_node_map[hypergraph_node] = reservoirpy_node
            except Exception as e:
                # Skip nodes that can't be converted
                print(f"Warning: Could not convert node {hypergraph_node.name}: {e}")
                continue
        
        # Create model
        try:
            from ..model import Model
            model = Model(nodes[0] if nodes else None)
            
            # Add remaining nodes
            for node in nodes[1:]:
                model = model >> node
                
        except Exception:
            # Fallback: return first node or empty model
            if nodes:
                return nodes[0]
            else:
                from ..node import Node
                return Node()
        
        return model
    
    def validate_round_trip(
        self, 
        original_model: Model, 
        tolerance: float = 1e-6
    ) -> Dict[str, Any]:
        """
        Validate round-trip translation fidelity.
        
        Parameters
        ----------
        original_model : Model
            Original ReservoirPy model
        tolerance : float
            Numerical tolerance for comparison
            
        Returns
        -------
        dict
            Validation results and metrics
        """
        # Convert to hypergraph and back
        atomspace = self.model_to_atomspace(original_model, include_states=True)
        reconstructed_model = self.atomspace_to_model(atomspace)
        
        results = {
            "conversion_successful": True,
            "node_count_match": False,
            "state_fidelity": {},
            "structure_preserved": False,
            "errors": []
        }
        
        try:
            # Check node count
            orig_nodes = getattr(original_model, 'nodes', [])
            recon_nodes = getattr(reconstructed_model, 'nodes', [])
            results["node_count_match"] = len(orig_nodes) == len(recon_nodes)
            
            # Check state fidelity for each node
            for i, (orig_node, recon_node) in enumerate(zip(orig_nodes, recon_nodes)):
                if hasattr(orig_node, 'state') and hasattr(recon_node, 'state'):
                    if orig_node.state is not None and recon_node.state is not None:
                        try:
                            # Compare states
                            orig_flat = orig_node.state.flatten()
                            recon_flat = recon_node.state.flatten()
                            
                            # Handle different lengths
                            min_len = min(len(orig_flat), len(recon_flat))
                            if min_len > 0:
                                mse = np.mean((orig_flat[:min_len] - recon_flat[:min_len]) ** 2)
                                correlation = np.corrcoef(orig_flat[:min_len], recon_flat[:min_len])[0, 1]
                                
                                results["state_fidelity"][f"node_{i}"] = {
                                    "mse": float(mse),
                                    "correlation": float(correlation) if not np.isnan(correlation) else 0.0,
                                    "within_tolerance": mse < tolerance
                                }
                        except Exception as e:
                            results["errors"].append(f"State comparison error for node {i}: {e}")
            
            # Overall structure preservation check
            results["structure_preserved"] = (
                results["node_count_match"] and 
                len(results["errors"]) == 0
            )
            
        except Exception as e:
            results["conversion_successful"] = False
            results["errors"].append(f"Validation error: {e}")
        
        return results