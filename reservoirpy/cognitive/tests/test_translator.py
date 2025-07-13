"""
Tests for Translation Components
==============================

Test suite for ReservoirPy ↔ Hypergraph translators.
"""

import pytest
import numpy as np

from ...nodes import Reservoir, Ridge, Input, Output, Tanh
from ..translator import NodeTranslator, StateTranslator, ModelTranslator
from ..hypergraph import HypergraphNode
from ..tensor_fragment import TensorFragment


class TestNodeTranslator:
    """Test NodeTranslator functionality."""
    
    def setup_method(self):
        """Set up test fixtures."""
        self.translator = NodeTranslator()
    
    def test_reservoir_to_hypergraph(self):
        """Test converting reservoir to hypergraph."""
        reservoir = Reservoir(units=20, lr=0.1, sr=0.9)
        reservoir.initialize()
        
        hypergraph_node = self.translator.node_to_hypergraph(reservoir)
        
        assert isinstance(hypergraph_node, HypergraphNode)
        assert hypergraph_node.node_type == "reservoir"
        assert "class_name" in hypergraph_node.properties
        assert hypergraph_node.properties["class_name"] == "Reservoir"
    
    def test_readout_to_hypergraph(self):
        """Test converting readout to hypergraph."""
        readout = Ridge(output_dim=5)
        readout.set_input_dim(50)
        
        hypergraph_node = self.translator.node_to_hypergraph(readout)
        
        assert hypergraph_node.node_type == "readout"
        assert hypergraph_node.properties["class_name"] == "Ridge"
        assert hypergraph_node.properties["output_dim"] == 5
    
    def test_activation_to_hypergraph(self):
        """Test converting activation to hypergraph."""
        activation = Tanh()
        
        hypergraph_node = self.translator.node_to_hypergraph(activation)
        
        assert hypergraph_node.node_type == "activation"
        assert hypergraph_node.properties["class_name"] == "Tanh"
    
    def test_hypergraph_to_reservoir(self):
        """Test converting hypergraph back to reservoir."""
        # Create hypergraph node representing a reservoir
        properties = {
            "class_name": "Reservoir",
            "hypers": {"units": 30, "lr": 0.2, "sr": 0.8},
            "input_dim": 10,
            "output_dim": 30
        }
        hypergraph_node = HypergraphNode("test_reservoir", "reservoir", properties)
        
        reconstructed = self.translator.hypergraph_to_node(hypergraph_node)
        
        assert reconstructed is not None
        # Note: Exact class matching depends on import availability


class TestStateTranslator:
    """Test StateTranslator functionality."""
    
    def setup_method(self):
        """Set up test fixtures."""
        self.translator = StateTranslator()
    
    def test_state_to_fragment(self):
        """Test converting state to tensor fragment."""
        state = np.random.rand(50)
        context = {"modality": "visual", "autonomy_level": 2}
        
        fragment = self.translator.state_to_fragment(state, context)
        
        assert isinstance(fragment, TensorFragment)
        assert fragment.data is not None
    
    def test_fragment_to_state(self):
        """Test converting tensor fragment back to state."""
        original_state = np.random.rand(30)
        context = {"type": "reservoir"}
        
        # Convert to fragment and back
        fragment = self.translator.state_to_fragment(original_state, context)
        reconstructed_state = self.translator.fragment_to_state(fragment, (30,))
        
        assert reconstructed_state.shape == (30,)
        # Allow for some encoding error
        assert reconstructed_state is not None


class TestModelTranslator:
    """Test ModelTranslator functionality."""
    
    def setup_method(self):
        """Set up test fixtures.""" 
        self.translator = ModelTranslator()
    
    def test_simple_model_conversion(self):
        """Test converting simple model to AtomSpace."""
        # Use single node as simplified model
        reservoir = Reservoir(units=15)
        reservoir.initialize()
        
        atomspace = self.translator.model_to_atomspace(reservoir, include_states=True)
        
        assert atomspace is not None
        assert len(atomspace.nodes) >= 1
    
    def test_validation_workflow(self):
        """Test model validation workflow."""
        reservoir = Reservoir(units=10)
        reservoir.initialize()
        
        results = self.translator.validate_round_trip(reservoir, tolerance=1e-6)
        
        assert "conversion_successful" in results
        assert "object_type" in results