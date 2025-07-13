"""
Tests for Main Hypergraph Encoder
=================================

Test suite for the main HypergraphEncoder interface.
"""

import pytest
import numpy as np

from ...nodes import Reservoir, Ridge, Tanh
from ..encoder import HypergraphEncoder
from ..hypergraph import HypergraphNode
from ..scheme_adapter import SchemeExpression


class TestHypergraphEncoder:
    """Test HypergraphEncoder functionality."""
    
    def setup_method(self):
        """Set up test fixtures."""
        self.encoder = HypergraphEncoder()
    
    def test_encode_simple_node(self):
        """Test encoding simple node."""
        node = Tanh()
        node.set_input_dim(10)
        
        encoded = self.encoder.encode_node(node, include_state=False)
        
        assert isinstance(encoded, HypergraphNode)
        assert encoded.node_type == "activation"
    
    def test_encode_node_with_state(self):
        """Test encoding node with state."""
        reservoir = Reservoir(units=20)
        reservoir.initialize()
        
        # Set a state
        test_state = np.random.rand(20)
        reservoir.set_state_proxy(test_state)
        
        encoded = self.encoder.encode_node(reservoir, include_state=True)
        
        assert encoded.tensor_data is not None
        assert "tensor_signature" in encoded.properties
    
    def test_scheme_mode(self):
        """Test encoding in Scheme mode."""
        self.encoder.set_scheme_mode(True)
        
        node = Tanh()
        encoded = self.encoder.encode_node(node)
        
        assert isinstance(encoded, SchemeExpression)
        assert encoded.operator == "AtomSpace"
    
    def test_decode_node(self):
        """Test decoding node."""
        original = Tanh()
        original.set_input_dim(5)
        
        encoded = self.encoder.encode_node(original)
        decoded = self.encoder.decode_node(encoded)
        
        assert decoded is not None
    
    def test_encoding_statistics(self):
        """Test encoding statistics tracking."""
        self.encoder.reset_statistics()
        
        node1 = Tanh()
        node2 = Reservoir(units=10)
        
        self.encoder.encode_node(node1)
        self.encoder.encode_node(node2)
        
        stats = self.encoder.get_encoding_statistics()
        assert stats["nodes_encoded"] == 2
    
    def test_cognitive_pattern_creation(self):
        """Test creating cognitive patterns."""
        nodes = [
            HypergraphNode("input", "input"),
            HypergraphNode("reservoir", "reservoir"),
            HypergraphNode("output", "output")
        ]
        relationships = [
            ("input", "reservoir", "connection"),
            ("reservoir", "output", "connection")
        ]
        
        pattern = self.encoder.create_cognitive_pattern(
            "test_pattern", nodes, relationships, "Test cognitive pattern"
        )
        
        assert pattern.name == "test_pattern"
        assert len(pattern.nodes) >= 3  # Original nodes plus potential metadata
        assert len(pattern.links) == 2
    
    def test_error_handling(self):
        """Test error handling in encoder."""
        # Test with problematic node
        class BadNode:
            def __init__(self):
                self.name = "bad_node"
        
        bad_node = BadNode()
        
        with pytest.raises(ValueError):
            self.encoder.encode_node(bad_node)
    
    def test_tensor_fragment_mode(self):
        """Test tensor fragment encoding mode."""
        self.encoder.set_tensor_fragments(True)
        
        state = np.random.rand(50)
        context = {"modality": "visual"}
        
        fragment = self.encoder.encode_state(state, context)
        decoded_state = self.encoder.decode_state(fragment, (50,))
        
        assert decoded_state.shape == (50,)
    
    def test_disable_tensor_fragments(self):
        """Test disabling tensor fragment encoding."""
        self.encoder.set_tensor_fragments(False)
        
        state = np.random.rand(50)
        
        with pytest.raises(ValueError, match="Tensor fragment encoding is disabled"):
            self.encoder.encode_state(state)