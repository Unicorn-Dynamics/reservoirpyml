"""
Tests for Hypergraph Primitives
==============================

Test suite for basic hypergraph data structures and operations.
"""

import pytest
import numpy as np

from ..hypergraph import HypergraphNode, HypergraphLink, AtomSpace


class TestHypergraphNode:
    """Test HypergraphNode functionality."""
    
    def test_node_creation(self):
        """Test basic node creation."""
        node = HypergraphNode("test_node", "reservoir")
        
        assert node.name == "test_node"
        assert node.node_type == "reservoir"
        assert node.properties == {}
        assert node.tensor_data is None
        assert node.id is not None
    
    def test_node_with_properties(self):
        """Test node creation with properties."""
        properties = {"input_dim": 10, "output_dim": 5}
        node = HypergraphNode("test_node", "readout", properties)
        
        assert node.properties == properties
        assert node.properties["input_dim"] == 10
    
    def test_node_with_tensor_data(self):
        """Test node creation with tensor data."""
        tensor_data = np.random.rand(3, 4, 5)
        node = HypergraphNode("test_node", "tensor_fragment", tensor_data=tensor_data)
        
        assert node.tensor_data is not None
        assert node.tensor_data.shape == (3, 4, 5)
        np.testing.assert_array_equal(node.tensor_data, tensor_data)
    
    def test_node_equality(self):
        """Test node equality based on ID."""
        node1 = HypergraphNode("test", "reservoir")
        node2 = HypergraphNode("test", "reservoir")
        
        # Different nodes should not be equal (different IDs)
        assert node1 != node2
        
        # Same node should be equal to itself
        assert node1 == node1
    
    def test_node_signature(self):
        """Test node signature generation."""
        properties = {"param1": "value1", "param2": 42}
        tensor_data = np.ones((2, 3))
        node = HypergraphNode("test", "reservoir", properties, tensor_data)
        
        signature = node.get_signature()
        assert isinstance(signature, str)
        assert len(signature) == 32  # MD5 hash length
        
        # Same content should produce same signature
        node2 = HypergraphNode("test", "reservoir", properties, tensor_data)
        assert node.get_signature() == node2.get_signature()


class TestHypergraphLink:
    """Test HypergraphLink functionality."""
    
    def test_link_creation(self):
        """Test basic link creation."""
        node1 = HypergraphNode("node1", "input")
        node2 = HypergraphNode("node2", "reservoir")
        
        link = HypergraphLink([node1, node2], "connection")
        
        assert link.link_type == "connection"
        assert len(link.nodes) == 2
        assert node1 in link.nodes
        assert node2 in link.nodes
        assert link.strength == 1.0
    
    def test_link_insufficient_nodes(self):
        """Test that link requires at least 2 nodes."""
        node1 = HypergraphNode("node1", "input")
        
        with pytest.raises(ValueError, match="at least 2 nodes"):
            HypergraphLink([node1], "connection")
    
    def test_link_with_properties(self):
        """Test link creation with properties."""
        node1 = HypergraphNode("node1", "input")
        node2 = HypergraphNode("node2", "reservoir")
        properties = {"weight": 0.5, "delay": 1}
        
        link = HypergraphLink([node1, node2], "connection", properties, strength=0.8)
        
        assert link.properties == properties
        assert link.strength == 0.8
    
    def test_link_contains_node(self):
        """Test contains_node method."""
        node1 = HypergraphNode("node1", "input")
        node2 = HypergraphNode("node2", "reservoir")
        node3 = HypergraphNode("node3", "output")
        
        link = HypergraphLink([node1, node2], "connection")
        
        assert link.contains_node(node1)
        assert link.contains_node(node2)
        assert not link.contains_node(node3)
    
    def test_get_connected_nodes(self):
        """Test get_connected_nodes method."""
        node1 = HypergraphNode("node1", "input")
        node2 = HypergraphNode("node2", "reservoir")
        node3 = HypergraphNode("node3", "output")
        
        # Test binary link
        link = HypergraphLink([node1, node2], "connection")
        connected = link.get_connected_nodes(node1)
        assert len(connected) == 1
        assert node2 in connected
        
        # Test hypergraph link (more than 2 nodes)
        hyperlink = HypergraphLink([node1, node2, node3], "state_flow")
        connected = hyperlink.get_connected_nodes(node1)
        assert len(connected) == 2
        assert node2 in connected
        assert node3 in connected
        
        # Test with node not in link
        node4 = HypergraphNode("node4", "activation")
        connected = link.get_connected_nodes(node4)
        assert len(connected) == 0


class TestAtomSpace:
    """Test AtomSpace functionality."""
    
    def test_atomspace_creation(self):
        """Test basic AtomSpace creation."""
        atomspace = AtomSpace("test_space")
        
        assert atomspace.name == "test_space"
        assert len(atomspace.nodes) == 0
        assert len(atomspace.links) == 0
    
    def test_add_node(self):
        """Test adding nodes to AtomSpace."""
        atomspace = AtomSpace()
        node = HypergraphNode("test_node", "reservoir")
        
        atomspace.add_node(node)
        
        assert len(atomspace.nodes) == 1
        assert node in atomspace.nodes
        assert atomspace.get_node("test_node") == node
    
    def test_add_link(self):
        """Test adding links to AtomSpace."""
        atomspace = AtomSpace()
        node1 = HypergraphNode("node1", "input")
        node2 = HypergraphNode("node2", "reservoir")
        link = HypergraphLink([node1, node2], "connection")
        
        atomspace.add_link(link)
        
        assert len(atomspace.links) == 1
        assert link in atomspace.links
        # Nodes should be automatically added
        assert len(atomspace.nodes) == 2
        assert node1 in atomspace.nodes
        assert node2 in atomspace.nodes
    
    def test_get_nodes_by_type(self):
        """Test getting nodes by type."""
        atomspace = AtomSpace()
        
        node1 = HypergraphNode("node1", "reservoir")
        node2 = HypergraphNode("node2", "reservoir") 
        node3 = HypergraphNode("node3", "readout")
        
        atomspace.add_node(node1)
        atomspace.add_node(node2)
        atomspace.add_node(node3)
        
        reservoir_nodes = atomspace.get_nodes_by_type("reservoir")
        readout_nodes = atomspace.get_nodes_by_type("readout")
        
        assert len(reservoir_nodes) == 2
        assert node1 in reservoir_nodes
        assert node2 in reservoir_nodes
        assert len(readout_nodes) == 1
        assert node3 in readout_nodes
    
    def test_get_links_for_node(self):
        """Test getting links containing a specific node."""
        atomspace = AtomSpace()
        
        node1 = HypergraphNode("node1", "input")
        node2 = HypergraphNode("node2", "reservoir")
        node3 = HypergraphNode("node3", "output")
        
        link1 = HypergraphLink([node1, node2], "connection")
        link2 = HypergraphLink([node2, node3], "connection")
        
        atomspace.add_link(link1)
        atomspace.add_link(link2)
        
        # Node2 should be in both links
        node2_links = atomspace.get_links_for_node(node2)
        assert len(node2_links) == 2
        assert link1 in node2_links
        assert link2 in node2_links
        
        # Node1 should be in only one link
        node1_links = atomspace.get_links_for_node(node1)
        assert len(node1_links) == 1
        assert link1 in node1_links
    
    def test_get_connected_nodes(self):
        """Test getting all nodes connected to a specific node."""
        atomspace = AtomSpace()
        
        node1 = HypergraphNode("node1", "input")
        node2 = HypergraphNode("node2", "reservoir")
        node3 = HypergraphNode("node3", "output")
        node4 = HypergraphNode("node4", "activation")
        
        link1 = HypergraphLink([node1, node2], "connection")
        link2 = HypergraphLink([node2, node3], "connection")
        link3 = HypergraphLink([node1, node4], "connection")
        
        atomspace.add_link(link1)
        atomspace.add_link(link2)
        atomspace.add_link(link3)
        
        # Node1 should be connected to node2 and node4
        connected = atomspace.get_connected_nodes(node1)
        assert len(connected) == 2
        assert node2 in connected
        assert node4 in connected
        
        # Node2 should be connected to node1 and node3
        connected = atomspace.get_connected_nodes(node2)
        assert len(connected) == 2
        assert node1 in connected
        assert node3 in connected
    
    def test_query_pattern(self):
        """Test pattern querying functionality."""
        atomspace = AtomSpace()
        
        # Create a simple pattern: input -> reservoir -> output
        input_node = HypergraphNode("input1", "input")
        reservoir_node = HypergraphNode("reservoir1", "reservoir") 
        output_node = HypergraphNode("output1", "output")
        
        link1 = HypergraphLink([input_node, reservoir_node], "connection")
        link2 = HypergraphLink([reservoir_node, output_node], "connection")
        
        atomspace.add_link(link1)
        atomspace.add_link(link2)
        
        # Query for input -> reservoir pattern
        patterns = atomspace.query_pattern(["input", "reservoir"], "connection")
        assert len(patterns) == 1
        assert input_node in patterns[0]
        assert reservoir_node in patterns[0]
        
        # Query for reservoir -> output pattern
        patterns = atomspace.query_pattern(["reservoir", "output"], "connection")
        assert len(patterns) == 1
        assert reservoir_node in patterns[0]
        assert output_node in patterns[0]
        
        # Query for non-existent pattern
        patterns = atomspace.query_pattern(["input", "output"], "connection")
        assert len(patterns) == 0
    
    def test_clear(self):
        """Test clearing AtomSpace."""
        atomspace = AtomSpace()
        
        node = HypergraphNode("node1", "reservoir")
        atomspace.add_node(node)
        
        assert len(atomspace.nodes) == 1
        
        atomspace.clear()
        
        assert len(atomspace.nodes) == 0
        assert len(atomspace.links) == 0
        assert atomspace.get_node("node1") is None
    
    def test_size(self):
        """Test AtomSpace size statistics."""
        atomspace = AtomSpace()
        
        node1 = HypergraphNode("node1", "reservoir")
        node2 = HypergraphNode("node2", "readout")
        link = HypergraphLink([node1, node2], "connection")
        
        atomspace.add_link(link)
        
        size_stats = atomspace.size()
        assert size_stats["nodes"] == 2
        assert size_stats["links"] == 1
        assert size_stats["node_types"] == 2
    
    def test_complex_hypergraph(self):
        """Test complex hypergraph with multiple node types and link types."""
        atomspace = AtomSpace("complex_test")
        
        # Create various node types
        input_node = HypergraphNode("input", "input")
        reservoir1 = HypergraphNode("reservoir1", "reservoir")
        reservoir2 = HypergraphNode("reservoir2", "reservoir")
        readout = HypergraphNode("readout", "readout")
        output_node = HypergraphNode("output", "output")
        
        # Create various link types
        input_link = HypergraphLink([input_node, reservoir1], "connection")
        lateral_link = HypergraphLink([reservoir1, reservoir2], "connection")
        feedback_link = HypergraphLink([reservoir2, reservoir1], "feedback")
        readout_link = HypergraphLink([reservoir2, readout], "connection")
        output_link = HypergraphLink([readout, output_node], "connection")
        
        # Add to AtomSpace
        for link in [input_link, lateral_link, feedback_link, readout_link, output_link]:
            atomspace.add_link(link)
        
        # Verify structure
        assert len(atomspace.nodes) == 5
        assert len(atomspace.links) == 5
        
        # Test connectivity
        reservoir1_connected = atomspace.get_connected_nodes(reservoir1)
        assert len(reservoir1_connected) == 2  # input and reservoir2
        assert input_node in reservoir1_connected
        assert reservoir2 in reservoir1_connected
        
        # Test pattern queries
        reservoir_patterns = atomspace.query_pattern(["reservoir", "reservoir"], "connection")
        assert len(reservoir_patterns) == 1
        
        feedback_patterns = atomspace.query_pattern(["reservoir", "reservoir"], "feedback")
        assert len(feedback_patterns) == 1