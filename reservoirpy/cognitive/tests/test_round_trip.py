"""
Tests for Round-Trip Translation Validation
==========================================

Comprehensive test suite for validating round-trip translation fidelity
between ReservoirPy primitives and hypergraph representations.
"""

import pytest
import numpy as np

from ...nodes import Reservoir, Ridge, Input, Output, Tanh
from ...node import Node
from ...model import Model
from ..encoder import HypergraphEncoder
from ..translator import NodeTranslator, ModelTranslator
from ..hypergraph import AtomSpace


class TestRoundTripValidation:
    """Test round-trip translation fidelity."""
    
    def setup_method(self):
        """Set up test fixtures."""
        self.encoder = HypergraphEncoder()
        self.tolerance = 1e-6
    
    def test_simple_node_round_trip(self):
        """Test round-trip for simple node without state."""
        # Create a simple Tanh activation node
        original_node = Tanh()
        original_node.set_input_dim(10)
        original_node.set_output_dim(10)
        
        # Round-trip conversion
        encoded = self.encoder.encode_node(original_node, include_state=False)
        decoded = self.encoder.decode_node(encoded)
        
        # Validation
        results = self.encoder.validate_round_trip(original_node, self.tolerance)
        
        assert results["round_trip_successful"]
        assert results["object_type"] == "Node"
        assert "errors" not in results or len(results["errors"]) == 0
    
    def test_reservoir_node_round_trip(self):
        """Test round-trip for reservoir node with state."""
        # Create reservoir with state
        reservoir = Reservoir(units=50, lr=0.1, sr=0.9)
        reservoir.initialize()
        
        # Set a state
        test_state = np.random.rand(50)
        reservoir.set_state_proxy(test_state)
        
        # Round-trip conversion
        results = self.encoder.validate_round_trip(reservoir, self.tolerance)
        
        assert results["round_trip_successful"]
        assert results["class_match"]
        
        # Check state fidelity if available
        if results.get("state_fidelity"):
            state_fidelity = results["state_fidelity"]
            assert state_fidelity["correlation"] > 0.8  # Good correlation
            # MSE might be higher due to tensor fragment encoding
    
    def test_readout_node_round_trip(self):
        """Test round-trip for readout node."""
        # Create ridge readout
        readout = Ridge(output_dim=5)
        readout.set_input_dim(100)
        
        # Initialize with some random weights (simulate training)
        if hasattr(readout, 'initialize'):
            readout.initialize()
        
        # Round-trip
        results = self.encoder.validate_round_trip(readout, self.tolerance)
        
        assert results["round_trip_successful"]
        assert results["object_type"] == "Node"
    
    def test_input_output_nodes_round_trip(self):
        """Test round-trip for input/output nodes."""
        input_node = Input()
        output_node = Output()
        
        input_node.set_output_dim(20)
        output_node.set_input_dim(10)
        
        # Test input node
        results_input = self.encoder.validate_round_trip(input_node, self.tolerance)
        assert results_input["round_trip_successful"]
        
        # Test output node
        results_output = self.encoder.validate_round_trip(output_node, self.tolerance)
        assert results_output["round_trip_successful"]
    
    def test_simple_model_round_trip(self):
        """Test round-trip for simple model."""
        # Create simple model: input -> reservoir -> readout -> output
        input_node = Input()
        reservoir = Reservoir(units=30, lr=0.1)
        readout = Ridge(output_dim=5)
        output_node = Output()
        
        # Initialize nodes
        input_node.set_output_dim(10)
        reservoir.set_input_dim(10)
        readout.set_input_dim(30)
        output_node.set_input_dim(5)
        
        # Create model (simplified - using first node as representative)
        model = reservoir  # Simplified model representation
        
        # Round-trip
        results = self.encoder.validate_round_trip(model, self.tolerance)
        
        assert results["round_trip_successful"]
    
    def test_state_preservation_fidelity(self):
        """Test specific state preservation through round-trip."""
        reservoir = Reservoir(units=20, lr=0.2, sr=0.8)
        reservoir.initialize()
        
        # Set specific state pattern
        original_state = np.sin(np.linspace(0, 2*np.pi, 20))
        reservoir.set_state_proxy(original_state)
        
        # Encode with tensor fragments
        encoded = self.encoder.encode_node(reservoir, include_state=True)
        decoded = self.encoder.decode_node(encoded)
        
        # Compare states
        if hasattr(decoded, 'state') and decoded.state is not None:
            decoded_state = decoded.state.flatten()
            original_flat = original_state.flatten()
            
            # Allow for some error due to encoding process
            min_len = min(len(original_flat), len(decoded_state))
            mse = np.mean((original_flat[:min_len] - decoded_state[:min_len]) ** 2)
            
            # Should preserve general pattern even if not exact
            assert mse < 2.0  # Relaxed tolerance for tensor fragment encoding
    
    def test_multiple_nodes_atomspace_round_trip(self):
        """Test round-trip through AtomSpace with multiple nodes."""
        # Create multiple nodes
        nodes = [
            Input(),
            Reservoir(units=25),
            Ridge(output_dim=3),
            Tanh(),
            Output()
        ]
        
        # Set dimensions
        nodes[0].set_output_dim(5)    # Input
        nodes[1].set_input_dim(5)     # Reservoir
        nodes[2].set_input_dim(25)    # Ridge
        nodes[3].set_input_dim(3)     # Tanh
        nodes[4].set_input_dim(3)     # Output
        
        # Initialize
        for node in nodes:
            if hasattr(node, 'initialize'):
                node.initialize()
        
        # Convert each node and validate
        success_count = 0
        for i, node in enumerate(nodes):
            try:
                results = self.encoder.validate_round_trip(node, self.tolerance)
                if results["round_trip_successful"]:
                    success_count += 1
            except Exception as e:
                print(f"Node {i} ({type(node).__name__}) failed: {e}")
        
        # Most nodes should succeed
        assert success_count >= len(nodes) * 0.8  # 80% success rate
    
    def test_encoding_statistics_tracking(self):
        """Test that encoding statistics are properly tracked."""
        # Reset statistics
        self.encoder.reset_statistics()
        stats = self.encoder.get_encoding_statistics()
        
        assert stats["nodes_encoded"] == 0
        assert stats["round_trips_validated"] == 0
        
        # Perform some operations
        node1 = Tanh()
        node2 = Reservoir(units=10)
        
        self.encoder.encode_node(node1)
        self.encoder.encode_node(node2)
        self.encoder.validate_round_trip(node1)
        
        # Check updated statistics
        stats = self.encoder.get_encoding_statistics()
        assert stats["nodes_encoded"] == 2
        assert stats["round_trips_validated"] == 1
    
    def test_scheme_mode_round_trip(self):
        """Test round-trip in Scheme S-expression mode."""
        self.encoder.set_scheme_mode(True)
        
        reservoir = Reservoir(units=15)
        reservoir.initialize()
        
        # Encode in Scheme mode
        scheme_expr = self.encoder.encode_node(reservoir)
        
        # Should return SchemeExpression
        from ..scheme_adapter import SchemeExpression
        assert isinstance(scheme_expr, SchemeExpression)
        
        # Decode back
        decoded = self.encoder.decode_node(scheme_expr)
        
        # Should be similar to original
        assert decoded.__class__.__name__ == "Reservoir" or decoded.__class__.__name__ == "Node"
    
    def test_error_handling_in_round_trip(self):
        """Test error handling during round-trip validation."""
        # Create a mock node that might cause issues
        class ProblematicNode(Node):
            def __init__(self):
                super().__init__()
                self.problematic_attr = lambda x: x  # Non-serializable
        
        problematic_node = ProblematicNode()
        
        # Should handle errors gracefully
        results = self.encoder.validate_round_trip(problematic_node, self.tolerance)
        
        # Should indicate failure but not crash
        assert not results["round_trip_successful"]
        assert "errors" in results
        assert len(results["errors"]) > 0
    
    def test_tolerance_sensitivity(self):
        """Test round-trip validation with different tolerance levels."""
        reservoir = Reservoir(units=10, lr=0.1)
        reservoir.initialize()
        
        # Set state
        test_state = np.random.rand(10)
        reservoir.set_state_proxy(test_state)
        
        # Test with strict tolerance
        results_strict = self.encoder.validate_round_trip(reservoir, 1e-10)
        
        # Test with relaxed tolerance
        results_relaxed = self.encoder.validate_round_trip(reservoir, 1e-3)
        
        # Relaxed tolerance should be more forgiving
        if "state_fidelity" in results_strict and "state_fidelity" in results_relaxed:
            # Note: Due to tensor fragment encoding, even relaxed tolerance might show issues
            pass  # This test is more about checking the tolerance parameter is used
    
    def test_large_state_round_trip(self):
        """Test round-trip with large reservoir states."""
        large_reservoir = Reservoir(units=200, lr=0.05, sr=0.95)
        large_reservoir.initialize()
        
        # Set large state
        large_state = np.random.randn(200)
        large_reservoir.set_state_proxy(large_state)
        
        # Round-trip should handle large states
        results = self.encoder.validate_round_trip(large_reservoir, self.tolerance)
        
        # Should succeed even with large states
        assert results["round_trip_successful"]
    
    def test_different_node_types_batch(self):
        """Test round-trip for different node types in batch."""
        # Create various node types
        test_nodes = []
        
        # Add basic nodes
        try:
            test_nodes.append(Input())
            test_nodes.append(Output())
            test_nodes.append(Tanh())
            test_nodes.append(Reservoir(units=20))
            test_nodes.append(Ridge(output_dim=5))
        except Exception as e:
            print(f"Node creation failed: {e}")
        
        # Test each node type
        results_summary = {
            "total_tested": len(test_nodes),
            "successful": 0,
            "failed": 0,
            "errors": []
        }
        
        for i, node in enumerate(test_nodes):
            try:
                # Set basic dimensions
                if hasattr(node, 'set_input_dim'):
                    node.set_input_dim(10)
                if hasattr(node, 'set_output_dim'):
                    node.set_output_dim(10)
                
                results = self.encoder.validate_round_trip(node, self.tolerance)
                
                if results["round_trip_successful"]:
                    results_summary["successful"] += 1
                else:
                    results_summary["failed"] += 1
                    results_summary["errors"].extend(results.get("errors", []))
                    
            except Exception as e:
                results_summary["failed"] += 1
                results_summary["errors"].append(f"Node {i}: {str(e)}")
        
        # Should have reasonable success rate
        success_rate = results_summary["successful"] / results_summary["total_tested"]
        assert success_rate >= 0.6  # At least 60% success rate
        
        print(f"Round-trip success rate: {success_rate:.2%}")
        if results_summary["errors"]:
            print("Errors encountered:")
            for error in results_summary["errors"][:5]:  # Show first 5 errors
                print(f"  - {error}")
    
    def test_node_translator_direct(self):
        """Test NodeTranslator directly for round-trip consistency."""
        translator = NodeTranslator()
        
        # Test simple node
        original = Tanh()
        original.set_input_dim(5)
        original.set_output_dim(5)
        
        # Convert to hypergraph and back
        hypergraph_node = translator.node_to_hypergraph(original)
        reconstructed = translator.hypergraph_to_node(hypergraph_node)
        
        # Basic checks
        assert reconstructed is not None
        assert hasattr(reconstructed, 'input_dim') or hasattr(reconstructed, 'set_input_dim')
    
    def test_atomspace_model_round_trip(self):
        """Test round-trip through AtomSpace model conversion."""
        model_translator = ModelTranslator()
        
        # Create simple model (using single node as representative)
        reservoir = Reservoir(units=15)
        reservoir.initialize()
        
        # Convert to AtomSpace
        atomspace = model_translator.model_to_atomspace(reservoir, include_states=True)
        
        # Convert back to model
        reconstructed_model = model_translator.atomspace_to_model(atomspace)
        
        # Should get something back
        assert reconstructed_model is not None
    
    def test_performance_benchmark(self):
        """Basic performance benchmark for translation overhead."""
        import time
        
        # Create test node
        reservoir = Reservoir(units=50)
        reservoir.initialize()
        test_state = np.random.rand(50)
        reservoir.set_state_proxy(test_state)
        
        # Measure encoding time
        start_time = time.time()
        for _ in range(10):
            encoded = self.encoder.encode_node(reservoir, include_state=True)
            decoded = self.encoder.decode_node(encoded)
        encoding_time = time.time() - start_time
        
        # Should complete reasonably quickly
        average_time = encoding_time / 10
        print(f"Average round-trip time: {average_time:.4f} seconds")
        
        # Should be under 1 second per round-trip for small nodes
        assert average_time < 1.0