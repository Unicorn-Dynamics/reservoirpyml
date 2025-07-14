"""
Tests for GGML Neural-Symbolic Integration
========================================

Test suite for Phase 3: Neural-Symbolic Synthesis via Custom ggml Kernels.
"""

import pytest
import numpy as np

from reservoirpy.cognitive.ggml import (
    GGMLTensor, GGMLContext, SymbolicKernel, NeuralSymbolicBridge,
    hypergraph_conv, symbolic_activation, attention_weighted_op,
    GGMLReservoir, SymbolicReadout
)
from reservoirpy.cognitive.ggml.core import TensorType, SymbolicAnnotation
from reservoirpy.cognitive.ggml.kernels import (
    HypergraphConvKernel, SymbolicActivationKernel, AttentionWeightedOpKernel
)
from reservoirpy.cognitive.hypergraph import AtomSpace
from reservoirpy.cognitive.attention.ecan import ECANAttentionSystem
from reservoirpy.nodes import Reservoir, Ridge


class TestGGMLCore:
    """Test core GGML functionality."""
    
    def test_ggml_tensor_creation(self):
        """Test GGMLTensor creation and basic operations."""
        data = np.random.randn(10, 5)
        tensor = GGMLTensor(data=data, tensor_type=TensorType.NEURAL, name="test_tensor")
        
        assert tensor.shape == data.shape
        assert tensor.tensor_type == TensorType.NEURAL
        assert tensor.name == "test_tensor"
        assert np.array_equal(tensor.data, data)
        
    def test_ggml_tensor_arithmetic(self):
        """Test tensor arithmetic operations."""
        data1 = np.random.randn(5, 3)
        data2 = np.random.randn(5, 3)
        
        tensor1 = GGMLTensor(data=data1, name="t1")
        tensor2 = GGMLTensor(data=data2, name="t2")
        
        # Test addition
        result_add = tensor1 + tensor2
        assert np.allclose(result_add.data, data1 + data2)
        assert len(result_add.computation_graph) == 1
        
        # Test multiplication
        result_mul = tensor1 * tensor2
        assert np.allclose(result_mul.data, data1 * data2)
        
        # Test scalar multiplication
        result_scalar = tensor1 * 2.0
        assert np.allclose(result_scalar.data, data1 * 2.0)
        
    def test_ggml_tensor_to_symbolic(self):
        """Test conversion to symbolic representation."""
        data = np.random.randn(8, 4)
        tensor = GGMLTensor(data=data, tensor_type=TensorType.NEURAL)
        atom_space = AtomSpace()
        
        symbolic_tensor = tensor.to_symbolic(atom_space)
        
        assert symbolic_tensor.tensor_type == TensorType.SYMBOLIC
        assert len(atom_space.nodes) == 1  # One node added to AtomSpace
        assert len(symbolic_tensor.symbolic_annotation.hypergraph_nodes) == 1
        
    def test_ggml_context(self):
        """Test GGML computation context."""
        context = GGMLContext(mem_size=1024*1024, enable_gradients=True)
        
        data = np.random.randn(6, 8)
        tensor = context.create_tensor(data=data, tensor_type=TensorType.NEURAL, name="ctx_tensor")
        
        assert tensor.id in context.tensors
        assert context.memory_usage > 0
        
        # Test memory usage tracking
        memory_stats = context.get_memory_usage()
        assert "total_bytes" in memory_stats
        assert "tensor_count" in memory_stats
        assert memory_stats["tensor_count"] == 1


class TestSymbolicKernels:
    """Test symbolic kernel operations."""
    
    def test_hypergraph_conv_kernel(self):
        """Test hypergraph convolution kernel."""
        kernel = HypergraphConvKernel(hidden_dim=32, num_heads=2)
        
        # Create test data
        node_features = GGMLTensor(data=np.random.randn(5, 32), name="nodes")
        edge_indices = GGMLTensor(data=np.array([[0, 1], [1, 2], [2, 0]]), name="edges")
        
        result = kernel(node_features, edge_indices)
        
        assert result.tensor_type == TensorType.SYMBOLIC
        assert result.shape == (5, 32)  # Should maintain node feature dimensions
        assert "hypergraph_convolution" in result.symbolic_annotation.semantic_context["operation"]
        
    def test_symbolic_activation_kernel(self):
        """Test symbolic activation kernel."""
        kernel = SymbolicActivationKernel(activation_type="cognitive_tanh")
        
        input_data = np.random.randn(10, 16)
        input_tensor = GGMLTensor(data=input_data, name="activation_input")
        
        result = kernel(input_tensor)
        
        assert result.tensor_type == TensorType.SYMBOLIC
        assert result.shape == input_data.shape
        assert result.symbolic_annotation.semantic_context["activation_type"] == "cognitive_tanh"
        
        # Test different activation types
        relu_kernel = SymbolicActivationKernel(activation_type="symbolic_relu")
        relu_result = relu_kernel(input_tensor)
        assert np.all(relu_result.data >= 0)  # ReLU property
        
    def test_attention_weighted_op_kernel(self):
        """Test attention-weighted operation kernel."""
        kernel = AttentionWeightedOpKernel(operation_type="attention_gating")
        
        input_data = np.random.randn(8, 12)
        attention_data = np.random.rand(8, 12)  # Positive attention weights
        
        input_tensor = GGMLTensor(data=input_data, name="input")
        attention_tensor = GGMLTensor(data=attention_data, name="attention")
        
        result = kernel(input_tensor, attention_tensor)
        
        assert result.tensor_type == TensorType.HYBRID
        assert result.shape == input_data.shape
        assert "attention_gating" in result.symbolic_annotation.semantic_context["operation"]
        
    def test_convenience_functions(self):
        """Test convenience functions for kernel operations."""
        node_features = GGMLTensor(data=np.random.randn(4, 16), name="nodes")
        edge_indices = GGMLTensor(data=np.array([[0, 1], [1, 2], [2, 3]]), name="edges")
        
        # Test hypergraph convolution
        conv_result = hypergraph_conv(node_features, edge_indices, hidden_dim=16)
        assert conv_result.tensor_type == TensorType.SYMBOLIC
        
        # Test symbolic activation
        activation_result = symbolic_activation(node_features, activation_type="cognitive_tanh")
        assert activation_result.tensor_type == TensorType.SYMBOLIC
        
        # Test attention weighted operation
        attention_weights = GGMLTensor(data=np.random.rand(4, 16), name="weights")
        attention_result = attention_weighted_op(
            node_features, attention_weights, operation_type="attention_gating"
        )
        assert attention_result.tensor_type == TensorType.HYBRID


class TestNeuralSymbolicBridge:
    """Test neural-symbolic bridge functionality."""
    
    def test_bridge_initialization(self):
        """Test bridge initialization."""
        context = GGMLContext()
        atom_space = AtomSpace()
        attention_system = ECANAttentionSystem(atomspace=atom_space)
        
        bridge = NeuralSymbolicBridge(context, atom_space, attention_system)
        
        assert bridge.ggml_context == context
        assert bridge.atom_space == atom_space
        assert bridge.attention_system == attention_system
        assert bridge.bridge_operations == 0
        
    def test_reservoir_to_symbolic(self):
        """Test conversion of reservoir to symbolic representation."""
        context = GGMLContext()
        atom_space = AtomSpace()
        bridge = NeuralSymbolicBridge(context, atom_space)
        
        # Create a simple reservoir
        reservoir = Reservoir(units=20, lr=0.3, sr=1.0, input_dim=3)
        input_data = np.random.randn(1, 3)  # Single timestep format
        
        # Initialize reservoir
        reservoir.initialize(input_data)
        
        # Convert to symbolic
        symbolic_tensor = bridge.reservoir_to_symbolic(reservoir, input_data)
        
        assert symbolic_tensor.tensor_type == TensorType.SYMBOLIC
        assert len(atom_space.nodes) == 2  # Reservoir node + tensor node added
        assert id(reservoir) in bridge.neural_to_symbolic
        
    def test_hybrid_computation(self):
        """Test hybrid neural-symbolic computation pipeline."""
        context = GGMLContext()
        atom_space = AtomSpace()
        bridge = NeuralSymbolicBridge(context, atom_space)
        
        # Create reservoir and input
        reservoir = Reservoir(units=15, lr=0.3, sr=1.0, input_dim=2)
        input_data = np.random.randn(1, 2)  # Single timestep format
        reservoir.initialize(input_data)
        
        # Define symbolic reasoning steps
        reasoning_steps = [
            {"type": "symbolic_activation", "params": {"activation_type": "cognitive_tanh"}},
            {"type": "hypergraph_conv", "params": {"hidden_dim": 16, "num_heads": 2}}
        ]
        
        # Execute hybrid computation
        result_tensor, stats = bridge.create_hybrid_computation(
            reservoir, reasoning_steps, input_data
        )
        
        assert result_tensor.tensor_type in [TensorType.SYMBOLIC, TensorType.HYBRID]
        assert stats["reasoning_steps"] == 2
        assert stats["neural_operations"] >= 1
        assert stats["symbolic_operations"] >= 2


class TestGGMLReservoirBackend:
    """Test GGML-backed ReservoirPy nodes."""
    
    def test_ggml_reservoir_creation(self):
        """Test GGMLReservoir creation and basic properties."""
        ggml_reservoir = GGMLReservoir(
            units=25,
            symbolic_enhancement=True,
            lr=0.3,
            sr=1.25
        )
        
        assert ggml_reservoir.units == 25
        assert ggml_reservoir.symbolic_enhancement == True
        assert ggml_reservoir.lr == 0.3
        assert ggml_reservoir.sr == 1.25
        assert isinstance(ggml_reservoir.ggml_context, GGMLContext)
        assert isinstance(ggml_reservoir.atom_space, AtomSpace)
        
    def test_ggml_reservoir_computation(self):
        """Test GGMLReservoir forward computation."""
        ggml_reservoir = GGMLReservoir(units=20, symbolic_enhancement=True)
        
        # Test input - use single timestep format
        input_data = np.random.randn(1, 4)
        
        # Initialize and run
        ggml_reservoir.initialize(input_data)
        output = ggml_reservoir(input_data)
        
        assert output.shape[0] == input_data.shape[0]  # Batch dimension preserved
        assert len(ggml_reservoir.computation_history) > 0
        
        # Test symbolic state
        symbolic_state = ggml_reservoir.get_symbolic_state()
        assert symbolic_state is not None
        assert symbolic_state.tensor_type in [TensorType.SYMBOLIC, TensorType.HYBRID]
        
    def test_ggml_reservoir_without_enhancement(self):
        """Test GGMLReservoir without symbolic enhancement."""
        ggml_reservoir = GGMLReservoir(units=15, symbolic_enhancement=False)
        
        input_data = np.random.randn(1, 3)  # Single timestep format
        ggml_reservoir.initialize(input_data)
        output = ggml_reservoir(input_data)
        
        # Should behave like standard reservoir
        assert output.shape[0] == input_data.shape[0]
        # Should have fewer computation steps (no symbolic enhancement)
        stats = ggml_reservoir.get_computation_stats()
        assert stats["symbolic_enhancement"] == False
        
    def test_symbolic_readout(self):
        """Test SymbolicReadout functionality."""
        symbolic_readout = SymbolicReadout(
            output_dim=3,
            symbolic_interpretation=True
        )
        
        # Create training data - use single timestep format
        train_x = np.random.randn(1, 10)
        train_y = np.random.randn(1, 3)
        
        # Fit readout
        symbolic_readout.initialize(train_x, train_y)
        symbolic_readout.fit(train_x, train_y)
        
        # Test inference - use single timestep format
        test_x = np.random.randn(1, 10)
        output = symbolic_readout(test_x)
        
        assert output.shape == (1, 3)
        
        # Test symbolic interpretation
        interpretation = symbolic_readout.get_symbolic_interpretation()
        assert interpretation["symbolic_interpretation"] == True
        assert "aggregate_stats" in interpretation
        assert "symbolic_patterns" in interpretation


class TestIntegrationScenarios:
    """Test end-to-end integration scenarios."""
    
    def test_complete_neural_symbolic_pipeline(self):
        """Test complete neural-symbolic computation pipeline."""
        # Create GGML-backed components
        ggml_reservoir = GGMLReservoir(units=30, symbolic_enhancement=True)
        symbolic_readout = SymbolicReadout(output_dim=2, symbolic_interpretation=True)
        
        # Generate sample data - use single timestep format
        input_dim = 4
        X = np.random.randn(1, input_dim)  # Single timestep
        y = np.random.randn(1, 2)
        
        # Initialize components
        ggml_reservoir.initialize(X)
        
        # Process through reservoir
        reservoir_state = ggml_reservoir(X)
        
        # Train readout
        symbolic_readout.initialize(reservoir_state, y)
        symbolic_readout.fit(reservoir_state, y)
        
        # Test prediction - single timestep
        prediction = symbolic_readout(reservoir_state)
        
        assert prediction.shape == (1, 2)
        
        # Verify symbolic enhancements
        reservoir_stats = ggml_reservoir.get_computation_stats()
        readout_stats = symbolic_readout.get_symbolic_interpretation()
        
        assert reservoir_stats["symbolic_enhancement"] == True
        assert readout_stats["symbolic_interpretation"] == True
        assert len(ggml_reservoir.atom_space.nodes) > 0  # Symbolic nodes created
        
    def test_attention_integration(self):
        """Test integration with ECAN attention system."""
        # Create attention system
        atom_space = AtomSpace()
        attention_system = ECANAttentionSystem(atomspace=atom_space)
        
        # Add some concepts with attention values
        concepts = ["concept_a", "concept_b", "concept_c"]
        for concept in concepts:
            attention_system.set_sti(concept, np.random.rand() * 100)
            
        # Create GGML context with attention
        context = GGMLContext()
        bridge = NeuralSymbolicBridge(context, atom_space, attention_system)
        
        # Create test tensor
        test_data = np.random.randn(len(concepts), 8)
        test_tensor = context.create_tensor(data=test_data, name="attention_test")
        
        # Apply attention-based reasoning
        result = bridge.apply_symbolic_reasoning(
            test_tensor,
            reasoning_type="attention_reasoning",
            operation_type="attention_gating"
        )
        
        assert result.tensor_type in [TensorType.SYMBOLIC, TensorType.HYBRID]
        assert "attention_statistics" in result.symbolic_annotation.semantic_context
        
    def test_performance_benchmarking(self):
        """Test performance and memory usage of GGML operations."""
        context = GGMLContext(mem_size=2*1024*1024)  # 2MB
        
        # Create multiple tensors for performance testing
        tensors = []
        for i in range(10):
            data = np.random.randn(50, 20)
            tensor = context.create_tensor(data=data, name=f"perf_tensor_{i}")
            tensors.append(tensor)
            
        # Perform operations
        for i in range(len(tensors) - 1):
            result = tensors[i] + tensors[i + 1]
            
        # Check memory usage
        memory_stats = context.get_memory_usage()
        computation_stats = context.get_computation_stats()
        
        assert memory_stats["tensor_count"] >= 10
        assert computation_stats["total_operations"] >= 0
        assert memory_stats["total_mb"] < 10  # Should be reasonable memory usage