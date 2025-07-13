"""
Tests for Tensor Fragment Architecture
=====================================

Test suite for tensor fragment encoding and cognitive state representation.
"""

import pytest
import numpy as np

from ..tensor_fragment import TensorSignature, TensorFragment
from ..hypergraph import HypergraphNode


class TestTensorSignature:
    """Test TensorSignature functionality."""
    
    def test_signature_creation_defaults(self):
        """Test tensor signature creation with default parameters."""
        signature = TensorSignature.create_signature()
        
        assert signature.modality == 7  # multimodal
        assert signature.depth == 2     # prime for depth 1
        assert signature.context > 0
        assert signature.salience > 0
        assert signature.autonomy_index > 0
    
    def test_signature_creation_custom(self):
        """Test tensor signature creation with custom parameters."""
        signature = TensorSignature.create_signature(
            modality_type="visual",
            processing_depth=3,
            context_span=15,
            salience_weight=0.8,
            autonomy_level=2
        )
        
        assert signature.modality == 2  # visual
        assert signature.depth == 5     # prime for depth 3
        assert signature.context > 10   # Fibonacci for span 15
        assert signature.salience > 1
        assert signature.autonomy_index == 6  # 2*3 for level 2
    
    def test_modality_encoding(self):
        """Test different modality encodings."""
        visual_sig = TensorSignature.create_signature(modality_type="visual")
        audio_sig = TensorSignature.create_signature(modality_type="audio")
        text_sig = TensorSignature.create_signature(modality_type="text")
        multi_sig = TensorSignature.create_signature(modality_type="multimodal")
        
        assert visual_sig.modality == 2
        assert audio_sig.modality == 3
        assert text_sig.modality == 5
        assert multi_sig.modality == 7
    
    def test_depth_prime_factorization(self):
        """Test depth encoding using prime factorization."""
        sig1 = TensorSignature.create_signature(processing_depth=1)
        sig2 = TensorSignature.create_signature(processing_depth=2)
        sig3 = TensorSignature.create_signature(processing_depth=3)
        
        assert sig1.depth == 2   # First prime
        assert sig2.depth == 3   # Second prime
        assert sig3.depth == 5   # Third prime
    
    def test_context_fibonacci(self):
        """Test context encoding using Fibonacci sequence."""
        sig1 = TensorSignature.create_signature(context_span=1)
        sig2 = TensorSignature.create_signature(context_span=2)
        sig5 = TensorSignature.create_signature(context_span=5)
        
        assert sig1.context == 1
        assert sig2.context == 1
        assert sig5.context == 5  # 5th Fibonacci number
    
    def test_salience_encoding(self):
        """Test salience weight encoding."""
        low_sig = TensorSignature.create_signature(salience_weight=0.1)
        high_sig = TensorSignature.create_signature(salience_weight=1.0)
        
        assert low_sig.salience >= 2
        assert high_sig.salience >= 2
        assert low_sig.salience <= high_sig.salience
    
    def test_autonomy_encoding(self):
        """Test autonomy level encoding."""
        sig1 = TensorSignature.create_signature(autonomy_level=1)
        sig2 = TensorSignature.create_signature(autonomy_level=2)
        sig3 = TensorSignature.create_signature(autonomy_level=3)
        
        assert sig1.autonomy_index == 2      # Basic reactive
        assert sig2.autonomy_index == 6      # 2*3 Simple planning
        assert sig3.autonomy_index == 30     # 2*3*5 Multi-goal
    
    def test_tensor_shape(self):
        """Test tensor shape generation."""
        signature = TensorSignature(2, 3, 5, 7, 11)
        shape = signature.get_tensor_shape()
        
        assert shape == (2, 3, 5, 7, 11)
        assert len(shape) == 5
    
    def test_total_dimensions(self):
        """Test total dimension calculation."""
        signature = TensorSignature(2, 3, 5, 7, 11)
        total = signature.get_total_dimensions()
        
        assert total == 2 * 3 * 5 * 7 * 11
        assert total == 2310
    
    def test_prime_factorization(self):
        """Test prime factorization functionality."""
        signature = TensorSignature(6, 12, 15, 21, 30)
        factorization = signature.get_prime_factorization()
        
        assert factorization["modality"] == [2, 3]         # 6 = 2*3
        assert factorization["depth"] == [2, 2, 3]         # 12 = 2²*3
        assert factorization["context"] == [3, 5]          # 15 = 3*5
        assert factorization["salience"] == [3, 7]         # 21 = 3*7
        assert factorization["autonomy_index"] == [2, 3, 5] # 30 = 2*3*5
    
    def test_edge_cases(self):
        """Test edge cases for signature creation."""
        # Zero and negative values
        sig = TensorSignature.create_signature(
            processing_depth=0,
            context_span=0,
            salience_weight=0.0,
            autonomy_level=0
        )
        
        assert sig.depth == 1       # Minimum value
        assert sig.context == 1     # Minimum value
        assert sig.salience == 1    # Minimum value
        assert sig.autonomy_index == 1  # Minimum value


class TestTensorFragment:
    """Test TensorFragment functionality."""
    
    def test_fragment_creation_default(self):
        """Test tensor fragment creation with default data."""
        signature = TensorSignature.create_signature()
        fragment = TensorFragment(signature)
        
        assert fragment.signature == signature
        assert fragment.data is not None
        assert fragment.data.shape == signature.get_tensor_shape()
        assert fragment.data.dtype == np.float32
        assert np.allclose(fragment.data, 0)  # Should be zeros
    
    def test_fragment_creation_with_data(self):
        """Test tensor fragment creation with provided data."""
        signature = TensorSignature(2, 3, 4, 5, 6)
        data = np.random.rand(2, 3, 4, 5, 6).astype(np.float32)
        fragment = TensorFragment(signature, data)
        
        assert fragment.data.shape == (2, 3, 4, 5, 6)
        np.testing.assert_array_equal(fragment.data, data)
    
    def test_fragment_data_shape_mismatch(self):
        """Test error handling for mismatched data shape."""
        signature = TensorSignature(2, 3, 4, 5, 6)
        wrong_data = np.random.rand(3, 4, 5)  # Wrong shape
        
        with pytest.raises(ValueError, match="doesn't match signature shape"):
            TensorFragment(signature, wrong_data)
    
    def test_encode_reservoir_state(self):
        """Test encoding reservoir state into tensor fragment."""
        signature = TensorSignature.create_signature()
        fragment = TensorFragment(signature)
        
        # Create a reservoir state
        reservoir_state = np.random.rand(50)
        reservoir_info = {"type": "reservoir", "units": 50}
        
        fragment.encode_reservoir_state(reservoir_state, reservoir_info)
        
        # Check that data was encoded
        assert not np.allclose(fragment.data, 0)
        assert "original_state_shape" in fragment.metadata
        assert "reservoir_info" in fragment.metadata
        assert fragment.metadata["original_state_shape"] == (50,)
    
    def test_encode_large_state(self):
        """Test encoding state larger than tensor capacity."""
        signature = TensorSignature(2, 3, 4, 5, 2)  # Small tensor
        fragment = TensorFragment(signature)
        
        # Large state that needs truncation
        large_state = np.random.rand(1000)
        reservoir_info = {"type": "reservoir", "units": 1000}
        
        fragment.encode_reservoir_state(large_state, reservoir_info)
        
        # Should be truncated to fit tensor
        total_elements = np.prod(signature.get_tensor_shape())
        assert fragment.data.size == total_elements
    
    def test_encode_small_state(self):
        """Test encoding state smaller than tensor capacity."""
        signature = TensorSignature(10, 10, 10, 10, 10)  # Large tensor
        fragment = TensorFragment(signature)
        
        # Small state that needs padding
        small_state = np.random.rand(5)
        reservoir_info = {"type": "reservoir", "units": 5}
        
        fragment.encode_reservoir_state(small_state, reservoir_info)
        
        # Should be padded with zeros
        flat_data = fragment.data.flatten()
        assert not np.allclose(flat_data[:5], 0)  # First 5 should have data
        # Note: remaining might be zero from padding
    
    def test_decode_to_reservoir_state(self):
        """Test decoding tensor fragment back to reservoir state."""
        signature = TensorSignature.create_signature()
        fragment = TensorFragment(signature)
        
        # Encode a state
        original_state = np.random.rand(30)
        reservoir_info = {"type": "reservoir"}
        fragment.encode_reservoir_state(original_state, reservoir_info)
        
        # Decode back
        decoded_state = fragment.decode_to_reservoir_state()
        
        # Should recover something close to original
        assert decoded_state.shape == original_state.shape
    
    def test_decode_with_target_shape(self):
        """Test decoding with specific target shape."""
        signature = TensorSignature.create_signature()
        fragment = TensorFragment(signature)
        
        # Set some data
        fragment.data = np.random.rand(*signature.get_tensor_shape())
        
        # Decode to specific shape
        target_shape = (20, 5)
        decoded = fragment.decode_to_reservoir_state(target_shape)
        
        assert decoded.shape == target_shape
    
    def test_to_hypergraph_node(self):
        """Test conversion to hypergraph node."""
        signature = TensorSignature.create_signature()
        fragment = TensorFragment(signature)
        fragment.metadata = {"test_key": "test_value"}
        
        node = fragment.to_hypergraph_node("test_fragment", "tensor_node")
        
        assert isinstance(node, HypergraphNode)
        assert node.name == "test_fragment"
        assert node.node_type == "tensor_node"
        assert "signature" in node.properties
        assert "tensor_shape" in node.properties
        assert "metadata" in node.properties
        assert node.properties["metadata"]["test_key"] == "test_value"
        np.testing.assert_array_equal(node.tensor_data, fragment.data)
    
    def test_from_hypergraph_node(self):
        """Test creation from hypergraph node."""
        # Create original fragment
        signature = TensorSignature.create_signature()
        original_fragment = TensorFragment(signature)
        original_fragment.metadata = {"test_key": "test_value"}
        
        # Convert to node
        node = original_fragment.to_hypergraph_node("test")
        
        # Convert back to fragment
        reconstructed_fragment = TensorFragment.from_hypergraph_node(node)
        
        assert reconstructed_fragment.signature.modality == signature.modality
        assert reconstructed_fragment.signature.depth == signature.depth
        assert reconstructed_fragment.metadata["test_key"] == "test_value"
        np.testing.assert_array_equal(reconstructed_fragment.data, original_fragment.data)
    
    def test_from_hypergraph_node_error(self):
        """Test error handling when creating from invalid node."""
        node = HypergraphNode("test", "invalid", {"no_signature": True})
        
        with pytest.raises(ValueError, match="does not contain tensor signature"):
            TensorFragment.from_hypergraph_node(node)
    
    def test_compute_similarity_identical(self):
        """Test similarity computation for identical fragments."""
        signature = TensorSignature.create_signature()
        data = np.random.rand(*signature.get_tensor_shape())
        
        fragment1 = TensorFragment(signature, data.copy())
        fragment2 = TensorFragment(signature, data.copy())
        
        similarity = fragment1.compute_similarity(fragment2)
        assert similarity > 0.9  # Should be very similar
    
    def test_compute_similarity_different(self):
        """Test similarity computation for different fragments."""
        signature1 = TensorSignature.create_signature(modality_type="visual")
        signature2 = TensorSignature.create_signature(modality_type="audio")
        
        fragment1 = TensorFragment(signature1)
        fragment2 = TensorFragment(signature2)
        
        # Fill with different random data
        fragment1.data = np.random.rand(*signature1.get_tensor_shape())
        fragment2.data = np.random.rand(*signature2.get_tensor_shape())
        
        similarity = fragment1.compute_similarity(fragment2)
        assert 0 <= similarity <= 1  # Valid similarity range
    
    def test_compute_similarity_different_shapes(self):
        """Test similarity computation for fragments with different shapes."""
        sig1 = TensorSignature(2, 3, 4, 5, 6)
        sig2 = TensorSignature(3, 4, 5, 6, 7)
        
        fragment1 = TensorFragment(sig1)
        fragment2 = TensorFragment(sig2)
        
        fragment1.data = np.random.rand(*sig1.get_tensor_shape())
        fragment2.data = np.random.rand(*sig2.get_tensor_shape())
        
        similarity = fragment1.compute_similarity(fragment2)
        assert 0 <= similarity <= 1
    
    def test_round_trip_consistency(self):
        """Test round-trip encoding/decoding consistency."""
        signature = TensorSignature.create_signature()
        fragment = TensorFragment(signature)
        
        # Original state
        original_state = np.random.rand(100)
        reservoir_info = {"type": "reservoir", "units": 100}
        
        # Encode
        fragment.encode_reservoir_state(original_state, reservoir_info)
        
        # Decode back
        decoded_state = fragment.decode_to_reservoir_state((100,))
        
        # Should be reasonably close (allowing for truncation/padding effects)
        mse = np.mean((original_state - decoded_state) ** 2)
        assert mse < 1.0  # Allow some error due to encoding process
    
    def test_empty_state_handling(self):
        """Test handling of empty or None states."""
        signature = TensorSignature.create_signature()
        fragment = TensorFragment(signature)
        
        # Test None state
        fragment.encode_reservoir_state(None, {})
        assert np.allclose(fragment.data, 0)  # Should remain zeros
        
        # Test empty array
        empty_state = np.array([])
        fragment.encode_reservoir_state(empty_state, {})
        assert np.allclose(fragment.data, 0)  # Should remain zeros
    
    def test_metadata_preservation(self):
        """Test that metadata is preserved through operations."""
        signature = TensorSignature.create_signature()
        metadata = {
            "source": "test_reservoir",
            "timestamp": "2024-01-01",
            "parameters": {"lr": 0.1, "sr": 0.9}
        }
        fragment = TensorFragment(signature, metadata=metadata)
        
        # Encode some state
        state = np.random.rand(50)
        reservoir_info = {"type": "test"}
        fragment.encode_reservoir_state(state, reservoir_info)
        
        # Check metadata preservation and augmentation
        assert fragment.metadata["source"] == "test_reservoir"
        assert fragment.metadata["timestamp"] == "2024-01-01"
        assert fragment.metadata["parameters"]["lr"] == 0.1
        assert fragment.metadata["reservoir_info"]["type"] == "test"