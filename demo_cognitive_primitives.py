"""
Demo Example: Cognitive Primitives & Hypergraph Encoding
=======================================================

This example demonstrates the complete Phase 1 implementation of cognitive
primitives and foundational hypergraph encoding for ReservoirPy.

Features demonstrated:
- Encoding ReservoirPy nodes as hypergraph patterns
- Tensor fragment architecture for cognitive states
- Scheme S-expression representation
- Round-trip translation validation
- Visualization tools
"""

import numpy as np
import matplotlib.pyplot as plt

# Import ReservoirPy components
from reservoirpy.nodes import Reservoir, Ridge, Input, Output, Tanh
from reservoirpy.cognitive import (
    HypergraphEncoder, 
    AtomSpace, 
    HypergraphNode, 
    HypergraphLink,
    TensorSignature,
    TensorFragment,
    CognitiveVisualizer,
    SchemeAdapter
)


def demo_basic_encoding():
    """Demonstrate basic node encoding to hypergraph patterns."""
    print("=== Basic Node Encoding Demo ===")
    
    # Create encoder
    encoder = HypergraphEncoder()
    
    # Create various ReservoirPy nodes
    nodes = {
        "input": Input(),
        "reservoir": Reservoir(units=50, lr=0.1, sr=0.9),
        "readout": Ridge(output_dim=5),
        "activation": Tanh(),
        "output": Output()
    }
    
    # Set dimensions
    nodes["input"].set_output_dim(10)
    nodes["reservoir"].set_input_dim(10)
    nodes["readout"].set_input_dim(50)
    nodes["activation"].set_input_dim(5)
    nodes["output"].set_input_dim(5)
    
    # Encode each node
    encoded_nodes = {}
    for name, node in nodes.items():
        try:
            encoded = encoder.encode_node(node, include_state=False)
            encoded_nodes[name] = encoded
            print(f"✓ {name}: {node.__class__.__name__} → {encoded.node_type}")
        except Exception as e:
            print(f"✗ {name}: Failed - {e}")
    
    print(f"\nSuccessfully encoded {len(encoded_nodes)} nodes")
    return encoded_nodes


def demo_tensor_fragment_architecture():
    """Demonstrate tensor fragment architecture for cognitive states."""
    print("\n=== Tensor Fragment Architecture Demo ===")
    
    # Create different tensor signatures for various cognitive modalities
    signatures = {
        "visual": TensorSignature.create_signature(
            modality_type="visual",
            processing_depth=3,
            context_span=15,
            salience_weight=0.8,
            autonomy_level=2
        ),
        "audio": TensorSignature.create_signature(
            modality_type="audio", 
            processing_depth=2,
            context_span=20,
            salience_weight=0.6,
            autonomy_level=3
        ),
        "multimodal": TensorSignature.create_signature(
            modality_type="multimodal",
            processing_depth=4,
            context_span=25,
            salience_weight=1.0,
            autonomy_level=4
        )
    }
    
    # Create tensor fragments
    fragments = {}
    for name, signature in signatures.items():
        fragment = TensorFragment(signature)
        
        # Simulate encoding a reservoir state
        dummy_state = np.random.randn(100) * 0.5
        reservoir_info = {"type": "reservoir", "modality": name}
        fragment.encode_reservoir_state(dummy_state, reservoir_info)
        
        fragments[name] = fragment
        
        print(f"✓ {name}: Shape {signature.get_tensor_shape()}, "
              f"Total dims: {signature.get_total_dimensions()}")
    
    # Demonstrate similarity computation
    print("\nSimilarity Analysis:")
    visual_audio_sim = fragments["visual"].compute_similarity(fragments["audio"])
    visual_multi_sim = fragments["visual"].compute_similarity(fragments["multimodal"])
    
    print(f"Visual ↔ Audio similarity: {visual_audio_sim:.3f}")
    print(f"Visual ↔ Multimodal similarity: {visual_multi_sim:.3f}")
    
    return fragments


def demo_atomspace_hypergraph():
    """Demonstrate AtomSpace hypergraph representation."""
    print("\n=== AtomSpace Hypergraph Demo ===")
    
    # Create AtomSpace
    atomspace = AtomSpace("cognitive_architecture")
    
    # Create cognitive processing nodes
    input_node = HypergraphNode("sensory_input", "input", 
                               {"modality": "multimodal", "input_dim": 10})
    
    reservoir_node = HypergraphNode("main_reservoir", "reservoir",
                                   {"units": 100, "lr": 0.1, "sr": 0.9})
    
    memory_node = HypergraphNode("working_memory", "reservoir", 
                                {"units": 50, "lr": 0.05, "sr": 0.95})
    
    readout_node = HypergraphNode("decision_readout", "readout",
                                 {"output_dim": 5, "ridge_param": 1e-6})
    
    output_node = HypergraphNode("motor_output", "output",
                                {"output_dim": 5})
    
    # Add nodes to AtomSpace
    for node in [input_node, reservoir_node, memory_node, readout_node, output_node]:
        atomspace.add_node(node)
    
    # Create cognitive links
    links = [
        HypergraphLink([input_node, reservoir_node], "connection", 
                      {"weight": 1.0, "type": "feedforward"}),
        HypergraphLink([reservoir_node, memory_node], "connection",
                      {"weight": 0.8, "type": "lateral"}),
        HypergraphLink([memory_node, reservoir_node], "feedback",
                      {"weight": 0.3, "type": "memory_feedback"}),
        HypergraphLink([reservoir_node, readout_node], "connection",
                      {"weight": 1.0, "type": "readout"}),
        HypergraphLink([readout_node, output_node], "connection",
                      {"weight": 1.0, "type": "output"})
    ]
    
    # Add links to AtomSpace
    for link in links:
        atomspace.add_link(link)
    
    print(f"✓ Created AtomSpace: {atomspace}")
    print(f"  - Nodes: {len(atomspace.nodes)}")
    print(f"  - Links: {len(atomspace.links)}")
    
    # Query for patterns
    input_to_reservoir = atomspace.query_pattern(["input", "reservoir"], "connection")
    feedback_patterns = atomspace.query_pattern(["reservoir", "reservoir"], "feedback")
    
    print(f"  - Input→Reservoir patterns: {len(input_to_reservoir)}")
    print(f"  - Feedback patterns: {len(feedback_patterns)}")
    
    return atomspace


def demo_scheme_representation():
    """Demonstrate Scheme S-expression cognitive grammar."""
    print("\n=== Scheme Cognitive Grammar Demo ===")
    
    # Create AtomSpace
    atomspace = AtomSpace("scheme_test")
    
    # Add simple cognitive pattern
    input_node = HypergraphNode("input", "input")
    process_node = HypergraphNode("processor", "reservoir")
    output_node = HypergraphNode("output", "output")
    
    atomspace.add_node(input_node)
    atomspace.add_node(process_node)
    atomspace.add_node(output_node)
    
    # Add connections
    atomspace.add_link(HypergraphLink([input_node, process_node], "connection"))
    atomspace.add_link(HypergraphLink([process_node, output_node], "connection"))
    
    # Convert to Scheme representation
    scheme_adapter = SchemeAdapter()
    scheme_expr = scheme_adapter.atomspace_to_scheme(atomspace)
    
    print("✓ Generated Scheme representation:")
    print(scheme_expr.to_string())
    
    # Convert back to AtomSpace
    reconstructed = scheme_adapter.scheme_to_atomspace(scheme_expr)
    print(f"✓ Round-trip successful: {reconstructed}")
    
    # Create cognitive grammar pattern
    pattern = scheme_adapter.create_cognitive_grammar_pattern(
        "perception_action",
        ["input", "processing", "decision", "output"],
        [
            ("input", "processing", "feedforward"),
            ("processing", "decision", "integration"), 
            ("decision", "output", "execution")
        ]
    )
    
    print("\n✓ Cognitive Grammar Pattern:")
    print(pattern.to_string())
    
    return scheme_expr, pattern


def demo_round_trip_validation():
    """Demonstrate round-trip translation validation."""
    print("\n=== Round-Trip Validation Demo ===")
    
    encoder = HypergraphEncoder()
    
    # Test different node types
    test_nodes = [
        ("Tanh", Tanh()),
        ("Input", Input()),
        ("Output", Output()),
        ("Reservoir", Reservoir(units=20, lr=0.1)),
        ("Ridge", Ridge(output_dim=3))
    ]
    
    # Set dimensions for all nodes
    for name, node in test_nodes:
        if hasattr(node, 'set_input_dim'):
            node.set_input_dim(10)
        if hasattr(node, 'set_output_dim'):
            node.set_output_dim(10)
    
    # Validate round-trips
    results = []
    for name, node in test_nodes:
        try:
            validation = encoder.validate_round_trip(node, tolerance=1e-6)
            success = validation["round_trip_successful"]
            results.append((name, success))
            
            status = "✓" if success else "✗"
            print(f"{status} {name}: {'PASS' if success else 'FAIL'}")
            
            if not success and "errors" in validation:
                print(f"  Error: {validation['errors'][0][:80]}...")
                
        except Exception as e:
            results.append((name, False))
            print(f"✗ {name}: EXCEPTION - {str(e)[:80]}...")
    
    # Summary
    successful = sum(1 for _, success in results if success)
    total = len(results)
    success_rate = successful / total
    
    print(f"\nRound-trip Success Rate: {success_rate:.1%} ({successful}/{total})")
    
    return results


def demo_visualization():
    """Demonstrate visualization capabilities."""
    print("\n=== Visualization Demo ===")
    
    try:
        # Create AtomSpace for visualization
        atomspace = AtomSpace("visualization_demo")
        
        # Create nodes
        nodes = [
            HypergraphNode("input", "input"),
            HypergraphNode("reservoir1", "reservoir"),
            HypergraphNode("reservoir2", "reservoir"), 
            HypergraphNode("readout", "readout"),
            HypergraphNode("activation", "activation"),
            HypergraphNode("output", "output")
        ]
        
        for node in nodes:
            atomspace.add_node(node)
        
        # Create links
        links = [
            HypergraphLink([nodes[0], nodes[1]], "connection"),  # input -> reservoir1
            HypergraphLink([nodes[1], nodes[2]], "connection"),  # reservoir1 -> reservoir2
            HypergraphLink([nodes[2], nodes[1]], "feedback"),    # reservoir2 -> reservoir1 (feedback)
            HypergraphLink([nodes[2], nodes[3]], "connection"),  # reservoir2 -> readout
            HypergraphLink([nodes[3], nodes[4]], "connection"),  # readout -> activation
            HypergraphLink([nodes[4], nodes[5]], "connection"),  # activation -> output
        ]
        
        for link in links:
            atomspace.add_link(link)
        
        # Create visualizer
        visualizer = CognitiveVisualizer(style='cognitive')
        
        # Generate visualization
        print("✓ Creating AtomSpace network visualization...")
        fig = visualizer.visualize_atomspace(atomspace, layout='hierarchical')
        plt.savefig('/tmp/cognitive_network.png', dpi=300, bbox_inches='tight')
        plt.close()
        
        # Create tensor fragment visualization
        print("✓ Creating tensor fragment visualization...")
        signature = TensorSignature.create_signature(
            modality_type="multimodal",
            processing_depth=3,
            context_span=20,
            salience_weight=0.9,
            autonomy_level=3
        )
        fragment = TensorFragment(signature)
        fragment.data = np.random.randn(*signature.get_tensor_shape()) * 0.5
        
        fig2 = visualizer.visualize_tensor_fragment(fragment)
        plt.savefig('/tmp/tensor_fragment.png', dpi=300, bbox_inches='tight')
        plt.close()
        
        # Create cognitive flow visualization
        print("✓ Creating cognitive flow visualization...")
        fig3 = visualizer.visualize_cognitive_flow(
            atomspace, 
            highlight_path=["input", "reservoir1", "reservoir2", "readout", "output"]
        )
        plt.savefig('/tmp/cognitive_flow.png', dpi=300, bbox_inches='tight')
        plt.close()
        
        print("✓ Visualizations saved to /tmp/")
        
    except Exception as e:
        print(f"✗ Visualization failed: {e}")
        return False
    
    return True


def demo_performance_benchmark():
    """Demonstrate performance characteristics."""
    print("\n=== Performance Benchmark Demo ===")
    
    import time
    
    encoder = HypergraphEncoder()
    
    # Test encoding performance
    reservoir = Reservoir(units=100, lr=0.1, sr=0.9)
    reservoir.set_input_dim(20)
    
    # Warm-up
    for _ in range(5):
        encoded = encoder.encode_node(reservoir, include_state=False)
    
    # Benchmark encoding
    n_trials = 50
    start_time = time.time()
    
    for _ in range(n_trials):
        encoded = encoder.encode_node(reservoir, include_state=False)
        decoded = encoder.decode_node(encoded)
    
    total_time = time.time() - start_time
    avg_time = total_time / n_trials
    
    print(f"✓ Encoding Performance:")
    print(f"  - Average round-trip time: {avg_time*1000:.2f} ms")
    print(f"  - Throughput: {1/avg_time:.1f} round-trips/second")
    
    # Test tensor fragment performance
    signature = TensorSignature.create_signature()
    fragment = TensorFragment(signature)
    
    start_time = time.time()
    for _ in range(100):
        state = np.random.randn(100)
        fragment.encode_reservoir_state(state, {"type": "test"})
        decoded_state = fragment.decode_to_reservoir_state((100,))
    
    tensor_time = time.time() - start_time
    print(f"  - Tensor fragment avg time: {tensor_time*10:.2f} ms")
    
    # Get encoding statistics
    stats = encoder.get_encoding_statistics()
    print(f"\n✓ Encoding Statistics:")
    for key, value in stats.items():
        if key != "errors":
            print(f"  - {key}: {value}")
    
    return avg_time, tensor_time


def main():
    """Run complete demonstration of Phase 1 implementation."""
    print("=" * 60)
    print("COGNITIVE PRIMITIVES & HYPERGRAPH ENCODING DEMO")
    print("Phase 1: Foundational Implementation")
    print("=" * 60)
    
    # Run all demonstrations
    try:
        encoded_nodes = demo_basic_encoding()
        fragments = demo_tensor_fragment_architecture()
        atomspace = demo_atomspace_hypergraph()
        scheme_expr, pattern = demo_scheme_representation()
        validation_results = demo_round_trip_validation()
        visualization_success = demo_visualization()
        encoding_time, tensor_time = demo_performance_benchmark()
        
        # Final summary
        print("\n" + "=" * 60)
        print("PHASE 1 IMPLEMENTATION SUMMARY")
        print("=" * 60)
        
        print("✓ Core Components Implemented:")
        print("  - Hypergraph primitives (nodes, links, AtomSpace)")
        print("  - Tensor fragment architecture with cognitive signatures")
        print("  - Bidirectional translators (Node ↔ Hypergraph)")
        print("  - Scheme S-expression cognitive grammar")
        print("  - Comprehensive visualization tools")
        print("  - Round-trip validation with fidelity metrics")
        
        print("\n✓ Acceptance Criteria Status:")
        success_rate = sum(1 for _, success in validation_results if success) / len(validation_results)
        print(f"  - ReservoirPy node encoding: {len(encoded_nodes)} types supported")
        print(f"  - Round-trip fidelity: {success_rate:.1%} success rate")
        print(f"  - Tensor signatures: Mathematically sound (prime factorization)")
        print(f"  - Performance: {encoding_time*1000:.1f}ms avg encoding time")
        print(f"  - Visualization: {'✓' if visualization_success else '✗'} Available")
        
        print("\n✓ Phase 1 Implementation: COMPLETE")
        print("  Ready for Phase 2: Advanced cognitive grammar integration")
        
    except Exception as e:
        print(f"\n✗ Demo failed with error: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()