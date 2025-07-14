# GGML Neural-Symbolic API Reference

## Module: `reservoirpy.cognitive.ggml`

### Overview

GGML Interface for Neural-Symbolic Computation
==============================================

Lightweight ggml-compatible interface for neural-symbolic tensor operations
integrated with ReservoirPy's cognitive computing framework.

This module provides custom kernels and tensor operations that bridge
neural computation (via ReservoirPy) with symbolic reasoning (via AtomSpace
and hypergraph representations).

.. currentmodule:: reservoirpy.cognitive.ggml

Core Components
===============

.. autosummary::
   :toctree: generated/
   :template: autosummary/class.rst

   GGMLTensor - Core tensor representation with symbolic annotations
   GGMLContext - Computation context for neural-symbolic operations
   SymbolicKernel - Custom kernels for symbolic tensor operations
   NeuralSymbolicBridge - Bridge between ReservoirPy and symbolic computation

Symbolic Operations
==================

.. autosummary::
   :toctree: generated/
   :template: autosummary/function.rst

   hypergraph_conv - Hypergraph convolution operations
   symbolic_activation - Symbolic activation functions
   attention_weighted_op - Attention-weighted tensor operations
   pattern_matching - Symbolic pattern matching kernels

ReservoirPy Integration
======================

.. autosummary::
   :toctree: generated/
   :template: autosummary/class.rst

   GGMLReservoir - GGML-backed reservoir computing node
   SymbolicReadout - Symbolic reasoning readout layer


### Classes and Functions

#### `GGMLContext`

**Type:** Class

**Description:** 
    Computation context for neural-symbolic operations.
    
    Manages the computational graph, memory allocation, and symbolic reasoning
    context for GGML tensor operations.
    
    Parameters
    ----------
    mem_size : int
        Memory size for tensor operations (bytes)
    atom_space : AtomSpace, optional
        AtomSpace for symbolic reasoning
    enable_gradients : bool
        Whether to compute gradients
    

**Methods:**
- `create_tensor()`: 
        Create a new tensor in this context.
        
        Parameters
        ----------
       ...
- `get_computation_stats()`: Get computation statistics....
- `get_memory_usage()`: Get current memory usage statistics....
- `symbolic_op()`: 
        Execute a symbolic operation on tensors.
        
        Parameters
        ----------
   ...

#### `GGMLReservoir`

**Type:** Class

**Description:** 
    GGML-backed reservoir computing node with neural-symbolic capabilities.
    
    Extends ReservoirPy's Node architecture with GGML tensor operations
    and symbolic reasoning capabilities.
    
    Parameters
    ----------
    units : int
        Number of reservoir units
    ggml_context : GGMLContext
        GGML computation context
    atom_space : AtomSpace, optional
        AtomSpace for symbolic representation
    symbolic_enhancement : bool
        Whether to apply symbolic enhancement
    lr : float
        Leak rate for reservoir dynamics
    sr : float
        Spectral radius for reservoir initialization
    input_scaling : float
        Input scaling factor
    **kwargs
        Additional parameters for reservoir initialization
    

**Methods:**
- `call()`: 
        Forward pass through GGML reservoir.
        
        Parameters
        ----------
       ...
- `clean_buffers()`: Clean Node's buffer arrays....
- `copy()`: Returns a copy of the Node.

        Parameters
        ----------
        name : str
            Na...
- `create_buffer()`: Create a buffer array on disk, using numpy.memmap. This can be
        used to store transient varia...

#### `GGMLTensor`

**Type:** Class

**Description:** 
    GGML-compatible tensor with symbolic annotations.
    
    A tensor that combines numerical computation with symbolic representations,
    enabling neural-symbolic reasoning in reservoir computing.
    
    Parameters
    ----------
    data : np.ndarray
        Numerical tensor data
    tensor_type : TensorType
        Type of tensor (neural, symbolic, or hybrid)
    symbolic_annotation : SymbolicAnnotation, optional
        Symbolic annotation linking to hypergraph
    tensor_signature : TensorSignature, optional
        Cognitive tensor signature
    name : str, optional
        Human-readable name for the tensor
    

**Methods:**
- `matmul()`: Matrix multiplication with symbolic tracking....
- `to_symbolic()`: 
        Convert tensor to symbolic representation.
        
        Parameters
        ----------
 ...

#### `NeuralSymbolicBridge`

**Type:** Class

**Description:** 
    Bridge between neural computation (ReservoirPy) and symbolic reasoning (AtomSpace).
    
    This class provides seamless integration between reservoir computing and
    symbolic AI, enabling hybrid neural-symbolic computation.
    
    Parameters
    ----------
    ggml_context : GGMLContext
        GGML computation context
    atom_space : AtomSpace
        AtomSpace for symbolic reasoning
    attention_system : ECANAttentionSystem, optional
        ECAN attention system for resource allocation
    

**Methods:**
- `apply_symbolic_reasoning()`: 
        Apply symbolic reasoning to tensor data.
        
        Parameters
        ----------
   ...
- `create_hybrid_computation()`: 
        Create hybrid neural-symbolic computation pipeline.
        
        Parameters
        ---...
- `create_tensor_signature_from_reservoir()`: 
        Create tensor signature from reservoir computation.
        
        Parameters
        ---...
- `get_bridge_statistics()`: Get comprehensive statistics about bridge operations....
- `integrate_with_attention_reservoir()`: 
        Integrate GGML symbolic reasoning with AttentionReservoir.
        
        Parameters
    ...

#### `SymbolicAnnotation`

**Type:** Class

**Description:** 
    Symbolic annotation for tensor data linking to hypergraph representations.
    
    Attributes
    ----------
    atom_space_ref : str
        Reference to AtomSpace representation
    hypergraph_nodes : List[str]
        Associated hypergraph node IDs
    semantic_context : Dict[str, Any]
        Semantic context information
    attention_weight : float
        Attention weight for this annotation
    

**Methods:**

#### `SymbolicKernel`

**Type:** Class

**Description:** 
    Abstract base class for symbolic tensor operation kernels.
    
    Symbolic kernels implement neural-symbolic operations that bridge
    numerical computation with symbolic reasoning.
    

**Methods:**
- `forward()`: Execute the forward pass of the kernel....

#### `SymbolicReadout`

**Type:** Class

**Description:** 
    Symbolic reasoning readout layer with GGML backend.
    
    Implements readout operations with symbolic interpretation capabilities,
    enabling explicit reasoning about reservoir dynamics.
    
    Parameters
    ----------
    output_dim : int
        Output dimension
    ggml_context : GGMLContext, optional
        GGML computation context
    symbolic_interpretation : bool
        Whether to provide symbolic interpretation
    ridge : float
        Ridge regression regularization parameter
    **kwargs
        Additional parameters
    

**Methods:**
- `call()`: 
        Forward pass through symbolic readout.
        
        Parameters
        ----------
     ...
- `clean_buffers()`: Clean Node's buffer arrays....
- `copy()`: Returns a copy of the Node.

        Parameters
        ----------
        name : str
            Na...
- `create_buffer()`: Create a buffer array on disk, using numpy.memmap. This can be
        used to store transient varia...

#### `TensorType`

**Type:** Class

**Description:** Tensor type enumeration for neural-symbolic operations.

**Methods:**

#### `attention_weighted_op()`

**Type:** Function

**Description:** Attention-weighted tensor operation.

#### `hypergraph_conv()`

**Type:** Function

**Description:** Hypergraph convolution operation.

#### `symbolic_activation()`

**Type:** Function

**Description:** Symbolic activation function.

