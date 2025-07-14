"""
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
"""

from .core import GGMLTensor, GGMLContext, TensorType, SymbolicAnnotation
from .kernels import SymbolicKernel, hypergraph_conv, symbolic_activation, attention_weighted_op
from .bridge import NeuralSymbolicBridge
from .reservoir_backend import GGMLReservoir, SymbolicReadout

__all__ = [
    # Core components
    "GGMLTensor",
    "GGMLContext", 
    "TensorType",
    "SymbolicAnnotation",
    "SymbolicKernel",
    "NeuralSymbolicBridge",
    # Symbolic operations
    "hypergraph_conv",
    "symbolic_activation", 
    "attention_weighted_op",
    # ReservoirPy integration
    "GGMLReservoir",
    "SymbolicReadout",
]