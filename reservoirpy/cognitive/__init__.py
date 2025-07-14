"""
============================================
Cognitive Primitives & Hypergraph Encoding
============================================

This module provides foundational cognitive encoding capabilities for ReservoirPy,
including bidirectional translation between ReservoirPy primitives and hypergraph patterns.

.. currentmodule:: reservoirpy.cognitive

Core Components
===============

.. autosummary::
   :toctree: generated/
   :template: autosummary/class.rst

   HypergraphEncoder - Main encoder for ReservoirPy → Hypergraph translation
   SchemeAdapter - Scheme-based cognitive grammar adapter
   TensorFragment - Tensor fragment architecture for cognitive states
   CognitiveVisualizer - Visualization tools for hypergraph dynamics

Hypergraph Primitives
====================

.. autosummary::
   :toctree: generated/
   :template: autosummary/class.rst

   HypergraphNode - Basic hypergraph node representation
   HypergraphLink - Hypergraph link connecting multiple nodes
   AtomSpace - Simple AtomSpace-like hypergraph container

Translation Components
=====================

.. autosummary::
   :toctree: generated/
   :template: autosummary/class.rst

   NodeTranslator - Translator for ReservoirPy Node objects
   StateTranslator - Translator for reservoir states
   ModelTranslator - Translator for ReservoirPy Model objects
"""

from .hypergraph import AtomSpace, HypergraphLink, HypergraphNode
from .encoder import HypergraphEncoder
from .scheme_adapter import SchemeAdapter
from .tensor_fragment import TensorFragment, TensorSignature
from .translator import NodeTranslator, StateTranslator, ModelTranslator
from .visualizer import CognitiveVisualizer

# Phase 2: ECAN Attention Allocation & Resource Kernel Construction
from .attention import (
    ECANAttentionSystem, AttentionValue, ResourceAllocator, AttentionReservoir,
    AttentionMarket, ResourceScheduler, AttentionBank, MeshTopology, 
    TopologyModifier, AttentionFlow
)

# Phase 3: Neural-Symbolic Synthesis via Custom ggml Kernels  
from .ggml import (
    GGMLTensor, GGMLContext, SymbolicKernel, NeuralSymbolicBridge,
    hypergraph_conv, symbolic_activation, attention_weighted_op,
    GGMLReservoir, SymbolicReadout
)

__all__ = [
    # Core components
    "HypergraphEncoder",
    "SchemeAdapter", 
    "TensorFragment",
    "TensorSignature",
    "CognitiveVisualizer",
    # Hypergraph primitives
    "HypergraphNode",
    "HypergraphLink", 
    "AtomSpace",
    # Translation components
    "NodeTranslator",
    "StateTranslator",
    "ModelTranslator",
    # Phase 2: ECAN Attention components
    "ECANAttentionSystem",
    "AttentionValue",
    "ResourceAllocator", 
    "AttentionReservoir",
    "AttentionMarket",
    "ResourceScheduler",
    "AttentionBank",
    "MeshTopology",
    "TopologyModifier",
    "AttentionFlow",
    # Phase 3: GGML Neural-Symbolic components
    "GGMLTensor",
    "GGMLContext",
    "SymbolicKernel", 
    "NeuralSymbolicBridge",
    "hypergraph_conv",
    "symbolic_activation",
    "attention_weighted_op",
    "GGMLReservoir",
    "SymbolicReadout",
]