"""
====================================
ECAN Attention Allocation System
====================================

This module implements Economic Cognitive Attention Networks (ECAN) for dynamic
attention allocation and resource management in ReservoirPy cognitive architectures.

Core ECAN Components
===================

.. autosummary::
   :toctree: generated/
   :template: autosummary/class.rst

   ECANAttentionSystem - Main ECAN attention allocation and spreading system
   AttentionValue - Attention value container (STI/LTI)
   ResourceAllocator - Economic attention markets and resource scheduling
   AttentionReservoir - Attention-aware reservoir node implementation

Dynamic Mesh Integration
=======================

.. autosummary::
   :toctree: generated/
   :template: autosummary/class.rst

   MeshTopology - Dynamic mesh topology management
   AttentionFlow - Attention cascade and propagation algorithms
   TopologyModifier - Adaptive topology modification based on attention

Scheduling & Economics
=====================

.. autosummary::
   :toctree: generated/
   :template: autosummary/class.rst

   AttentionMarket - Economic attention market mechanisms
   ResourceScheduler - Real-time cognitive resource scheduling
   AttentionBank - Attention banking and lending operations
"""

from .ecan import ECANAttentionSystem, AttentionValue
from .resource_allocator import ResourceAllocator, AttentionMarket, ResourceScheduler, AttentionBank
from .attention_reservoir import AttentionReservoir, AttentionFlow
from .mesh_topology import MeshTopology, TopologyModifier

__all__ = [
    # Core ECAN components
    "ECANAttentionSystem",
    "AttentionValue", 
    "ResourceAllocator",
    "AttentionReservoir",
    
    # Economic mechanisms
    "AttentionMarket",
    "ResourceScheduler", 
    "AttentionBank",
    
    # Dynamic mesh
    "MeshTopology",
    "TopologyModifier",
    "AttentionFlow",
]