"""
==================================================
Phase 5: Recursive Meta-Cognition & Evolutionary Optimization
==================================================

This module enables the system to observe, analyze, and recursively improve itself
using evolutionary algorithms, with ReservoirPy serving as both the computational
substrate and the target for optimization.

.. currentmodule:: reservoirpy.cognitive.meta_optimization

Core Components
===============

.. autosummary::
   :toctree: generated/
   :template: autosummary/class.rst

   MetaCognitiveSystem - Main meta-cognitive analysis and feedback system
   EvolutionaryOptimizer - Evolutionary algorithms for reservoir optimization
   PerformanceBenchmarker - Continuous performance monitoring and metrics
   RecursiveImprovement - Recursive feedback loops for self-improvement

Meta-Cognitive Components
========================

.. autosummary::
   :toctree: generated/
   :template: autosummary/class.rst

   ReservoirIntrospector - Reservoir performance monitoring and analysis
   MetaAttentionAllocator - Meta-cognitive attention allocation system
   SelfAnalysisModule - Feedback-driven self-analysis capabilities

Evolutionary Components
======================

.. autosummary::
   :toctree: generated/
   :template: autosummary/class.rst

   TopologyEvolver - Evolutionary optimization of reservoir topology
   ParameterEvolver - Adaptive evolution of spectral radius and learning rates
   ConnectionEvolver - Genetic programming for reservoir connection patterns
   ArchitectureMutator - Real-time reservoir architecture mutation system

Optimization Components
======================

.. autosummary::
   :toctree: generated/
   :template: autosummary/class.rst

   ContinuousBenchmarker - Real-time performance benchmarking system
   SelfTuner - Adaptive self-tuning of kernels and attention mechanisms
   MultiObjectiveOptimizer - Multi-objective optimization for cognitive trade-offs
   FitnessEvaluator - Fitness evaluation for evolutionary optimization

Recursive Improvement
====================

.. autosummary::
   :toctree: generated/
   :template: autosummary/class.rst

   EmbodiedFeedback - Feedback mechanisms from embodied agent performance
   EnsembleEvolver - Reservoir ensemble evolution for improved robustness
   MetaLearner - Meta-learning algorithms for rapid adaptation
   HierarchicalOptimizer - Hierarchical optimization across cognitive mesh layers
"""

from .meta_cognitive import (
    MetaCognitiveSystem, ReservoirIntrospector, MetaAttentionAllocator, 
    SelfAnalysisModule
)
from .evolutionary import (
    EvolutionaryOptimizer, TopologyEvolver, ParameterEvolver, 
    ConnectionEvolver, ArchitectureMutator
)
from .benchmarking import (
    PerformanceBenchmarker, ContinuousBenchmarker, SelfTuner,
    MultiObjectiveOptimizer, FitnessEvaluator
)
from .recursive_loops import (
    RecursiveImprovement, EmbodiedFeedback, EnsembleEvolver,
    MetaLearner, HierarchicalOptimizer
)

__all__ = [
    # Core components
    "MetaCognitiveSystem",
    "EvolutionaryOptimizer", 
    "PerformanceBenchmarker",
    "RecursiveImprovement",
    # Meta-cognitive components
    "ReservoirIntrospector",
    "MetaAttentionAllocator",
    "SelfAnalysisModule",
    # Evolutionary components
    "TopologyEvolver",
    "ParameterEvolver",
    "ConnectionEvolver", 
    "ArchitectureMutator",
    # Optimization components
    "ContinuousBenchmarker",
    "SelfTuner",
    "MultiObjectiveOptimizer",
    "FitnessEvaluator",
    # Recursive improvement
    "EmbodiedFeedback",
    "EnsembleEvolver",
    "MetaLearner",
    "HierarchicalOptimizer",
]